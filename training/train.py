"""Train the multi-task (classification + segmentation) model.

  python -m training.train --config training/configs/mri.yaml --seed 0
  python -m training.train --config training/configs/mri.yaml --seed 0 --scheme official   # leakage experiment
  python -m training.train --config training/configs/mri.yaml --limit 64 --epochs 1          # smoke test

Split schemes
  grouped   our leakage-controlled split (manifest column `split`): train / val / test + externals
  official  BRISC's published split: official train (minus a grouped 15% val) -> evaluated on the
            official test set AND on its leakage-free subset (`official_test_clean`)

Loss = CE(logits, y) + w_seg * [(focal) BCE + soft-Dice](seg, mask | mask known)
       + w_seg * w_aux * same loss on the coarsest FPN level (deep supervision, v1)
       + w_guid * (1 - energy of CAM_y inside the tumor mask)        [ablation only, Rao et al. 2023]
Optimisation: AdamW (discriminative LRs), linear warm-up + cosine decay per step, bf16 autocast,
EMA of weights (evaluated + saved), early stopping on validation macro-F1.
Outputs (outside OneDrive): <RUNS>/<task>_<scheme>/seed<k>/{best.pt, history.json, metrics.json,
pred_<split>.npz}. Prediction files hold logits, labels, Dice and multi-stage GAP embeddings for
the calibration / conformal / OOD / XAI stages.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from neuroedge.config import TaskConfig, load_config
from neuroedge.metrics import classification_metrics
from neuroedge.models import MultiTaskNet
from training.data.paths import DATA_ROOT, RUNS, manifest_path
from training.dataset import TumorDataset, build_cache


# ------------------------------------------------------------------------------ splits
def group_holdout(labels: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    _, hold = next(StratifiedGroupKFold(max(2, round(1 / frac)), shuffle=True, random_state=seed)
                   .split(np.zeros(len(labels)), labels, groups))
    m = np.zeros(len(labels), bool)
    m[hold] = True
    return m


def _as_bool(s: pd.Series) -> pd.Series:
    """CSV booleans arrive as strings; astype(bool) would turn "False" into True."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


def make_splits(df: pd.DataFrame, scheme: str) -> dict[str, pd.DataFrame]:
    ext = {s: g for s, g in df.groupby("split") if s.startswith("external_")}
    if scheme == "grouped":
        out = {"train": df[df.split == "train"], "val": df[df.split == "val"], "test": df[df.split == "test"]}
        if (df.split == "cal").any():          # split v2: dedicated calibration split
            out["cal"] = df[df.split == "cal"]
        return {**out, **ext}
    if scheme == "official":
        b = df[df.dataset == "brisc"]
        tr = b[b.source_split == "train"]
        is_val = group_holdout(tr.label.to_numpy(), tr.group.to_numpy(), 0.15, seed=7)
        te = b[b.source_split == "test"]
        return {"train": tr[~is_val], "val": tr[is_val], "test_official": te,
                "test_official_clean": te[_as_bool(te.official_test_clean)], **ext}
    raise ValueError(scheme)


# ------------------------------------------------------------------------------- losses
def seg_loss(logits: torch.Tensor, target: torch.Tensor, has_mask: torch.Tensor,
             focal_gamma: float = 0.0) -> torch.Tensor:
    """(focal) BCE + soft Dice over images with a known mask (tumour annotation or no_tumor = empty).
    focal_gamma > 0 down-weights the easy background pixels (Lin et al. ICCV'17): tumours are ~1.5% of
    the image, so plain BCE is dominated by background."""
    sel = has_mask > 0
    if not sel.any():
        return logits.sum() * 0.0
    lg, tg = logits[sel].float(), target[sel]
    if focal_gamma > 0:
        ce = F.binary_cross_entropy_with_logits(lg, tg, reduction="none")
        pt = torch.exp(-ce)
        bce = ((1 - pt) ** focal_gamma * ce).mean()
    else:
        bce = F.binary_cross_entropy_with_logits(lg, tg)
    p = torch.sigmoid(lg)
    inter = (p * tg).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + tg.sum((1, 2, 3)) + 1)  # smooth=1: empty masks OK
    return bce + dice.mean()


def guidance_loss(features, weight, labels, mask, has_tumor_mask) -> torch.Tensor:
    """1 - fraction of positive CAM energy inside the tumor mask (energy loss, Rao et al. ICCV'23)."""
    sel = has_tumor_mask > 0
    if not sel.any():
        return features.sum() * 0.0
    cam = torch.relu(torch.einsum("bkhw,bk->bhw", features[sel].float(), weight[labels[sel]].float()))
    cam = F.interpolate(cam[:, None], size=mask.shape[-2:], mode="bilinear", align_corners=False)
    inside = (cam * mask[sel]).sum((1, 2, 3)) / (cam.sum((1, 2, 3)) + 1e-6)
    return (1 - inside).mean()


# ---------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, loader, device, amp: bool = False) -> dict[str, np.ndarray]:
    """amp=True (bf16) only for the per-epoch validation signal; reported metrics use FP32,
    matching the exported FP32 graph the INT8 model is derived from."""
    model.eval()
    logits, labels, dice, idx, embs = [], [], [], [], [[], [], [], []]
    for b in loader:
        x = b["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(x)
        logits.append(out["logits"].float().cpu())
        labels.append(b["label"])
        idx.append(b["index"])
        for k, e in enumerate(out["embeddings"]):
            embs[k].append(e.float().cpu().half())
        if "seg_logits" in out:
            p = (torch.sigmoid(out["seg_logits"].float()) > 0.5).float().cpu()
            t = b["mask"]
            inter, tot = (p * t).sum((1, 2, 3)), p.sum((1, 2, 3)) + t.sum((1, 2, 3))
            d = torch.where(t.sum((1, 2, 3)) > 0, 2 * inter / tot.clamp(min=1), torch.full_like(inter, float("nan")))
            dice.append(d)
    res = {"logits": torch.cat(logits).numpy(), "labels": torch.cat(labels).numpy(),
           "index": torch.cat(idx).numpy()}
    for k in range(4):
        res[f"emb{k + 1}"] = torch.cat(embs[k]).numpy()
    if dice:
        res["dice"] = torch.cat(dice).numpy()
    return res


def summarize(pred: dict, class_names) -> dict:
    probs = torch.softmax(torch.from_numpy(pred["logits"]), 1).numpy()
    m = classification_metrics(pred["labels"], probs, list(class_names))
    if "dice" in pred and np.isfinite(pred["dice"]).any():
        m["seg_dice_mean"] = float(np.nanmean(pred["dice"]))
        m["seg_dice_n"] = int(np.isfinite(pred["dice"]).sum())
    return m


# --------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheme", default="grouped", choices=["grouped", "official"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="subsample each split (smoke test)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config field, e.g. --set model.attn_eps=0.005 (repeatable)")
    args = ap.parse_args()

    cfg: TaskConfig = load_config(args.config)
    for item in args.set:                               # ablations without one YAML per variant
        key, val = item.split("=", 1)
        sect, field = key.split(".", 1)
        obj = getattr(cfg, sect)
        old = getattr(obj, field)                       # AttributeError on typos: fail loudly
        setattr(obj, field, type(old)(val) if not isinstance(old, bool) else val.lower() in ("1", "true"))
        print(f"override {key} = {getattr(obj, field)!r}")
    tc = cfg.train
    epochs = args.epochs or tc.epochs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(manifest_path(cfg), keep_default_na=False)
    splits = make_splits(df, args.scheme)
    if args.limit:
        splits = {k: v.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, args.limit // cfg.num_classes)), random_state=0))
            for k, v in splits.items()}
    # Canonical (sorted) id list -> one cache per distinct image set, independent of split scheme.
    needed = pd.concat(splits.values()).drop_duplicates("id").sort_values("id")
    set_hash = hashlib.md5("\n".join(needed["id"]).encode()).hexdigest()[:10]
    cache = build_cache(needed, cfg.preprocess, DATA_ROOT / "cache", f"{cfg.name}_{set_hash}")

    out_dir = RUNS / f"{cfg.name}_{args.scheme}{args.tag}" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = tc.num_workers if args.workers is None else args.workers
    # Evaluation datasets never augment — including the train split (its embeddings fit the OOD model).
    ds = {k: TumorDataset(v, cache, cfg.class_names, cfg.preprocess, train=False) for k, v in splits.items()}
    train_ds = TumorDataset(splits["train"], cache, cfg.class_names, cfg.preprocess, train=True, seed=args.seed)
    dl_kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, drop_last=True,
                          generator=torch.Generator().manual_seed(args.seed), **dl_kw)
    # val is evaluated every epoch -> persistent workers (Windows spawns are slow); others run once.
    eval_dl = {k: DataLoader(v, batch_size=64, shuffle=False, num_workers=min(workers, 2), pin_memory=True,
                             persistent_workers=(k == "val" and workers > 0))
               for k, v in ds.items()}
    print(f"device={device} | " + " | ".join(f"{k}={len(v)}" for k, v in ds.items()))

    model = MultiTaskNet(cfg.num_classes, cfg.model).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.param_groups(tc.lr_backbone, tc.lr_head, tc.weight_decay))
    steps = epochs * len(train_dl)
    warm = min(tc.warmup_epochs * len(train_dl), steps // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tc.ema_decay), use_buffers=True)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=tc.amp and device.type == "cuda")
    no_tumor_idx = torch.tensor([i for i, c in enumerate(cfg.class_names) if c in ("no_tumor", "healthy")],
                                dtype=torch.long, device=device)

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), {"ce": 0.0, "seg": 0.0, "guid": 0.0}, 0
        for b in train_dl:
            x = b["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y, m, hm = (b[k].to(device, non_blocking=True) for k in ("label", "mask", "has_mask"))
            with torch.autocast(**amp):
                out = model(x)
            ce = F.cross_entropy(out["logits"].float(), y)
            sl = seg_loss(out["seg_logits"], m, hm, tc.seg_focal_gamma) if "seg_logits" in out else ce * 0
            if "seg_aux_logits" in out:
                sl = sl + cfg.model.seg_aux_weight * seg_loss(out["seg_aux_logits"], m, hm, tc.seg_focal_gamma)
            gl = ce * 0
            if tc.guidance_loss_weight > 0:
                tumor = hm * (~torch.isin(y, no_tumor_idx)).float() * (m.sum((1, 2, 3)) > 0).float()
                gl = guidance_loss(out["features"], model.classifier.weight, y, m, tumor)
            loss = ce + tc.seg_loss_weight * sl + tc.guidance_loss_weight * gl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema.update_parameters(model)
            bs = len(y)
            n += bs
            for k, v in (("ce", ce), ("seg", sl), ("guid", gl)):
                tot[k] += float(v) * bs

        val = summarize(predict(ema.module, eval_dl["val"], device, amp=tc.amp), cfg.class_names)
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "time_s": round(time.time() - t0, 1),
               **{f"train_{k}": v / n for k, v in tot.items()},
               "val_macro_f1": val["macro_f1"], "val_acc": val["accuracy"], "val_dice": val.get("seg_dice_mean")}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))   # survive crashes
        if val["macro_f1"] > best:
            best, best_epoch = val["macro_f1"], epoch
            torch.save({"state_dict": ema.module.state_dict(), "class_names": cfg.class_names,
                        "config": dataclasses.asdict(cfg), "preprocess_fingerprint": cfg.preprocess.fingerprint(),
                        "epoch": epoch, "seed": args.seed, "scheme": args.scheme}, out_dir / "best.pt")
        elif epoch - best_epoch >= tc.early_stop_patience:
            print(f"early stop at epoch {epoch} (best {best_epoch})")
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Final evaluation of the best EMA checkpoint on every split
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    metrics = {"best_epoch": best_epoch, "seed": args.seed, "scheme": args.scheme}
    for k, dl in eval_dl.items():
        pred = predict(model, dl, device)
        pred["ids"] = np.array(ds[k].ids)
        np.savez_compressed(out_dir / f"pred_{k}.npz", **pred)
        metrics[k] = summarize(pred, cfg.class_names)
        print(f"{k:>22}: acc={metrics[k]['accuracy']:.4f} macroF1={metrics[k]['macro_f1']:.4f} "
              f"AUROC={metrics[k]['macro_auroc']:.4f}" +
              (f" dice={metrics[k]['seg_dice_mean']:.3f}" if "seg_dice_mean" in metrics[k] else ""))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()

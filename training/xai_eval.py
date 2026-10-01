"""Quantitative evaluation of explanation methods on held-out images with expert tumor masks.

  python -m training.xai_eval --run <RUNS>/mri_grouped/seed0 [--split test] [--n-per-class 40]

Methods (all explain the model's PREDICTED class, computed on logits, FP32):
  CAM (exact)   = HiResCAM; ReLU of it = Grad-CAM (identity proven in tests/test_cam.py)
  Finer-CAM     S_pred - S_runner-up (Zhang et al., CVPR 2025), gamma = 1
  Grad-CAM++    Chattopadhay et al., WACV 2018 (on logits, as its derivation requires)
  LayerCAM      layer3 + layer4 fused (Jiang et al., TIP 2021)
  ShapleyCAM    gradient + Hessian-vector product with the ReST target (Cai et al., 2025);
                with a plain logit target the Hessian of a GAP->Linear head is 0 and it would
                reduce exactly to Grad-CAM
  Score-CAM     gradient-free (Wang et al., CVPRW 2020) — opt-in (--with-scorecam): 2048 forward
                passes per image, so it cannot run on the Pi and is not an app candidate
  Purpose       compare the cheap candidates and select <= 3 for the app on measured evidence
  Seg head      the segmentation decoder's tumor probability (a localiser, NOT an explanation;
                reported as the localization reference)
Metrics
  localization  pointing game, energy-based pointing (fraction of evidence inside the mask),
                top-15% IoU and Dice vs. the expert mask
  faithfulness  ROAD combined score (Rong et al., ICML 2022): (LeRF - MoRF)/2 of softmax confidence
                with noisy-linear imputation; higher = more faithful
  sanity        cascading model-parameter randomization (Adebayo et al., NeurIPS 2018): Spearman /
                SSIM between original and randomized-model maps; must DROP towards 0
All metrics: mean with 95% bootstrap CI over images.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuroedge import cam as necam
from neuroedge.config import PreprocessConfig
from neuroedge.metrics import bootstrap_ci, energy_pointing, map_similarity, pointing_game, topk_overlap
from neuroedge.preprocess import normalize
from training.data.paths import DATA_ROOT, MANIFESTS
from training.export import load_run


class LogitsOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["logits"]


def load_split(bundle_cfg: dict, class_names, split: str, n_per_class: int, seed: int):
    pcfg = PreprocessConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in bundle_cfg["preprocess"].items()})
    df = pd.read_csv(MANIFESTS / f"{bundle_cfg['name']}.csv", keep_default_na=False)
    df = df[(df["split"] == split) & (df["mask_path"] != "") & (df["label"] != "no_tumor")]
    for cache in sorted((DATA_ROOT / "cache").glob(f"*_{pcfg.fingerprint()}")):
        ids = (cache / "ids.txt").read_text(encoding="utf-8").splitlines()
        pos = {k: i for i, k in enumerate(ids)}
        if all(i in pos for i in df["id"]):
            imgs = np.load(cache / "images.npy", mmap_mode="r")
            masks = np.load(cache / "masks.npy", mmap_mode="r")
            sel = [pos[i] for i in df["id"]]
            x = np.stack([normalize(imgs[i].astype(np.float32) / 255.0, pcfg) for i in sel])
            return (x, np.stack([masks[i] > 0 for i in sel]).astype(np.uint8),
                    np.stack([imgs[i] for i in sel]),
                    np.array([class_names.index(l) for l in df["label"]]), df["id"].tolist())
    raise FileNotFoundError("cache with the split's images not found")


@torch.no_grad()
def forward_all(net, x, device, bs=32):
    outs = {"logits": [], "features": [], "seg": []}
    for s in range(0, len(x), bs):
        o = net(torch.from_numpy(x[s:s + bs]).to(device))
        outs["logits"].append(o["logits"].float().cpu())
        outs["features"].append(o["features"].float().cpu())
        outs["seg"].append(torch.sigmoid(o["seg_logits"].float()).cpu())
    return {k: torch.cat(v).numpy() for k, v in outs.items()}


def lib_cams(method_cls, wrapper, layers, x, targets, device, bs=16, **attrs):
    out = []
    with method_cls(model=wrapper, target_layers=layers) as m:
        for k, v in attrs.items():
            setattr(m, k, v)
        for s in range(0, len(x), bs):
            out.append(m(input_tensor=torch.from_numpy(x[s:s + bs]).to(device), targets=targets[s:s + bs]))
    return np.concatenate(out)


def exact_maps(feats, w, cls, size, finer_ref=None):
    maps = []
    for i in range(len(feats)):
        m = necam.finer_cam(feats[i], w, cls[i], finer_ref[i]) if finer_ref is not None else necam.exact_cam(feats[i], w, cls[i])
        maps.append(necam.scale_for_display(necam.upsample(m, size)))
    return np.stack(maps)


def randomize(module: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def ci(vals) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m, lo, hi = bootstrap_ci(lambda idx: float(v[idx].mean()), len(v), n_boot=1000)
    return {"mean": m, "lo": lo, "hi": hi, "n": int(len(v))}


def main() -> None:
    import sys

    from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, ScoreCAM, ShapleyCAM
    from pytorch_grad_cam.metrics.road import ROADCombined
    from pytorch_grad_cam.utils.model_targets import (ClassifierOutputReST, ClassifierOutputSoftmaxTarget,
                                                      ClassifierOutputTarget)

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--road-percentiles", default="20,40,60,80")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-scorecam", action="store_true", help="also evaluate Score-CAM (slow)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = load_run(args.run)
    net = net.to(device).eval()
    class_names = list(ckpt["class_names"])
    w, _ = net.head_weights()
    x, gt, raw, y, ids = load_split(ckpt["config"], class_names, args.split, 0, args.seed)
    o = forward_all(net, x, device)
    pred = o["logits"].argmax(1)

    # explain correctly classified tumor images (balanced subsample per class)
    rng = np.random.default_rng(args.seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero((y == c) & (pred == c))
        keep += sorted(rng.choice(idx, min(args.n_per_class, len(idx)), replace=False).tolist())
    keep = np.array(keep)
    x, gt, raw, y, pred = x[keep], gt[keep], raw[keep], y[keep], pred[keep]
    feats, seg, logits = o["features"][keep], o["seg"][keep][:, 0], o["logits"][keep]
    size = x.shape[-1]
    ref = np.array([necam.most_similar_class(l, p) for l, p in zip(logits, pred)])
    print(f"{len(keep)} correctly classified tumor images with masks ({args.split})", flush=True)

    wrapper = LogitsOnly(net).eval()
    L3, L4 = net.encoder.layer3, net.encoder.layer4
    t_logit = [ClassifierOutputTarget(int(c)) for c in pred]
    maps = {
        "CAM (exact) / Grad-CAM": exact_maps(feats, w, pred, size),
        "Finer-CAM": exact_maps(feats, w, pred, size, finer_ref=ref),
        "Grad-CAM++": lib_cams(GradCAMPlusPlus, wrapper, [L4], x, t_logit, device),
        "LayerCAM (L3+L4)": lib_cams(LayerCAM, wrapper, [L3, L4], x, t_logit, device),
        "ShapleyCAM (ReST)": lib_cams(ShapleyCAM, wrapper, [L4], x, [ClassifierOutputReST(int(c)) for c in pred], device, bs=8),
    }
    if args.with_scorecam:   # opt-in: 2048 forward passes per image, unusable on the Pi
        # bs=1: Score-CAM materialises 2048 masked copies of each input (~1.2 GB per image)
        maps["Score-CAM"] = lib_cams(ScoreCAM, wrapper, [L4], x, t_logit, device, bs=1, batch_size=64)
    maps["Seg head (reference)"] = seg
    print("maps computed", flush=True)

    road = ROADCombined(percentiles=[int(p) for p in args.road_percentiles.split(",")])
    t_soft = [ClassifierOutputSoftmaxTarget(int(c)) for c in pred]
    results = {}
    for name, m in maps.items():
        r = {"pointing_game": ci([float(pointing_game(m[i], gt[i], tolerance_px=7)) for i in range(len(m))]),
             "energy_pointing": ci([energy_pointing(m[i], gt[i]) for i in range(len(m))]),
             "top15_iou": ci([topk_overlap(m[i], gt[i], 0.15)["iou"] for i in range(len(m))]),
             "top15_dice": ci([topk_overlap(m[i], gt[i], 0.15)["dice"] for i in range(len(m))])}
        if not name.startswith("Seg head"):
            scores = []
            for s in range(0, len(x), 16):
                scores.append(road(torch.from_numpy(x[s:s + 16]).to(device), m[s:s + 16].astype(np.float32),
                                   t_soft[s:s + 16], wrapper))
            r["road_combined"] = ci(np.concatenate(scores).ravel())
        results[name] = r
        print(f"{name:>24}: energy-in-mask={r['energy_pointing']['mean']:.3f}  pointing={r['pointing_game']['mean']:.3f}  "
              f"top15 IoU={r['top15_iou']['mean']:.3f}" +
              (f"  ROAD={r['road_combined']['mean']:.4f}" if "road_combined" in r else ""), flush=True)

    # ---- cascading randomization sanity check (exact CAM; classifier first, then deeper)
    sanity = []
    rnd = copy.deepcopy(net)
    base = exact_maps(feats, w, pred, size)
    for k, stage in enumerate(["classifier", "layer4", "layer3", "layer2", "layer1"]):
        randomize(rnd.classifier if stage == "classifier" else getattr(rnd.encoder, stage), seed=100 + k)
        of = forward_all(rnd, x, device)
        wr, _ = rnd.head_weights()
        mr = exact_maps(of["features"], wr, pred, size)
        sims = [map_similarity(base[i], mr[i]) for i in range(len(base))]
        sanity.append({"randomized_up_to": stage,
                       "spearman": ci([s["spearman"] for s in sims]), "ssim": ci([s["ssim"] for s in sims])})
        print(f"sanity: randomized up to {stage:>10}: Spearman={sanity[-1]['spearman']['mean']:.3f} "
              f"SSIM={sanity[-1]['ssim']['mean']:.3f}", flush=True)

    out = args.run / "xai"
    out.mkdir(exist_ok=True)
    report = {"split": args.split, "n_images": int(len(keep)), "per_class": {class_names[c]: int((y == c).sum()) for c in np.unique(y)},
              "methods": results, "sanity_cascading_randomization": sanity}
    (out / "xai_report.json").write_text(json.dumps(report, indent=2))

    fmt = lambda r: f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"  # noqa: E731
    lines = [f"# XAI evaluation — {args.run.parent.name}/{args.run.name} ({args.split}, n={len(keep)})", "",
             "Correctly classified tumor images with expert masks; explanations of the predicted class. "
             "Mean [95% bootstrap CI].", "",
             "| Method | Energy in mask ↑ | Pointing game ↑ | Top-15% IoU ↑ | ROAD combined ↑ |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {fmt(r['energy_pointing'])} | {fmt(r['pointing_game'])} | {fmt(r['top15_iou'])} | "
                     f"{fmt(r['road_combined']) if 'road_combined' in r else '—'} |")
    lines += ["", "## Sanity check — cascading randomization (exact CAM)", "",
              "| Randomized up to | Spearman ↓ | SSIM ↓ |", "|---|---|---|"]
    lines += [f"| {s['randomized_up_to']} | {fmt(s['spearman'])} | {fmt(s['ssim'])} |" for s in sanity]
    (out / "xai_report.md").write_text("\n".join(lines), encoding="utf-8")
    make_figure(out / "xai_examples.png", raw, gt, maps, y, class_names, rng)
    print("\n".join(lines))


def make_figure(path, raw, gt, maps, y, class_names, rng, per_class=2):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [i for c in np.unique(y) for i in rng.choice(np.flatnonzero(y == c), min(per_class, (y == c).sum()), replace=False)]
    cols = ["Input + expert mask"] + list(maps)
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.0 * len(cols), 2.0 * len(picks)))
    for r, i in enumerate(picks):
        base = np.repeat(raw[i][..., None], 3, -1)
        contour = base.copy()
        cnts, _ = cv2.findContours(gt[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(contour, cnts, -1, (0, 255, 0), 1)
        axes[r, 0].imshow(contour)
        axes[r, 0].set_ylabel(class_names[y[i]], fontsize=8)
        for c, (name, m) in enumerate(maps.items(), start=1):
            ov = necam.overlay(raw[i], m[i])
            cv2.drawContours(ov, cnts, -1, (0, 255, 0), 1)
            axes[r, c].imshow(ov)
        for c, name in enumerate(cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(name, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

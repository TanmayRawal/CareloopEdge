"""Post-hoc calibration, conformal prediction and OOD detection from a finished training run.

  python -m training.calibrate --run <RUNS>/mri_grouped/seed0 [--alpha 0.05]

Uses ONLY the validation split for fitting (temperature, conformal thresholds, OOD threshold) and
the training split for the OOD class means/covariance; test + external splits are evaluation-only.
Writes  calibration.json, ood.npz, calibration_report.md  into the run directory — these files are
shipped inside the edge bundle.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch

from neuroedge.metrics import calibration_error
from neuroedge.uncertainty import ClassConditionalConformal, MahalanobisOOD, TemperatureScaler, softmax


def load(run: Path, split: str) -> dict | None:
    p = run / f"pred_{split}.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else None


def cam_in_lesion(o: dict, w: np.ndarray, size: int, seg_p: float = 0.5, min_px: int = 20) -> np.ndarray:
    """Per image: share of positive exact-CAM evidence inside the predicted lesion (NaN if no lesion).
    Mirrors edge/engine.py exactly, so thresholds fitted here apply to the engine."""
    from neuroedge import cam as necam

    vals = np.full(len(o["logits"]), np.nan)
    for i in range(len(vals)):
        lesion = (1 / (1 + np.exp(-o["seg_logits"][i, 0])) >= seg_p).astype(np.uint8)
        if lesion.sum() < min_px:
            continue
        c = necam.upsample(necam.exact_cam(o["features"][i], w, int(np.argmax(o["logits"][i]))), size)
        vals[i] = necam.evidence_inside_fraction(c, lesion)
    return vals


def onnx_predictions(run: Path, onnx_path: Path, out: Path) -> None:
    """Recompute pred_<split>.npz with the DEPLOYED ONNX graph (e.g. INT8): calibration, conformal
    thresholds and OOD statistics must be fitted on the outputs of the model that actually runs."""
    import onnxruntime as ort
    import pandas as pd

    from training.data.paths import MANIFESTS
    from training.quantize import rows_arrays

    bundle = json.loads((run / "export" / "bundle.json").read_text())
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sets: dict[str, pd.DataFrame] = {}
    main = pd.read_csv(MANIFESTS / f"{bundle['task']}.csv", keep_default_na=False)
    for s in ["train", "val", "test"] + sorted(x for x in main["split"].unique() if x.startswith("external_")):
        sets[s] = main[main["split"] == s]
    for p in sorted(MANIFESTS.glob("external_*.csv")):
        ext = pd.read_csv(p, keep_default_na=False)
        sets[p.stem] = ext[ext["split"] == p.stem]
    out.mkdir(parents=True, exist_ok=True)
    head = np.load(run / "export" / "head.npz")
    small = ["logits", "emb1", "emb2", "emb3", "emb4"]
    for name, df in sets.items():
        if df.empty:
            continue
        # Chunked: whole splits (images + 2048x14x14 feature maps) do not fit in RAM.
        acc: dict[str, list] = {k: [] for k in small + ["labels", "ids", "cam_in_lesion"]}
        need = small + (["features", "seg_logits"] if name == "val" else [])
        for s in range(0, len(df), 128):
            x, y, ids = rows_arrays(bundle, df.iloc[s:s + 128], name)
            o = dict(zip(need, sess.run(need, {"image": x})))
            for k in small:
                acc[k].append(o[k] if k == "logits" else o[k].astype(np.float16))
            acc["labels"].append(y)
            acc["ids"] += ids
            if name == "val":   # statistics for calibrating the engine's review triggers
                acc["cam_in_lesion"].append(cam_in_lesion(o, head["W"], x.shape[-1]))
            del x, o
        extra = {"cam_in_lesion": np.concatenate(acc["cam_in_lesion"])} if name == "val" else {}
        np.savez_compressed(out / f"pred_{name}.npz", labels=np.concatenate(acc["labels"]), ids=np.array(acc["ids"]),
                            **{k: np.concatenate(acc[k]) for k in small}, **extra)
        print(f"  {name}: {len(acc['ids'])} images via {onnx_path.name}", flush=True)


def embeddings(pred: dict) -> list[np.ndarray]:
    return [pred[f"emb{k}"].astype(np.float32) for k in range(1, 5)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # reports contain ≥ → etc.; Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--ood-tpr", type=float, default=0.95)
    ap.add_argument("--onnx", type=Path, default=None,
                    help="calibrate the deployed ONNX graph (e.g. <run>/export/model_int8.onnx) instead of PyTorch")
    args = ap.parse_args()

    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    class_names = list(ckpt["class_names"])
    alpha = args.alpha if args.alpha is not None else ckpt["config"].get("conformal_alpha", 0.05)
    src = args.run                          # where pred_*.npz come from and outputs go
    if args.onnx is not None:
        src = args.run / "export" / f"calib_{args.onnx.stem}"
        onnx_predictions(args.run, args.onnx, src)
    train, val = load(src, "train"), load(src, "val")
    evals = {p.stem[5:]: load(src, p.stem[5:]) for p in sorted(src.glob("pred_*.npz"))
             if p.stem[5:] not in ("train", "val")}
    args.run = src                          # write calibration.json / ood.npz / report next to the preds

    ts = TemperatureScaler().fit(val["logits"], val["labels"])
    cp = ClassConditionalConformal(alpha).fit(softmax(ts.transform(val["logits"])), val["labels"])
    ood = MahalanobisOOD().fit(embeddings(train), train["labels"], embeddings(val), tpr=args.ood_tpr)
    ood.save(args.run / "ood.npz")

    # Two-tier OOD: above ood.threshold (95% of val in-distribution scores) -> warning + review;
    # above the 99.5th percentile -> hard rejection (no diagnosis shown). Rejecting at the 95% level
    # would turn away ~1 in 20 legitimate scans.
    val_scores = ood.score(embeddings(val))
    report = {"temperature": ts.temperature, "alpha": alpha, "conformal": cp.to_dict(),
              "ood_threshold": ood.threshold, "ood_tpr": ood.tpr,
              "ood_reject_threshold": float(np.quantile(val_scores, 0.995)), "ood_reject_tpr": 0.995,
              "class_names": class_names, "splits": {}}
    if "cam_in_lesion" in val:
        tumor = np.array([class_names[i] not in ("no_tumor", "healthy") for i in val["labels"]])
        correct = val["logits"].argmax(1) == val["labels"]
        v = val["cam_in_lesion"][tumor & correct]
        v = v[np.isfinite(v)]
        if len(v) >= 20:   # 5th percentile: ~5% of typical correct cases trigger review by construction
            report["review_thresholds"] = {"min_cam_seg_consistency": float(np.percentile(v, 5))}
            report["cam_in_lesion_val"] = {"n": int(len(v)), "median": float(np.median(v)),
                                           "p5": float(np.percentile(v, 5))}
    for name, pred in {"val": val, **evals}.items():
        raw, cal = softmax(pred["logits"]), softmax(ts.transform(pred["logits"]))
        sets = cp.predict_sets(cal)
        cov = cp.coverage_report(sets, pred["labels"])
        report["splits"][name] = {
            "n": int(len(pred["labels"])),
            "ece_before": calibration_error(raw, pred["labels"])["ece"],
            "ece_after": calibration_error(cal, pred["labels"])["ece"],
            "nll_before": calibration_error(raw, pred["labels"])["nll"],
            "nll_after": calibration_error(cal, pred["labels"])["nll"],
            "coverage": cov,
            "ood_flag_rate": float(ood.is_ood(embeddings(pred)).mean()),
        }
    (args.run / "calibration.json").write_text(json.dumps(report, indent=2))

    lines = [f"# Calibration report — {args.run.parent.name}/{args.run.name}", "",
             f"* Temperature T = {ts.temperature:.3f} (fitted on val)",
             f"* Class-conditional conformal, alpha = {alpha} (target per-class coverage ≥ {1 - alpha:.0%})",
             f"* Mahalanobis OOD threshold at {args.ood_tpr:.0%} in-distribution TPR (val)", "",
             "| split | n | ECE before → after | NLL before → after | marginal cov. | min per-class cov. | "
             "mean set size | singleton | OOD-flag rate |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["splits"].items():
        c = r["coverage"]
        lines.append(f"| {name} | {r['n']} | {r['ece_before']:.3f} → {r['ece_after']:.3f} | "
                     f"{r['nll_before']:.3f} → {r['nll_after']:.3f} | {c['marginal_coverage']:.3f} | "
                     f"{min(c['per_class_coverage'].values()):.3f} | {c['mean_set_size']:.2f} | "
                     f"{c['frac_singleton']:.2f} | {r['ood_flag_rate']:.3f} |")
    lines += ["", "Coverage guarantees hold under exchangeability with the calibration (val) data; external "
              "sets are distribution-shifted, so their coverage is an empirical robustness check, not a guarantee."]
    (args.run / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

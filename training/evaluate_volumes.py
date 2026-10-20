"""Patient-level evaluation of a deployed bundle on 3D studies (FINDINGS §11).

    python -m training.evaluate_volumes --bundle ~/neuroedge_data/bundles/mri_v1_0.2.0_int8 [--per-plane 12]

For every study in manifests/volumes.csv: load the volume, take evenly spaced slices from the axial, coronal and
sagittal planes (neuroedge.volume), run the DEPLOYED engine on each slice exactly as the Pi does (edge.engine), and
cache the per-slice results. Then combine slices into one answer per patient with each rule in neuroedge.aggregate.

Honesty rules
  * Pre-registered targets (fixed before any result was seen): patient macro-F1 >= 0.85 over the three tumour types,
    slice-level tumour/no-tumour AUROC >= 0.90 (slices with a ground-truth mask), IXI false-positive rate <= 10%.
  * The aggregation rule is chosen on CALIBRATION patients only (30% of the "dev" studies, chosen by a hash of the
    study id); every number in the report is computed on the remaining patients and on the "external" datasets,
    which are never used for any choice.
  * Results are reported per dataset as well as pooled, because each dataset is a different hospital/scanner and a
    pooled number can hide a dataset shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from edge.engine import Engine
from neuroedge import aggregate, volume
from neuroedge.metrics import bootstrap_ci, classification_metrics
from training.data.paths import DATA_ROOT, RUNS

TARGETS = {"patient_macro_f1_tumour_types": 0.85, "slice_auroc_tumour": 0.90, "ixi_false_positive_rate": 0.10}
MIN_MASK_PX = 30           # a slice "has tumour" in the ground truth if the mask covers at least this many pixels


def is_cal(study_id: str) -> bool:
    return int(hashlib.sha256(study_id.encode()).hexdigest(), 16) % 10 < 3


def run_study(engine: Engine, row, per_plane: int) -> dict:
    vol = volume.load_volume(row.t1c_path)
    mask = None
    if isinstance(row.seg_path, str) and row.seg_path:
        try:
            mask = volume.load_mask(row.seg_path, like=vol)
        except ValueError:            # mask stored on a different grid (e.g. the pituitary "_f" masks): patient label only
            mask = None
    slices = volume.select_slices(vol, per_plane=per_plane, mask=mask)
    probs, lesion, rejected, planes, truth = [], [], [], [], []
    for s in slices:
        r = engine.analyze(s.raw, with_overlays=False)
        probs.append([r["probabilities"][c] for c in engine.class_names])
        lesion.append(r["lesion"]["area_px_model"] if r["lesion"]["present"] else 0)
        rejected.append(r["status"] != "ok")
        planes.append(s.plane)
        truth.append(-1 if s.mask is None else int(s.mask.sum() >= MIN_MASK_PX))
    return {"probs": np.asarray(probs, np.float32), "lesion": np.asarray(lesion, np.float32),
            "rejected": np.asarray(rejected, bool), "planes": np.asarray(planes), "truth": np.asarray(truth, np.int8)}


def collect(bundle: Path, manifest: Path, cache: Path, per_plane: int, limit: int | None) -> pd.DataFrame:
    engine = Engine.from_dir(bundle, threads=8)
    rows = pd.read_csv(manifest, keep_default_na=False)
    if "overlap" in rows:                             # same patient as a training image / another dataset
        rows = rows[~rows["overlap"].astype(str).str.lower().eq("true")]
    if limit:
        rows = rows.groupby("dataset", group_keys=False).head(limit)
    cache.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    for row in rows.itertuples():
        out = cache / f"{row.dataset}__{row.study_id}.npz"
        if out.exists():
            continue
        try:
            res = run_study(engine, row, per_plane)
        except Exception as e:                       # unreadable study: record and continue
            (cache / f"{row.dataset}__{row.study_id}.error").write_text(str(e)[:500], encoding="utf-8")
            continue
        np.savez_compressed(out, **res)
        done += 1
        if done % 20 == 0:
            print(f"  {done} studies, {(time.time() - t0) / done:.1f} s/study", flush=True)
    return rows


def patient_table(rows: pd.DataFrame, cache: Path, classes: list[str], rule: str, k: int) -> pd.DataFrame:
    nt = classes.index("no_tumor")
    recs = []
    for row in rows.itertuples():
        f = cache / f"{row.dataset}__{row.study_id}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p = aggregate.aggregate(z["probs"], rule, nt, k=k, lesion_area=z["lesion"], ok=~z["rejected"])
        recs.append({"study_id": row.study_id, "dataset": row.dataset, "label": row.label, "role": row.role,
                     "cal": row.role == "dev" and is_cal(row.study_id), **{c: float(v) for c, v in zip(classes, p)},
                     "rejected_frac": float(z["rejected"].mean())})
    return pd.DataFrame(recs)


def metrics_for(df: pd.DataFrame, classes: list[str]) -> dict:
    if df.empty:
        return {}
    y = np.array([classes.index(l) for l in df["label"]])
    P = df[classes].to_numpy()
    m = classification_metrics(y, P, classes)
    pred = P.argmax(1)
    tumour_types = [classes.index(c) for c in ("glioma", "meningioma", "pituitary")]
    sel = np.isin(y, tumour_types)
    out = {"n": int(len(df)), "accuracy": float((pred == y).mean()), "macro_f1": m.get("macro_f1"),
           "per_class": m.get("per_class")}
    if sel.sum() >= 5:
        from sklearn.metrics import f1_score
        f = lambda idx: f1_score(y[sel][idx], pred[sel][idx], labels=tumour_types, average="macro", zero_division=0)  # noqa: E731
        v, lo, hi = bootstrap_ci(f, int(sel.sum()))
        out["macro_f1_tumour_types"] = {"value": v, "ci95": [lo, hi], "n": int(sel.sum())}
    return out


def slice_auroc(rows: pd.DataFrame, cache: Path, classes: list[str]) -> dict:
    """Tumour vs no-tumour per slice (slices with a ground-truth mask), on patients not used for any choice."""
    from sklearn.metrics import roc_auc_score
    nt = classes.index("no_tumor")
    ys, ps = [], []
    for row in rows.itertuples():
        if row.role == "dev" and is_cal(row.study_id):
            continue
        f = cache / f"{row.dataset}__{row.study_id}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        keep = z["truth"] >= 0
        ys.append(z["truth"][keep])
        ps.append(1.0 - z["probs"][keep, nt])
    if not ys:
        return {}
    y, p = np.concatenate(ys), np.concatenate(ps)
    if y.min() == y.max():
        return {"n_slices": int(len(y))}
    v, lo, hi = bootstrap_ci(lambda i: roc_auc_score(y[i], p[i]) if y[i].min() != y[i].max() else np.nan, len(y))
    return {"value": v, "ci95": [lo, hi], "n_slices": int(len(y)), "positive_slices": int(y.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DATA_ROOT / "bundles" / "mri_v1_0.2.0_int8")
    ap.add_argument("--manifest", type=Path, default=DATA_ROOT / "manifests" / "volumes.csv")
    ap.add_argument("--per-plane", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="at most N studies per dataset (smoke runs)")
    ap.add_argument("--out", type=Path, default=RUNS / "volumes_eval")
    a = ap.parse_args()

    tag = f"{a.bundle.name}_pp{a.per_plane}"
    cache = a.out / tag / "slices"
    rows = collect(a.bundle, a.manifest, cache, a.per_plane, a.limit)
    classes = json.loads((a.bundle / "bundle.json").read_text())["class_names"]

    # 1. choose the rule on calibration patients only (tumour-type macro-F1, then tumour/no-tumour accuracy)
    choice = {}
    for rule in aggregate.RULES:
        t = patient_table(rows, cache, classes, rule, a.k)
        cal = t[t["cal"]] if not t.empty else t
        mm = metrics_for(cal, classes)
        choice[rule] = (mm.get("macro_f1_tumour_types") or {}).get("value", float("nan"))
    best = max(choice, key=lambda r: -1 if np.isnan(choice[r]) else choice[r])

    # 2. report on everything that was not used for the choice
    t = patient_table(rows, cache, classes, best, a.k)
    report = {"bundle": a.bundle.name, "per_plane": a.per_plane, "k": a.k, "rule_selection_on_cal": choice,
              "rule": best, "targets": TARGETS, "datasets": {}, "pooled": {}}
    test = t[~t["cal"]] if not t.empty else t
    for ds, g in test.groupby("dataset"):
        report["datasets"][ds] = metrics_for(g, classes)
        if ds == "ixi":
            report["datasets"][ds]["false_positive_rate"] = float((g[classes].to_numpy().argmax(1)
                                                                   != classes.index("no_tumor")).mean())
    report["pooled"] = metrics_for(test, classes)
    report["slice_auroc_tumour"] = slice_auroc(rows, cache, classes)

    f1 = (report["pooled"].get("macro_f1_tumour_types") or {}).get("value")
    auc = report["slice_auroc_tumour"].get("value")
    fpr = report["datasets"].get("ixi", {}).get("false_positive_rate")
    report["gate"] = {
        "patient_macro_f1_tumour_types": None if f1 is None else bool(f1 >= TARGETS["patient_macro_f1_tumour_types"]),
        "slice_auroc_tumour": None if auc is None else bool(auc >= TARGETS["slice_auroc_tumour"]),
        "ixi_false_positive_rate": None if fpr is None else bool(fpr <= TARGETS["ixi_false_positive_rate"]),
    }
    out = a.out / tag / "report.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    t.to_csv(a.out / tag / "patients.csv", index=False)
    print(json.dumps({"rule": best, "pooled": report["pooled"].get("macro_f1_tumour_types"),
                      "slice_auroc": report["slice_auroc_tumour"].get("value"), "ixi_fpr": fpr,
                      "gate": report["gate"]}, indent=2, default=float))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()

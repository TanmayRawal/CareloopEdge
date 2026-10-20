"""Same-patient leakage check for the 3D datasets (FINDINGS §11).

    python -m training.data.check_overlap          # reads manifests/volumes.csv, writes manifests/volumes_overlap.json

Why: public datasets are often assembled from each other (BraTS includes cases from older TCIA collections), and the
old 2D training images (BRISC, figshare, Nickparvar) may be slices of the same patients. A patient whose scan is in
training must never be counted as a test patient.

How: each study becomes thumbnails of a few evenly spaced slices per plane (the same flip-aware NCC on 64x64
thumbnails as training/data/dedupe.py, with the thresholds calibrated there).
  * study vs study (different datasets): the median over the first study's slices of the best NCC to any slice of the
    other study. A median >= t_patient means "same scan" (one matching slice could be chance; most slices matching
    cannot).
  * study vs 2D training images: any slice >= t_dup, or at least 3 slices >= t_patient.
Matched studies are marked `overlap` in volumes.csv; evaluate_volumes.py excludes them from every test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from neuroedge import volume
from training.data.dedupe import THUMB, best_similarity, thumbnails
from training.data.paths import DATA_ROOT, MANIFESTS


def thumb_from_array(img01: np.ndarray, size: int = THUMB) -> np.ndarray:
    """Same as dedupe._thumb_pair, for an in-memory [0,1] image."""
    g = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    t = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for v in (t, t[:, ::-1]):
        v = (v - v.mean()) / (v.std() + 1e-6)
        out.append(v.ravel() / size)
    return np.stack(out)


def study_thumbs(path: str, per_plane: int = 6) -> np.ndarray:
    vol = volume.load_volume(path)
    return np.stack([thumb_from_array(s.raw.pixels) for s in volume.select_slices(vol, per_plane=per_plane)])


def study_match(a: np.ndarray, b: np.ndarray) -> float:
    """Median over a's slices of the best flip-aware NCC to any slice of b."""
    return float(np.median(best_similarity(a, b).max(axis=1)))


def check(rows: pd.DataFrame, t_dup: float, t_patient: float, train_paths: list[str],
          per_plane: int = 6) -> tuple[dict, dict]:
    thumbs = {}
    for r in rows.itertuples():
        try:
            thumbs[(r.dataset, r.study_id)] = study_thumbs(r.t1c_path, per_plane)
        except Exception as e:
            print(f"  skip {r.dataset}/{r.study_id}: {e}")
    keys = list(thumbs)
    cross = []
    for i, ka in enumerate(keys):                          # different datasets only
        for kb in keys[i + 1:]:
            if ka[0] == kb[0]:
                continue
            s = max(study_match(thumbs[ka], thumbs[kb]), study_match(thumbs[kb], thumbs[ka]))
            if s >= t_patient:
                cross.append({"a": list(ka), "b": list(kb), "ncc_median": round(s, 4)})
    vs_train = {}
    if train_paths:
        t_train = thumbnails(train_paths)
        for k in keys:
            best = best_similarity(thumbs[k], t_train).max(axis=1)
            if (best >= t_dup).any() or (best >= t_patient).sum() >= 3:
                vs_train["/".join(k)] = {"max_ncc": round(float(best.max()), 4),
                                         "slices_over_t_patient": int((best >= t_patient).sum())}
    return {"pairs": cross}, vs_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFESTS / "volumes.csv")
    ap.add_argument("--train-manifest", type=Path, default=MANIFESTS / "mri_v2.csv")
    ap.add_argument("--audit", type=Path, default=MANIFESTS / "mri_audit.json")
    a = ap.parse_args()
    audit = json.loads(a.audit.read_text())
    t_dup, t_patient = float(audit["t_dup"]), float(audit["t_patient"])
    rows = pd.read_csv(a.manifest, keep_default_na=False)
    train = pd.read_csv(a.train_manifest)
    train_paths = [str(DATA_ROOT / p) if not Path(p).is_absolute() else p
                   for p in train.loc[train["split"].isin(["train", "val", "cal", "test"]), "path"]]
    cross, vs_train = check(rows, t_dup, t_patient, train_paths)
    flagged = {tuple(p["b"]) for p in cross["pairs"]} | {tuple(k.split("/", 1)) for k in vs_train}
    rows["overlap"] = [(r.dataset, r.study_id) in flagged for r in rows.itertuples()]
    rows.to_csv(a.manifest, index=False)
    out = {"t_dup": t_dup, "t_patient": t_patient, "studies": int(len(rows)), "cross_dataset": cross["pairs"],
           "vs_2d_training": vs_train, "excluded": int(rows["overlap"].sum())}
    (MANIFESTS / "volumes_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{out['excluded']} of {len(rows)} studies overlap (cross-dataset pairs: {len(cross['pairs'])}, "
          f"vs 2D training: {len(vs_train)})")


if __name__ == "__main__":
    main()

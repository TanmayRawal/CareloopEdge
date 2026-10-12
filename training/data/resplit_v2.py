"""Split v2: add a dedicated CALIBRATION split, keeping v0's test set unchanged.

  python -m training.data.resplit_v2            ->  <DATA_ROOT>/manifests/mri_v2.csv (+ mri_v2_audit.json)

Why: in v0 the validation split served both model selection (early stopping) and calibration
(temperature, conformal, OOD, review thresholds); conformal sets then under-covered on test (0.87 at a
nominal 0.95). v2 re-splits v0's train+val pool by the SAME leakage-safe groups into train / val / cal
(~65 / 10 / 10 % of BRISC) with StratifiedGroupKFold; test (15 %) is copied verbatim so v0 and v1 are
compared on identical images. External splits are untouched.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from training.data.build_manifest import leakage_checks
from training.data.paths import MANIFESTS


def carve(labels: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of ~frac of the rows, whole groups only, stratified by label."""
    k = max(2, round(1 / frac))
    _, idx = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels, groups))
    m = np.zeros(len(labels), bool)
    m[idx] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="mri.csv")
    ap.add_argument("--dst", default="mri_v2.csv")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df = pd.read_csv(MANIFESTS / args.src, keep_default_na=False)
    pool = df["split"].isin(["train", "val"]).to_numpy()
    sub = df[pool]
    n_brisc = int((df["split"].isin(["train", "val", "test"])).sum())
    lab, grp = sub["label"].to_numpy(), sub["group"].astype(str).to_numpy()
    # fractions of the pool (85 % of BRISC): cal 10/85, then val 10/75 of the remainder
    is_cal = carve(lab, grp, 0.10 / 0.85, args.seed)
    rest = np.flatnonzero(~is_cal)
    is_val_rel = carve(lab[rest], grp[rest], 0.10 / 0.75, args.seed + 1)
    new = np.full(len(sub), "train", dtype=object)
    new[is_cal] = "cal"
    new[rest[is_val_rel]] = "val"
    df.loc[pool, "split"] = new

    spanning = leakage_checks(df, splits=("train", "val", "cal", "test"))
    known = df[df["split"].isin(["train", "val", "cal", "test"]) & (df["patient_id_inferred"].astype(str) != "")]
    known_spanning = int((known.groupby("patient_id_inferred")["split"].nunique() > 1).sum())
    counts = {s: int((df["split"] == s).sum()) for s in ("train", "val", "cal", "test")}
    per_class = df[df["split"].isin(counts)].groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index")
    rep = {"source": args.src, "seed": args.seed, "counts": counts,
           "fractions_of_brisc": {s: round(c / n_brisc, 3) for s, c in counts.items()},
           "per_class": per_class, "groups_spanning_splits": spanning,
           "known_patients_spanning_splits": known_spanning,
           "test_unchanged_from_v0": True}
    assert spanning == 0 and known_spanning == 0, rep
    df.to_csv(MANIFESTS / args.dst, index=False)
    (MANIFESTS / args.dst.replace(".csv", "_audit.json")).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

"""Build leakage-controlled manifests (CSV) for the MRI and CT tasks, with audit reports.

Audit findings that shaped this design (see dedupe.py and reports/leakage_audit.md):
  * 4,775 / 7,200 Nickparvar-v2 images are byte-identical to BRISC images (shared sources).
  * 566 / 1,000 BRISC official-test images are copies of BRISC-train images — 462 of them
    left-right mirrored — and most others have same-patient neighbouring slices in train.
Therefore:
  MRI  * BRISC train+test are POOLED and re-split 70/15/15, stratified by class and grouped by
         same-patient proxy groups (no group spans splits).
       * `official_split` and `official_test_clean` columns support a leakage experiment:
         train on the official train split, test on the official test split vs. its clean subset.
       * figshare / Nickparvar images form external test sets, excluding any image whose group
         touches BRISC.
  CT   * Kaggle multimodal CT subset, grouped the same way, split 70/15/15. Known confounds
         (burned-in text, file format / resolution correlated with class) are recorded per image
         for the shortcut audit.

Usage:
  python -m training.data.build_manifest              # calibrate, group, split, write manifests
"""

from __future__ import annotations

import argparse
import sys
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from neuroedge.config import CT_CLASSES_BINARY, MRI_CLASSES

from .dedupe import best_similarity, group_ids, md5_all, similar_pairs, thumbnails
from .paths import MANIFESTS, PROCESSED, RAW

COLUMNS = ["id", "dataset", "modality", "source_split", "path", "mask_path", "label", "plane",
           "patient_id"]


# ----------------------------------------------------------------------------- collectors
def collect_brisc(verify: bool = True) -> pd.DataFrame:
    root = RAW / "brisc2025" / "brisc2025"
    man = pd.read_csv(root / "manifest.csv")
    man = man[(man["task"] == "classification") & (~man["is_mask"].astype(bool))]
    rows = []
    for r in man.itertuples():
        path = root / Path(r.relative_path.replace("\\", "/"))
        if verify and hashlib.sha256(path.read_bytes()).hexdigest() != r.sha256:
            raise ValueError(f"SHA-256 mismatch: {path}")
        stem = Path(r.filename).stem
        mask = root / "segmentation_task" / r.split / "masks" / f"{stem}.png"
        rows.append({"id": f"brisc/{stem}", "dataset": "brisc", "modality": "mri",
                     "source_split": r.split, "path": str(path),
                     "mask_path": str(mask) if mask.exists() else "",
                     "label": r.tumor_label, "plane": r.plane_label, "patient_id": ""})
    return pd.DataFrame(rows, columns=COLUMNS)


def collect_figshare() -> pd.DataFrame:
    base = PROCESSED / "figshare"
    meta = pd.read_csv(base / "figshare.csv", dtype={"patient_id": str})
    return pd.DataFrame({
        "id": "figshare/" + meta["file"].str.replace(".png", "", regex=False),
        "dataset": "figshare", "modality": "mri", "source_split": "all",
        "path": [str(base / "images" / f) for f in meta["file"]],
        "mask_path": [str(base / "masks" / f) for f in meta["file"]],
        "label": meta["label"], "plane": "", "patient_id": meta["patient_id"],
    }, columns=COLUMNS)


def collect_nickparvar() -> pd.DataFrame:
    root = RAW / "nickparvar"
    name_map = {"glioma": "glioma", "meningioma": "meningioma", "notumor": "no_tumor", "pituitary": "pituitary"}
    rows = []
    for split in ("Training", "Testing"):
        for folder, label in name_map.items():
            for p in sorted((root / split / folder).glob("*")):
                rows.append({"id": f"nickparvar/{split}/{p.stem}", "dataset": "nickparvar", "modality": "mri",
                             "source_split": split.lower(), "path": str(p), "mask_path": "",
                             "label": label, "plane": "", "patient_id": ""})
    return pd.DataFrame(rows, columns=COLUMNS)


def collect_ct_multimodal() -> pd.DataFrame:
    root = RAW / "ct_mri_multimodal" / "Dataset" / "Brain Tumor CT scan Images"
    rows = []
    for folder, label in (("Healthy", "healthy"), ("Tumor", "tumor")):
        for p in sorted((root / folder).iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            rows.append({"id": f"ctmm/{folder}/{p.stem}", "dataset": "ct_multimodal", "modality": "ct",
                         "source_split": "all", "path": str(p), "mask_path": "", "label": label,
                         "plane": "", "patient_id": "",
                         # confound attributes for the shortcut audit
                         "height": img.shape[0], "width": img.shape[1], "file_ext": p.suffix.lower(),
                         "stored_channels": 1 if img.ndim == 2 else img.shape[2]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ calibration
def calibrate_patient_threshold(t: np.ndarray, pids: np.ndarray, max_false_link: float,
                                t_dup: float) -> tuple[float, dict]:
    """Smallest NCC threshold at which at most `max_false_link` of images have a
    *different-patient* neighbour above it (true patient IDs from figshare).

    Pairs above t_dup across different patient IDs are reported separately: they are true
    duplicates filed under two IDs, not a property of the threshold.
    """
    sim = best_similarity(t, t)
    np.fill_diagonal(sim, -1.0)
    same = pids[:, None] == pids[None, :]
    diff_best = np.where(same, -1.0, sim).max(axis=1)
    same_best = np.where(same, sim, -1.0).max(axis=1)
    cross_dups = diff_best >= t_dup
    grid = np.round(np.arange(0.70, 0.991, 0.01), 2)
    curve = []
    for thr in grid:
        false_link = float(((diff_best >= thr) & ~cross_dups).mean())
        has_sibling = same_best > -1
        recall = float((same_best[has_sibling] >= thr).mean())
        curve.append({"t": float(thr), "false_link_rate": false_link, "same_patient_recall": recall})
    ok = [c["t"] for c in curve if c["false_link_rate"] <= max_false_link]
    chosen = min(ok) if ok else float(grid[-1])
    return chosen, {"chosen": chosen, "max_false_link": max_false_link,
                    "cross_id_duplicates": int(cross_dups.sum()), "curve": curve}


# ---------------------------------------------------------------------------- splitting
def stratified_group_split(labels: np.ndarray, groups: np.ndarray, fracs: tuple[float, float, float],
                           seed: int) -> np.ndarray:
    """Assign 'train'/'val'/'test' with whole groups, approximately stratified by label."""
    from sklearn.model_selection import StratifiedGroupKFold

    out = np.full(len(labels), "train", dtype=object)
    k_test = max(2, round(1 / fracs[2]))
    _, test_idx = next(StratifiedGroupKFold(k_test, shuffle=True, random_state=seed)
                       .split(np.zeros(len(labels)), labels, groups))
    out[test_idx] = "test"
    rest = np.flatnonzero(out != "test")
    k_val = max(2, round((1 - fracs[2]) / fracs[1]))
    _, val_rel = next(StratifiedGroupKFold(k_val, shuffle=True, random_state=seed + 1)
                      .split(np.zeros(len(rest)), labels[rest], groups[rest]))
    out[rest[val_rel]] = "val"
    return out


def assign_mri(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    brisc = (df["dataset"] == "brisc").to_numpy()
    df["split"] = ""
    df.loc[brisc, "split"] = stratified_group_split(df.loc[brisc, "label"].to_numpy(),
                                                    df.loc[brisc, "group"].to_numpy(), (0.70, 0.15, 0.15), seed)
    # leakage experiment flags on the OFFICIAL BRISC split
    df["official_split"] = np.where(brisc, df["source_split"], "")
    off_train_groups = set(df.loc[brisc & (df["source_split"] == "train").to_numpy(), "group"])
    df["official_test_clean"] = brisc & (df["source_split"] == "test").to_numpy() & \
        ~df["group"].isin(off_train_groups).to_numpy()
    brisc_groups = set(df.loc[brisc, "group"])
    for name in ("figshare", "nickparvar"):
        ext = (df["dataset"] == name).to_numpy()
        overlap = ext & df["group"].isin(brisc_groups).to_numpy()
        df.loc[overlap, "split"] = "excluded_overlap_brisc"
        df.loc[ext & ~overlap, "split"] = f"external_{name}"
    assert (df["split"] != "").all()
    return df


# --------------------------------------------------------------------------------- audit
def leakage_checks(df: pd.DataFrame, splits=("train", "val", "test")) -> int:
    sub = df[df["split"].isin(splits)]
    return int((sub.groupby("group")["split"].nunique() > 1).sum())


def overlap_table(df: pd.DataFrame, col: str) -> dict:
    sets = {d: set(g[col]) for d, g in df.groupby("dataset")}
    return {f"{a}->{b}": int(g[col].isin(sets[b]).sum())
            for a, g in df.groupby("dataset") for b in sets if a != b}


def write_report(path: Path, title: str, rep: dict, classes: tuple[str, ...]) -> None:
    L = [f"# Leakage audit — {title}", "",
         f"* Duplicate rule: flip-aware NCC (64x64) ≥ {rep['t_dup']} or identical MD5.",
         f"* Same-patient grouping rule: flip-aware NCC ≥ {rep['t_patient']} "
         f"({rep.get('t_patient_source', '')}).",
         f"* Images: {rep['n_images']}; groups: {rep['n_groups']}; largest group: {rep['largest_group']}.",
         f"* Groups spanning train/val/test: **{rep['groups_spanning_splits']}** (must be 0).", ""]
    if "brisc_with_known_patient" in rep:
        L += [f"* BRISC images with a true patient ID (inherited from figshare duplicates): "
              f"{rep['brisc_with_known_patient']}; known patients spanning splits: "
              f"**{rep['known_patients_spanning_splits']}** (must be 0).", ""]
    if "official_test" in rep:
        o = rep["official_test"]
        L += ["## BRISC official test split", "",
              f"* {o['dup_of_train']} / {o['n']} official-test images are copies or mirrors of official-train images.",
              f"* {o['known_patient_in_train']} / {o['n']} official-test images belong to a patient (true figshare ID) "
              "that also appears in official train.",
              f"* {o['n'] - o['clean']} / {o['n']} share a same-patient group with official-train images; "
              f"clean subset = {o['clean']}.", ""]
    L += ["## Cross-dataset overlap (images of A that have a duplicate / group-mate in B)", "",
          "| A -> B | duplicates | same group |", "|---|---|---|"]
    for k in rep["overlap_dup"]:
        L.append(f"| {k} | {rep['overlap_dup'][k]} | {rep['overlap_group'][k]} |")
    L += ["", "## Split counts", "", "| split | " + " | ".join(classes) + " | total |",
          "|---|" + "---|" * (len(classes) + 1)]
    for s, c in rep["split_counts"].items():
        L.append(f"| {s} | " + " | ".join(str(c.get(k, 0)) for k in classes) + f" | {sum(c.values())} |")
    if "calibration" in rep:
        L += ["", "## Same-patient threshold calibration (figshare patient IDs)", "",
              "| NCC threshold | different-patient false-link rate | same-patient recall |", "|---|---|---|"]
        L += [f"| {c['t']:.2f} | {c['false_link_rate']:.4f} | {c['same_patient_recall']:.3f} |"
              for c in rep["calibration"]["curve"] if c["t"] >= 0.80]
    path.write_text("\n".join(L), encoding="utf-8")


def add_groups(df: pd.DataFrame, t: np.ndarray, md5: list[str], t_dup: float, t_patient: float) -> pd.DataFrame:
    """Grouping used for splitting.

    dup_group  copies / mirrors / re-encodes only (NCC >= t_dup or identical MD5)
    patient_id_inferred
               true patient IDs (figshare) propagated to every copy of that image in other
               datasets — BRISC and Nickparvar redistribute most figshare images
    group      union of: NCC >= t_patient pairs, identical MD5, identical inferred patient ID
    """
    df = df.copy()
    pairs, sims = similar_pairs(t, t_patient)
    df["md5"] = md5
    df["dup_group"] = group_ids(len(df), pairs[sims >= t_dup], md5)
    known = df[df["patient_id"].astype(str) != ""]
    pid_of = known.groupby("dup_group")["patient_id"].agg(lambda s: ";".join(sorted(set(map(str, s)))))
    df["patient_id_inferred"] = df["dup_group"].map(pid_of).fillna("")
    multi = int(df["patient_id_inferred"].str.contains(";").sum())
    if multi:
        print(f"[warn] {multi} images are duplicates of slices filed under different patient IDs")
    df["group"] = group_ids(len(df), pairs, md5, df["patient_id_inferred"].tolist())
    return df


# ----------------------------------------------------------------------------------- main
def build_mri(args) -> None:
    df = pd.concat([collect_brisc(not args.no_verify), collect_figshare(), collect_nickparvar()], ignore_index=True)
    assert set(df["label"]) <= set(MRI_CLASSES), set(df["label"])
    print(df.groupby(["dataset", "label"]).size().unstack(fill_value=0))
    t = thumbnails(df["path"].tolist())
    md5 = md5_all(df["path"].tolist())

    fig = (df["dataset"] == "figshare").to_numpy()
    t_patient, calib = calibrate_patient_threshold(t[fig], df.loc[fig, "patient_id"].to_numpy(),
                                                   args.max_false_link, args.t_dup)
    if args.t_patient is not None:
        t_patient = args.t_patient
    print(f"same-patient threshold: {t_patient} (calibrated {calib['chosen']})")
    df = add_groups(df, t, md5, args.t_dup, t_patient)
    df = assign_mri(df, args.seed)

    brisc = df[df["dataset"] == "brisc"]
    off_test = brisc[brisc["source_split"] == "test"]
    off_train_dup = set(brisc.loc[brisc["source_split"] == "train", "dup_group"])
    sizes = df.groupby("group").size()
    rep = {
        "t_dup": args.t_dup, "t_patient": t_patient,
        "t_patient_source": f"calibrated on figshare patient IDs, false-link ≤ {args.max_false_link:.1%}"
                            if args.t_patient is None else "manual override",
        "n_images": len(df), "n_groups": int(sizes.size), "largest_group": int(sizes.max()),
        "groups_spanning_splits": leakage_checks(df),
        "brisc_with_known_patient": int((brisc["patient_id_inferred"] != "").sum()),
        "known_patients_spanning_splits": int(
            df[df["split"].isin(["train", "val", "test"]) & (df["patient_id_inferred"] != "")]
            .groupby("patient_id_inferred")["split"].nunique().gt(1).sum()),
        "official_test": {"n": len(off_test), "dup_of_train": int(off_test["dup_group"].isin(off_train_dup).sum()),
                          "known_patient_in_train": int(off_test["patient_id_inferred"].ne("").to_numpy().__and__(
                              off_test["patient_id_inferred"].isin(
                                  set(brisc.loc[brisc["source_split"] == "train", "patient_id_inferred"]) - {""}
                              ).to_numpy()).sum()),
                          "clean": int(off_test["official_test_clean"].sum())},
        "overlap_dup": overlap_table(df, "dup_group"), "overlap_group": overlap_table(df, "group"),
        "split_counts": {s: g["label"].value_counts().to_dict() for s, g in df.groupby("split")},
        "calibration": calib,
    }
    if rep["groups_spanning_splits"] or rep["known_patients_spanning_splits"]:
        raise AssertionError("Leakage: a group or a known patient spans train/val/test")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFESTS / "mri.csv", index=False)
    (MANIFESTS / "mri_audit.json").write_text(json.dumps(rep, indent=2))
    write_report(MANIFESTS / "mri_audit.md", "MRI", rep, MRI_CLASSES)
    print(json.dumps({k: rep[k] for k in ("n_images", "n_groups", "largest_group", "groups_spanning_splits",
                                          "official_test", "overlap_dup")}, indent=2))
    print(pd.crosstab(df["split"], df["label"], margins=True))


def build_ct(args, t_patient: float) -> None:
    df = collect_ct_multimodal()
    t = thumbnails(df["path"].tolist())
    md5 = md5_all(df["path"].tolist())
    df = add_groups(df, t, md5, args.t_dup, t_patient)
    df["split"] = stratified_group_split(df["label"].to_numpy(), df["group"].to_numpy(), (0.70, 0.15, 0.15), args.seed)
    sizes = df.groupby("group").size()
    rep = {"t_dup": args.t_dup, "t_patient": t_patient, "t_patient_source": "MRI-calibrated value (no CT patient IDs)",
           "n_images": len(df), "n_groups": int(sizes.size), "largest_group": int(sizes.max()),
           "groups_spanning_splits": leakage_checks(df),
           "overlap_dup": {}, "overlap_group": {},
           "split_counts": {s: g["label"].value_counts().to_dict() for s, g in df.groupby("split")},
           "confounds": {
               "size_512_by_class": pd.crosstab(df["label"], (df["height"] == 512) & (df["width"] == 512)).to_dict(),
               "ext_by_class": pd.crosstab(df["label"], df["file_ext"]).to_dict()}}
    if rep["groups_spanning_splits"]:
        raise AssertionError("Leakage: a group spans train/val/test")
    df.to_csv(MANIFESTS / "ct.csv", index=False)
    (MANIFESTS / "ct_audit.json").write_text(json.dumps(rep, indent=2, default=str))
    write_report(MANIFESTS / "ct_audit.md", "CT", rep, CT_CLASSES_BINARY)
    print(pd.crosstab(df["split"], df["label"], margins=True))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # reports contain ≥ → etc.; Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-dup", type=float, default=0.98)
    ap.add_argument("--t-patient", type=float, default=None, help="override the calibrated value")
    ap.add_argument("--max-false-link", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--skip-ct", action="store_true")
    args = ap.parse_args()
    build_mri(args)
    if not args.skip_ct:
        rep = json.loads((MANIFESTS / "mri_audit.json").read_text())
        build_ct(args, rep["t_patient"])


if __name__ == "__main__":
    main()

"""Index the 3D datasets into manifests/volumes.csv: one row per patient study.

    python -m training.data.build_volumes            # scans ~/neuroedge_data/raw/<dataset>/

Columns: study_id, dataset, site, label, t1c_path, seg_path, role
  role = "external"  -> test only (never used to choose the aggregation rule or to train)
         "dev"       -> may be split into calibration / test patients by evaluate_volumes.py
Layouts recognised (raw/<folder>):
  brats_gli, brats_africa, brats_men   BraTS 2023 folders: <ID>/<ID>-t1c.nii.gz + <ID>-seg.nii.gz
  ixi                                   IXI-T1/IXI<nnn>-<Site>-<nnnn>-T1.nii.gz (healthy volunteers; non-contrast T1)
  pituitary_nhnn                        NIfTI image + tumour mask per patient (see PITUITARY_* below)
  tcia_meningioma                       DICOM series folders; the post-contrast T1 series is picked by description
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from training.data.paths import DATA_ROOT

DATASETS = {  # folder -> (label, role)
    "brats_gli": ("glioma", "dev"),
    "brats_africa": ("glioma", "external"),
    "brats_men": ("meningioma", "dev"),
    "tcia_meningioma": ("meningioma", "external"),
    "pituitary_nhnn": ("pituitary", "dev"),
    "ixi": ("no_tumor", "external"),
}
T1C_WORDS = re.compile(r"(t1.*(post|\+c|gad|gd|ce|contrast|mprage.*c))|(post.*t1)|(t1c)", re.I)
FIELDS = ("study_id", "dataset", "site", "label", "t1c_path", "seg_path", "role")


def brats(root: Path, dataset: str):
    label, role = DATASETS[dataset]
    for t1c in sorted(root.rglob("*-t1c.nii.gz")):
        sid = t1c.name[: -len("-t1c.nii.gz")]
        seg = t1c.with_name(f"{sid}-seg.nii.gz")
        site = "africa" if dataset == "brats_africa" else ""
        yield dict(study_id=sid, dataset=dataset, site=site, label=label, t1c_path=str(t1c),
                   seg_path=str(seg) if seg.exists() else "", role=role)


def ixi(root: Path):
    label, role = DATASETS["ixi"]
    for f in sorted(root.rglob("IXI*-T1.nii.gz")):
        m = re.match(r"(IXI\d+)-([A-Za-z]+)-", f.name)
        yield dict(study_id=m.group(1) if m else f.stem, dataset="ixi", site=m.group(2) if m else "",
                   label=label, t1c_path=str(f), seg_path="", role=role)


# The NHNN pituitary release (Scientific Data 2025) names files per patient; image and tumour-mask names are matched
# by these patterns once the archive is extracted (adjust if the release uses different names).
PITUITARY_IMAGE = re.compile(r"(t1|image|img|scan)", re.I)
PITUITARY_MASK = re.compile(r"(tumou?r|seg|mask|label)", re.I)


def pituitary(root: Path):
    label, role = DATASETS["pituitary_nhnn"]
    by_patient: dict[str, dict] = {}
    for f in sorted(root.rglob("*.nii*")):
        pid = re.match(r"(\d+)", f.name)
        key = pid.group(1) if pid else f.parent.name
        slot = by_patient.setdefault(key, {})
        low = f.name.lower()
        if "carotid" in low:
            continue                                  # carotid artery masks are not tumour labels
        if PITUITARY_MASK.search(low):
            slot["seg"] = f
        elif PITUITARY_IMAGE.search(low) or "img" not in slot:
            slot.setdefault("img", f)
    for key, s in sorted(by_patient.items()):
        if "img" in s:
            yield dict(study_id=f"PIT-{key}", dataset="pituitary_nhnn", site="nhnn", label=label,
                       t1c_path=str(s["img"]), seg_path=str(s.get("seg", "")), role=role)


def tcia_meningioma(root: Path):
    """Picks, per patient, the DICOM series folder whose SeriesDescription looks like post-contrast T1."""
    import pydicom
    label, role = DATASETS["tcia_meningioma"]
    for patient in sorted(p for p in root.iterdir() if p.is_dir()):
        best = None
        for series in sorted({f.parent for f in patient.rglob("*.dcm")}):
            first = next(series.glob("*.dcm"))
            ds = pydicom.dcmread(str(first), stop_before_pixels=True, force=True)
            if str(getattr(ds, "Modality", "")) != "MR":
                continue
            if T1C_WORDS.search(str(getattr(ds, "SeriesDescription", ""))):
                n = len(list(series.glob("*.dcm")))
                if best is None or n > best[1]:
                    best = (series, n)
        if best:
            yield dict(study_id=patient.name, dataset="tcia_meningioma", site="tcia", label=label,
                       t1c_path=str(best[0]), seg_path="", role=role)


def build(raw: Path) -> list[dict]:
    rows: list[dict] = []
    for folder in DATASETS:
        root = raw / folder
        if not root.exists():
            continue
        if folder.startswith("brats_"):
            rows += list(brats(root, folder))
        elif folder == "ixi":
            rows += list(ixi(root))
        elif folder == "pituitary_nhnn":
            rows += list(pituitary(root))
        elif folder == "tcia_meningioma":
            rows += list(tcia_meningioma(root))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DATA_ROOT / "raw")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "manifests" / "volumes.csv")
    a = ap.parse_args()
    rows = build(a.raw)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} studies -> {a.out}")
    for (ds, lab), n in sorted(Counter((r["dataset"], r["label"]) for r in rows).items()):
        print(f"  {ds:<16} {lab:<11} {n}")


if __name__ == "__main__":
    main()

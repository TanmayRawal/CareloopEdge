"""Audit and register an EXTERNAL test dataset against the frozen MRI manifest.

  python -m training.data.build_external --dataset bdneuro

External datasets are never used for training, and adding one never changes the BRISC splits
(the MRI manifest is read-only here). Every external image is compared with every BRISC image
using the same flip-aware NCC rule as the main manifest:
  * NCC >= t_dup      -> excluded (copy / mirror of a BRISC image)
  * NCC >= t_patient  -> excluded (likely same-patient slice)
Copies inside the external set itself are collapsed to one representative.
Writes  <MANIFESTS>/external_<name>.csv  and  external_<name>_audit.md.
"""

from __future__ import annotations

import argparse
import sys
import json

import numpy as np
import pandas as pd

from .dedupe import best_similarity, group_ids, md5_all, similar_pairs, thumbnails
from .paths import MANIFESTS, RAW

LABEL_MAP = {"glioma": "glioma", "meningioma": "meningioma", "no_tumor": "no_tumor", "notumor": "no_tumor",
             "pituitary": "pituitary"}


def collect_bdneuro() -> pd.DataFrame:
    root = RAW / "bdneuro"
    rows = []
    for split in ("train", "val", "test"):
        for folder in ("glioma", "meningioma", "no_tumor", "pituitary"):
            for p in sorted((root / split / folder).iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append({"id": f"bdneuro/{split}/{folder}/{p.stem}", "dataset": "bdneuro", "modality": "mri",
                                 "source_split": split, "path": str(p), "mask_path": "",
                                 "label": LABEL_MAP[folder], "plane": "", "patient_id": ""})
    return pd.DataFrame(rows)


COLLECTORS = {"bdneuro": collect_bdneuro}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # reports contain ≥ → etc.; Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(COLLECTORS))
    args = ap.parse_args()

    audit = json.loads((MANIFESTS / "mri_audit.json").read_text())
    t_dup, t_patient = audit["t_dup"], audit["t_patient"]
    ext = COLLECTORS[args.dataset]()
    mri = pd.read_csv(MANIFESTS / "mri.csv", keep_default_na=False)
    brisc = mri[mri["dataset"] == "brisc"]

    t_ext = thumbnails(ext["path"].tolist())
    t_brisc = thumbnails(brisc["path"].tolist())
    sim_to_brisc = np.concatenate([best_similarity(t_ext[s:s + 1024], t_brisc).max(axis=1)
                                   for s in range(0, len(ext), 1024)])
    md5_ext = md5_all(ext["path"].tolist())
    md5_brisc = set(md5_all(brisc["path"].tolist()))

    pairs, sims = similar_pairs(t_ext, t_dup)
    ext["dup_group"] = group_ids(len(ext), pairs, md5_ext)
    ext["max_ncc_to_brisc"] = sim_to_brisc
    ext["md5"] = md5_ext

    split = np.full(len(ext), f"external_{args.dataset}", dtype=object)
    split[ext["dup_group"].duplicated().to_numpy()] = "excluded_dup_within"
    split[(sim_to_brisc >= t_patient)] = "excluded_overlap_brisc"
    split[ext["md5"].isin(md5_brisc).to_numpy() | (sim_to_brisc >= t_dup)] = "excluded_dup_of_brisc"
    ext["split"] = split
    ext.to_csv(MANIFESTS / f"external_{args.dataset}.csv", index=False)

    counts = pd.crosstab(ext["split"], ext["label"], margins=True)
    lines = [f"# External dataset audit — {args.dataset}", "",
             f"Rules (from the MRI manifest): copy/mirror NCC ≥ {t_dup}; same-patient NCC ≥ {t_patient}.", "",
             "| split | " + " | ".join(counts.columns) + " |", "|---|" + "---|" * len(counts.columns)]
    lines += [f"| {idx} | " + " | ".join(str(v) for v in row) + " |" for idx, row in counts.iterrows()]
    lines += ["", "Max NCC to any BRISC image (external images kept):",
              pd.Series(sim_to_brisc[split == f"external_{args.dataset}"]).describe().round(3).to_string()]
    (MANIFESTS / f"external_{args.dataset}_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

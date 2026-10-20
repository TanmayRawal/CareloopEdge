"""Phase 2 training data: slices from the 3D development volumes, added to the 2D manifest (manifests/mri_v3.csv).

    python -m training.data.export_volume_slices [--per-plane 12 --tumour-per-plane 8]

For each development patient in the train / val / cal splits of manifests/volumes.csv (never test, never external,
never IXI):
  * evenly spaced slices in all three planes (as at inference: neuroedge.volume.select_slices), and
  * extra slices through the tumour (evenly spaced over the mask's extent in each plane), so tumour slices are not
    rare in training.
Each slice is saved as an 8-bit PNG (exactly the intensities the model sees at inference) with its mask PNG. Labels:
  tumour area >= MIN_TUMOUR_PX -> the patient's tumour type
  no tumour on the slice       -> no_tumor  (same scan and contrast, so "no tumour" cannot be learned from contrast)
  a sliver (0 < area < MIN)    -> skipped (ambiguous)
The 2D rows of mri_v2.csv are kept unchanged, so the existing train.py / calibrate.py / export path is reused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from neuroedge import volume
from training.data.paths import DATA_ROOT, MANIFESTS, PROCESSED

MIN_TUMOUR_PX = 30
SPLITS = ("train", "val", "cal")


def tumour_slices(vol: volume.Volume, mask: np.ndarray, per_plane: int) -> list[volume.VolumeSlice]:
    out = []
    for plane in volume.PLANES:
        ax = volume.axis_for(plane)
        present = np.where(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        if present.size == 0:
            continue
        idx = np.unique(np.linspace(present[0], present[-1], min(per_plane, present.size)).round().astype(int))
        sp = volume._plane_spacing(vol.spacing_mm, plane)
        for k in idx:
            img, mm = volume._isotropic(volume._plane_image(vol.data, plane, int(k)), sp, cv2.INTER_LINEAR)
            m, _ = volume._isotropic(volume._plane_image(mask, plane, int(k)).astype(np.uint8), sp, cv2.INTER_NEAREST)
            raw = volume.RawImage(np.ascontiguousarray(img, np.float32), False, None, (mm, mm), {"plane": plane})
            out.append(volume.VolumeSlice(plane, int(k), 0.0, raw, m))
    return out


def export(rows: pd.DataFrame, out_dir: Path, per_plane: int, tumour_per_plane: int) -> list[dict]:
    recs = []
    for r in rows.itertuples():
        if not (isinstance(r.seg_path, str) and r.seg_path):
            continue                                     # no ground truth for slice labels
        try:
            vol = volume.load_volume(r.t1c_path)
            mask = volume.load_mask(r.seg_path, like=vol, image_path=r.t1c_path)
        except Exception as e:
            print(f"  skip {r.dataset}/{r.study_id}: {e}")
            continue
        d = out_dir / r.dataset / r.study_id
        d.mkdir(parents=True, exist_ok=True)
        slices = volume.select_slices(vol, per_plane=per_plane, mask=mask) + tumour_slices(vol, mask, tumour_per_plane)
        seen = set()
        for s in slices:
            if (s.plane, s.index) in seen:
                continue
            seen.add((s.plane, s.index))
            area = int(s.mask.sum())
            if 0 < area < MIN_TUMOUR_PX:
                continue
            label = r.label if area >= MIN_TUMOUR_PX else "no_tumor"
            sid = f"{r.dataset}__{r.study_id}__{s.plane[:3]}{s.index:03d}"
            img_p, msk_p = d / f"{s.plane[:3]}{s.index:03d}.png", d / f"{s.plane[:3]}{s.index:03d}_mask.png"
            cv2.imwrite(str(img_p), (np.clip(s.raw.pixels, 0, 1) * 255).round().astype(np.uint8))
            cv2.imwrite(str(msk_p), (s.mask > 0).astype(np.uint8) * 255)
            recs.append({"id": sid, "dataset": f"vol_{r.dataset}", "modality": "MR", "source_split": r.split,
                         "path": str(img_p), "mask_path": str(msk_p), "label": label, "plane": s.plane,
                         "patient_id": r.study_id, "md5": "", "dup_group": "", "patient_id_inferred": r.study_id,
                         "group": f"{r.dataset}:{r.study_id}", "split": r.split})
        print(f"  {r.dataset}/{r.study_id}: {len(seen)} slices", flush=True)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=Path, default=MANIFESTS / "volumes.csv")
    ap.add_argument("--base", type=Path, default=MANIFESTS / "mri_v2.csv")
    ap.add_argument("--out", type=Path, default=MANIFESTS / "mri_v3.csv")
    ap.add_argument("--per-plane", type=int, default=12)
    ap.add_argument("--tumour-per-plane", type=int, default=8)
    a = ap.parse_args()
    vols = pd.read_csv(a.volumes, keep_default_na=False)
    if "overlap" in vols:
        vols = vols[~vols["overlap"].astype(str).str.lower().eq("true")]
    vols = vols[(vols["role"] == "dev") & vols["split"].isin(SPLITS)]
    recs = export(vols, PROCESSED / "volume_slices", a.per_plane, a.tumour_per_plane)
    base = pd.read_csv(a.base)
    new = pd.DataFrame(recs)
    out = pd.concat([base, new.reindex(columns=base.columns)], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"{len(new)} volume slices added -> {a.out}")
    print(pd.crosstab(new["dataset"] + "/" + new["split"], new["label"]))


if __name__ == "__main__":
    main()

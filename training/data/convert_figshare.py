"""Convert the figshare brain-tumor dataset (Cheng et al., 2015) from MATLAB v7.3 to PNG.

Each .mat (HDF5) file holds `cjdata` with: label (1 meningioma, 2 glioma, 3 pituitary),
PID (patient id, uint16 char codes), image (int16, 512x512 or 256x256), tumorMask, tumorBorder.
MATLAB stores arrays column-major, so HDF5 arrays are transposed on read.

Images are min-max scaled to 8-bit (the same convention as the redistributed Kaggle copies, which
keeps near-duplicate detection against them meaningful). Masks are written as 0/255 PNG.

Usage:  python -m training.data.convert_figshare
"""

from __future__ import annotations

import csv

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from .paths import PROCESSED, RAW

LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def read_mat(path) -> dict:
    with h5py.File(path, "r") as f:
        cj = f["cjdata"]
        image = np.asarray(cj["image"][()], np.float64).T
        mask = np.asarray(cj["tumorMask"][()], np.uint8).T
        label = int(np.asarray(cj["label"][()]).ravel()[0])
        pid = "".join(chr(int(c)) for c in np.asarray(cj["PID"][()]).ravel())
    return {"image": image, "mask": mask, "label": LABELS[label], "patient_id": pid.strip()}


def to_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = float(image.min()), float(image.max())
    return np.zeros(image.shape, np.uint8) if hi <= lo else ((image - lo) / (hi - lo) * 255).round().astype(np.uint8)


def main() -> None:
    src = RAW / "figshare" / "mat"
    out_img, out_mask = PROCESSED / "figshare" / "images", PROCESSED / "figshare" / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.mat"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"No .mat files in {src}")
    rows = []
    for p in tqdm(files, desc="figshare"):
        d = read_mat(p)
        if d["mask"].shape != d["image"].shape:
            raise ValueError(f"{p.name}: mask {d['mask'].shape} != image {d['image'].shape}")
        name = f"figshare_{int(p.stem):04d}.png"
        cv2.imwrite(str(out_img / name), to_uint8(d["image"]))
        cv2.imwrite(str(out_mask / name), (d["mask"] > 0).astype(np.uint8) * 255)
        rows.append({"file": name, "label": d["label"], "patient_id": d["patient_id"],
                     "height": d["image"].shape[0], "width": d["image"].shape[1],
                     "mask_pixels": int((d["mask"] > 0).sum())})
    with open(PROCESSED / "figshare" / "figshare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    labels, pids = [r["label"] for r in rows], {r["patient_id"] for r in rows}
    print(f"{len(rows)} images, {len(pids)} patients, " +
          ", ".join(f"{k}={labels.count(k)}" for k in sorted(set(labels))))


if __name__ == "__main__":
    main()

"""Brain surface point cloud for the device home screen (edge/portal/static/brain.bin).

    python -m training.make_brain_points [--study PATH_TO_T1C.nii.gz] [--cell-mm 4]

Built from a real skull-stripped T1 post-contrast MRI (BraTS 2023 adult glioma, CC BY-NC; research use, as for the
model): brain tissue above a low intensity threshold (removes the CSF in the sulci so the folds show), largest
component, holes filled; its outer surface voxels are pooled into cells of CELL mm (one point per cell) with an outward
normal from the smoothed mask gradient. Output: int8 x, y, z, nx, ny, nz per point (RAS+, centred, scaled to +-127; normals of the smoothed mask),
read by static/portal.js and drawn on a 2D canvas with back-face culling and simple shading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

DEFAULT = Path("C:/Users/Tanmay/neuroedge_data/raw/brats_gli/BraTS-GLI-00005-000/BraTS-GLI-00005-000-t1c.nii.gz")
OUT = Path(__file__).resolve().parents[1] / "edge/portal/static/brain.bin"


def surface_points(t1c: Path, cell_mm: float) -> tuple[np.ndarray, np.ndarray]:
    img = nib.as_closest_canonical(nib.load(str(t1c)))
    v = np.asarray(img.dataobj, np.float32)
    zooms = np.array(img.header.get_zooms()[:3], np.float32)
    brain = v > 0
    m = (ndi.gaussian_filter(v, 0.8) > np.percentile(v[brain], 30)) & brain
    m = ndi.binary_opening(m, iterations=1)
    lab, _ = ndi.label(m)
    m = ndi.binary_fill_holes(lab == np.bincount(lab.ravel())[1:].argmax() + 1)
    # outer surface only: voxels of the filled mask that touch the outside
    outside = ndi.binary_dilation(~m) & m
    pts = np.argwhere(outside).astype(np.float32) * zooms
    g = np.stack(np.gradient(ndi.gaussian_filter(m.astype(np.float32), 1.5)), -1)[outside]
    nrm = -g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
    # one real surface voxel per cell, chosen at random (fixed seed): even density without a visible lattice
    order = np.random.default_rng(0).permutation(len(pts))
    keys = np.floor(pts[order] / cell_mm).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    P, N = pts[order][first], nrm[order][first]
    P -= (P.max(0) + P.min(0)) / 2
    return P, N


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=Path, default=DEFAULT)
    ap.add_argument("--cell-mm", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    P, N = surface_points(a.study, a.cell_mm)
    q = np.concatenate([np.round(P / np.abs(P).max() * 127), np.round(N * 127)], 1).clip(-127, 127).astype(np.int8)
    a.out.write_bytes(q.tobytes())
    print(f"{len(q)} points -> {a.out} ({a.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

"""3D MRI volumes -> the 2D slices the model reads, shared by training/evaluation and the Pi.

A radiologist never decides from one image: they scroll through the study in all three planes. This module loads a
whole study (NIfTI, or a DICOM series given as files / a zip) and returns evenly spaced slices from the axial,
coronal and sagittal planes, each as a `RawImage` that goes through the unchanged `preprocess()` (so the
preprocessing fingerprint, and therefore the deployed model contract, does not change).

Orientation: volumes are reoriented to the closest RAS+ axes, and slices are displayed the way the training images
look (radiological convention): axial/coronal with the patient's right on the image left, superior/anterior up.
Intensities are scaled once per volume (robust percentiles over head voxels), so all slices of a study share one scale.
Anisotropic voxels (e.g. 5 mm slices) are resampled in-plane to square pixels so anatomy is not squashed.
Only non-identifying information is kept from DICOM headers.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .preprocess import RawImage

PLANES = ("axial", "coronal", "sagittal")
MAX_FILES = 2000                     # DICOM files per series
MAX_VOXELS = 512 * 512 * 400         # refuse absurd volumes (decompression bombs)


@dataclass
class Volume:
    data: np.ndarray                         # (X, Y, Z) float32 in RAS+ axis order, intensities in [0, 1]
    spacing_mm: tuple[float, float, float]   # voxel size along X, Y, Z
    source: str                              # "nifti" | "dicom"
    meta: dict = field(default_factory=dict)


@dataclass
class VolumeSlice:
    plane: str
    index: int            # index along the plane's normal axis
    position: float       # 0..1 along that axis (for display: "slice 12 of 30")
    raw: RawImage
    mask: np.ndarray | None = None


# ----------------------------------------------------------------------------------------------- loading
def _scale_unit(data: np.ndarray, percentiles=(0.5, 99.5)) -> np.ndarray:
    """Robust per-volume scaling to [0, 1]: the low end from the whole volume (background stays black, as in the
    training images), the high end from the head voxels (so a large dark background does not compress the head)."""
    data = np.nan_to_num(data.astype(np.float32))
    lo = float(np.percentile(data, percentiles[0]))
    fg = data[data > lo]
    if fg.size < 100:
        fg = data.ravel()
    hi = float(np.percentile(fg, percentiles[1]))
    return np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _canonical(arr: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Reorder/flip axes to the closest RAS+ orientation; return data and voxel spacing in that order."""
    import nibabel as nib
    ornt = nib.orientations.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ornt, target)
    out = nib.orientations.apply_orientation(arr, transform)
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))      # voxel size of each input axis
    sp = [0.0, 0.0, 0.0]
    for src_axis, (dst_axis, _flip) in enumerate(transform):   # input axis i becomes output axis transform[i, 0]
        sp[int(dst_axis)] = float(zooms[src_axis])
    return np.ascontiguousarray(out), (sp[0], sp[1], sp[2])


def load_nifti(source: str | Path | bytes) -> Volume:
    import nibabel as nib
    if isinstance(source, (bytes, bytearray)):
        import gzip
        raw = bytes(source)
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        img = nib.Nifti1Image.from_bytes(raw)
    else:
        img = nib.load(str(source))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.ndim == 4:                      # some exports carry a singleton / multi-echo 4th axis
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {arr.shape}")
    if arr.size > MAX_VOXELS:
        raise ValueError("volume too large")
    data, spacing = _canonical(arr, img.affine)
    return Volume(_scale_unit(data), spacing, "nifti", {"shape": data.shape})


def _dicom_datasets(files: list[bytes]):
    import pydicom
    out = []
    for b in files[:MAX_FILES]:
        try:
            ds = pydicom.dcmread(io.BytesIO(b), force=True)
            if hasattr(ds, "PixelData") and hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
                out.append(ds)
        except Exception:
            continue
    return out


def load_dicom_series(files: list[bytes]) -> Volume:
    """One DICOM series (one file per slice) -> volume. Slices are ordered by position along the slice normal, the
    affine is built from the DICOM geometry, then reoriented to RAS+ like NIfTI."""
    dss = _dicom_datasets(files)
    if len(dss) < 3:
        raise ValueError("A series needs at least 3 DICOM slices with position information.")
    # One study = one patient, one series. Mixed uploads are refused rather than silently reduced to one series:
    # the doctor must know that every slice shown belongs to the same scan.
    for tag, what in (("PatientID", "patients"), ("StudyInstanceUID", "studies"), ("SeriesInstanceUID", "series")):
        vals = {str(getattr(d, tag, "")) for d in dss} - {""}
        if len(vals) > 1:
            raise ValueError(f"The upload mixes {len(vals)} different {what}; upload the files of one series only.")
    iop = np.asarray(dss[0].ImageOrientationPatient, dtype=np.float64)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    if any(not np.allclose(np.asarray(d.ImageOrientationPatient, dtype=np.float64), iop, atol=1e-3) for d in dss):
        raise ValueError("The slices have different orientations; upload one series only.")
    dss.sort(key=lambda d: float(np.dot(normal, np.asarray(d.ImagePositionPatient, dtype=np.float64))))
    pos = np.array([np.dot(normal, np.asarray(d.ImagePositionPatient, dtype=np.float64)) for d in dss])
    if np.any(np.diff(pos) < 1e-3):
        raise ValueError("Some slices are duplicated (same position); upload each slice of the series once.")
    rows, cols = int(dss[0].Rows), int(dss[0].Columns)
    if rows * cols * len(dss) > MAX_VOXELS:
        raise ValueError("volume too large")
    stack = []
    for d in dss:
        if (int(d.Rows), int(d.Columns)) != (rows, cols):
            continue
        a = d.pixel_array.astype(np.float32)
        if a.ndim != 2:
            raise ValueError("Enhanced (multi-frame) DICOM is not supported; export the series as one file per slice.")
        a = a * float(getattr(d, "RescaleSlope", 1) or 1) + float(getattr(d, "RescaleIntercept", 0) or 0)
        if str(getattr(d, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            a = a.max() - a
        stack.append(a)
    vol = np.stack(stack, axis=-1)                      # (rows, cols, n) = (i, j, k)
    ps = [float(x) for x in getattr(dss[0], "PixelSpacing", [1.0, 1.0])]
    p0 = np.asarray(dss[0].ImagePositionPatient, dtype=np.float64)
    p1 = np.asarray(dss[-1].ImagePositionPatient, dtype=np.float64)
    step = (p1 - p0) / max(len(stack) - 1, 1)
    # DICOM patient coordinates are LPS. PixelSpacing = (between rows, between columns). The row index i advances
    # along the column direction cosine, the column index j along the row direction cosine.
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = col * ps[0]        # i (row index) advances along the column direction
    affine_lps[:3, 1] = row * ps[1]        # j (column index) advances along the row direction
    affine_lps[:3, 2] = step
    affine_lps[:3, 3] = p0
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    data, spacing = _canonical(vol, lps_to_ras @ affine_lps)
    meta = {"modality": str(getattr(dss[0], "Modality", "")) or None, "n_files": len(stack),
            "series_description": str(getattr(dss[0], "SeriesDescription", ""))[:64]}
    return Volume(_scale_unit(data), spacing, "dicom", meta)


def load_volume(source: str | Path | bytes | list[bytes], filename: str | None = None) -> Volume:
    """NIfTI path/bytes, a DICOM series directory, a list of DICOM file bytes, or a .zip of a series."""
    if isinstance(source, list):
        return load_dicom_series(source)
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return load_dicom_series([f.read_bytes() for f in sorted(p.rglob("*")) if f.is_file()][:MAX_FILES])
        filename = filename or p.name
        source = p.read_bytes()
    name = (filename or "").lower()
    if name.endswith((".nii", ".nii.gz")) or source[:2] == b"\x1f\x8b" or source[344:348] in (b"n+1\x00", b"ni1\x00"):
        return load_nifti(source)
    if name.endswith(".zip") or source[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(source)) as z:
            infos = [i for i in z.infolist() if not i.is_dir()][:MAX_FILES]
            if sum(i.file_size for i in infos) > 1_000_000_000:
                raise ValueError("series archive too large")
            nifti = [i for i in infos if i.filename.lower().endswith((".nii", ".nii.gz"))]
            if nifti:
                return load_nifti(z.read(nifti[0]))
            return load_dicom_series([z.read(i) for i in infos])
    raise ValueError("Unsupported series format (use NIfTI, a DICOM series, or a .zip of one).")


# ---------------------------------------------------------------------------------------------- slicing
def _plane_image(data: np.ndarray, plane: str, k: int) -> np.ndarray:
    """2D image in display convention from a RAS+ volume."""
    if plane == "axial":        # (X, Y) at z=k -> rows: anterior at top; cols: patient right on image left
        s = data[:, :, k]
        return s.T[::-1, ::-1]
    if plane == "coronal":      # (X, Z) at y=k -> rows: superior at top; cols: patient right on image left
        s = data[:, k, :]
        return s.T[::-1, ::-1]
    s = data[k, :, :]           # sagittal (Y, Z) at x=k -> rows: superior at top; cols: anterior on image left
    return s.T[::-1, ::-1]


def _plane_spacing(spacing: tuple[float, float, float], plane: str) -> tuple[float, float]:
    sx, sy, sz = spacing
    return {"axial": (sy, sx), "coronal": (sz, sx), "sagittal": (sz, sy)}[plane]   # (row, col) mm


def _isotropic(img: np.ndarray, sp_rc: tuple[float, float], interpolation: int) -> tuple[np.ndarray, float]:
    """Resample so rows and columns have the same mm size (the finer of the two)."""
    r, c = sp_rc
    if r <= 0 or c <= 0 or abs(r - c) / max(r, c) < 0.05:
        return img, float(min(r, c) or 1.0)
    target = min(r, c)
    h, w = img.shape
    out = cv2.resize(img, (max(1, round(w * c / target)), max(1, round(h * r / target))), interpolation=interpolation)
    return out, float(target)


def axis_for(plane: str) -> int:
    return {"sagittal": 0, "coronal": 1, "axial": 2}[plane]


def head_extent(data: np.ndarray, axis: int, min_fraction: float = 0.08) -> tuple[int, int]:
    """First/last index along `axis` whose slice contains enough head (bright) area."""
    thr = max(0.05, float(np.percentile(data, 60)) * 0.5)
    frac = (data > thr).mean(axis=tuple(a for a in range(3) if a != axis))   # head area per slice
    keep = np.where(frac >= min_fraction * max(float(frac.max()), 1e-6))[0]
    if keep.size == 0:
        return 0, data.shape[axis] - 1
    return int(keep[0]), int(keep[-1])


def select_slices(vol: Volume, per_plane: int = 12, planes: tuple[str, ...] = PLANES,
                  mask: np.ndarray | None = None, trim: float = 0.08) -> list[VolumeSlice]:
    """Evenly spaced slices across the head in each plane (the outer `trim` of the head extent is skipped: those
    slices show only skull/vertex and carry no diagnostic content)."""
    out = []
    for plane in planes:
        ax = axis_for(plane)
        lo, hi = head_extent(vol.data, ax)
        span = hi - lo
        lo, hi = lo + int(trim * span), hi - int(trim * span)
        n = min(per_plane, hi - lo + 1)
        idx = np.unique(np.linspace(lo, hi, n).round().astype(int)) if n > 0 else []
        sp = _plane_spacing(vol.spacing_mm, plane)
        for k in idx:
            img, mm = _isotropic(_plane_image(vol.data, plane, int(k)), sp, cv2.INTER_LINEAR)
            m = None
            if mask is not None:
                m, _ = _isotropic(_plane_image(mask, plane, int(k)).astype(np.uint8), sp, cv2.INTER_NEAREST)
            raw = RawImage(np.ascontiguousarray(img, dtype=np.float32), False, None, (mm, mm),
                           {"plane": plane, "slice": int(k)})
            out.append(VolumeSlice(plane, int(k), (k - lo) / max(hi - lo, 1), raw, m))
    return out


def load_mask(path: str | Path, like: Volume | None = None, image_path: str | Path | None = None) -> np.ndarray:
    """Segmentation NIfTI -> binary (X, Y, Z) uint8 in the same RAS+ orientation as `load_nifti`.

    If the mask is stored on a different voxel grid than the image (e.g. the NHNN pituitary masks), it is resampled
    onto the image grid through the two affines (nearest neighbour), given `image_path`."""
    import nibabel as nib
    img = nib.load(str(path))
    if image_path is not None:
        ref = nib.load(str(image_path))
        if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-3):
            from nibabel.processing import resample_from_to
            data3 = np.asarray(img.dataobj)
            if data3.ndim == 4:
                data3 = data3[..., 0]
            img = resample_from_to(nib.Nifti1Image((data3 > 0).astype(np.uint8), img.affine), (ref.shape[:3], ref.affine),
                                   order=0)
    arr = np.asarray(img.dataobj)
    if arr.ndim == 4:
        arr = arr[..., 0]
    data, _ = _canonical((arr > 0).astype(np.uint8), img.affine)
    if like is not None and data.shape != like.data.shape:
        raise ValueError(f"mask shape {data.shape} != volume shape {like.data.shape}")
    return data

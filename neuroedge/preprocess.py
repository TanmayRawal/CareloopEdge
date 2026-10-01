"""Image loading and preprocessing — the ONLY implementation, shared by training and the Pi.

Pipeline:  read (PNG/JPG/DICOM) -> unit-range intensities -> head-foreground crop
           -> pad to square -> resize -> [optional CLAHE] -> replicate to 3 ch -> normalize.

Every geometric step is recorded in a `Geometry`, so maps produced in model space (CAM,
segmentation) can be mapped back exactly onto the original image (`inverse_map`).

Notes on honesty of naming: the crop is a *head-foreground bounding-box crop*; it is not skull
stripping. DICOM uploads are reduced to a whitelist of non-identifying tags at read time, so
patient-identifying information never leaves `read_image`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CT_WINDOWS, PreprocessConfig

# DICOM tags that are safe to keep (no patient identity). Everything else is dropped.
_SAFE_DICOM_TAGS = ("Modality", "PixelSpacing", "SliceThickness", "Rows", "Columns",
                    "BodyPartExamined", "PhotometricInterpretation")


@dataclass
class RawImage:
    pixels: np.ndarray            # (H, W) float32; HU for CT DICOM, [0, 1] otherwise
    is_hu: bool                   # True when pixels are Hounsfield units
    modality: str | None          # "MR" / "CT" from DICOM, else None
    pixel_spacing_mm: tuple[float, float] | None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Geometry:
    orig_hw: tuple[int, int]
    crop_box: tuple[int, int, int, int]   # x0, y0, x1, y1 in original pixels (x1/y1 exclusive)
    square: int                           # side of the zero-padded square crop
    pad_tl: tuple[int, int]               # (top, left) offset of the crop inside the square
    size: int                             # model input side

    @property
    def scale(self) -> float:
        """Model pixels per original pixel."""
        return self.size / self.square


@dataclass
class PreprocessResult:
    tensor: np.ndarray            # (3, S, S) float32, normalized — model input
    display: np.ndarray           # (S, S) uint8 — exactly what the model saw, for overlays
    original: np.ndarray          # (H, W) uint8 — original image for full-resolution overlays
    geometry: Geometry
    modality: str | None
    mm_per_model_px: float | None  # physical size of one model pixel (if spacing known)
    quality_flags: list[str]
    mask: np.ndarray | None = None  # (S, S) uint8 {0,1} if a mask was supplied


# --------------------------------------------------------------------------------------- reading
def _is_dicom(data: bytes, filename: str | None) -> bool:
    if filename and filename.lower().endswith((".dcm", ".dicom")):
        return True
    return len(data) > 132 and data[128:132] == b"DICM"


def _read_dicom(data: bytes) -> RawImage:
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array.astype(np.float32)
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if arr.ndim == 3 and photometric not in ("RGB", "YBR_FULL", "YBR_FULL_422"):
        raise ValueError("Multi-frame DICOM is not supported by the 2D model; upload one slice.")
    if arr.ndim == 3:  # colour DICOM (rare: screenshots / secondary captures)
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    modality = str(getattr(ds, "Modality", "")) or None
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    is_hu = modality == "CT"
    if not is_hu and photometric == "MONOCHROME1":
        arr = arr.max() - arr

    spacing = getattr(ds, "PixelSpacing", None)
    spacing_mm = (float(spacing[0]), float(spacing[1])) if spacing is not None else None
    meta = {t: str(getattr(ds, t)) for t in _SAFE_DICOM_TAGS if hasattr(ds, t)}
    return RawImage(arr, is_hu, modality, spacing_mm, meta)


def _read_raster(data: bytes) -> RawImage:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image (supported: PNG, JPG, BMP, TIFF, DICOM).")
    max_val = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    colorfulness = 0.0
    if img.ndim == 3:
        img = img[..., :3]
        colorfulness = _colorfulness(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = img.astype(np.float32) / max_val
    return RawImage(pixels, False, None, None, {"colorfulness": colorfulness})


def read_image(source: str | Path | bytes, filename: str | None = None) -> RawImage:
    """Read PNG/JPG/DICOM from a path or raw bytes (web uploads)."""
    if isinstance(source, (str, Path)):
        filename = filename or str(source)
        data = Path(source).read_bytes()
    else:
        data = source
    return _read_dicom(data) if _is_dicom(data, filename) else _read_raster(data)


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness. Grayscale scans ~0; natural photos >> 10."""
    b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


# ------------------------------------------------------------------------------ intensity mapping
def window_ct(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """Linear CT window to [0, 1]: values below L-W/2 -> 0, above L+W/2 -> 1."""
    lo = level - width / 2.0
    return np.clip((hu - lo) / width, 0.0, 1.0).astype(np.float32)


def to_unit_range(raw: RawImage, cfg: PreprocessConfig, ct_mode: str = "brain") -> np.ndarray:
    """Map raw pixels to float32 [0, 1]. Returns (H, W) or (H, W, 3) for multi-window CT.

    ct_mode="brain": single brain window, replicated later — matches public CT datasets, which
    are distributed as already-windowed grayscale PNG/JPG. ct_mode="multi" (brain/subdural/bone
    stacked) is only valid for a model trained on multi-window DICOM.
    """
    if raw.is_hu:
        if ct_mode == "multi":
            return np.stack([window_ct(raw.pixels, l, w) for l, w in CT_WINDOWS], axis=-1)
        level, width = CT_WINDOWS[0]
        return window_ct(raw.pixels, level, width)
    px = raw.pixels
    if raw.modality is not None:  # DICOM MR: robust percentile scaling
        lo, hi = np.percentile(px, cfg.mr_percentiles)
        px = (px - lo) / max(hi - lo, 1e-6)
    return np.clip(px, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------- geometry
def foreground_crop_box(gray01: np.ndarray, margin: float = 0.04) -> tuple[int, int, int, int]:
    """Bounding box of the largest bright connected component (the head), with margin.

    Falls back to the full image when no plausible foreground is found (<5% of the area).
    """
    h, w = gray01.shape
    u8 = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0, 0, w, h
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[idx]
    if area < 0.05 * h * w:
        return 0, 0, w, h
    m = int(round(margin * max(bw, bh)))
    return max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)


def make_geometry(orig_hw: tuple[int, int], box: tuple[int, int, int, int], size: int) -> Geometry:
    x0, y0, x1, y1 = box
    ch, cw = y1 - y0, x1 - x0
    side = max(ch, cw)
    return Geometry(orig_hw, box, side, ((side - ch) // 2, (side - cw) // 2), size)


def apply_geometry(img: np.ndarray, g: Geometry, interpolation: int) -> np.ndarray:
    """Crop -> zero-pad to square (centred) -> resize to model size."""
    x0, y0, x1, y1 = g.crop_box
    crop = img[y0:y1, x0:x1]
    sq_shape = (g.square, g.square) + crop.shape[2:]
    square = np.zeros(sq_shape, dtype=img.dtype)
    t, l = g.pad_tl
    square[t:t + crop.shape[0], l:l + crop.shape[1]] = crop
    if g.square == g.size:
        return square
    return cv2.resize(square, (g.size, g.size), interpolation=interpolation)


def inverse_map(model_map: np.ndarray, g: Geometry, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Map an (S, S) model-space map back to original image coordinates.

    Pixels outside the crop were never seen by the model and are returned as 0.
    """
    square = cv2.resize(model_map.astype(np.float32), (g.square, g.square), interpolation=interpolation)
    x0, y0, x1, y1 = g.crop_box
    t, l = g.pad_tl
    out = np.zeros(g.orig_hw, dtype=np.float32)
    out[y0:y1, x0:x1] = square[t:t + (y1 - y0), l:l + (x1 - x0)]
    return out


# -------------------------------------------------------------------------------------- quality
def quality_flags(gray01: np.ndarray, raw: RawImage) -> list[str]:
    """Cheap input-quality checks. Thresholds are provisional; calibrate on validation data."""
    flags = []
    h, w = gray01.shape[:2]
    if min(h, w) < 128:
        flags.append("low_resolution")
    u8 = (np.clip(gray01 if gray01.ndim == 2 else gray01[..., 0], 0, 1) * 255).astype(np.uint8)
    if cv2.Laplacian(u8, cv2.CV_64F).var() < 10.0:
        flags.append("blurry_or_flat")
    if (u8 > 20).mean() < 0.05:
        flags.append("mostly_empty")
    if raw.meta.get("colorfulness", 0.0) > 15.0:
        flags.append("color_image")
    return flags


# --------------------------------------------------------------------------------------- driver
def normalize(img01: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """(S,S) or (S,S,3) in [0,1] -> (3,S,S) float32 normalized with ImageNet statistics."""
    if img01.ndim == 2:
        img01 = np.repeat(img01[..., None], 3, axis=-1)
    mean = np.asarray(cfg.mean, np.float32)
    std = np.asarray(cfg.std, np.float32)
    return np.ascontiguousarray(((img01 - mean) / std).transpose(2, 0, 1), dtype=np.float32)


def preprocess(source: str | Path | bytes | RawImage, cfg: PreprocessConfig, *,
               filename: str | None = None, mask: np.ndarray | None = None,
               ct_mode: str = "brain") -> PreprocessResult:
    raw = source if isinstance(source, RawImage) else read_image(source, filename)
    unit = to_unit_range(raw, cfg, ct_mode)
    gray = unit if unit.ndim == 2 else unit[..., 0]
    h, w = gray.shape

    box = foreground_crop_box(gray, cfg.crop_margin) if cfg.brain_crop else (0, 0, w, h)
    geom = make_geometry((h, w), box, cfg.image_size)
    model_img = apply_geometry(unit, geom, cv2.INTER_AREA if geom.square > geom.size else cv2.INTER_LINEAR)
    model_img = np.clip(model_img, 0.0, 1.0).astype(np.float32)

    if cfg.clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid,) * 2)
        chans = model_img if model_img.ndim == 3 else model_img[..., None]
        chans = np.stack([clahe.apply((c * 255).astype(np.uint8)) / 255.0
                          for c in np.moveaxis(chans, -1, 0)], -1).astype(np.float32)
        model_img = chans if model_img.ndim == 3 else chans[..., 0]

    display = ((model_img if model_img.ndim == 2 else model_img[..., 0]) * 255).round().astype(np.uint8)
    out_mask = None
    if mask is not None:
        if mask.shape[:2] != (h, w):
            raise ValueError(f"mask shape {mask.shape} != image shape {(h, w)}")
        out_mask = (apply_geometry((mask > 0).astype(np.uint8), geom, cv2.INTER_NEAREST) > 0).astype(np.uint8)

    mm = None
    if raw.pixel_spacing_mm is not None:
        mm = float(np.sqrt(raw.pixel_spacing_mm[0] * raw.pixel_spacing_mm[1])) / geom.scale

    return PreprocessResult(
        tensor=normalize(model_img, cfg),
        display=display,
        original=(np.clip(gray, 0, 1) * 255).round().astype(np.uint8),
        geometry=geom,
        modality=raw.modality,
        mm_per_model_px=mm,
        quality_flags=quality_flags(unit, raw),
        mask=out_mask,
    )

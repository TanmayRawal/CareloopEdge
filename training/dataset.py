"""Datasets for training/evaluation.

Every image goes through neuroedge.preprocess.preprocess() ONCE (the exact code the Pi runs) and
the model-space result (uint8 SxS, plus mask) is cached in a .npz keyed by the preprocessing
fingerprint — changing the preprocessing config can never silently reuse a stale cache.

Augmentations (train only) act jointly on image and mask in model space:
  random-resized-crop (scale 0.8-1.0), horizontal flip, rotation +-10 deg,
  contrast/brightness/gamma jitter, mild Gaussian noise.
Deliberately NOT used: MixUp/CutMix/CutOut (they corrupt spatial attribution targets and the
calibration of soft labels) and label smoothing (interferes with temperature scaling and OOD).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from neuroedge.config import PreprocessConfig
from neuroedge.preprocess import normalize, preprocess


# ------------------------------------------------------------------------------- caching
def _load_one(row, cfg: PreprocessConfig):
    mask = None
    if isinstance(row.mask_path, str) and row.mask_path:
        m = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"unreadable mask {row.mask_path}")
        mask = m
        img = cv2.imread(row.path, cv2.IMREAD_GRAYSCALE)
        if img is not None and m.shape != img.shape:   # a few masks differ in size: resize nearest
            mask = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    r = preprocess(row.path, cfg, mask=mask)
    out_mask = r.mask if r.mask is not None else np.zeros_like(r.display)
    return r.display, out_mask


def build_cache(df: pd.DataFrame, cfg: PreprocessConfig, cache_root: Path, name: str,
                workers: int = 8) -> Path:
    """Cache dir with images.npy / masks.npy (memory-mappable, shared by DataLoader workers)."""
    path = cache_root / f"{name}_{cfg.fingerprint()}"
    ids_file = path / "ids.txt"
    if ids_file.exists() and ids_file.read_text(encoding="utf-8").splitlines() == df["id"].tolist():
        return path
    path.mkdir(parents=True, exist_ok=True)
    rows = list(df.itertuples())
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(lambda r: _load_one(r, cfg), rows))
    np.save(path / "images.npy", np.stack([o[0] for o in out]))
    np.save(path / "masks.npy", np.stack([o[1] for o in out]))
    ids_file.write_text("\n".join(df["id"].tolist()), encoding="utf-8")   # written last = commit marker
    return path


# -------------------------------------------------------------------------- augmentation
def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """img float32 [0,1] (S,S); mask uint8 {0,1} (S,S)."""
    s = img.shape[0]
    if rng.random() < 0.8:                                   # random resized crop
        scale = rng.uniform(0.8, 1.0)
        ratio = math.exp(rng.uniform(math.log(0.9), math.log(1.1)))
        w = min(s, int(round(s * math.sqrt(scale * ratio))))
        h = min(s, int(round(s * math.sqrt(scale / ratio))))
        x0, y0 = int(rng.integers(0, s - w + 1)), int(rng.integers(0, s - h + 1))
        img = cv2.resize(img[y0:y0 + h, x0:x0 + w], (s, s), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask[y0:y0 + h, x0:x0 + w], (s, s), interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.5:                                   # horizontal flip
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:                                   # small rotation
        m = cv2.getRotationMatrix2D((s / 2, s / 2), rng.uniform(-10, 10), 1.0)
        img = cv2.warpAffine(np.ascontiguousarray(img), m, (s, s), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), m, (s, s), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.8:                                   # intensity jitter (image only)
        c, b = rng.uniform(0.85, 1.15), rng.uniform(-0.06, 0.06)
        img = np.clip((img - 0.5) * c + 0.5 + b, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, rng.uniform(0, 0.02), img.shape), 0, 1)
    return np.ascontiguousarray(img, np.float32), np.ascontiguousarray(mask)


# ------------------------------------------------------------------------------- dataset
class TumorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_path: Path, class_names: tuple[str, ...],
                 cfg: PreprocessConfig, train: bool, seed: int = 0):
        cached_ids = (Path(cache_path) / "ids.txt").read_text(encoding="utf-8").splitlines()
        idx = {k: i for i, k in enumerate(cached_ids)}
        self.sel = np.array([idx[i] for i in df["id"]])
        self.cache_path = Path(cache_path)
        self._images = self._masks = None              # opened lazily (memory-mapped) per process
        self.labels = np.array([class_names.index(l) for l in df["label"]], np.int64)
        no_tumor = df["label"].isin(["no_tumor", "healthy"]).to_numpy()
        has_mask_file = df["mask_path"].fillna("").astype(str).ne("").to_numpy() if "mask_path" in df else False
        # A tumor-free image has a known all-zero mask; tumor images need an annotation.
        self.has_mask = (has_mask_file | no_tumor).astype(np.float32)
        self.ids = df["id"].tolist()
        self.cfg, self.train, self.seed = cfg, train, seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = state["_masks"] = None      # never pickle the arrays into workers
        state["_rng"] = None
        return state

    def _generator(self) -> np.random.Generator:
        """One generator per worker process, seeded from torch's per-worker seed (itself derived
        from the run seed), advancing across epochs — persistent workers get fresh augmentations
        every epoch while runs stay reproducible."""
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(info.seed if info is not None else self.seed)
        return self._rng

    def _arrays(self):
        if self._images is None:
            self._images = np.load(self.cache_path / "images.npy", mmap_mode="r")
            self._masks = np.load(self.cache_path / "masks.npy", mmap_mode="r")
        return self._images, self._masks

    def __getitem__(self, i: int) -> dict:
        images, masks = self._arrays()
        img = images[self.sel[i]].astype(np.float32) / 255.0
        mask = (masks[self.sel[i]] > 0).astype(np.uint8)
        if self.train:
            img, mask = augment(img, mask, self._generator())
        return {"image": torch.from_numpy(normalize(img, self.cfg)),
                "label": torch.tensor(self.labels[i]),
                "mask": torch.from_numpy(mask[None].astype(np.float32)),
                "has_mask": torch.tensor(self.has_mask[i]),
                "index": torch.tensor(i)}

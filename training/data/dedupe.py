"""Leakage control: flip-aware near-duplicate and same-patient grouping across all image sources.

Why not perceptual hashes alone: on brain MRI, 64-bit pHash distances of *distinct* slices are
small (all brains look alike at 8x8 DCT resolution), so no hash threshold separates copies from
different images. Auditing BRISC 2025 also showed that most of its test/train duplicates are
LEFT-RIGHT MIRRORED copies (augmentation leakage), which hashes do not catch.

Method
  1. Each image -> 64x64 grayscale thumbnail, z-normalised, plus its horizontal mirror.
  2. Similarity = max(NCC(a, b), NCC(a, mirror(b))). NCC(a, mirror(b)) = NCC(mirror(a), b), so
     one mirrored copy per image suffices.
  3. Two thresholds, both chosen from data and recorded in the audit report:
       * t_dup     — copies / mirrors / re-encodes. The NCC distribution is bimodal with an
                     empty gap around 0.95-0.98 (BRISC: 566 pairs >= 0.98, 3 in [0.95, 0.98)).
       * t_patient — "probably the same patient" (adjacent slices). Calibrated on figshare, which
                     has true patient IDs: the largest value at which different-patient pairs are
                     still essentially never linked.
  4. Union-find over pairs >= t_patient, plus exact MD5 matches -> group ids. A group never spans
     train/val/test; external-test images whose group touches the training source are excluded.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

THUMB = 64


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _thumb_pair(path: str, size: int = THUMB) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"unreadable image: {path}")
    t = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    out = []
    for v in (t, t[:, ::-1]):
        v = (v - v.mean()) / (v.std() + 1e-6)
        out.append(v.ravel() / size)          # unit L2 norm -> dot product == NCC
    return np.stack(out)


def thumbnails(paths: list[str], workers: int = 8) -> np.ndarray:
    """(N, 2, size*size) float32: [original, mirrored], each unit-norm."""
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(_thumb_pair, paths, chunksize=64))).astype(np.float32)


def md5_all(paths: list[str], workers: int = 8) -> list[str]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(file_md5, paths, chunksize=64))


def best_similarity(t_src: np.ndarray, t_dst: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Flip-aware NCC matrix (len(src), len(dst))."""
    out = np.empty((len(t_src), len(t_dst)), np.float32)
    for s in range(0, len(t_src), chunk):
        a = t_src[s:s + chunk, 0]
        out[s:s + chunk] = np.maximum(a @ t_dst[:, 0].T, a @ t_dst[:, 1].T)
    return out


def similar_pairs(t: np.ndarray, threshold: float, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """All pairs (i < j) with flip-aware NCC >= threshold, and their similarities."""
    pairs, sims = [], []
    for s in range(0, len(t), chunk):
        sim = best_similarity(t[s:s + chunk], t, chunk)
        i, j = np.nonzero(sim >= threshold)
        keep = (i + s) < j
        pairs.append(np.stack([i[keep] + s, j[keep]], axis=1))
        sims.append(sim[i[keep], j[keep]])
    return np.concatenate(pairs), np.concatenate(sims)


def nearest_other(t: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """For each image, the highest flip-aware NCC to any *other* image."""
    best = np.empty(len(t), np.float32)
    for s in range(0, len(t), chunk):
        sim = best_similarity(t[s:s + chunk], t, chunk)
        sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -1.0
        best[s:s + chunk] = sim.max(axis=1)
    return best


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_ids(n: int, pairs: np.ndarray, *key_lists: list[str]) -> np.ndarray:
    """Connected components over similarity pairs plus "same key" links -> dense group ids.

    Each key list (e.g. MD5 digests, known patient IDs) links all items sharing an identical
    non-empty key.
    """
    uf = UnionFind(n)
    for a, b in pairs:
        uf.union(int(a), int(b))
    for keys in key_lists:
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            if not k:
                continue
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    roots = np.array([uf.find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]

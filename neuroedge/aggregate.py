"""Slice results -> one patient-level answer (shared by evaluation and the Pi).

A study is many slices; most show no tumour even when the patient has one. Each rule turns the per-slice calibrated
probabilities (N, C) into one (C,) distribution:

  mean      plain average over slices            (penalises small tumours seen on few slices)
  topk      average of the k slices with the highest tumour probability; the decision "tumour or not" and the
            tumour type both come from the slices that actually show something
  max       topk with k = 1
  lesion    slices weighted by the segmentation head's lesion area (slices without a lesion get a small weight)

Which rule is used in the product is chosen on calibration patients only (training/evaluate_volumes.py) and stored in
the bundle; the other rules stay available for research.
"""

from __future__ import annotations

import numpy as np

RULES = ("mean", "topk", "max", "lesion")


def tumour_prob(probs: np.ndarray, no_tumour_idx: int) -> np.ndarray:
    return 1.0 - probs[:, no_tumour_idx]


def aggregate(probs: np.ndarray, rule: str, no_tumour_idx: int, k: int = 5,
              lesion_area: np.ndarray | None = None, ok: np.ndarray | None = None) -> np.ndarray:
    """probs: (N, C) per-slice calibrated probabilities. ok: slices to use (e.g. not rejected as out-of-distribution);
    if no slice is usable, all are used. Returns a (C,) distribution summing to 1."""
    probs = np.asarray(probs, np.float64)
    if probs.ndim != 2 or len(probs) == 0:
        raise ValueError("need at least one slice")
    if ok is not None and np.asarray(ok, bool).any():
        sel = np.asarray(ok, bool)
        probs = probs[sel]
        lesion_area = None if lesion_area is None else np.asarray(lesion_area, np.float64)[sel]
    if rule == "mean":
        out = probs.mean(axis=0)
    elif rule in ("topk", "max"):
        kk = 1 if rule == "max" else max(1, min(k, len(probs)))
        top = np.argsort(-tumour_prob(probs, no_tumour_idx), kind="stable")[:kk]
        out = probs[top].mean(axis=0)
    elif rule == "lesion":
        if lesion_area is None:
            raise ValueError("rule 'lesion' needs lesion_area")
        w = np.asarray(lesion_area, np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(len(probs), 1.0 / len(probs))
        w = 0.9 * w + 0.1 / len(probs)              # slices without a lesion still count a little
        out = (probs * w[:, None]).sum(axis=0)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return out / out.sum()


def key_slices(probs: np.ndarray, no_tumour_idx: int, n: int = 3) -> list[int]:
    """Indices of the slices that most support 'tumour' (shown to the doctor first)."""
    return [int(i) for i in np.argsort(-tumour_prob(np.asarray(probs), no_tumour_idx), kind="stable")[:n]]

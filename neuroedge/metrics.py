"""Evaluation metrics: classification, calibration, statistics, and explanation quality.

Classification   accuracy, balanced accuracy, macro-F1, per-class sensitivity/specificity/
                 precision/F1/AUROC (one-vs-rest), Cohen's kappa, MCC.
Calibration      top-label ECE / MCE with equal-width bins (Guo et al. 2017).
Statistics       percentile bootstrap CIs; exact McNemar test for paired model comparison
                 (e.g. FP32 vs INT8 on the same test set).
XAI              pointing game (Zhang et al. 2018), energy-based pointing (Wang et al. 2020),
                 top-k% IoU / Dice against expert masks, and map-similarity (Pearson, Spearman,
                 SSIM, top-k IoU) for quantization-drift and cross-method agreement.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from scipy import stats


# ------------------------------------------------------------------------- classification
def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, roc_auc_score)

    y_true = np.asarray(y_true, np.int64)
    probs = np.asarray(probs, np.float64)
    y_pred = probs.argmax(axis=1)
    n_cls = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        both_pos = (y_true == i).any() and (y_true != i).any()
        per_class[name] = {
            "sensitivity": float(sens),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "precision": float(prec),
            "f1": float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0,
            "auroc": float(roc_auc_score(y_true == i, probs[:, i])) if both_pos else float("nan"),
            "support": int(tp + fn),
        }
    # Macro averages run over classes PRESENT in y_true: an external set without a class (e.g.
    # figshare has no "no_tumor") must not score that class as F1 = 0. Predictions INTO an absent
    # class still count as errors (they lower the recall of the true class).
    present = np.unique(y_true)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=present.tolist())),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_auroc": float(np.nanmean([per_class[class_names[i]]["auroc"] for i in present])),
        "classes_present": [class_names[i] for i in present],
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------- calibration
def calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Top-label ECE/MCE with equal-width confidence bins; also returns the reliability table."""
    probs = np.asarray(probs, np.float64)
    labels = np.asarray(labels, np.int64)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, table = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not sel.any():
            continue
        gap = abs(correct[sel].mean() - conf[sel].mean())
        ece += sel.mean() * gap
        mce = max(mce, gap)
        table.append({"bin": [float(lo), float(hi)], "count": int(sel.sum()),
                      "accuracy": float(correct[sel].mean()), "confidence": float(conf[sel].mean())})
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    return {"ece": float(ece), "mce": float(mce), "nll": nll, "reliability": table}


# ------------------------------------------------------------------------------ statistics
def bootstrap_ci(metric: Callable[[np.ndarray], float], n_items: int, n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap. `metric(idx)` evaluates the statistic on resampled indices."""
    rng = np.random.default_rng(seed)
    vals = np.array([metric(rng.integers(0, n_items, n_items)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(metric(np.arange(n_items))), float(lo), float(hi)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on paired correctness vectors."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}


# --------------------------------------------------------------------------------- XAI maps
def _fit(cam: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if cam.shape == shape:
        return cam.astype(np.float32)
    return cv2.resize(cam.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def pointing_game(cam: np.ndarray, mask: np.ndarray, tolerance_px: int = 15) -> bool:
    """Hit if the CAM maximum falls inside the mask dilated by `tolerance_px` (Zhang et al.)."""
    cam = _fit(cam, mask.shape)
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    k = 2 * tolerance_px + 1
    dil = cv2.dilate((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return bool(dil[y, x])


def energy_pointing(cam: np.ndarray, mask: np.ndarray) -> float:
    cam = np.maximum(_fit(cam, mask.shape), 0)
    tot = float(cam.sum())
    return float(cam[mask > 0].sum() / tot) if tot > 0 else float("nan")


def _topk_binary(cam: np.ndarray, frac: float) -> np.ndarray:
    thr = np.quantile(cam, 1.0 - frac)
    return cam >= thr


def topk_overlap(cam: np.ndarray, mask: np.ndarray, frac: float = 0.15) -> dict:
    """IoU and Dice between the top-`frac` CAM pixels and the mask."""
    cam = _fit(cam, mask.shape)
    pred, gt = _topk_binary(cam, frac), mask > 0
    inter, union = float((pred & gt).sum()), float((pred | gt).sum())
    denom = float(pred.sum() + gt.sum())
    return {"iou": inter / union if union else float("nan"),
            "dice": 2 * inter / denom if denom else float("nan")}


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Gaussian-window SSIM (Wang et al. 2004), sigma=1.5, 11x11."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    s_aa, s_bb, s_ab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2)
    return float((num / den).mean())


def map_similarity(a: np.ndarray, b: np.ndarray, frac: float = 0.2) -> dict:
    """Agreement between two explanation maps of the same image (drift / cross-method)."""
    b = _fit(b, a.shape)
    fa, fb = a.ravel(), b.ravel()
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)  # noqa: E731
    ta, tb = _topk_binary(a, frac), _topk_binary(b, frac)
    union = float((ta | tb).sum())
    return {
        "pearson": float(np.corrcoef(fa, fb)[0, 1]) if fa.std() > 0 and fb.std() > 0 else float("nan"),
        "spearman": float(stats.spearmanr(fa, fb)[0]),
        "ssim": ssim(norm(a), norm(b)),
        f"top{int(frac * 100)}_iou": float((ta & tb).sum()) / union if union else float("nan"),
    }

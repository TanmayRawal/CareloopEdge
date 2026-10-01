"""Calibration, conformal prediction sets and OOD detection — numpy-only at inference time.

TemperatureScaler       Guo et al., ICML 2017. One scalar T fitted by NLL on a calibration split.
ClassConditionalConformal
                        Split conformal with the LAC score s(x,y) = 1 - p_y(x), calibrated per
                        class (Mondrian / Sadinle et al., JASA 2019). Guarantee, under
                        exchangeability within each class:  P(y in C(x) | Y = y) >= 1 - alpha.
                        Per-class calibration is used because marginal coverage can hide
                        failures on individual classes (Pitfalls of CP for medical imaging, 2025).
                        NOT valid under distribution shift -> always placed behind the OOD gate.
MahalanobisOOD          Class-conditional Gaussians with a shared, Ledoit-Wolf-shrunk covariance
                        on GAP embeddings of several stages, ensembled (Lee et al., NeurIPS 2018).
                        Feature-space detectors beat logit-based ones on medical data
                        (OpenMIBOOD, CVPR 2025).
energy_score            Liu et al., NeurIPS 2020 — kept only as a secondary signal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------------ basics
def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(np.asarray(logits, np.float64)))


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """E(x) = -T log sum_c exp(y_c / T). Higher = more out-of-distribution."""
    z = np.asarray(logits, np.float64) / temperature
    m = z.max(axis=-1)
    return -temperature * (m + np.log(np.exp(z - m[..., None]).sum(axis=-1)))


# ------------------------------------------------------------------------ temperature scaling
class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    @staticmethod
    def _nll(logits: np.ndarray, labels: np.ndarray, beta: float) -> float:
        return float(-log_softmax(logits * beta)[np.arange(len(labels)), labels].mean())

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Minimise NLL over beta = 1/T. NLL is convex in beta, so the bounded 1-D search is global."""
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, np.float64)
        labels = np.asarray(labels, np.int64)
        res = minimize_scalar(lambda b: self._nll(logits, labels, b), bounds=(0.05, 20.0),
                              method="bounded", options={"xatol": 1e-6})
        self.temperature = float(1.0 / res.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, np.float64) / self.temperature

    def to_dict(self) -> dict:
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureScaler":
        return cls(d["temperature"])


# --------------------------------------------------------------------------------- conformal
class ClassConditionalConformal:
    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.thresholds: np.ndarray | None = None   # per-class q_y on the score scale
        self.counts: np.ndarray | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "ClassConditionalConformal":
        probs = np.asarray(probs, np.float64)
        labels = np.asarray(labels, np.int64)
        n_cls = probs.shape[1]
        q = np.empty(n_cls)
        counts = np.zeros(n_cls, np.int64)
        for y in range(n_cls):
            scores = np.sort(1.0 - probs[labels == y, y])
            n = len(scores)
            counts[y] = n
            k = math.ceil((n + 1) * (1 - self.alpha))   # finite-sample corrected rank
            # With too few calibration samples the guarantee forces including y always.
            q[y] = np.inf if (n == 0 or k > n) else scores[k - 1]
        self.thresholds, self.counts = q, counts
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Boolean (N, C) membership. Empty sets are possible: no class is plausible -> refer/OOD."""
        if self.thresholds is None:
            raise RuntimeError("fit() first")
        probs = np.atleast_2d(np.asarray(probs, np.float64))
        return (1.0 - probs) <= self.thresholds[None, :]

    @staticmethod
    def coverage_report(sets: np.ndarray, labels: np.ndarray) -> dict:
        labels = np.asarray(labels, np.int64)
        hit = sets[np.arange(len(labels)), labels]
        per_class = {int(y): float(hit[labels == y].mean()) for y in np.unique(labels)}
        sizes = sets.sum(axis=1)
        return {"marginal_coverage": float(hit.mean()), "per_class_coverage": per_class,
                "mean_set_size": float(sizes.mean()), "frac_singleton": float((sizes == 1).mean()),
                "frac_empty": float((sizes == 0).mean())}

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "thresholds": [float(t) for t in self.thresholds],
                "counts": [int(c) for c in self.counts]}

    @classmethod
    def from_dict(cls, d: dict) -> "ClassConditionalConformal":
        obj = cls(d["alpha"])
        obj.thresholds = np.asarray(d["thresholds"], np.float64)
        obj.counts = np.asarray(d["counts"], np.int64)
        return obj


# ------------------------------------------------------------------------------ Mahalanobis
class MahalanobisOOD:
    """Multi-layer Mahalanobis ensemble. Higher score = more OOD."""

    def __init__(self):
        self.means: list[np.ndarray] = []        # per layer (C, D)
        self.precisions: list[np.ndarray] = []   # per layer (D, D)
        self.mu: np.ndarray | None = None        # per-layer standardisation of ID scores
        self.sigma: np.ndarray | None = None
        self.threshold: float | None = None
        self.tpr: float | None = None

    @staticmethod
    def _layer_scores(x: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
        d = x[:, None, :] - means[None, :, :]                       # (N, C, D)
        return np.einsum("ncd,de,nce->nc", d, prec, d).min(axis=1)  # min over classes

    def fit(self, train_emb: list[np.ndarray], train_labels: np.ndarray,
            calib_emb: list[np.ndarray], tpr: float = 0.95) -> "MahalanobisOOD":
        """train_emb: per-layer (N, D_l) in-distribution features (training split).
        calib_emb: held-out in-distribution features used for standardisation + threshold."""
        from sklearn.covariance import LedoitWolf

        labels = np.asarray(train_labels, np.int64)
        classes = np.unique(labels)
        self.means, self.precisions = [], []
        for x in train_emb:
            x = np.asarray(x, np.float64)
            means = np.stack([x[labels == c].mean(axis=0) for c in classes])
            centered = x - means[np.searchsorted(classes, labels)]
            lw = LedoitWolf(assume_centered=True).fit(centered)
            self.means.append(means)
            self.precisions.append(lw.precision_)
        raw = self._raw(calib_emb)
        self.mu, self.sigma = raw.mean(axis=0), raw.std(axis=0) + 1e-12
        ens = ((raw - self.mu) / self.sigma).mean(axis=1)
        self.tpr = float(tpr)
        self.threshold = float(np.quantile(ens, tpr))  # tpr of ID samples fall below threshold
        return self

    def _raw(self, emb: list[np.ndarray]) -> np.ndarray:
        return np.stack([self._layer_scores(np.atleast_2d(np.asarray(x, np.float64)), m, p)
                         for x, m, p in zip(emb, self.means, self.precisions)], axis=1)

    def score(self, emb: list[np.ndarray]) -> np.ndarray:
        return ((self._raw(emb) - self.mu) / self.sigma).mean(axis=1)

    def is_ood(self, emb: list[np.ndarray]) -> np.ndarray:
        return self.score(emb) > self.threshold

    def save(self, path: str | Path) -> None:
        arrays = {f"mean_{i}": m for i, m in enumerate(self.means)}
        arrays |= {f"prec_{i}": p.astype(np.float32) for i, p in enumerate(self.precisions)}
        meta = {"n_layers": len(self.means), "threshold": self.threshold, "tpr": self.tpr}
        np.savez_compressed(path, mu=self.mu, sigma=self.sigma, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisOOD":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        obj = cls()
        obj.means = [z[f"mean_{i}"] for i in range(meta["n_layers"])]
        obj.precisions = [z[f"prec_{i}"].astype(np.float64) for i in range(meta["n_layers"])]
        obj.mu, obj.sigma = z["mu"], z["sigma"]
        obj.threshold, obj.tpr = meta["threshold"], meta["tpr"]
        return obj

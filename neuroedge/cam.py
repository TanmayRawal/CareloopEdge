r"""Class activation maps that are *exact* for a GAP -> Linear head.

Notation: A in R^{K x h x w} is the last feature map, W in R^{C x K}, b in R^C the linear head.

    y_c = b_c + (1/hw) * sum_{ij} sum_k W_ck A_k(i,j)

Define the per-cell contribution map

    S_c(i,j) = (1/hw) * sum_k W_ck A_k(i,j)          =>   sum_{ij} S_c(i,j) + b_c = y_c   (exact)

so S_c is an additive decomposition of the logit, measured in logit units. Furthermore,
dy_c/dA_k(i,j) = W_ck/(hw) at every cell, hence:

    HiResCAM_c = sum_k (dy_c/dA_k) * A_k            = S_c
    Grad-CAM_c = ReLU(sum_k mean_ij(dy_c/dA_k) A_k) = ReLU(S_c)

(Selvaraju et al. ICCV'17; Draelos & Carin '20).

Weighted pooling (v1 lesion-attention head): with pooling weights a_ij >= 0, sum a = 1 (GAP is a = 1/hw),

    y_c = b_c + sum_{ij} a_ij sum_k W_ck A_k(i,j),     S_c(i,j) = a_ij * sum_k W_ck A_k(i,j)

and S_c is again an exact additive decomposition of the logit. Every function below takes an optional
`attention` (h,w); None means GAP. Note: dy_c/dA_k(i,j) = a_ij W_ck now varies over cells, so HiResCAM
(sum_k grad*A) still equals S_c, but classic Grad-CAM (spatially averaged gradient) does NOT: it would drop
the a_ij weighting (the known Grad-CAM failure mode, Draelos & Carin '20). The app therefore shows ReLU(S_c),
the HiResCAM form of Grad-CAM, which is faithful for both heads. Finer-CAM (Zhang et al. CVPR'25) explains
y_t - gamma * y_r, which by linearity equals S_t - gamma * S_r: what distinguishes the target
from its most similar class. All of this needs only a forward pass -> works on the INT8 Pi model.

Display rules (scientific honesty): bilinear upsampling and division by the max only. No
percentile floors, gamma curves, blob isolation or masking — those fabricate localization.
"""

from __future__ import annotations

import cv2
import numpy as np


# ------------------------------------------------------------------------------- exact maps
def _weights(features: np.ndarray, attention: np.ndarray | None) -> np.ndarray:
    _, h, w = features.shape
    if attention is None:
        return np.full((h, w), 1.0 / (h * w))
    a = np.asarray(attention, np.float64).reshape(h, w)
    return a


def contribution_maps(features: np.ndarray, weight: np.ndarray, attention: np.ndarray | None = None) -> np.ndarray:
    """All classes at once: (K,h,w), (C,K) -> (C,h,w) per-cell logit contributions S_c."""
    raw = np.einsum("ck,khw->chw", weight.astype(np.float64), features.astype(np.float64))
    return raw * _weights(features, attention)[None]


def exact_cam(features: np.ndarray, weight: np.ndarray, class_idx: int,
              attention: np.ndarray | None = None) -> np.ndarray:
    """Signed contribution map S_c (h, w). For GAP it equals HiResCAM; ReLU of it equals Grad-CAM."""
    return contribution_maps(features, weight[class_idx:class_idx + 1], attention)[0]


def decomposition_error(features: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                        logits: np.ndarray, attention: np.ndarray | None = None) -> float:
    """max_c |sum S_c + b_c - y_c|. ~0 (float error) for FP32; quantifies drift for INT8 models."""
    s = contribution_maps(features, weight, attention).sum(axis=(1, 2)) + bias
    return float(np.max(np.abs(s - logits)))


def most_similar_class(logits: np.ndarray, target: int) -> int:
    """Highest-scoring class other than the target (Finer-CAM's comparison class)."""
    masked = np.asarray(logits, np.float64).copy()
    masked[target] = -np.inf
    return int(np.argmax(masked))


def finer_cam(features: np.ndarray, weight: np.ndarray, target: int, reference: int,
              gamma: float = 1.0, attention: np.ndarray | None = None) -> np.ndarray:
    """S_target - gamma * S_reference: exact decomposition of the margin y_t - gamma*y_r (- biases).
    Kept for research/evaluation; the v1 app shows Grad-CAM + lesion only."""
    maps = contribution_maps(features, weight[[target, reference]], attention)
    return maps[0] - gamma * maps[1]


# ----------------------------------------------------------------------------- display utils
def upsample(cam: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    hw = (size, size) if isinstance(size, int) else size
    return cv2.resize(cam.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def scale_for_display(cam: np.ndarray, signed: bool = False, ref_max: float | None = None) -> np.ndarray:
    """Linear scaling only. Unsigned: ReLU then /max -> [0,1]. Signed: /max|.| -> [-1,1].

    ref_max lets several maps share one scale (e.g. comparing methods or images fairly).
    """
    x = cam if signed else np.maximum(cam, 0.0)
    m = float(ref_max) if ref_max is not None else float(np.max(np.abs(x)))
    if m <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / m, -1.0 if signed else 0.0, 1.0).astype(np.float32)


_CMAPS = {"inferno": cv2.COLORMAP_INFERNO, "turbo": cv2.COLORMAP_TURBO,
          "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS}


def overlay(gray_u8: np.ndarray, cam01: np.ndarray, cmap: str = "turbo", alpha: float = 0.7,
            fade: tuple[float, float] = (0.1, 0.4)) -> np.ndarray:
    """RGB overlay. Colour = cmap(cam01); per-pixel opacity = alpha * smoothstep(fade[0], fade[1], cam01).

    Both mappings are monotone in the CAM value. Opacity fades to zero below fade[0] (10% of the map's maximum),
    so weak evidence leaves the anatomy clean grey instead of tinting the whole brain.
    """
    if cam01.shape != gray_u8.shape:
        cam01 = upsample(cam01, gray_u8.shape)
    c = np.clip(cam01, 0, 1)
    heat = cv2.applyColorMap((c * 255).astype(np.uint8), _CMAPS[cmap])
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = np.repeat(gray_u8[..., None], 3, axis=-1).astype(np.float32)
    t = np.clip((c - fade[0]) / (fade[1] - fade[0]), 0, 1)
    a = (alpha * t * t * (3 - 2 * t))[..., None]
    return np.clip(base * (1 - a) + heat * a, 0, 255).astype(np.uint8)


def diverging_overlay(gray_u8: np.ndarray, cam_signed: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Signed maps (e.g. Finer-CAM): red = evidence for target, blue = evidence for the reference."""
    if cam_signed.shape != gray_u8.shape:
        cam_signed = upsample(cam_signed, gray_u8.shape)
    v = np.clip(cam_signed, -1, 1)[..., None]
    red, blue = np.array([220, 40, 40], np.float32), np.array([40, 90, 230], np.float32)
    color = np.where(v >= 0, red, blue)
    base = np.repeat(gray_u8[..., None], 3, axis=-1).astype(np.float32)
    a = alpha * np.abs(v)
    return np.clip(base * (1 - a) + color * a, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------- consistency with a mask
def evidence_inside_fraction(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of positive evidence inside a mask (energy-based pointing game, Wang et al. '20).

    Used both as an XAI metric (vs. expert masks) and at inference as CAM–segmentation
    consistency (vs. the predicted lesion mask). Returns NaN if there is no positive evidence.
    """
    if cam.shape != mask.shape:
        cam = upsample(cam, mask.shape)
    pos = np.maximum(cam, 0.0)
    total = float(pos.sum())
    return float(pos[mask > 0].sum() / total) if total > 1e-12 else float("nan")


# ------------------------------------------------------------- autograd reference (training PC)
def autograd_cams(model, x, target_fn):
    """Reference Grad-CAM and HiResCAM via autograd, for verifying the closed forms.

    model(x) must return a dict with "features" (B,K,h,w) and "logits"; target_fn maps logits
    (B,C) -> scalar per sample (B,). Returns numpy (gradcam, hirescam), each (B,h,w).
    """
    import torch

    model.eval()
    out = model(x)
    a = out["features"]
    a.retain_grad()
    target_fn(out["logits"]).sum().backward()
    g = a.grad
    gradcam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))
    hires = (g * a).sum(1)
    return gradcam.detach().cpu().numpy(), hires.detach().cpu().numpy()

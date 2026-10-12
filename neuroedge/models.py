"""Model definitions (PyTorch; used on the training PC and for export — not needed on the Pi).

MultiTaskNet
    encoder : timm ResNet-50, output_stride=16 (14x14 maps at 224 px) or 8 (28x28)
    cls head: weighted pooling -> Dropout -> Linear. The pooling weights a (sum to 1 over cells) are
              "gap"    a_ij = 1/hw                               (v0)
              "lesion" a_ij = (p_ij + eps) / sum(p + eps),  p = sigmoid(seg) pooled to h x w, DETACHED
              Either way  y_c = b_c + sum_ij a_ij (W_c . A_ij)  -> the CAM a_ij (W_c . A_ij) is an exact
              decomposition of the logit (cam.py). With no lesion (p ~ 0) "lesion" reduces to GAP.
              p is detached so the classification loss cannot reshape the lesion mask to suit itself:
              the mask is learned only from expert annotations.
    seg head: light FPN over stages 1-4 -> 1-channel tumor logit map at input resolution
    extras  : per-stage GAP embeddings for the Mahalanobis OOD detector

Gate
    MobileNetV3-Small, 3 classes (brain_mri / brain_ct / other). Needs no CAM, so the INT8
    explanation drift reported for depthwise networks does not affect it.
"""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class _ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FPNSegHead(nn.Module):
    """Top-down FPN (Lin et al., 2017) fused to stride 4, then upsampled to the input."""

    def __init__(self, in_channels: list[int], ch: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, ch, 1) for c in in_channels)
        self.smooth = _ConvBNReLU(ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def __init__(self, in_channels: list[int], ch: int = 128, aux: bool = False):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, ch, 1) for c in in_channels)
        self.smooth = _ConvBNReLU(ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)
        self.aux = nn.Conv2d(ch, 1, 1) if aux else None     # deep supervision on the coarsest level

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int]):
        p = self.lateral[-1](feats[-1])
        aux = self.aux(p) if self.aux is not None else None
        for lat, f in zip(reversed(self.lateral[:-1]), reversed(feats[:-1])):
            p = lat(f) + F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=False)
        seg = F.interpolate(self.out(self.smooth(p)), size=out_hw, mode="bilinear", align_corners=False)
        if aux is not None:
            aux = F.interpolate(aux, size=out_hw, mode="bilinear", align_corners=False)
        return seg, aux


def lesion_attention(seg_logits: torch.Tensor, cell: int, eps: float) -> torch.Tensor:
    """(B,1,H,W) seg logits -> (B,H/cell,W/cell) pooling weights a = (p + eps) / sum(p + eps), p detached.

    p is the tumour probability averaged over each CAM cell (area pooling with a constant cell size =
    the encoder's output stride, which keeps the ONNX graph static), so a cell's weight is its tumour
    fraction. Detaching makes the attention a pure function of the segmentation branch."""
    p = F.avg_pool2d(torch.sigmoid(seg_logits.detach().float()), kernel_size=cell, stride=cell)[:, 0]
    w = p + eps
    return w / w.sum(dim=(1, 2), keepdim=True)


class MultiTaskNet(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        if cfg.pooling not in ("gap", "lesion"):
            raise ValueError(f"unknown pooling {cfg.pooling!r}")
        if cfg.pooling == "lesion" and not cfg.segmentation:
            raise ValueError("lesion pooling needs the segmentation head")
        self.pooling, self.attn_eps, self.output_stride = cfg.pooling, cfg.attn_eps, cfg.output_stride
        self.encoder = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, features_only=True,
                                         out_indices=(1, 2, 3, 4), output_stride=cfg.output_stride)
        chans = self.encoder.feature_info.channels()
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(chans[-1], num_classes)
        self.seg_head = (FPNSegHead(chans, cfg.fpn_channels, aux=cfg.seg_aux_weight > 0)
                         if cfg.segmentation else None)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        feats = self.encoder(x)
        a = feats[-1]                                   # (B, K, h, w) — the CAM feature map A
        b, _, h, w = a.shape
        out: dict = {"features": a, "embeddings": [f.mean(dim=(2, 3)) for f in feats]}
        if self.seg_head is not None:
            out["seg_logits"], aux = self.seg_head(feats, x.shape[-2:])
            if aux is not None:
                out["seg_aux_logits"] = aux
        if self.pooling == "lesion":
            att = lesion_attention(out["seg_logits"], self.output_stride, self.attn_eps).to(a.dtype)
        else:
            att = torch.full((b, h, w), 1.0 / (h * w), dtype=a.dtype, device=a.device)
        out["attention"] = att                           # (B, h, w), sums to 1 per image
        pooled = torch.einsum("bkhw,bhw->bk", a, att)
        out["logits"] = self.classifier(self.dropout(pooled))
        return out

    def head_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(W, b) of the linear head as float32 numpy — shipped to the Pi for exact CAM."""
        w = self.classifier.weight.detach().float().cpu().numpy()
        b = self.classifier.bias.detach().float().cpu().numpy()
        return w, b

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
        """Discriminative LRs; no weight decay on biases / norm parameters."""
        groups = []
        heads = [m for m in (self.classifier, self.seg_head) if m is not None]
        for module, lr in [(self.encoder, lr_backbone)] + [(h, lr_head) for h in heads]:
            decay, no_decay = [], []
            for name, p in module.named_parameters():
                if p.requires_grad:
                    (no_decay if p.ndim <= 1 else decay).append(p)
            groups += [{"params": decay, "lr": lr, "weight_decay": weight_decay},
                       {"params": no_decay, "lr": lr, "weight_decay": 0.0}]
        return groups


class ExportWrapper(nn.Module):
    """Flat tuple outputs for ONNX: (logits, features, seg_logits, emb1..emb4, attention)."""

    def __init__(self, net: MultiTaskNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        o = self.net(x)
        seg = o.get("seg_logits", torch.zeros_like(x[:, :1]))
        return (o["logits"], o["features"], seg, *o["embeddings"], o["attention"])


def build_gate(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    return timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=num_classes)

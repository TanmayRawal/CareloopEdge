"""Model definitions (PyTorch; used on the training PC and for export — not needed on the Pi).

MultiTaskNet
    encoder : timm ResNet-50, output_stride=16 (dilated stage 4 -> 14x14 maps at 224 px)
    cls head: GAP -> Dropout -> Linear      (keeps CAM == Grad-CAM == HiResCAM exact; see cam.py)
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

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        p = self.lateral[-1](feats[-1])
        for lat, f in zip(reversed(self.lateral[:-1]), reversed(feats[:-1])):
            p = lat(f) + F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(self.out(self.smooth(p)), size=out_hw, mode="bilinear", align_corners=False)


class MultiTaskNet(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.encoder = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, features_only=True,
                                         out_indices=(1, 2, 3, 4), output_stride=cfg.output_stride)
        chans = self.encoder.feature_info.channels()
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(chans[-1], num_classes)
        self.seg_head = FPNSegHead(chans, cfg.fpn_channels) if cfg.segmentation else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        feats = self.encoder(x)
        a = feats[-1]                                   # (B, K, h, w) — the CAM feature map A
        logits = self.classifier(self.dropout(a.mean(dim=(2, 3))))
        out: dict = {"logits": logits, "features": a,
                     "embeddings": [f.mean(dim=(2, 3)) for f in feats]}
        if self.seg_head is not None:
            out["seg_logits"] = self.seg_head(feats, x.shape[-2:])
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
    """Flat tuple outputs for ONNX: (logits, features, seg_logits, emb1..emb4)."""

    def __init__(self, net: MultiTaskNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        o = self.net(x)
        seg = o.get("seg_logits", torch.zeros_like(x[:, :1]))
        return (o["logits"], o["features"], seg, *o["embeddings"])


def build_gate(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    return timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=num_classes)

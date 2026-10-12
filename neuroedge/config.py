"""Configuration: one place for class names, input geometry and normalization.

Every other module (training, export, Pi inference) reads these values from here or from a
YAML file parsed by `load_config`, so the train/serve contract cannot silently drift.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Canonical label orders. Index order is part of the model contract — never reorder.
MRI_CLASSES: tuple[str, ...] = ("glioma", "meningioma", "no_tumor", "pituitary")
CT_CLASSES_BINARY: tuple[str, ...] = ("healthy", "tumor")
CT_CLASSES_4: tuple[str, ...] = ("healthy", "hemorrhage", "ischemia", "tumor")
GATE_CLASSES: tuple[str, ...] = ("brain_mri", "brain_ct", "other")

# ImageNet statistics (backbones are ImageNet-initialised).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# CT display windows as (level, width) in Hounsfield units: brain, subdural, bone.
CT_WINDOWS: tuple[tuple[float, float], ...] = ((40.0, 80.0), (80.0, 200.0), (600.0, 2000.0))


@dataclass
class PreprocessConfig:
    image_size: int = 224
    brain_crop: bool = True
    crop_margin: float = 0.04          # fraction of the crop side added on each border
    clahe: bool = False                # ablation only; off by default
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    mr_percentiles: tuple[float, float] = (0.5, 99.5)  # DICOM MR intensity clipping
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def fingerprint(self) -> str:
        """Stable hash shipped inside the model bundle; the Pi refuses mismatches."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    output_stride: int = 16            # 16 -> 14x14 CAM at 224 px (vs 7x7 at stride 32)
    pretrained: bool = True
    dropout: float = 0.3
    segmentation: bool = True
    fpn_channels: int = 128
    # Classifier pooling. "gap": mean over cells (v0). "lesion": weights a_ij = (p_ij + eps) / sum(p + eps)
    # from the DETACHED segmentation probability p (pooled to the CAM grid); exact CAM is preserved.
    pooling: str = "gap"
    attn_eps: float = 0.02
    seg_aux_weight: float = 0.0        # deep supervision on the coarsest FPN level (0 = off)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    ema_decay: float = 0.999
    seg_loss_weight: float = 1.0
    seg_focal_gamma: float = 0.0       # 0 = plain BCE (v0); 2.0 = focal BCE for small lesions
    guidance_loss_weight: float = 0.0  # explanation-guidance ablation (energy-in-mask)
    early_stop_patience: int = 8
    amp: bool = True
    num_workers: int = 4
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass
class TaskConfig:
    name: str = "mri"
    class_names: tuple[str, ...] = MRI_CLASSES
    manifest: str = "data/manifests/mri.csv"
    output_dir: str = "runs/mri"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    conformal_alpha: float = 0.05

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _build(cls: type, data: dict[str, Any] | None):
    """Construct a dataclass from a dict, converting lists to tuples and rejecting unknown keys."""
    data = dict(data or {})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    for k, v in list(data.items()):
        if isinstance(v, list):
            data[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return cls(**data)


def load_config(path: str | Path) -> TaskConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sub = {
        "preprocess": _build(PreprocessConfig, raw.pop("preprocess", None)),
        "model": _build(ModelConfig, raw.pop("model", None)),
        "train": _build(TrainConfig, raw.pop("train", None)),
    }
    cfg = _build(TaskConfig, raw)
    for k, v in sub.items():
        setattr(cfg, k, v)
    return cfg

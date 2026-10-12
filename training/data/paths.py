"""Data locations. Data lives OUTSIDE the OneDrive-synced project folder (several GB, patient data).

Override with the NEUROEDGE_DATA environment variable.
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("NEUROEDGE_DATA", Path.home() / "neuroedge_data"))
RAW = DATA_ROOT / "raw"
PROCESSED = DATA_ROOT / "processed"
MANIFESTS = DATA_ROOT / "manifests"
RUNS = DATA_ROOT / "runs"


def manifest_path(cfg) -> Path:
    """Manifest named by a config (TaskConfig or its asdict()): its `manifest` basename inside MANIFESTS.
    v0 runs used <task>.csv; v1 runs use mri_v2.csv (split with a calibration set)."""
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    name = Path(get("manifest") or f"{get('name')}.csv").name
    return MANIFESTS / name

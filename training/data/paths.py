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

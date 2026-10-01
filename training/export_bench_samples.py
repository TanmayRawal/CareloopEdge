"""Copy a class-balanced set of held-out test images for benchmarking on the Pi (copy the folder over with scp).

  python -m training.export_bench_samples --per-class 10 --out C:/Users/Tanmay/neuroedge_data/bench_samples
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from training.data.paths import DATA_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "bench_samples")
    args = ap.parse_args()
    df = pd.read_csv(DATA_ROOT / "manifests" / "mri.csv")
    test = df[df["split"] == "test"]
    args.out.mkdir(parents=True, exist_ok=True)
    for label, g in test.groupby("label"):
        for _, r in g.sample(min(args.per_class, len(g)), random_state=args.seed).iterrows():
            shutil.copyfile(r["path"], args.out / f"{label}__{Path(r['path']).name}")
    print(f"{len(list(args.out.iterdir()))} images in {args.out}")


if __name__ == "__main__":
    main()

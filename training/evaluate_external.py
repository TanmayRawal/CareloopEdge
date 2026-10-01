"""Evaluate a trained run on an audited external dataset (never used for training).

  python -m training.evaluate_external --run <RUNS>/mri_grouped/seed0 --dataset bdneuro

Reads <MANIFESTS>/external_<dataset>.csv (rows with split == external_<dataset>), preprocesses with the
run's own preprocessing config, and writes pred_external_<dataset>.npz + metrics into the run dir
(the same format as train.py, so calibrate.py picks it up automatically).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neuroedge.config import PreprocessConfig
from training.data.paths import DATA_ROOT, MANIFESTS
from training.dataset import TumorDataset, build_cache
from training.export import load_run
from training.train import predict, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    net, ckpt = load_run(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).to(memory_format=torch.channels_last)
    pcfg = PreprocessConfig(**{k: tuple(v) if isinstance(v, list) else v
                               for k, v in ckpt["config"]["preprocess"].items()})
    df = pd.read_csv(MANIFESTS / f"external_{args.dataset}.csv", keep_default_na=False)
    df = df[df["split"] == f"external_{args.dataset}"].sort_values("id")
    tag = hashlib.md5("\n".join(df["id"]).encode()).hexdigest()[:10]
    cache = build_cache(df, pcfg, DATA_ROOT / "cache", f"ext_{args.dataset}_{tag}")
    ds = TumorDataset(df, cache, tuple(ckpt["class_names"]), pcfg, train=False)
    pred = predict(net, DataLoader(ds, batch_size=64, num_workers=0), device)
    pred["ids"] = np.array(ds.ids)
    name = f"external_{args.dataset}"
    np.savez_compressed(args.run / f"pred_{name}.npz", **pred)
    m = summarize(pred, ckpt["class_names"])
    metrics_path = args.run / "metrics.json"
    allm = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    allm[name] = m
    metrics_path.write_text(json.dumps(allm, indent=2))
    print(f"{name}: n={len(df)} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} AUROC={m['macro_auroc']:.4f}")
    for c, v in m["per_class"].items():
        print(f"  {c:>11}: sensitivity={v['sensitivity']:.3f} precision={v['precision']:.3f} n={v['support']}")
    print("confusion (rows=true, cols=pred):", m["confusion_matrix"])


if __name__ == "__main__":
    main()

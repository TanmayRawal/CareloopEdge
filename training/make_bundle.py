"""Assemble the plaintext edge bundle from a finished run (input to server/publish.py).

  python -m training.make_bundle --run <RUNS>/mri_grouped/seed0 --precision int8 --out <dir>

Bundle = bundle.json (+ SHA-256 of every file), model.onnx (INT8 by default), head.npz,
calibration.json, ood.npz. The Pi engine (edge/engine.py) consumes exactly these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-deep-explain", dest="deep_explain", action="store_false",
                    help="omit FP32 weights (smaller bundle; no on-device LayerCAM)")
    args = ap.parse_args()
    exp = args.run / "export"
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exp / f"model_{args.precision}.onnx", args.out / "model.onnx")
    shutil.copyfile(exp / "head.npz", args.out / "head.npz")
    # Calibration / conformal / OOD statistics must come from the SAME graph that is shipped.
    calib = exp / f"calib_model_{args.precision}"
    if not (calib / "calibration.json").exists():
        raise SystemExit(f"missing {calib}: run  python -m training.calibrate --run {args.run} "
                         f"--onnx {exp / f'model_{args.precision}.onnx'}")
    shutil.copyfile(calib / "calibration.json", args.out / "calibration.json")
    shutil.copyfile(calib / "ood.npz", args.out / "ood.npz")
    if args.deep_explain:   # FP32 state_dict only (no optimizer / config objects) for on-device LayerCAM
        import torch
        ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
        torch.save(ckpt["state_dict"], args.out / "model_fp32.pt")
    meta = json.loads((exp / "bundle.json").read_text())
    meta["precision"] = args.precision
    if (exp / "quant_report.json").exists() and args.precision == "int8":
        q = json.loads((exp / "quant_report.json").read_text())
        meta["quantization"] = {"test": q["test"], "acceptance": q["acceptance"]}
    meta["files"] = {f: hashlib.sha256((args.out / f).read_bytes()).hexdigest()
                     for f in ("model.onnx", "head.npz", "calibration.json", "ood.npz", "model_fp32.pt")
                     if (args.out / f).exists()}
    (args.out / "bundle.json").write_text(json.dumps(meta, indent=2))
    print(f"bundle written to {args.out}: {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main()

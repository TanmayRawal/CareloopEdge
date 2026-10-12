"""Export a trained run to ONNX (FP32) + the metadata the edge engine needs, with parity checks.

  python -m training.export --run <RUNS>/mri_grouped/seed0

Writes <run>/export/:
  model_fp32.onnx      inputs: image (N,3,S,S) float32
                       outputs: logits, features (A), seg_logits, emb1..emb4
  head.npz             W (C,K), b (C,) of the linear head — exact CAM on the Pi needs only these
  bundle.json          class names, preprocessing config + fingerprint, I/O names, SHA-256s, metrics
Parity (asserted): ONNX vs PyTorch max |logit diff| < 1e-3 and exact-CAM decomposition error < 1e-3.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from neuroedge.cam import decomposition_error
from neuroedge.config import ModelConfig, PreprocessConfig
from neuroedge.models import ExportWrapper, MultiTaskNet
from training.data.paths import manifest_path

OUTPUTS = ["logits", "features", "seg_logits", "emb1", "emb2", "emb3", "emb4", "attention"]


def load_run(run: Path) -> tuple[MultiTaskNet, dict]:
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**{**ckpt["config"]["model"], "pretrained": False})
    net = MultiTaskNet(len(ckpt["class_names"]), mcfg)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt


def sample_inputs(run: Path, n: int, seed: int = 0) -> np.ndarray:
    """Real validation images (normalized) for parity / calibration, read from the training cache."""
    from neuroedge.preprocess import normalize
    from training.data.paths import DATA_ROOT

    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    pcfg = PreprocessConfig(**{k: tuple(v) if isinstance(v, list) else v
                               for k, v in ckpt["config"]["preprocess"].items()})
    caches = sorted((DATA_ROOT / "cache").glob(f"*_{pcfg.fingerprint()}"))
    if not caches:
        raise FileNotFoundError("no preprocessing cache found for this run")
    imgs = np.load(caches[0] / "images.npy", mmap_mode="r")
    idx = np.random.default_rng(seed).choice(len(imgs), size=min(n, len(imgs)), replace=False)
    return np.stack([normalize(imgs[i].astype(np.float32) / 255.0, pcfg) for i in np.sort(idx)])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = args.run / "export"
    out.mkdir(exist_ok=True)
    net, ckpt = load_run(args.run)
    size = ckpt["config"]["preprocess"]["image_size"]
    wrapper = ExportWrapper(net).eval()
    dummy = torch.zeros(1, 3, size, size)
    onnx_path = out / "model_fp32.onnx"
    torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=["image"], output_names=OUTPUTS,
                      dynamic_axes={"image": {0: "batch"}, **{o: {0: "batch"} for o in OUTPUTS}},
                      opset_version=args.opset, do_constant_folding=True)

    w, b = net.head_weights()
    np.savez(out / "head.npz", W=w, b=b)

    # ---------------- parity on real validation images
    import onnxruntime as ort

    x = sample_inputs(args.run, 32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_onnx = dict(zip(OUTPUTS, sess.run(OUTPUTS, {"image": x})))
    with torch.no_grad():
        o_torch = dict(zip(OUTPUTS, [t.numpy() for t in wrapper(torch.from_numpy(x))]))
    logit_diff = float(np.abs(o_onnx["logits"] - o_torch["logits"]).max())
    feat_diff = float(np.abs(o_onnx["features"] - o_torch["features"]).max())
    decomp = max(decomposition_error(o_onnx["features"][i], w, b, o_onnx["logits"][i], o_onnx["attention"][i])
                 for i in range(len(x)))
    agree = float((o_onnx["logits"].argmax(1) == o_torch["logits"].argmax(1)).mean())
    print(f"parity: max|Δlogit|={logit_diff:.2e}  max|Δfeature|={feat_diff:.2e}  "
          f"CAM decomposition error={decomp:.2e}  argmax agreement={agree:.3f}")
    assert logit_diff < 1e-3 and decomp < 1e-3 and agree == 1.0, "ONNX export parity check failed"

    metrics = json.loads((args.run / "metrics.json").read_text()) if (args.run / "metrics.json").exists() else {}
    bundle = {
        "task": ckpt["config"]["name"], "class_names": list(ckpt["class_names"]),
        "manifest": manifest_path(ckpt["config"]).name,
        "image_size": size, "preprocess": ckpt["config"]["preprocess"],
        "preprocess_fingerprint": ckpt["preprocess_fingerprint"], "model": ckpt["config"]["model"],
        "inputs": ["image"], "outputs": OUTPUTS, "cam_layer_output": "features",
        "files": {"model_fp32.onnx": sha256(onnx_path), "head.npz": sha256(out / "head.npz")},
        "training": {"seed": ckpt["seed"], "scheme": ckpt["scheme"], "best_epoch": ckpt["epoch"]},
        "metrics": {k: {m: v[m] for m in ("accuracy", "macro_f1", "macro_auroc") if m in v}
                    for k, v in metrics.items() if isinstance(v, dict)},
        "parity": {"max_abs_logit_diff": logit_diff, "cam_decomposition_error": decomp},
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    print(f"exported to {out}")


if __name__ == "__main__":
    main()

"""INT8 post-training static quantization (ONNX Runtime, QDQ) + accuracy / explanation-drift audit.

  python -m training.quantize --run <RUNS>/mri_grouped/seed0

* Per-channel symmetric INT8 weights, asymmetric UINT8 activations, percentile (99.99) calibration
  on 512 training images.
* The global-average-pool (ReduceMean) and classifier (Gemm) nodes stay FP32, so the INT8 model's
  logits are exactly W·mean(A) + b over its own (dequantized) feature map A: the exact-CAM identity
  of neuroedge.cam holds for the deployed model, not only for the FP32 one.
* Acceptance (reported, not silently enforced): macro-F1 drop ≤ 1 pt, McNemar p > 0.05 vs FP32,
  and CAM drift (Pearson, SSIM, top-20% IoU) — quantization can preserve accuracy while changing
  explanations (Same Predictions, Different Reasons, 2026; Quant-XAI).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from neuroedge.cam import decomposition_error, exact_cam
from neuroedge.config import PreprocessConfig
from neuroedge.metrics import classification_metrics, map_similarity, mcnemar_exact
from neuroedge.preprocess import normalize
from training.data.paths import DATA_ROOT, MANIFESTS
from training.export import OUTPUTS


class _Reader(CalibrationDataReader):
    """Supports len()/set_range() so ORT can calibrate in strides (CalibStridedMinMax)."""

    def __init__(self, x: np.ndarray):
        self.x = x
        self.set_range(0, len(x))

    def __len__(self) -> int:
        return len(self.x)

    def set_range(self, start_index: int, end_index: int) -> None:
        self._it = iter(self.x[i:i + 1] for i in range(start_index, min(end_index, len(self.x))))

    def get_next(self):
        nxt = next(self._it, None)
        return None if nxt is None else {"image": nxt}


def split_arrays(bundle: dict, split: str, limit: int | None = None, seed: int = 0, manifest: str | None = None):
    """(normalized images, labels, ids) for a manifest split, read from the preprocessing cache."""
    df = pd.read_csv(MANIFESTS / (manifest or f"{bundle['task']}.csv"), keep_default_na=False)
    df = df[df["split"] == split]
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=seed)
    return rows_arrays(bundle, df, split)


def rows_arrays(bundle: dict, df: pd.DataFrame, what: str = "rows"):
    pcfg = PreprocessConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in bundle["preprocess"].items()})
    for cache in sorted((DATA_ROOT / "cache").glob(f"*_{pcfg.fingerprint()}")):
        ids = (cache / "ids.txt").read_text(encoding="utf-8").splitlines()
        pos = {k: i for i, k in enumerate(ids)}
        if all(i in pos for i in df["id"]):
            imgs = np.load(cache / "images.npy", mmap_mode="r")
            x = np.stack([normalize(imgs[pos[i]].astype(np.float32) / 255.0, pcfg) for i in df["id"]])
            y = np.array([bundle["class_names"].index(l) for l in df["label"]])
            return x, y, df["id"].tolist()
    raise FileNotFoundError(f"no cache contains all '{what}' images")


def float_nodes(model_path: Path) -> list[str]:
    """Names of the pooling + classifier nodes to keep in FP32."""
    g = onnx.load(str(model_path)).graph
    producer = {o: n for n in g.node for o in n.output}
    keep = [n.name for n in g.node if n.op_type in ("ReduceMean", "GlobalAveragePool")]
    head = producer["logits"]
    keep.append(head.name)
    if head.op_type == "Add":                      # MatMul + Add form of a Linear layer
        keep += [producer[i].name for i in head.input if i in producer and producer[i].op_type == "MatMul"]
    return keep


def run(sess: ort.InferenceSession, x: np.ndarray, bs: int = 32) -> dict[str, np.ndarray]:
    outs = [sess.run(OUTPUTS, {"image": x[s:s + bs]}) for s in range(0, len(x), bs)]
    return {name: np.concatenate([o[k] for o in outs]) for k, name in enumerate(OUTPUTS)}


def latency_ms(path: Path, size: int, threads: int, n: int = 40) -> float:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(1, 3, size, size)).astype(np.float32)
    for _ in range(5):
        sess.run(["logits"], {"image": x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(OUTPUTS, {"image": x})
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--calib", type=int, default=512)
    args = ap.parse_args()
    exp = args.run / "export"
    bundle = json.loads((exp / "bundle.json").read_text())
    fp32, pre, int8 = exp / "model_fp32.onnx", exp / "model_fp32_pre.onnx", exp / "model_int8.onnx"

    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=False)
    xc, _, _ = split_arrays(bundle, "train", limit=args.calib, seed=1)
    keep = float_nodes(pre)
    quantize_static(str(pre), str(int8), _Reader(xc), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile, nodes_to_exclude=keep,
                    # * CalibStridedMinMax=8: feed the (histogram) calibrator 8 images at a time; histograms
                    #   are merged across strides. Without it ORT keeps every intermediate tensor of all
                    #   calibration images in RAM (~100+ GB for 512 images -> bad allocation).
                    # * CalibTensorRangeSymmetric=False: post-ReLU activations are >= 0; a symmetric
                    #   [-a, a] range would waste half of the UINT8 codes.
                    extra_options={"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False,
                                   "WeightSymmetric": True, "CalibStridedMinMax": 8})
    print(f"quantized; FP32 nodes kept: {keep}")

    head = np.load(exp / "head.npz")
    w, b = head["W"], head["b"]
    sess = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for k, p in (("fp32", fp32), ("int8", int8))}
    report = {"fp32_nodes": keep, "size_mb": {k: round(p.stat().st_size / 1e6, 1) for k, p in (("fp32", fp32), ("int8", int8))}}
    for split in ("val", "test"):
        x, y, _ = split_arrays(bundle, split)
        o = {k: run(s, x) for k, s in sess.items()}
        probs = {k: np.exp(v["logits"] - v["logits"].max(1, keepdims=True)) for k, v in o.items()}
        probs = {k: p / p.sum(1, keepdims=True) for k, p in probs.items()}
        m = {k: classification_metrics(y, p, bundle["class_names"]) for k, p in probs.items()}
        pred = {k: p.argmax(1) for k, p in probs.items()}
        drift = [map_similarity(exact_cam(o["fp32"]["features"][i], w, pred["fp32"][i]),
                                exact_cam(o["int8"]["features"][i], w, pred["fp32"][i])) for i in range(len(y))]
        report[split] = {
            "n": len(y),
            "fp32": {k: m["fp32"][k] for k in ("accuracy", "macro_f1", "macro_auroc")},
            "int8": {k: m["int8"][k] for k in ("accuracy", "macro_f1", "macro_auroc")},
            "prediction_agreement": float((pred["fp32"] == pred["int8"]).mean()),
            "mcnemar": mcnemar_exact(pred["fp32"] == y, pred["int8"] == y),
            "cam_drift": {k: float(np.nanmean([d[k] for d in drift])) for k in drift[0]},
            "int8_cam_decomposition_error": max(decomposition_error(o["int8"]["features"][i], w, b, o["int8"]["logits"][i])
                                                for i in range(len(y))),
        }
        r = report[split]
        print(f"{split}: F1 fp32={r['fp32']['macro_f1']:.4f} int8={r['int8']['macro_f1']:.4f} "
              f"agree={r['prediction_agreement']:.4f} McNemar p={r['mcnemar']['p_value']:.3f} "
              f"CAM drift={json.dumps({k: round(v, 3) for k, v in r['cam_drift'].items()})} "
              f"decomp err={r['int8_cam_decomposition_error']:.1e}")
    size = bundle["image_size"]
    report["pc_cpu_latency_ms"] = {f"{k}_{t}thr": latency_ms(p, size, t)
                                   for k, p in (("fp32", fp32), ("int8", int8)) for t in (1, 4)}
    print("PC CPU latency (ms, batch 1):", report["pc_cpu_latency_ms"])
    d = report["test"]["int8"]["macro_f1"] - report["test"]["fp32"]["macro_f1"]
    report["acceptance"] = {"macro_f1_drop_le_1pt": d >= -0.01,
                            "mcnemar_p_gt_0.05": report["test"]["mcnemar"]["p_value"] > 0.05}
    (exp / "quant_report.json").write_text(json.dumps(report, indent=2))
    pre.unlink(missing_ok=True)
    print("acceptance:", report["acceptance"])


if __name__ == "__main__":
    main()

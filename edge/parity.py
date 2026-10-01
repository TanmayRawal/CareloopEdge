"""Cross-platform parity: does the Pi (ARM, its own numpy/OpenCV/onnxruntime) give the same outputs as the PC?

  python -m edge.parity dump --images DIR (--bundle-dir D | --model mri) --out parity_<host>.npz
  python -m edge.parity compare parity_pc.npz parity_pi.npz

dump stores, per image: calibrated probabilities, prediction, exact-CAM map, preprocessed input tensor and the
library versions. compare reports prediction agreement, max |dp|, CAM correlation and the input-tensor
difference, which tells whether any drift starts in preprocessing (decode/resize) or in the model runtime.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".bmp", ".tif", ".tiff"}


def dump(args) -> None:
    import cv2
    import onnxruntime as ort

    from edge.engine import Engine
    from neuroedge import cam as necam

    if args.bundle_dir:
        eng = Engine.from_dir(args.bundle_dir, threads=4)
    else:
        from edge.sync import Device
        files, _ = Device().load_bundle(args.model)
        eng = Engine(files, threads=4)
    imgs = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTS)
    names, probs, preds, cams, tensors = [], [], [], [], []
    for p in imgs:
        r = eng.analyze(p.read_bytes(), filename=p.name, with_overlays=False)
        st = r["_state"]
        names.append(p.name)
        probs.append([r["probabilities"][c] for c in eng.class_names])
        preds.append(eng.class_names.index(r["raw_argmax"]))
        cams.append(necam.exact_cam(st["features"], eng.W, st["pred"]).ravel())
        tensors.append(st["pre"].tensor.astype(np.float32))
    meta = {"host": platform.node(), "machine": platform.machine(), "numpy": np.__version__, "opencv": cv2.__version__,
            "onnxruntime": ort.__version__, "python": platform.python_version()}
    np.savez_compressed(args.out, names=np.array(names), probs=np.array(probs), preds=np.array(preds),
                        cams=np.array(cams), tensors=np.array(tensors), meta=json.dumps(meta))
    print(json.dumps({**meta, "n": len(names)}))


def compare(a_path: str, b_path: str) -> None:
    a, b = np.load(a_path), np.load(b_path)
    assert list(a["names"]) == list(b["names"]), "different image sets"
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    dp = np.abs(a["probs"] - b["probs"])
    dt = np.abs(a["tensors"] - b["tensors"])
    cc = [np.corrcoef(x, y)[0, 1] for x, y in zip(a["cams"], b["cams"])]
    rep = {"A": ma, "B": mb, "n": int(len(a["names"])),
           "prediction_agreement": float((a["preds"] == b["preds"]).mean()),
           "max_abs_prob_diff": float(dp.max()), "mean_abs_prob_diff": float(dp.mean()),
           "input_tensor_max_abs_diff": float(dt.max()), "images_with_any_input_diff": int((dt.reshape(len(dt), -1).max(1) > 0).sum()),
           "cam_pearson_min": float(np.min(cc)), "cam_pearson_mean": float(np.mean(cc))}
    worst = int(dp.max(1).argmax())
    rep["worst_image"] = {"name": str(a["names"][worst]), "probs_A": a["probs"][worst].round(4).tolist(),
                          "probs_B": b["probs"][worst].round(4).tolist()}
    print(json.dumps(rep, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--images", required=True)
    g = d.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundle-dir")
    g.add_argument("--model")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())

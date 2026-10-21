"""NeuroEdge inference engine for the Raspberry Pi — numpy + OpenCV + ONNX Runtime only (no PyTorch).

    engine = Engine.from_dir("bundle/")            # or Engine(files) with decrypted in-memory bytes
    result = engine.analyze(upload_bytes, filename="scan.png")

Pipeline per image
  preprocess (shared neuroedge.preprocess)  ->  INT8 ONNX model  ->  logits, A, seg, embeddings
  calibrated probabilities   softmax(logits / T)                       (temperature scaling)
  prediction set             class-conditional conformal (1 - alpha per class)
  OOD                        multi-layer Mahalanobis on embeddings (+ energy, secondary)
  explanation                Grad-CAM in its exact (HiResCAM) form: S_c = a_ij W_c.A_ij sums to the logit; with the
                             v1 lesion-attention head a_ij comes from the segmentation (exported as "attention")
  localisation               segmentation head -> lesion area / centroid / extent
  consistency                share of positive CAM evidence inside the predicted lesion
  triage flags               ood, uncertain (set size != 1), low consistency, input quality
Everything returned is JSON-serialisable except the overlay PNG bytes.

Orientation note: for PNG/JPG uploads image orientation is unknown, so positions are reported in
IMAGE coordinates ("image-left"), never as patient hemispheres.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from neuroedge import cam as necam
from neuroedge.bundle import BUNDLE_FILES, OPTIONAL_FILES
from neuroedge.config import PreprocessConfig
from neuroedge.preprocess import inverse_map, preprocess
from neuroedge.uncertainty import ClassConditionalConformal, MahalanobisOOD, energy_score, softmax


@dataclass
class Thresholds:
    # Review triggers are CALIBRATED, not guessed: calibration.json may carry
    # "review_thresholds" fitted on validation data (e.g. the 5th percentile of CAM-in-lesion
    # evidence among correctly classified validation tumors); these defaults are fallbacks.
    min_cam_seg_consistency: float = 0.02
    seg_probability: float = 0.5
    min_lesion_px: int = 20                # in model space (224x224); smaller = no lesion


REJECT_FLAGS = {"out_of_distribution", "color_image", "mostly_empty"}   # no diagnosis is shown


class Engine:
    def __init__(self, files: dict[str, bytes], threads: int = 4, thresholds: Thresholds | None = None):
        missing = [f for f in BUNDLE_FILES if f not in files]
        if missing:
            raise ValueError(f"bundle incomplete, missing: {missing}")
        self.meta = json.loads(files["bundle.json"])
        self.class_names: list[str] = self.meta["class_names"]
        self.pcfg = PreprocessConfig(**{k: tuple(v) if isinstance(v, list) else v
                                        for k, v in self.meta["preprocess"].items()})
        if self.pcfg.fingerprint() != self.meta["preprocess_fingerprint"]:
            raise ValueError("preprocessing fingerprint mismatch: bundle was built with different preprocessing")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(files["model.onnx"], so, providers=["CPUExecutionProvider"])
        head = np.load(io.BytesIO(files["head.npz"]))
        self.W, self.b = head["W"], head["b"]
        cal = json.loads(files["calibration.json"])
        self.temperature = float(cal["temperature"])
        self.conformal = ClassConditionalConformal.from_dict(cal["conformal"])
        self.ood = MahalanobisOOD.load(io.BytesIO(files["ood.npz"]))
        # two tiers: warn above ood.threshold (95% val TPR), reject above the 99.5% level
        self.ood_reject = float(cal.get("ood_reject_threshold", self.ood.threshold))
        self.th = thresholds or Thresholds(**cal.get("review_thresholds", {}))
        self.outputs = self.meta["outputs"]

    @classmethod
    def from_dir(cls, path: str | Path, **kw) -> "Engine":
        p = Path(path)
        files = {f: (p / f).read_bytes() for f in BUNDLE_FILES}
        files |= {f: (p / f).read_bytes() for f in OPTIONAL_FILES if (p / f).exists()}
        return cls(files, **kw)

    # ------------------------------------------------------------------------------ analysis
    def analyze(self, source: bytes | str | Path, filename: str | None = None, with_overlays: bool = True) -> dict:
        t = {}
        t0 = time.perf_counter()
        pre = preprocess(source, self.pcfg, filename=filename)
        t["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        out = dict(zip(self.outputs, self.session.run(self.outputs, {"image": pre.tensor[None]})))
        t["model_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        logits = out["logits"][0].astype(np.float64)
        feats = out["features"][0]
        probs = softmax(logits / self.temperature)
        pred = int(np.argmax(probs))
        pset = self.conformal.predict_sets(probs[None])[0]
        emb = [out[f"emb{k}"] for k in range(1, 5)]
        ood_score = float(self.ood.score(emb)[0])
        is_ood = bool(ood_score > self.ood_reject)
        ood_warning = bool(not is_ood and ood_score > self.ood.threshold)

        # v1 bundles export the classifier's pooling weights; v0 (GAP) bundles do not -> uniform
        att = out["attention"][0] if "attention" in out else None
        cam = necam.exact_cam(feats, self.W, pred, att)
        runner_up = necam.most_similar_class(logits, pred)
        decomp = necam.decomposition_error(feats, self.W, self.b, logits, att)
        size = self.pcfg.image_size
        cam_up = necam.upsample(cam, size)

        seg_prob = 1.0 / (1.0 + np.exp(-out["seg_logits"][0, 0].astype(np.float64)))
        lesion = (seg_prob >= self.th.seg_probability).astype(np.uint8)
        lesion_stats = self._lesion_stats(lesion, pre.mm_per_model_px)
        consistency = (necam.evidence_inside_fraction(cam_up, lesion)
                       if lesion_stats["present"] else float("nan"))
        t["postprocess_ms"] = (time.perf_counter() - t0) * 1000

        flags = list(pre.quality_flags)
        if is_ood:
            flags.append("out_of_distribution")
        elif ood_warning:
            flags.append("unusual_image")
        if pset.sum() != 1:
            flags.append("uncertain_prediction_set" if pset.sum() > 1 else "empty_prediction_set")
        if lesion_stats["present"] and np.isfinite(consistency) and consistency < self.th.min_cam_seg_consistency:
            flags.append("explanation_outside_lesion")
        tumor_pred = self.class_names[pred] not in ("no_tumor", "healthy")
        if tumor_pred != lesion_stats["present"]:
            flags.append("classifier_segmenter_disagree")

        rejected = sorted(set(flags) & REJECT_FLAGS)
        result = {
            # "rejected": the input is outside what the model was validated on -> the UI must not
            # present a tumor class (the raw output is kept only for logging / research).
            "status": "rejected" if rejected else "ok",
            "reject_reasons": rejected,
            "prediction": None if rejected else self.class_names[pred],
            "raw_argmax": self.class_names[pred],
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
            "prediction_set": [c for c, m in zip(self.class_names, pset) if m],
            "conformal_alpha": self.conformal.alpha,
            "ood": {"mahalanobis_score": ood_score, "threshold": float(self.ood.threshold),
                    "reject_threshold": self.ood_reject, "is_ood": is_ood, "warning": ood_warning,
                    "energy": float(energy_score(logits[None], self.temperature)[0])},
            "explanation": {"method": "Grad-CAM (exact, HiResCAM form)", "explained_class": self.class_names[pred],
                            "runner_up": self.class_names[runner_up],
                            "lesion_attention_pooling": att is not None,
                            "logit_decomposition_error": decomp,
                            "evidence_inside_lesion": None if not np.isfinite(consistency) else float(consistency)},
            "lesion": lesion_stats,
            "modality_hint": pre.modality,
            "flags": flags,
            "needs_expert_review": bool(set(flags) & {"out_of_distribution", "unusual_image", "uncertain_prediction_set",
                                                       "empty_prediction_set", "explanation_outside_lesion",
                                                       "classifier_segmenter_disagree"}),
            "timings_ms": {k: round(v, 1) for k, v in t.items()},
        }
        if with_overlays:
            t0 = time.perf_counter()
            result["overlays"] = self._overlays(pre, cam_up, lesion)     # PNG encoding at full resolution
            result["timings_ms"]["render_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        # internal state (not JSON; research/parity tools only)
        result["_state"] = {"pre": pre, "features": feats, "pred": pred, "runner_up": runner_up, "attention": att}
        return result

    # ------------------------------------------------------------------------------ whole series
    def analyze_series(self, source, filename: str | None = None, with_overlays: bool = True) -> dict:
        """A whole study (NIfTI, a .zip of a DICOM series or NIfTI, or a list of DICOM file bytes) -> one answer.

        Slices are chosen as in training/evaluate_volumes.py (evenly spaced, three planes), each goes through
        analyze(), and the per-slice calibrated probabilities are combined with the bundle's aggregation rule
        (neuroedge/aggregate.py; chosen on calibration patients). The returned fields are those of analyze() for
        the key slice (the slice that most supports "tumour"), with the patient-level probabilities, prediction and
        prediction set in place of the slice's, plus result["series"]. Overlays: the key slice under the usual names,
        "s{i}_original" for every slice (the doctor scrolls through all of them), and "s{i}_{cam,lesion,cam_lesion}"
        for the key slices only (the three that most support "tumour")."""
        from neuroedge import aggregate, volume
        cfg = {"rule": "topk", "k": 5, "per_plane": 12, "key_slices": 3} | self.meta.get("series", {})
        t0 = time.perf_counter()
        vol = volume.load_volume(source, filename)
        slices = volume.select_slices(vol, per_plane=int(cfg["per_plane"]))
        if not slices:
            raise ValueError("no brain slices found in this series")
        load_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        per = [self.analyze(s.raw, with_overlays=False) for s in slices]
        model_ms = (time.perf_counter() - t0) * 1000
        probs = np.array([[r["probabilities"][c] for c in self.class_names] for r in per])
        lesion = np.array([r["lesion"]["area_px_model"] if r["lesion"]["present"] else 0 for r in per], np.float64)
        ok = np.array([r["status"] == "ok" for r in per])
        nt = self.class_names.index("no_tumor")
        p = aggregate.aggregate(probs, cfg["rule"], nt, k=int(cfg["k"]), lesion_area=lesion, ok=ok)
        usable = np.flatnonzero(ok) if ok.any() else np.arange(len(per))
        order = aggregate.key_slices(probs[usable], nt, n=int(cfg["key_slices"]))
        keys = [int(usable[i]) for i in order]
        pred = int(np.argmax(p))

        result = dict(per[keys[0]])
        result.pop("_state", None)
        flags = [f for f in result["flags"] if f not in ("uncertain_prediction_set", "empty_prediction_set")]
        pset = self.conformal.predict_sets(p[None])[0]
        if pset.sum() != 1:
            flags.append("uncertain_prediction_set" if pset.sum() > 1 else "empty_prediction_set")
        rejected = []
        if ok.mean() < 0.5:           # most of the study is outside what the model was validated on
            rejected = sorted({f for r in per for f in r["reject_reasons"]})
        flags = sorted(set(flags) | set(rejected))
        result.update({
            "status": "rejected" if rejected else "ok",
            "reject_reasons": rejected,
            "prediction": None if rejected else self.class_names[pred],
            "raw_argmax": self.class_names[pred],
            "probabilities": {c: float(v) for c, v in zip(self.class_names, p)},
            "prediction_set": [c for c, m in zip(self.class_names, pset) if m],
            "flags": flags,
            "needs_expert_review": bool(set(flags) & {"out_of_distribution", "unusual_image", "uncertain_prediction_set",
                                                       "empty_prediction_set", "explanation_outside_lesion",
                                                       "classifier_segmenter_disagree"}),
        })
        result["series"] = {
            "n_slices": len(slices), "source": vol.source, "spacing_mm": [round(float(v), 3) for v in vol.spacing_mm],
            "rule": cfg["rule"], "k": int(cfg["k"]), "key_slices": keys,
            "slices": [{"plane": s.plane, "index": s.index, "tumour_prob": round(float(1 - probs[i, nt]), 4),
                        "prediction": self.class_names[int(np.argmax(probs[i]))], "ok": bool(ok[i])}
                       for i, s in enumerate(slices)],
            "slices_rejected": int((~ok).sum()),
            "overlays": "key" if with_overlays else "none",
        }
        result["timings_ms"] = {"load_ms": round(load_ms, 1), "model_ms": round(model_ms, 1),
                                "per_slice_ms": round(model_ms / len(slices), 1)}
        if with_overlays:
            t0 = time.perf_counter()
            key_ov = {i: self.analyze(slices[i].raw)["overlays"] for i in keys}      # 3 extra runs, rendered
            result["overlays"] = dict(key_ov[keys[0]])
            result["overlays"] |= {f"s{i}_original": _png(np.repeat(r["_state"]["pre"].original[..., None], 3, -1))
                                   for i, r in enumerate(per)}
            result["overlays"] |= {f"s{i}_{k}": v for i, o in key_ov.items() for k, v in o.items() if k != "original"}
            result["timings_ms"]["render_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # ------------------------------------------------------------------------------- helpers
    def _lesion_stats(self, lesion: np.ndarray, mm_per_px: float | None) -> dict:
        area = int(lesion.sum())
        if area < self.th.min_lesion_px:
            return {"present": False}
        ys, xs = np.nonzero(lesion)
        h, w = lesion.shape
        cy, cx = float(ys.mean()), float(xs.mean())
        stats = {"present": True, "area_px_model": area, "area_fraction_of_image": area / (h * w),
                 "centroid_rel": [round(cx / w, 3), round(cy / h, 3)],
                 "image_side": "image-left" if cx < w / 2 else "image-right",
                 "image_vertical": "upper" if cy < h / 2 else "lower",
                 "bbox_model": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
        if mm_per_px:
            stats["area_mm2"] = round(area * mm_per_px ** 2, 1)
            stats["max_extent_mm"] = round(max(xs.max() - xs.min(), ys.max() - ys.min()) * mm_per_px, 1)
        return stats

    def _overlays(self, pre, cam_up, lesion) -> dict[str, bytes]:
        """The only two explanations shown (v1): Grad-CAM, and the lesion outline from the segmentation head;
        plus their combination and the original image. All at the upload's resolution."""
        g = pre.geometry
        orig = pre.original
        cam_o = inverse_map(necam.scale_for_display(cam_up), g)
        lesion_o = (inverse_map(lesion.astype(np.float32), g) >= 0.5).astype(np.uint8)
        cam_rgb = necam.overlay(orig, cam_o)
        seg_rgb = np.repeat(orig[..., None], 3, -1).copy()
        both_rgb = cam_rgb.copy()
        cnts, _ = cv2.findContours(lesion_o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick = max(1, round(max(orig.shape) / 300))
        cv2.drawContours(seg_rgb, cnts, -1, (0, 230, 0), thick)
        cv2.drawContours(both_rgb, cnts, -1, (0, 230, 0), thick)

        return {"original": _png(np.repeat(orig[..., None], 3, -1)), "cam": _png(cam_rgb),
                "lesion": _png(seg_rgb), "cam_lesion": _png(both_rgb)}


def _png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()

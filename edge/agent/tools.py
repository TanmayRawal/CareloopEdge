"""Grounded tools for the offline assistant.

case_facts()  the ONLY source of case-specific statements: a compact, display-rounded view of the
              engine result. The LLM never sees images or raw tensors.
KB            BM25 retrieval over the curated, cited markdown knowledge base in kb/.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

CLASS_TEXT = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary tumor", "no_tumor": "no tumor"}
FLAG_TEXT = {
    "unusual_image": "the image is less typical than 95% of validation scans",
    "out_of_distribution": "the image is outside what the model was validated on",
    "uncertain_prediction_set": "more than one class is plausible at the 95% level",
    "empty_prediction_set": "no class reaches the calibrated 95% level",
    "explanation_outside_lesion": "the model's evidence lies almost entirely outside the detected lesion",
    "classifier_segmenter_disagree": "the classifier and the lesion detector disagree",
    "blurry_or_flat": "the image has very little detail",
    "low_resolution": "the image resolution is very low",
}


def case_facts(result: dict) -> dict:
    """Verified, JSON-safe facts. Percentages are rounded to 1 decimal exactly as displayed."""
    pct = lambda v: round(100.0 * float(v), 1)  # noqa: E731
    probs = {CLASS_TEXT.get(k, k): pct(v) for k, v in sorted(result["probabilities"].items(), key=lambda kv: -kv[1])}
    lesion = result.get("lesion", {})
    facts = {
        "status": result["status"],
        "prediction": CLASS_TEXT.get(result["prediction"], result["prediction"]) if result["prediction"] else None,
        "probabilities_percent": probs,
        "prediction_set_95": [CLASS_TEXT.get(c, c) for c in result.get("prediction_set", [])],
        "needs_expert_review": bool(result.get("needs_expert_review")),
        "review_reasons": [FLAG_TEXT[f] for f in result.get("flags", []) if f in FLAG_TEXT],
        "runner_up": CLASS_TEXT.get(result["explanation"]["finer_cam_reference"], result["explanation"]["finer_cam_reference"]),
        "lesion_detected": bool(lesion.get("present")),
    }
    if lesion.get("present"):
        facts["lesion_position_in_image"] = f"{lesion['image_vertical']} {lesion['image_side']}"
        if lesion.get("area_mm2"):
            facts["lesion_area_mm2"] = lesion["area_mm2"]
        else:
            facts["lesion_area_percent_of_image"] = pct(lesion["area_fraction_of_image"])
        ev = result["explanation"].get("evidence_inside_lesion")
        if ev is not None:
            facts["model_evidence_inside_lesion_percent"] = pct(ev)
    return facts


def _set_meaning(f: dict) -> str:
    s = f["prediction_set_95"]
    if len(s) == 1:
        return f"Meaning: only {s[0]} remains plausible at the calibrated 95% level; no other class does."
    if not s:
        return "Meaning: no class reaches the calibrated 95% level, so the result is uncertain."
    return f"Meaning: {len(s)} classes remain plausible at the 95% level, so the result is uncertain between them."


def fact_sheet(f: dict) -> str:
    """The facts as short English sentences, with their interpretation spelled out deterministically.

    Small LLMs echo whatever format they are shown (so: no JSON) and misread statistical outputs (a one-class
    conformal set was called "uncertain" or "certain"; 5 % evidence-in-lesion became "concentrated in the
    lesion"), so the meaning of each value is stated here rather than left to the model to infer.
    The dict itself stays the verifier's source of truth."""
    if f["status"] != "ok":
        return "- The image was rejected and not classified: " + "; ".join(f["review_reasons"]) + "."
    lines = [f"- The model predicts: {f['prediction']}.",
             "- Calibrated probabilities: " + ", ".join(f"{k} {v:.1f}%" for k, v in f["probabilities_percent"].items()) + ".",
             "- 95% prediction set: " + (", ".join(f["prediction_set_95"]) or "empty") + ". " + _set_meaning(f),
             f"- Runner-up class: {f['runner_up']} ({f['probabilities_percent'].get(f['runner_up'], 0.0):.1f}%). "
             f"The Finer-CAM view shows which image regions favour {f['prediction']} over {f['runner_up']}; "
             "the model itself gives no reasons in words."]
    if f["lesion_detected"]:
        size = (f"{f['lesion_area_mm2']} mm²" if "lesion_area_mm2" in f
                else f"{f['lesion_area_percent_of_image']:.1f}% of the image area (no physical size: the file has no pixel spacing)")
        lines.append(f"- The segmentation network found a lesion in the {f['lesion_position_in_image']} of the image "
                     f"(image orientation, not patient side); size {size}.")
        if "model_evidence_inside_lesion_percent" in f:
            ev = f["model_evidence_inside_lesion_percent"]
            share = ("most of the evidence lies inside the lesion" if ev >= 50 else
                     "a substantial part lies inside the lesion, but most lies outside it" if ev >= 25 else
                     "only a small part lies inside the lesion; most of the evidence lies outside it")
            lines.append(f"- {ev:.1f}% of the model's positive Grad-CAM evidence lies inside that lesion: {share}. "
                         "(The coarse 14x14 map limits this value for small lesions.)")
    else:
        lines.append("- The segmentation network found no lesion in this slice.")
    lines.append("- Automatic review triggers: " + ("; ".join(f["review_reasons"]) if f["needs_expert_review"] else "none fired")
                 + ". Every result still needs review by a qualified radiologist.")
    lines.append("- Not available from this tool: tumor grade, volume, patient side, other slices, clinical history.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- KB
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of and or in on to is are be by for with as at it its this that from not can may".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KB:
    def __init__(self, folder: str | Path | None = None):
        folder = Path(folder or Path(__file__).parent / "kb")
        self.passages: list[dict] = []
        for md in sorted(folder.glob("*.md")):
            text = md.read_text(encoding="utf-8")
            refs = re.search(r"^References:(.*)$", text, re.M)
            for para in re.split(r"\n\s*\n", text):
                para = para.strip()
                if para and not para.startswith("#") and not para.startswith("References:"):
                    self.passages.append({"source": md.stem, "text": para, "references": refs.group(1).strip() if refs else ""})
        self.docs = [_tokens(p["text"] + " " + p["source"]) for p in self.passages]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.df = Counter(t for d in self.docs for t in set(d))

    def search(self, query: str, k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        q = _tokens(query)
        n = len(self.docs)
        scores = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / self.avgdl))
            scores.append(s)
        order = sorted(range(n), key=lambda i: -scores[i])
        return [self.passages[i] | {"score": round(scores[i], 3)} for i in order[:k] if scores[i] > 0]

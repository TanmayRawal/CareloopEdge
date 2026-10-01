"""Deterministic grounding verifier for LLM text (the "verifier role"; no LLM involved).

A draft passes only if:
  1. every number it states appears in the case facts or in the knowledge-base passages it was given
     (percentages must match a fact to within the 0.1 display rounding);
  2. it asserts no class other than the model's prediction as the result (other classes may only be
     mentioned in comparative / probability context);
  3. it contains no treatment directives and no definitive diagnostic language (negated uses are fine);
  4. it is prose, not an echo of the structured facts (JSON braces or snake_case field names);
  5. it does not contradict the meaning of the prediction set or of the evidence-inside-lesion share
     (the two misreadings observed in the evaluation transcripts).
List markers ("1.", "2)") are not treated as claimed numbers.
Failures are returned as human-readable problems; the orchestrator regenerates once with them and
otherwise falls back to the deterministic template.
"""

from __future__ import annotations

import re

CLASSES = ("glioma", "meningioma", "pituitary tumor", "pituitary", "no tumor")
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%|mm²|mm2|mm)?")
_TREATMENT = re.compile(r"\b(should|must|needs? to|recommend\w*|advis\w*)\b[^.]*\b(surgery|surgical|resect\w*|"
                        r"chemotherapy|radiotherapy|radiation|biopsy|treat\w*|medication|operat\w*)\b", re.I)
_DEFINITIVE = re.compile(r"\b(definitely|certainly|certain|certainty|confirm(s|ed)?|prove[sn]?|diagnosed with|"
                         r"is diagnosed|(primary|final|definitive) diagnosis|there is no doubt|100% sure|100(\.0)?% (confiden\w*|certain\w*))\b", re.I)
_NEGATION = re.compile(r"\b(not|cannot|can't|no|never|without)\b[\w\s,]{0,25}$", re.I)
_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")
_LEVEL_CTX = re.compile(r"^\s*%?\s*(\w+\s+){0,3}(prediction set|set|level|confidence|of validation|of the time|coverage)", re.I)
_SEES_IMAGE = re.compile(r"\b(the image shows|is seen|are seen|is visible|are visible|features are present|"
                         r"consistent with the model|support\w* the model's (prediction|classification)|"
                         r"these features support|absence of such features)\b", re.I)
_FIELD = re.compile(r"[{}]|\b[a-z]+_[a-z_]+\b")
_COMPARATIVE = re.compile(r"(runner-up|less likely|probabilit|%|prediction set|compared|rather than|than|not |"
                          r"instead|versus|vs\.?|alternative|second|no evidence|absence|absent|exclud|rule[sd]? out|"
                          r"none of|neither|nor\b|\bover\b|favou?r\w*|negligible|unlikely|low\b)", re.I)
_SET_UNCERTAIN = re.compile(r"\b(uncertain|not certain|multiple|more than one|several|two|three|cannot rule out|"
                            r"also considered)\b", re.I)
_OTHERS_OPEN = re.compile(r"\b(also (being )?considered|remain\w* plausible|cannot (be )?rule[d]? out|"
                          r"(does|do) not (exclude|rule out)|not excluded|not ruled out)\b", re.I)
_SET_SINGLE = re.compile(r"\b(only one|single class|one class|certain|confident)\b", re.I)
_EV_STRONG = re.compile(r"\b(concentrated|mostly|primarily|largely|mainly|predominantly|focused)\b", re.I)
_EV_HEDGE = re.compile(r"(\d+(\.\d+)?\s*%|\b(only|small|minority|part|some|little|fraction|outside)\b)", re.I)


def _fact_numbers(obj, out: set[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _fact_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _fact_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(round(float(obj), 1))
    elif isinstance(obj, str):
        for m in _NUM.finditer(obj):
            out.add(round(float(m.group(1)), 1))


_SET_VERB = re.compile(r"\b((?:prediction )?set) (confirms|confirmed|proves|establishes)\b", re.I)


def normalise(text: str) -> str:
    """Deterministic, meaning-preserving rewrites applied before verification and display.
    'The prediction set confirms X' is a correct statement about the set but reads like a diagnosis."""
    return _SET_VERB.sub(lambda m: m.group(1) + " indicates", text)


def verify(text: str, facts: dict, passages: list[dict] | None = None) -> tuple[bool, list[str]]:
    problems: list[str] = []
    allowed: set[float] = set()
    _fact_numbers(facts, allowed)
    for p in passages or []:
        _fact_numbers(p.get("text", ""), allowed)
    allowed |= {2.0, 1.0}            # fixed method constants: "2D", "1 slice"
    allowed.discard(95.0)            # 95 is only ever a level ("95% set", "95% of validation scans"), see below

    if _FIELD.search(text):
        problems.append("output looks like structured data (JSON or field names); write plain prose")
    body = _LIST_MARKER.sub("", text)
    for m in _NUM.finditer(body):
        v = round(float(m.group(1)), 1)
        if v == 95.0 and _LEVEL_CTX.search(body[m.end():m.end() + 45]):
            continue                 # "95% prediction set / level / of validation scans", not a probability
        if not any(abs(v - a) <= 0.11 for a in allowed):
            problems.append(f"number '{m.group(0).strip()}' is not in the case facts or cited passages")

    pred = (facts.get("prediction") or "").lower()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        for c in CLASSES:
            if c in low and c not in pred and not (c == "pituitary" and "pituitary" in pred):
                if not _COMPARATIVE.search(sent):
                    problems.append(f"sentence presents '{c}' without comparative context: {sent.strip()[:90]}")
                break
    if _SEES_IMAGE.search(text):
        problems.append("describes image content; the assistant cannot see the image")
    if _TREATMENT.search(text):
        problems.append("contains a treatment directive")
    for m in _DEFINITIVE.finditer(text):
        before, after = text[max(0, m.start() - 40):m.start()], text[m.end():m.end() + 25]
        purpose = re.search(r"\bto\s*$", before) or re.search(r"^\s*(diagnosis\s+)?(requires|needs|would|is not|cannot)", after, re.I)
        determiner = m.group(0).lower() == "certain" and re.match(r"\s+(regions?|areas?|parts?|features?|structures?|aspects?)\b", after, re.I)
        if not (_NEGATION.search(before) or purpose or determiner):
            problems.append(f"uses definitive diagnostic language ('{m.group(0)}')")
            break
    pset = facts.get("prediction_set_95")
    ev = facts.get("model_evidence_inside_lesion_percent")
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        if pset is not None and re.search(r"\b(prediction|the|this) set\b", low):
            if len(pset) == 1 and _SET_UNCERTAIN.search(sent):
                problems.append("misstates the prediction set (it contains exactly one class)")
            elif len(pset) != 1 and _SET_SINGLE.search(sent):
                problems.append(f"misstates the prediction set (it contains {len(pset)} classes)")
        if pset is not None and len(pset) == 1 and _OTHERS_OPEN.search(sent) and "elsewhere" not in low and any(
                c in low for c in CLASSES if c not in pset[0] and not (c == "pituitary" and "pituitary" in pset[0])):
            problems.append("implies other classes remain plausible (the prediction set contains one class)")
        if ev is not None and ev < 50 and "evidence" in low and re.search(r"\b(within|inside|in)\b[^.]{0,30}lesion", low):
            if _EV_STRONG.search(sent) or not _EV_HEDGE.search(sent):
                problems.append(f"overstates the evidence inside the lesion (only {ev:.1f}% lies inside it)")
    return not problems, problems

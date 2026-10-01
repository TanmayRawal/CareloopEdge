"""Offline assistant: grounded report drafting and Q&A over one analysed case.

Roles (after the 2025 radiology-agent review: RAG grounding, role separation, verification, human sign-off)
  facts      tools.case_facts — the only case-specific source of truth
  retriever  tools.KB (BM25 over cited passages)
  drafter    local LLM (Qwen3-1.7B via llama.cpp) — writes prose from facts + passages only
  verifier   verifier.verify — deterministic; one regeneration with the problems, then template fallback
  human      the report carries a "Reviewed by" sign-off field; nothing is final without it
The template report is always produced (instant, no LLM); the narrative is optional.
"""

from __future__ import annotations

import re

from edge.agent.llm import LocalLLM
from edge.agent.tools import KB, case_facts, fact_sheet
from edge.agent.verifier import normalise, verify

SYSTEM = (
    "You explain the output of a research brain-MRI classification model to a radiologist. "
    "Use only the FACTS and the numbered SOURCES you are given, and copy numbers exactly. "
    "Describe results as model outputs ('the model predicts'), never as a diagnosis, and never claim certainty. "
    "Never give treatment, surgery or biopsy advice. "
    "If the answer is not in the FACTS or SOURCES, say that this tool does not provide it. "
    "You cannot see the image: never say what the image shows or that features are present, seen or consistent; "
    "use SOURCES only as general background. "
    "Write 2 to 4 sentences of plain prose. Do not use JSON, lists, headings or field names, "
    "and do not repeat the facts list."
)
# Questions about management or outcome are answered by policy, not by the LLM (deterministic refusal).
_POLICY_Q = re.compile(r"\b(surg\w*|operat\w*|resect\w*|treat\w*|therap\w*|chemo\w*|radiat\w*|radiotherap\w*|"
                       r"biops\w*|medicat\w*|drugs?|prognos\w*|surviv\w*|life expectancy|cur(e|ed|able)|die|dying)\b", re.I)
# Questions whose correct answer is fully determined by the facts are answered by template: in the grounding
# evaluation (docs/FINDINGS.md §7) the 1.7B model misread the conformal set or invented reasons in 8 of 18 such answers.
_SET_Q = re.compile(r"\b(prediction set|conformal|95\s*%\s*set|the set|this set)\b", re.I)
_REVIEW_Q = re.compile(r"\b(review\w*|radiologist|flag\w*|trigger\w*)\b", re.I)
_WHY_Q = re.compile(r"\bwhy\b[^?]*\b(predict\w*|class\w*|glioma|meningioma|pituitary|tumou?r|rather|instead|not)\b|"
                    r"\bhow did (the model|it) (decide|choose)\b|\bwhat made (the model|it)\b", re.I)


def explain_set(f: dict) -> str:
    s = f["prediction_set_95"]
    if len(s) == 1:
        what = (f"The 95% prediction set contains only {s[0]}: at the calibrated 95% level no other class remains "
                "plausible. This does not make the result certain.")
    elif not s:
        what = "The 95% prediction set is empty: no class reaches the calibrated 95% level, so the result is uncertain."
    else:
        what = f"The 95% prediction set contains {', '.join(s)}: these classes remain plausible, so the result is uncertain between them."
    return (what + " The set comes from class-conditional conformal prediction calibrated on validation scans: for each "
            "true class, sets built this way contain it about 95% of the time on data like the calibration data. "
            "On our held-out test set coverage was lower than intended, so treat it as approximate.")


def explain_why(f: dict) -> str:
    pr, ru = f["prediction"], f["runner_up"]
    probs = f["probabilities_percent"]
    parts = ["The model does not give reasons in words; it is a pattern classifier trained on labelled MRI slices.",
             f"Its calibrated probabilities are {pr} {probs.get(pr, 0.0):.1f}% and {ru} {probs.get(ru, 0.0):.1f}% (runner-up).",
             f"The Finer-CAM tab shows which image regions push the decision towards {pr} (red) rather than {ru} (blue)."]
    ev = f.get("model_evidence_inside_lesion_percent")
    if ev is not None:
        parts.append(f"{ev:.1f}% of the positive Grad-CAM evidence lies inside the outlined lesion.")
    if pr == "pituitary tumor":
        parts.append("For pituitary predictions our evaluation found that the evidence lies largely in surrounding "
                     "anatomy rather than the tumor itself, so the lesion outline is the better guide to its location.")
    if pr == "no tumor":
        parts.append("A no-tumor output covers this single slice only and cannot exclude a lesion elsewhere.")
    return " ".join(parts)


def explain_review(f: dict) -> str:
    if f["needs_expert_review"]:
        head = "Yes. Automatic expert review was triggered because " + "; ".join(f["review_reasons"]) + "."
    else:
        head = ("No automatic review trigger fired. The triggers check for an image unlike the validation scans, more "
                "than one plausible class, evidence lying almost entirely outside the detected lesion, and disagreement "
                "between the classifier and the lesion detector.")
    return head + (" In every case the result must still be reviewed by a qualified radiologist: this is a research "
                   "tool reading a single 2D slice.")


REJECTED_ANSWER = ("This image was not classified: it is outside what the model was validated on, so the model makes "
                   "no statement about tumor type or about the absence of a tumor. Please check that it is a "
                   "grayscale brain MRI slice; review by a qualified radiologist is required.")
POLICY_ANSWER = ("This tool cannot advise on treatment or outcome. Those decisions belong to the patient's clinical team "
                 "after expert review of the full imaging study and clinical history; the output here is a research "
                 "model's reading of a single 2D slice.")


def template_report(f: dict) -> list[tuple[str, str]]:
    """Deterministic report sections (always available, no LLM)."""
    if f["status"] != "ok":
        return [("Result", "Not analysed: the input is outside what the model was validated on. "
                           + "; ".join(f["review_reasons"]).capitalize() + ".")]
    probs = ", ".join(f"{k} {v:.1f}%" for k, v in f["probabilities_percent"].items())
    pset = ", ".join(f["prediction_set_95"]) or "empty (no class reaches the calibrated 95% level)"
    secs = [("Model output", f"The model predicts {f['prediction']}. Calibrated probabilities: {probs}."),
            ("Uncertainty", f"95% prediction set: {pset}. Runner-up class: {f['runner_up']}.")]
    if f["lesion_detected"]:
        size = (f"{f['lesion_area_mm2']} mm²" if "lesion_area_mm2" in f
                else f"{f['lesion_area_percent_of_image']:.1f}% of the image")
        ev = f.get("model_evidence_inside_lesion_percent")
        secs.append(("Lesion", f"Segmentation detected a lesion in the {f['lesion_position_in_image']} of the image "
                               f"(image orientation, not patient side), size {size}."
                               + (f" {ev:.1f}% of the model's positive evidence lies inside it." if ev is not None else "")))
    else:
        secs.append(("Lesion", "The segmentation network detected no lesion in this slice."))
    secs.append(("Review", "Expert review recommended: " + "; ".join(f["review_reasons"]) + "."
                 if f["needs_expert_review"] else "No automatic review triggers fired; expert review is still required."))
    secs.append(("Limitations", "Research tool, not a medical device. Single 2D slice; trained on public data; "
                                "accuracy is lower on scans from other hospitals."))
    return secs


class Assistant:
    def __init__(self, llm: LocalLLM | None = None, kb: KB | None = None):
        self.llm = llm or LocalLLM()
        self.kb = kb or KB()

    def _grounded(self, messages: list[dict], facts: dict, passages: list[dict], max_tokens: int) -> dict:
        log = []
        base = messages
        for attempt in range(2):
            text = normalise(self.llm.chat(messages, max_tokens=max_tokens, temperature=0.2 if attempt == 0 else 0.7))
            ok, problems = verify(text, facts, passages)
            log.append({"attempt": attempt + 1, "ok": ok, "problems": problems, "draft": text})   # draft: audit only
            if ok:
                return {"text": text, "verified": True, "log": log}
            extra = (" A previous draft was rejected; avoid these errors: " + "; ".join(dict.fromkeys(problems))
                     + ". Avoid the words confirm, certain and definitive.")
            messages = [{"role": "system", "content": base[0]["content"] + extra}] + base[1:]
        return {"text": None, "verified": False, "log": log}

    def _context(self, facts: dict, passages: list[dict]) -> str:
        cited = "\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
        return f"FACTS:\n{fact_sheet(facts)}\n\nSOURCES:\n{cited or '(none)'}"

    def report(self, result: dict) -> dict:
        facts = case_facts(result)
        out = {"facts": facts, "sections": template_report(facts), "narrative": None, "llm_used": False}
        if facts["status"] != "ok" or not self.llm.available():
            return out
        passages = self.kb.search(f"{facts['prediction']} MRI appearance", k=2) + self.kb.search("heatmap prediction set", k=1)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": self._context(facts, passages) +
                 "\n\nWrite a short narrative summary of this model output for the radiologist."}]
        g = self._grounded(msgs, facts, passages, max_tokens=260)
        out.update(narrative=g["text"], llm_used=True, verification=g["log"], passages=passages)
        return out

    def ask(self, question: str, result: dict) -> dict:
        facts = case_facts(result)
        if facts["status"] != "ok":
            return {"answer": REJECTED_ANSWER, "verified": True, "verification": [], "fallback": None,
                    "passages": [], "llm_used": False, "policy": True, "route": "rejected"}
        routed = (("treatment", POLICY_ANSWER) if _POLICY_Q.search(question) else
                  ("set", explain_set(facts)) if _SET_Q.search(question) else
                  ("why", explain_why(facts)) if _WHY_Q.search(question) else
                  ("review", explain_review(facts)) if _REVIEW_Q.search(question) else None)
        if routed:
            return {"answer": routed[1], "verified": True, "verification": [], "fallback": None,
                    "passages": [], "llm_used": False, "policy": True, "route": routed[0]}
        passages = self.kb.search(question + " " + (facts["prediction"] or ""), k=3)
        if not self.llm.available():
            return {"answer": None, "fallback": passages, "facts": facts, "llm_used": False}
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": self._context(facts, passages) + f"\n\nQuestion: {question[:500]}"}]
        g = self._grounded(msgs, facts, passages, max_tokens=220)
        return {"answer": g["text"], "verified": g["verified"], "verification": g["log"],
                "fallback": None if g["verified"] else passages, "passages": passages, "llm_used": True}

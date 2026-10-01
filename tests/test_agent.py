from edge.agent.orchestrator import Assistant, template_report
from edge.agent.tools import KB, case_facts
from edge.agent.verifier import verify

RESULT = {
    "status": "ok", "prediction": "meningioma",
    "probabilities": {"glioma": 0.06, "meningioma": 0.9, "no_tumor": 0.01, "pituitary": 0.03},
    "prediction_set": ["meningioma"], "needs_expert_review": True, "flags": ["unusual_image"],
    "explanation": {"finer_cam_reference": "glioma", "evidence_inside_lesion": 0.534},
    "lesion": {"present": True, "image_vertical": "upper", "image_side": "image-left", "area_fraction_of_image": 0.096},
}
FACTS = case_facts(RESULT)


def test_facts_are_display_rounded():
    assert FACTS["prediction"] == "meningioma"
    assert FACTS["probabilities_percent"]["meningioma"] == 90.0
    assert FACTS["model_evidence_inside_lesion_percent"] == 53.4


def test_grounded_text_passes():
    ok, problems = verify("The model predicts meningioma with 90.0% probability; glioma is the runner-up at 6.0%. "
                          "Expert review is recommended because the image is unusual.", FACTS)
    assert ok, problems


def test_hallucinated_number_fails():
    ok, problems = verify("The model predicts meningioma with 93% probability.", FACTS)
    assert not ok and any("93" in p for p in problems)


def test_other_class_asserted_fails():
    ok, _ = verify("This scan shows a glioma.", FACTS)
    assert not ok


def test_treatment_and_definitive_language_fail():
    assert not verify("The patient should undergo surgery.", FACTS)[0]
    assert not verify("This confirms a meningioma.", FACTS)[0]


def test_kb_numbers_allowed_only_with_passage():
    kb = KB()
    passages = kb.search("pituitary macroadenoma size", k=2)
    text = "Macroadenomas measure 10 mm or more."
    assert verify(text, FACTS, passages)[0]
    assert not verify(text, FACTS, [])[0]


def test_template_report_without_llm():
    secs = dict(template_report(FACTS))
    assert "The model predicts meningioma" in secs["Model output"]
    assert "Expert review recommended" in secs["Review"]


class _NoLLM:
    def available(self):
        return False


def test_assistant_falls_back_without_llm():
    a = Assistant(llm=_NoLLM())
    rep = a.report(RESULT)
    assert rep["narrative"] is None and rep["sections"]
    ans = a.ask("What does a dural tail mean?", RESULT)
    assert ans["answer"] is None and ans["fallback"] and ans["fallback"][0]["source"] == "meningioma"


def test_fact_echo_and_list_markers():
    assert not verify('{ "prediction": "meningioma", "probabilities_percent": {"meningioma": 90.0} }', FACTS)[0]
    assert not verify("The needs_expert_review flag is true.", FACTS)[0]
    ok, problems = verify("1. The model predicts meningioma.\n2. Glioma is the runner-up at 6.0%.", FACTS)
    assert ok, problems


def test_certainty_language_with_negation():
    assert not verify("The model is certain of meningioma as the primary diagnosis.", FACTS)[0]
    assert verify("A single slice cannot give a certain answer; the model predicts meningioma.", FACTS)[0]
    assert verify("The model is uncertain between classes.", FACTS)[0]


def test_fact_sheet_has_no_json():
    from edge.agent.tools import fact_sheet
    sheet = fact_sheet(FACTS)
    assert "{" not in sheet and "_" not in sheet and "90.0%" in sheet and "Not available" in sheet


class _EchoLLM:
    calls = 0

    def available(self):
        return True

    def chat(self, messages, **kw):
        _EchoLLM.calls += 1
        return "should not be called"


def test_treatment_questions_answered_by_policy_without_llm():
    a = Assistant(llm=_EchoLLM())
    for q in ("Should this patient have surgery?", "What is the survival rate?", "Is a biopsy needed?"):
        r = a.ask(q, RESULT)
        assert r.get("policy") and "cannot advise" in r["answer"] and not r["llm_used"]
    assert _EchoLLM.calls == 0


def test_semantic_contradictions():
    low_ev = {**FACTS, "model_evidence_inside_lesion_percent": 5.4}
    assert not verify("The model's positive Grad-CAM evidence is concentrated within the lesion.", low_ev)[0]
    assert not verify("The model's evidence lies within that lesion.", low_ev)[0]
    assert verify("Only 5.4% of the model's evidence lies inside the lesion; most lies outside it.", low_ev)[0]
    assert not verify("Since the prediction set contains only meningioma, the result is uncertain.", FACTS)[0]
    multi = {**FACTS, "prediction_set_95": ["meningioma", "glioma"]}
    assert not verify("The prediction set contains only one class.", multi)[0]


def test_negations_and_purpose_clauses_are_not_flagged():
    nt = case_facts({**RESULT, "prediction": "no_tumor", "prediction_set": ["no_tumor"]})
    assert verify("The model found no evidence for glioma, meningioma or pituitary tumor in this slice.", nt)[0]
    assert verify("Review by a radiologist is necessary to confirm the diagnosis.", FACTS)[0]
    assert verify("A definitive diagnosis requires tissue.", FACTS)[0]


def test_rejected_case_never_reaches_llm():
    a = Assistant(llm=_EchoLLM())
    calls = _EchoLLM.calls
    rej = {**RESULT, "status": "rejected", "prediction": None, "prediction_set": [], "flags": ["out_of_distribution"]}
    r = a.ask("Is there a tumor?", rej)
    assert r["policy"] and "not classified" in r["answer"] and _EchoLLM.calls == calls


def test_routing_of_fact_determined_questions():
    a = Assistant(llm=_EchoLLM())
    calls = _EchoLLM.calls
    r = a.ask("What does the 95% prediction set mean?", RESULT)
    assert r["route"] == "set" and "contains only meningioma" in r["answer"]
    r = a.ask("Why meningioma and not glioma?", RESULT)
    assert r["route"] == "why" and "glioma 6.0%" in r["answer"]
    assert _EchoLLM.calls == calls
    assert a.ask("What MRI features are typical of meningioma?", RESULT).get("route") is None   # goes to the LLM
    for txt in (a.ask("Why meningioma?", RESULT)["answer"], a.ask("Explain the set", RESULT)["answer"]):
        assert verify(txt, FACTS, [])[0], txt                         # template answers satisfy the verifier too


def test_singleton_set_other_classes_not_open():
    assert not verify("Other classes like glioma are also considered.", FACTS)[0]
    assert verify("A single slice cannot rule out a lesion elsewhere in the brain.", FACTS)[0]


def test_95_only_as_method_constant_and_no_image_claims():
    assert verify("The 95% prediction set contains only meningioma.", FACTS)[0]
    assert not verify("The model predicts meningioma with a 95% probability.", FACTS)[0]
    assert not verify("Meningiomas are dural-based; these features are consistent with the model's prediction.", FACTS)[0]


def test_review_question_routed():
    a = Assistant(llm=_EchoLLM())
    r = a.ask("Does this case need expert review, and why?", RESULT)
    assert r["route"] == "review" and "Automatic expert review was triggered" in r["answer"]


def test_normalise_and_determiner():
    from edge.agent.verifier import normalise
    t = normalise("The 95% prediction set confirms that only meningioma remains plausible.")
    assert "indicates" in t and verify(t, FACTS)[0]
    assert verify("Certain regions of the image favour meningioma over glioma in the Finer-CAM view.", FACTS)[0]
    assert not verify("This scan certainly shows a meningioma.", FACTS)[0]


def test_audit_round3_patterns():
    assert not verify("The model does not rule out other classes such as glioma or no tumor.", FACTS)[0]
    assert not verify("These features support the model's classification of the lesion as a meningioma.", FACTS)[0]
    assert not verify("The model predicts meningioma with 100% confidence.", FACTS)[0]

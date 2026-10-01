"""Web app behaviour with a stub engine (no model needed): CSRF, rejection handling, headers."""

import io
import re

import pytest

pytest.importorskip("flask")
from edge.agent.orchestrator import Assistant  # noqa: E402
from edge.webapp.app import create_app  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


class StubEngine:
    has_deep_explain = False

    def __init__(self, status="ok"):
        self.status = status

    def analyze(self, data, filename=None):
        ok = self.status == "ok"
        return {"status": self.status, "reject_reasons": [] if ok else ["color_image"],
                "prediction": "meningioma" if ok else None, "raw_argmax": "meningioma",
                "probabilities": {"glioma": .05, "meningioma": .9, "no_tumor": .02, "pituitary": .03},
                "prediction_set": ["meningioma"], "conformal_alpha": 0.05,
                "ood": {"mahalanobis_score": 0.1, "threshold": 1.5, "is_ood": False, "energy": -5.0},
                "explanation": {"finer_cam_reference": "glioma", "evidence_inside_lesion": 0.4,
                                "logit_decomposition_error": 1e-6},
                "lesion": {"present": True, "image_side": "image-left", "image_vertical": "upper",
                           "area_fraction_of_image": 0.02},
                "flags": [] if ok else ["color_image"], "needs_expert_review": not ok,
                "timings_ms": {"model_ms": 12.0},
                "overlays": {k: PNG for k in ("cam", "finer_cam", "lesion", "original")}, "_state": {}}

    def finer_png(self, state, gamma):
        return PNG


class FakeLLM:
    """Deterministic stand-in for llama-server (tests never touch a real LLM)."""

    def __init__(self, reply=None):
        self.reply = reply

    def available(self):
        return self.reply is not None

    def chat(self, messages, max_tokens=350, temperature=0.2, json_schema=None):
        return self.reply


def _client(status="ok", llm_reply=None):
    app = create_app(StubEngine(status), {"class_names": ["glioma", "meningioma", "no_tumor", "pituitary"]},
                     assistant=Assistant(llm=FakeLLM(llm_reply)))
    app.testing = True
    return app.test_client()


def _csrf(client):
    html = client.get("/").get_data(as_text=True)
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def test_csrf_required():
    c = _client()
    r = c.post("/analyze", data={"scan": (io.BytesIO(PNG), "x.png")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_analyze_ok_flow_and_headers():
    c = _client()
    tok = _csrf(c)
    r = c.post("/analyze", data={"csrf": tok, "scan": (io.BytesIO(PNG), "x.png")}, content_type="multipart/form-data")
    assert r.status_code == 302
    page = c.get(r.headers["Location"])
    html = page.get_data(as_text=True)
    assert "Meningioma" in html and "95% prediction set" in html
    assert "default-src 'self'" in page.headers["Content-Security-Policy"]
    assert page.headers["X-Frame-Options"] == "DENY"
    cid = r.headers["Location"].rstrip("/").split("/")[-1]
    assert c.get(f"/case/{cid}/img/cam.png").status_code == 200
    assert c.get(f"/case/{cid}/finer.png?gamma=0.5").status_code == 200


def test_rejected_input_shows_no_class():
    c = _client("rejected")
    tok = _csrf(c)
    r = c.post("/analyze", data={"csrf": tok, "scan": (io.BytesIO(PNG), "x.png")}, content_type="multipart/form-data")
    html = c.get(r.headers["Location"]).get_data(as_text=True)
    assert "Not analysed" in html and "Meningioma" not in html


def test_unsupported_extension_rejected():
    c = _client()
    tok = _csrf(c)
    r = c.post("/analyze", data={"csrf": tok, "scan": (io.BytesIO(b"MZ"), "evil.exe")}, content_type="multipart/form-data")
    assert r.status_code == 400


def _case(c, status="ok"):
    tok = _csrf(c)
    r = c.post("/analyze", data={"csrf": tok, "scan": (io.BytesIO(PNG), "x.png")}, content_type="multipart/form-data")
    return tok, r.headers["Location"].rstrip("/").split("/")[-1]


def test_report_page_without_llm():
    c = _client()
    _, cid = _case(c)
    html = c.get(f"/case/{cid}/report").get_data(as_text=True)
    assert "The model predicts meningioma" in html and "Reviewed by" in html and "LLM offline" in html


def test_narrative_grounded_and_rejected():
    good = "The model predicts meningioma with 90.0% probability; glioma is the runner-up at 5.0%."
    c = _client(llm_reply=good)
    tok, cid = _case(c)
    j = c.post(f"/case/{cid}/narrative", headers={"X-CSRF-Token": tok}).get_json()
    assert j["narrative"] == good and j["llm_used"]
    assert good in c.get(f"/case/{cid}/report").get_data(as_text=True)      # kept for the printed report

    c = _client(llm_reply="This confirms a glioma; surgery is advised. Probability 97%.")
    tok, cid = _case(c)
    j = c.post(f"/case/{cid}/narrative", headers={"X-CSRF-Token": tok}).get_json()
    assert j["narrative"] is None and "grounding" in j["message"]
    assert [a["ok"] for a in j["verification"]] == [False, False]


def test_ask_requires_csrf_and_falls_back_to_sources():
    c = _client()
    tok, cid = _case(c)
    assert c.post(f"/case/{cid}/ask", json={"question": "What is a dural tail?"}).status_code == 400
    assert c.post(f"/case/{cid}/ask", json={"question": " "}, headers={"X-CSRF-Token": tok}).status_code == 400
    j = c.post(f"/case/{cid}/ask", json={"question": "What is a dural tail?"}, headers={"X-CSRF-Token": tok}).get_json()
    assert j["answer"] is None and not j["llm_used"] and j["sources"]


def test_ask_with_llm():
    ans = "The model predicts meningioma; the 95% prediction set contains only meningioma."
    c = _client(llm_reply=ans)
    tok, cid = _case(c)
    j = c.post(f"/case/{cid}/ask", json={"question": "What MRI features are typical of meningioma?"}, headers={"X-CSRF-Token": tok}).get_json()
    assert j["answer"] == ans and j["verified"]


def test_rejected_drafts_not_sent_to_browser():
    bad = "This confirms a glioma. Probability 97%."
    c = _client(llm_reply=bad)
    tok, cid = _case(c)
    r = c.post(f"/case/{cid}/narrative", headers={"X-CSRF-Token": tok})
    assert "97%" not in r.get_data(as_text=True)


def test_ask_routes_set_question_without_llm():
    c = _client(llm_reply="should not be used")
    tok, cid = _case(c)
    j = c.post(f"/case/{cid}/ask", json={"question": "What does the prediction set mean?"}, headers={"X-CSRF-Token": tok}).get_json()
    assert j["route"] == "set" and not j["llm_used"] and "contains only meningioma" in j["answer"]

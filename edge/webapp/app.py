"""NeuroEdge web app (runs on the Raspberry Pi, fully offline).

Privacy: uploads are processed in memory and kept in a small expiring in-RAM case store — nothing is
written to disk. Security: per-session CSRF token on every POST, 20 MB upload cap, type validation by
decoding (not by extension), strict Content-Security-Policy (no inline scripts, no external resources).
A rejected input (out-of-distribution, colour photo, empty image) never shows a tumor class.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
import uuid
from collections import OrderedDict

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}
DISCLAIMER = ("Research and education tool. Not a medical device. Outputs must not be used for diagnosis "
              "or treatment decisions; every result requires review by a qualified radiologist.")
FLAG_TEXT = {
    "out_of_distribution": "The image is unlike the MRI slices the model was validated on (feature-space OOD test).",
    "unusual_image": "The image is less typical than 95% of validation scans; interpret with extra care.",
    "color_image": "This looks like a colour photograph, not a grayscale scan.",
    "mostly_empty": "The image is almost empty.",
    "low_resolution": "Very low resolution (under 128 px).",
    "blurry_or_flat": "Very little image detail (blurred or flat).",
    "uncertain_prediction_set": "More than one class is plausible at the 95% confidence level.",
    "empty_prediction_set": "No class reaches the calibrated 95% confidence level.",
    "explanation_outside_lesion": "The model's evidence lies almost entirely outside the detected lesion.",
    "classifier_segmenter_disagree": "The classifier and the lesion detector disagree about whether a tumor is present.",
}
CLASS_TEXT = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary tumor", "no_tumor": "No tumor"}


class CaseStore:
    """Bounded, expiring in-memory store (no disk persistence of patient images)."""

    def __init__(self, max_cases: int = 50, ttl_s: int = 3600):
        self.max_cases, self.ttl_s = max_cases, ttl_s
        self._d: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, case: dict) -> str:
        cid = uuid.uuid4().hex
        with self._lock:
            self._d[cid] = {**case, "_t": time.time()}
            while len(self._d) > self.max_cases:
                self._d.popitem(last=False)
        return cid

    def get(self, cid: str) -> dict | None:
        with self._lock:
            c = self._d.get(cid)
            if c is None or time.time() - c["_t"] > self.ttl_s:
                self._d.pop(cid, None)
                return None
            return c


def create_app(engine, model_info: dict | None = None, secret_key: str | None = None, assistant=None) -> Flask:
    app = Flask(__name__)
    if assistant is None:
        from edge.agent.orchestrator import Assistant
        assistant = Assistant()
    app.config.update(SECRET_KEY=secret_key or secrets.token_hex(32), MAX_CONTENT_LENGTH=20 * 1024 * 1024,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
    cases = CaseStore()
    info = model_info or {}

    # ------------------------------------------------------------------ security plumbing
    def csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    @app.before_request
    def check_csrf():
        if request.method == "POST":
            sent = request.form.get("csrf") or request.headers.get("X-CSRF-Token", "")
            if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
                abort(400, "CSRF check failed")

    @app.after_request
    def headers(resp):
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; style-src 'self'; "
                                                   "script-src 'self'; frame-ancestors 'none'; form-action 'self'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def inject():
        return {"csrf_token": csrf_token, "disclaimer": DISCLAIMER, "model_info": info,
                "flag_text": FLAG_TEXT, "class_text": CLASS_TEXT}

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.post("/analyze")
    def analyze():
        f = request.files.get("scan")
        if f is None or not f.filename:
            return render_template("index.html", error="Please choose an image file."), 400
        ext = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXT:
            return render_template("index.html", error=f"Unsupported file type '{ext}'. Use PNG, JPG or DICOM."), 400
        data = f.read()
        try:
            res = engine.analyze(data, filename=f.filename)
        except ValueError as e:                       # undecodable / unsupported content
            return render_template("index.html", error=f"Could not read this file: {e}"), 400
        state = res.pop("_state")
        overlays = res.pop("overlays")
        cid = cases.put({"result": res, "overlays": overlays, "state": state, "raw": data, "filename": f.filename})
        return redirect(url_for("case", cid=cid))

    @app.get("/case/<cid>")
    def case(cid):
        c = cases.get(cid) or abort(404)
        return render_template("case.html", cid=cid, r=c["result"], deep=engine.has_deep_explain)

    @app.get("/case/<cid>/img/<kind>.png")
    def case_img(cid, kind):
        c = cases.get(cid) or abort(404)
        png = c["overlays"].get(kind) or c.get("extra", {}).get(kind) or abort(404)
        return app.response_class(png, mimetype="image/png")

    @app.get("/case/<cid>/finer.png")
    def finer(cid):
        c = cases.get(cid) or abort(404)
        try:
            gamma = float(request.args.get("gamma", "1.0"))
        except ValueError:
            abort(400)
        return app.response_class(engine.finer_png(c["state"], gamma), mimetype="image/png")

    @app.post("/case/<cid>/deep")
    def deep(cid):
        c = cases.get(cid) or abort(404)
        if c["result"]["status"] != "ok":
            return jsonify(error="Deep Explain is disabled for rejected inputs."), 400
        if not engine.has_deep_explain:
            return jsonify(error="This model bundle has no FP32 weights for Deep Explain."), 400
        r = engine.deep_explain(c["raw"], filename=c["filename"])
        c.setdefault("extra", {})["deep"] = r.pop("overlay")
        return jsonify(url=url_for("case_img", cid=cid, kind="deep"), **r)

    @app.get("/case/<cid>/report")
    def report(cid):
        c = cases.get(cid) or abort(404)
        from edge.agent.orchestrator import template_report
        from edge.agent.tools import case_facts
        facts = case_facts(c["result"])
        return render_template("report.html", cid=cid, r=c["result"], sections=template_report(facts),
                               narrative=c.get("narrative"), llm_ready=assistant.llm.available(),
                               generated=time.strftime("%Y-%m-%d %H:%M"), version=info.get("version", "—"))

    @app.post("/case/<cid>/narrative")
    def narrative(cid):
        c = cases.get(cid) or abort(404)
        rep = assistant.report(c["result"])
        if rep["narrative"]:
            c["narrative"] = rep["narrative"]
        # unverified drafts never leave the server (the problem messages quote them, so they stay server-side too)
        checks = [{"attempt": at["attempt"], "ok": at["ok"]} for at in rep.get("verification") or []]
        return jsonify(narrative=rep["narrative"], llm_used=rep["llm_used"], verification=checks,
                       message=None if rep["narrative"] else ("AI narrative unavailable (local LLM offline)."
                                                               if not rep["llm_used"] else
                                                               "The draft failed grounding checks twice; showing the template only."))

    @app.post("/case/<cid>/ask")
    def ask(cid):
        c = cases.get(cid) or abort(404)
        q = (request.get_json(silent=True) or {}).get("question", "").strip()
        if not q:
            return jsonify(error="Empty question."), 400
        res = assistant.ask(q[:500], c["result"])
        return jsonify(answer=res.get("answer"), verified=res.get("verified"), llm_used=res.get("llm_used"),
                       route=res.get("route"),
                       sources=[{"source": p["source"], "text": p["text"]} for p in (res.get("fallback") or res.get("passages") or [])])

    @app.get("/case/<cid>/json")
    def case_json(cid):
        c = cases.get(cid) or abort(404)
        return jsonify(c["result"])

    @app.errorhandler(413)
    def too_large(_):
        return render_template("index.html", error="File too large (max 20 MB)."), 413

    return app

"""NeuroEdge clinical portal (Flask; runs offline on the Raspberry Pi).

Roles
  patient  registers, uploads scans, sees case status and the DOCTOR's signed report (never AI output)
  doctor   prioritised worklist, case review with AI (calibrated probabilities, 95% set, Grad-CAM, lesion
           outline, review flags, assistant), signs / amends reports, can request a better image
  admin    manages doctor accounts, views the audit log
Security: argon2 passwords, lockout + rate limits, CSRF on every POST, strict CSP (no inline script, no
external resources), object-level access checks, every scan / overlay / report encrypted at rest, audit log.
On the Pi the app listens on loopback only; Caddy terminates HTTPS in front of it (behind_proxy=True: client IPs
from X-Forwarded-For for rate limits / audit, Secure __Host- session cookie).
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
import socket
import time
from pathlib import Path

from flask import Blueprint, Flask, abort, flash, g, jsonify, redirect, render_template, request, session, url_for

from edge.portal import auth, services
from edge.portal.db import DB, DIAGNOSES, STATUSES
from edge.portal.storage import BlobStore

DISCLAIMER = "NeuroEdge research prototype"
CLASS_TEXT = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary tumor", "no_tumor": "No tumor",
              "other": "Other finding", "inconclusive": "Inconclusive — further imaging needed"}
STATUS_TEXT = {"awaiting_review": "Awaiting doctor review", "in_review": "In review",
               "reupload_requested": "New image requested", "finalized": "Report ready"}
FLAG_TEXT = {
    "out_of_distribution": "Image differs from the scans the model was trained on.",
    "unusual_image": "Image is less typical than 95% of calibration scans.",
    "color_image": "Looks like a colour photo, not a grayscale scan.",
    "mostly_empty": "Image is almost empty.",
    "low_resolution": "Very low resolution (< 128 px).",
    "blurry_or_flat": "Very little image detail.",
    "uncertain_prediction_set": "More than one class is plausible at the 95% level.",
    "empty_prediction_set": "No class reaches the calibrated 95% level.",
    "explanation_outside_lesion": "The model's evidence lies almost entirely outside the detected lesion.",
    "classifier_segmenter_disagree": "Classifier and lesion detector disagree on whether a tumor is present.",
}
FLAG_SHORT = {"out_of_distribution": "Atypical image", "unusual_image": "Less typical", "color_image": "Colour image",
              "mostly_empty": "Mostly empty", "low_resolution": "Low resolution", "blurry_or_flat": "Low detail",
              "uncertain_prediction_set": "Two+ classes plausible", "empty_prediction_set": "Low confidence",
              "explanation_outside_lesion": "Evidence off lesion", "classifier_segmenter_disagree": "Detector disagrees"}


def tls_name_allowed(name: str) -> bool:
    """Names Caddy may issue an on-demand certificate for: this device's mDNS name, localhost, or a private /
    link-local / loopback IP (how browsers on the clinic LAN reach the Pi). Anything else is refused, so a
    stranger cannot make the Pi mint certificates for arbitrary names."""
    name = name.strip().lower().rstrip(".")
    if name in {"localhost", f"{socket.gethostname().lower()}.local"}:
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False
    return ip.is_private or ip.is_link_local or ip.is_loopback


def safe_next(nxt: str) -> str | None:
    """Only same-site paths: '/x' yes; '//evil', '/\\evil' (browsers read '\\' as '/'), 'https://evil' and
    control characters no."""
    if not nxt.startswith("/") or nxt.startswith(("//", "/\\")) or "\\" in nxt:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in nxt):
        return None
    return nxt


def create_app(engine, model_info: dict, data_dir: str | Path, key_file: str | Path,
               secret_key: str | None = None, assistant=None, behind_proxy: bool = False) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
    app.config.update(SECRET_KEY=secret_key or secrets.token_hex(32), MAX_CONTENT_LENGTH=21 * 1024 * 1024,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
                      # __Host- prefix: the browser only accepts it over HTTPS, host-only, path /
                      SESSION_COOKIE_SECURE=behind_proxy,
                      SESSION_COOKIE_NAME="__Host-ne_session" if behind_proxy else "ne_session")
    if behind_proxy:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)   # exactly one trusted proxy (Caddy)
    db = DB(Path(data_dir) / "portal.db")
    store = BlobStore.from_key_file(Path(data_dir), Path(key_file))
    ctx = services.Ctx(db=db, store=store, engine=engine, model_version=str(model_info.get("version", "unknown")))
    if assistant is None:
        from edge.agent.orchestrator import Assistant
        assistant = Assistant()
    app.extensions["portal"] = {"db": db, "store": store, "ctx": ctx, "assistant": assistant}
    login_limiter, register_limiter = auth.RateLimiter(20, 60), auth.RateLimiter(5, 3600)

    # ------------------------------------------------------------------ plumbing
    def csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        return session["csrf"]

    @app.before_request
    def before():
        auth.load_user(db)
        if request.method == "POST":
            sent = request.form.get("csrf") or request.headers.get("X-CSRF-Token", "")
            if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
                abort(400, "Security check failed — please reload the page and try again.")

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("Content-Security-Policy", (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "frame-ancestors 'none'; form-action 'self'; base-uri 'none'"))
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return resp

    @app.context_processor
    def inject():
        return {"csrf_token": csrf_token, "disclaimer": DISCLAIMER, "model_info": model_info, "me": g.get("user"),
                "class_text": CLASS_TEXT, "status_text": STATUS_TEXT, "flag_text": FLAG_TEXT, "flag_short": FLAG_SHORT,
                "fmt_time": lambda t: time.strftime("%d %b %Y, %H:%M", time.localtime(t)) if t else "—"}

    def ip():
        return request.remote_addr

    @app.get("/_tls_ask")
    def tls_ask():
        # Called by Caddy (on_demand_tls ask) directly on loopback. Requests relayed through the proxy carry
        # X-Forwarded-For, so their remote_addr is the real client and they are refused.
        if request.remote_addr != "127.0.0.1" or request.headers.get("X-Forwarded-For"):
            abort(404)
        return ("ok", 200) if tls_name_allowed(request.args.get("domain", "")) else ("refused", 403)

    def get_case_or_404(cid: str):
        case = db.case(cid)
        if not auth.can_view_case(g.user, case):
            abort(404)                                     # 404, not 403: do not reveal other patients' case ids
        return case

    # ------------------------------------------------------------------ auth
    bp_auth = Blueprint("auth", __name__)

    @bp_auth.get("/")
    def home():
        if g.user is None:
            return redirect(url_for("auth.login"))
        return redirect(url_for({"patient": "patient.dashboard", "doctor": "doctor.worklist",
                                 "admin": "admin.users"}[g.user["role"]]))

    @bp_auth.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("portal/login.html", expired=g.get("expired"))
        if not login_limiter.hit(ip() or "?"):
            return render_template("portal/login.html", error="Too many attempts. Wait a minute."), 429
        user, err = auth.check_login(db, request.form.get("email", ""), request.form.get("password", ""))
        if user is None:
            return render_template("portal/login.html", error=err, email=request.form.get("email", "")), 401
        auth.start_session(user)
        db.audit(user["id"], "login", ip=ip())
        return redirect(safe_next(request.args.get("next", "")) or url_for("auth.home"))

    @bp_auth.post("/logout")
    def logout():
        if g.user is not None:
            db.audit(g.user["id"], "logout", ip=ip())
        session.clear()
        return redirect(url_for("auth.login"))

    @bp_auth.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("portal/register.html")
        f = request.form
        email, name, pw = f.get("email", "").strip().lower(), f.get("name", "").strip(), f.get("password", "")
        err = None
        if not register_limiter.hit(ip() or "?"):
            err = "Too many registrations from this device. Try later."
        elif not name or "@" not in email or len(email) > 200:
            err = "Please enter your name and a valid e-mail address."
        elif pw != f.get("password2", ""):
            err = "Passwords do not match."
        elif f.get("consent") != "yes":
            err = "Please accept the research-use statement."
        else:
            err = auth.password_problem(pw, email)
        if err is None and db.user_by_email(email) is not None:
            err = "An account with this e-mail already exists."
        if err:
            return render_template("portal/register.html", error=err, form=f), 400
        uid = db.create_user(email, name, "patient", auth.hash_password(pw),
                             {"dob": f.get("dob") or None, "sex": f.get("sex") or None, "phone": f.get("phone") or None})
        db.audit(uid, "registered", ip=ip())
        auth.start_session(db.user(uid))
        return redirect(url_for("patient.dashboard"))

    # ------------------------------------------------------------------ shared: images & reports
    bp_case = Blueprint("case", __name__, url_prefix="/case")

    @bp_case.get("/<cid>/img/<kind>.png")
    @auth.require("patient", "doctor", "admin")
    def image(cid, kind):
        case = get_case_or_404(cid)
        base = kind.rsplit("_", 1)[0]
        if g.user["role"] == "patient" and base != "original":      # patients never receive AI overlays
            abort(404)
        if base not in ("original", "cam", "lesion", "cam_lesion") or not store.exists(case["id"], kind):
            abort(404)
        if g.user["role"] != "patient":
            db.audit(g.user["id"], "image_viewed", cid, {"kind": kind}, ip())
        return app.response_class(store.get(case["id"], kind), mimetype="image/png")

    @bp_case.get("/<cid>/report/<int:version>")
    @auth.require("patient", "doctor", "admin")
    def report_snapshot(cid, version):
        case = get_case_or_404(cid)
        kind = f"report_v{version}"
        if not store.exists(case["id"], kind):
            abort(404)
        db.audit(g.user["id"], "report_viewed", cid, {"version": version}, ip())
        resp = app.response_class(store.get(case["id"], kind), mimetype="text/html")
        # the archived snapshot is self-contained (embedded CSS); it may never run script
        resp.headers["Content-Security-Policy"] = ("default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; "
                                                   "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        return resp

    # ------------------------------------------------------------------ patient
    bp_pat = Blueprint("patient", __name__, url_prefix="/patient")

    @bp_pat.get("/")
    @auth.require("patient")
    def dashboard():
        cases = db.q("SELECT * FROM cases WHERE patient_id = ? ORDER BY created DESC", (g.user["id"],))
        return render_template("portal/patient_dashboard.html", cases=cases)

    @bp_pat.route("/upload", methods=["GET", "POST"])
    @auth.require("patient")
    def upload():
        if request.method == "GET":
            return render_template("portal/patient_upload.html")
        f = request.files.get("scan")
        try:
            cid = services.submit_scan(ctx, g.user, f.read() if f else b"", f.filename if f else "",
                                       request.form.get("note", ""), request.form.get("consent") == "yes", ip())
        except services.UploadError as e:
            return render_template("portal/patient_upload.html", error=str(e)), 400
        flash("Your scan was received. A doctor will review it; you will see the report here when it is ready.")
        return redirect(url_for("patient.case", cid=cid))

    @bp_pat.route("/case/<cid>", methods=["GET", "POST"])
    @auth.require("patient")
    def case(cid):
        c = get_case_or_404(cid)
        if request.method == "POST":                        # re-upload requested by the doctor
            f = request.files.get("scan")
            try:
                services.resubmit_scan(ctx, g.user, c, f.read() if f else b"", f.filename if f else "", ip())
                flash("Thank you — the new image was sent to your doctor.")
            except services.UploadError as e:
                flash(str(e), "error")
            return redirect(url_for("patient.case", cid=cid))
        attempt = (db.latest_ai(cid) or {}).get("result", {}).get("attempt", 1)
        reports = db.reports(cid)
        return render_template("portal/patient_case.html", case=c, reports=reports,
                               report=reports[0] if reports else None, attempt=attempt)

    # ------------------------------------------------------------------ doctor
    bp_doc = Blueprint("doctor", __name__, url_prefix="/doctor")

    @bp_doc.get("/")
    @auth.require("doctor")
    def worklist():
        status = request.args.get("status", "open")
        where = {"open": "c.status IN ('awaiting_review','in_review')", "all": "1=1"}.get(
            status, "c.status = ?" if status in STATUSES else "1=0")
        args = (status,) if status in STATUSES else ()
        rows = db.q(f"SELECT c.*, u.name AS patient_name, d.name AS doctor_name FROM cases c "
                    f"JOIN users u ON u.id = c.patient_id LEFT JOIN users d ON d.id = c.doctor_id "
                    f"WHERE {where} ORDER BY (c.status = 'finalized'), c.priority DESC, c.created LIMIT 500", args)
        items = []
        for r in rows:
            ai = db.latest_ai(r["id"])
            items.append({"case": r, "ai": ai["result"] if ai else None})
        counts = {s: db.one("SELECT COUNT(*) AS n FROM cases WHERE status = ?", (s,))["n"] for s in STATUSES}
        return render_template("portal/doctor_worklist.html", items=items, status=status, counts=counts)

    @bp_doc.get("/case/<cid>")
    @auth.require("doctor")
    def case_review(cid):
        c = get_case_or_404(cid)
        services.open_for_review(ctx, c, g.user, ip())
        c = db.case(cid)
        ai = db.latest_ai(cid)
        db.audit(g.user["id"], "case_viewed", cid, ip=ip())
        patient = db.user(c["patient_id"])
        profile = db.one("SELECT * FROM patient_profiles WHERE user_id = ?", (c["patient_id"],))
        return render_template("portal/doctor_case.html", case=c, ai=ai, r=ai["result"] if ai else None,
                               patient=patient, profile=profile, reports=db.reports(cid), diagnoses=DIAGNOSES,
                               llm_ready=assistant.llm.available())

    @bp_doc.post("/case/<cid>/sign")
    @auth.require("doctor")
    def sign(cid):
        c = get_case_or_404(cid)
        f = request.form
        if not auth.verify_password(g.user, f.get("password", "")):
            flash("Signature failed: password incorrect.", "error")
            return redirect(url_for("doctor.case_review", cid=cid))
        dx = f.get("diagnosis", "")
        agrees = {"yes": True, "no": False}.get(f.get("agrees"))
        try:
            v = services.sign_report(ctx, c, g.user, dx, agrees, f.get("findings", ""), f.get("recommendation", ""),
                                     f.get("amendment_reason") or None, ip())
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("doctor.case_review", cid=cid))
        flash(f"Report version {v} signed. The patient can now see it.")
        return redirect(url_for("doctor.worklist"))

    @bp_doc.post("/case/<cid>/reupload")
    @auth.require("doctor")
    def reupload(cid):
        c = get_case_or_404(cid)
        try:
            services.request_reupload(ctx, c, g.user, request.form.get("message", "").strip() or
                                      "Please upload a clearer MRI image.", ip())
            flash("The patient was asked for a new image.")
        except ValueError as e:
            flash(str(e), "error")
        return redirect(url_for("doctor.worklist"))

    @bp_doc.post("/case/<cid>/narrative")
    @auth.require("doctor")
    def narrative(cid):
        get_case_or_404(cid)
        ai = db.latest_ai(cid)
        if ai is None:
            return jsonify(error="No AI result."), 400
        rep = assistant.report(ai["result"])
        db.audit(g.user["id"], "assistant_narrative", cid, {"llm_used": rep["llm_used"]}, ip())
        return jsonify(narrative=rep["narrative"], sections=rep["sections"], llm_used=rep["llm_used"],
                       message=None if rep["narrative"] else ("Local LLM offline — template shown." if not rep["llm_used"]
                                                              else "Draft failed grounding checks; template shown."))

    @bp_doc.post("/case/<cid>/ask")
    @auth.require("doctor")
    def ask(cid):
        get_case_or_404(cid)
        ai = db.latest_ai(cid)
        q = (request.get_json(silent=True) or {}).get("question", "").strip()
        if not q or ai is None:
            return jsonify(error="Empty question."), 400
        res = assistant.ask(q[:500], ai["result"])
        return jsonify(answer=res.get("answer"), llm_used=res.get("llm_used"), route=res.get("route"),
                       sources=[{"source": p["source"], "text": p["text"]}
                                for p in (res.get("fallback") or res.get("passages") or [])])

    @bp_doc.get("/stats")
    @auth.require("doctor", "admin")
    def stats():
        rows = db.q("SELECT r.case_id, r.final_diagnosis, r.agrees_with_ai, r.signed_at, c.created FROM reports r "
                    "JOIN cases c ON c.id = r.case_id WHERE r.version = (SELECT MAX(version) FROM reports WHERE case_id = r.case_id)")
        pairs, turnaround = [], []
        for r in rows:
            ai = db.latest_ai(r["case_id"])
            res = ai["result"] if ai else {}
            if res.get("status") == "ok" and r["final_diagnosis"] in CLASS_TEXT:
                pairs.append((res["prediction"], r["final_diagnosis"]))
            turnaround.append((r["signed_at"] - r["created"]) / 3600)
        classes = ["glioma", "meningioma", "pituitary", "no_tumor"]
        confusion = {a: {d: sum(1 for p in pairs if p == (a, d)) for d in classes + ["other", "inconclusive"]}
                     for a in classes}
        agree = sum(1 for a, d in pairs if a == d)
        counts = {s: db.one("SELECT COUNT(*) AS n FROM cases WHERE status = ?", (s,))["n"] for s in STATUSES}
        med = sorted(turnaround)[len(turnaround) // 2] if turnaround else None
        return render_template("portal/doctor_stats.html", n_reports=len(rows), n_pairs=len(pairs),
                               agreement=(agree / len(pairs)) if pairs else None, confusion=confusion,
                               classes=classes, counts=counts, median_turnaround_h=med)

    # ------------------------------------------------------------------ admin
    bp_adm = Blueprint("admin", __name__, url_prefix="/admin")

    @bp_adm.route("/users", methods=["GET", "POST"])
    @auth.require("admin")
    def users():
        error = None
        if request.method == "POST":
            f = request.form
            email, name, pw, role = f.get("email", "").strip().lower(), f.get("name", "").strip(), f.get("password", ""), f.get("role")
            if role not in ("doctor", "admin") or not name or "@" not in email:
                error = "Name, e-mail and role (doctor/admin) are required."
            elif db.user_by_email(email) is not None:
                error = "An account with this e-mail already exists."
            else:
                error = auth.password_problem(pw, email)
            if error is None:
                uid = db.create_user(email, name, role, auth.hash_password(pw))
                db.audit(g.user["id"], "user_created", detail={"user": uid, "role": role}, ip=ip())
                flash(f"{role.title()} account created for {name}.")
                return redirect(url_for("admin.users"))
        rows = db.q("SELECT u.*, (SELECT COUNT(*) FROM cases WHERE patient_id = u.id) AS n_cases FROM users u ORDER BY role, name")
        return render_template("portal/admin_users.html", users=rows, error=error), (400 if error else 200)

    @bp_adm.post("/users/<int:uid>/toggle")
    @auth.require("admin")
    def toggle(uid):
        u = db.user(uid)
        if u is None or u["id"] == g.user["id"]:
            abort(400)
        with db.tx() as c:
            c.execute("UPDATE users SET active = 1 - active WHERE id = ?", (uid,))
        db.audit(g.user["id"], "user_" + ("disabled" if u["active"] else "enabled"), detail={"user": uid}, ip=ip())
        return redirect(url_for("admin.users"))

    @bp_adm.get("/audit")
    @auth.require("admin")
    def audit():
        rows = db.q("SELECT a.*, u.name AS user_name, u.role AS user_role FROM audit a LEFT JOIN users u ON u.id = a.user_id "
                    "ORDER BY a.ts DESC LIMIT 500")
        return render_template("portal/admin_audit.html", rows=rows)

    for bp in (bp_auth, bp_case, bp_pat, bp_doc, bp_adm):
        app.register_blueprint(bp)

    @app.errorhandler(413)
    def too_large(_):
        return render_template("portal/error.html", code=413, message="File too large (max 20 MB)."), 413

    for code in (400, 403, 404, 429):
        app.register_error_handler(code, lambda e, c=code: (render_template(
            "portal/error.html", code=c, message=getattr(e, "description", "")), c))
    return app

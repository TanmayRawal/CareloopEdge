"""NeuroEdge Imaging — radiology department portal (Flask; runs offline on the Raspberry Pi).

Roles
  patient  registers, uploads scans, follows the review, reads the DOCTOR's signed report (never AI output),
           books / reschedules / cancels a consultation with the reporting radiologist
  doctor   prioritised worklist with SLA badges, case review with AI (calibrated probabilities, 95% set, Grad-CAM,
           lesion outline, review flags, assistant), signs / amends reports, consultation schedule and attendance,
           working hours and days off
  admin    users, audit log, system status, e-mail outbox, department settings (reminder and booking policy)
Reminders (unread / unsigned studies, appointment reminders, daily agenda, attendance, no-shows) are produced by
`edge.portal.reminders`, run every 5 minutes by neuroedge-reminders.timer; e-mail via `edge.portal.mailer`.
Security: argon2 passwords, lockout + rate limits, CSRF on every POST, strict CSP (no inline script, no
external resources), object-level access checks, every scan / overlay / report encrypted at rest, audit log.
On the Pi the app listens on loopback only; Caddy terminates HTTPS in front of it (behind_proxy=True: client IPs
from X-Forwarded-For for rate limits / audit, Secure __Host- session cookie).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, abort, g, render_template, request, session, url_for

from edge.portal import auth, common, notify, scheduling, services
from edge.portal.common import safe_next, tls_name_allowed  # noqa: F401  (re-exported; tests import them here)
from edge.portal.db import DB
from edge.portal.storage import BlobStore


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

    # Static files are served with a content hash in the URL, so they can be cached for a year.
    static_dir = Path(app.static_folder)
    asset_ver = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:10] for p in static_dir.iterdir() if p.is_file()}

    def asset(name: str) -> str:
        return url_for("static", filename=name, v=asset_ver.get(name, "0"))

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
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.path.startswith("/static/") and request.args.get("v"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            resp.headers["Cache-Control"] = "no-store"        # pages contain patient data: never cached
        return resp

    def fmt(ts, pattern):
        return scheduling.local(ts, scheduling.tz(db)).strftime(pattern) if ts else "—"

    @app.context_processor
    def inject():
        me = g.get("user")
        bell = notify.unread(db, me["id"]) if me else (0, [])
        return {"csrf_token": csrf_token, "asset": asset, "disclaimer": common.DISCLAIMER, "model_info": model_info,
                "me": me, "clinic": db.setting("clinic_name"), "class_text": common.CLASS_TEXT,
                "status_text": common.STATUS_TEXT, "appt_text": common.APPT_TEXT, "role_text": common.ROLE_TEXT,
                "flag_text": common.FLAG_TEXT, "flag_short": common.FLAG_SHORT, "mrn": common.mrn,
                "age_from_dob": common.age_from_dob, "ago": common.ago, "bell_count": bell[0], "bell_items": bell[1],
                "open_count": lambda: db.one("SELECT COUNT(*) AS n FROM cases WHERE status IN ('awaiting_review','in_review')")["n"],
                "fmt_time": lambda t: fmt(t, "%d %b %Y, %H:%M"), "fmt_date": lambda t: fmt(t, "%d %b %Y"),
                "fmt_day": lambda t: fmt(t, "%a %d %b"), "fmt_clock": lambda t: fmt(t, "%H:%M"),
                "fmt_long": lambda t: fmt(t, "%A %d %B %Y")}

    def get_case_or_404(cid: str):
        case = db.case(cid)
        if not auth.can_view_case(g.user, case):
            abort(404)                                     # 404, not 403: do not reveal other patients' case ids
        return case

    @app.get("/_tls_ask")
    def tls_ask():
        # Called by Caddy (on_demand_tls ask) directly on loopback. Requests relayed through the proxy carry
        # X-Forwarded-For, so their remote_addr is the real client and they are refused.
        if request.remote_addr != "127.0.0.1" or request.headers.get("X-Forwarded-For"):
            abort(404)
        return ("ok", 200) if tls_name_allowed(request.args.get("domain", "")) else ("refused", 403)

    deps = SimpleNamespace(app=app, db=db, store=store, ctx=ctx, assistant=assistant, model_info=model_info,
                           data_dir=Path(data_dir), ip=lambda: request.remote_addr, get_case_or_404=get_case_or_404,
                           login_limiter=auth.RateLimiter(20, 60), register_limiter=auth.RateLimiter(5, 3600),
                           booking_limiter=auth.RateLimiter(20, 3600))
    from edge.portal.views import admin, auth as auth_views, case, doctor, notifications, patient
    for mod in (auth_views, case, patient, doctor, admin, notifications):
        app.register_blueprint(mod.blueprint(deps))

    @app.errorhandler(413)
    def too_large(_):
        return render_template("portal/error.html", code=413, message="File too large (max 20 MB)."), 413

    for code in (400, 403, 404, 429):
        app.register_error_handler(code, lambda e, c=code: (render_template(
            "portal/error.html", code=c, message=getattr(e, "description", "")), c))
    return app

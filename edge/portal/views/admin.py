"""Administration: users, audit log, system status, e-mail outbox and department settings."""

from __future__ import annotations

import shutil
import socket
import time
from zoneinfo import ZoneInfo

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from edge.portal import audit_text, auth, mailer, scheduling
from edge.portal.common import CLASS_TEXT, PAGE_SIZE, ago, human_bytes, page_args
from edge.portal.db import DEFAULT_SETTINGS

STARTED = time.time()

SETTING_FORM = [  # key, label, hint, kind
    ("clinic_name", "Department name", "", "text"),
    ("base_url", "Portal address", "e.g. https://rpi.local", "text"),
    ("timezone", "Time zone", "e.g. Asia/Kolkata", "text"),
    ("unread_hours", "Unread study reminder (hours)", "", "number"),
    ("unread_hours_urgent", "Urgent study reminder (hours)", "", "number"),
    ("unsigned_hours", "Unsigned study reminder (hours)", "", "number"),
    ("escalate_hours", "Escalate to admin (hours)", "", "number"),
    ("remind_before_hours", "Patient reminders (hours before)", "e.g. 24,2", "text"),
    ("unmarked_after_min", "Attendance reminder (minutes)", "", "number"),
    ("agenda_hour", "Daily agenda time", "e.g. 07:30", "text"),
    ("booking_lead_hours", "Minimum booking notice (hours)", "", "number"),
    ("booking_horizon_days", "Booking window (days)", "", "number"),
    ("cancel_cutoff_hours", "Change / cancel cutoff (hours)", "", "number"),
]


def validate(key: str, value: str) -> str | None:
    value = value.strip()
    if not value:
        return "is required"
    if key == "timezone":
        try:
            ZoneInfo(value)
        except Exception:
            return "is not a known time zone"
    elif key == "base_url" and not value.startswith(("https://", "http://")):
        return "must start with https://"
    elif key == "remind_before_hours":
        try:
            if not all(0 < float(x) <= 168 for x in value.split(",")):
                raise ValueError
        except ValueError:
            return "must be hours between 0 and 168, comma-separated"
    elif key == "agenda_hour":
        try:
            h, m = (int(x) for x in value.split(":"))
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError
        except ValueError:
            return "must be a time such as 07:30"
    elif dict((k, kind) for k, _, _, kind in SETTING_FORM)[key] == "number":
        try:
            if not 0 < float(value) <= 10000:
                raise ValueError
        except ValueError:
            return "must be a positive number"
    return None


def blueprint(d) -> Blueprint:
    bp = Blueprint("admin", __name__, url_prefix="/admin")
    db = d.db

    @bp.route("/users", methods=["GET", "POST"])
    @auth.require("admin")
    def users():
        error = None
        if request.method == "POST":
            f = request.form
            email, name, pw, role = f.get("email", "").strip().lower(), f.get("name", "").strip(), f.get("password", ""), f.get("role")
            if role not in ("doctor", "admin") or not name or "@" not in email:
                error = "Name, e-mail and role are required."
            elif db.user_by_email(email) is not None:
                error = "An account with this e-mail already exists."
            else:
                error = auth.password_problem(pw, email)
            if error is None:
                uid = db.create_user(email, name, role, auth.hash_password(pw))
                db.audit(g.user["id"], "user_created", detail={"user": uid, "role": role}, ip=d.ip())
                flash(f"Account created for {name}.")
                return redirect(url_for("admin.users"))
        role = request.args.get("role", "")
        where, args = ("WHERE u.role = ?", (role,)) if role in ("patient", "doctor", "admin") else ("", ())
        rows = db.q(f"SELECT u.*, (SELECT COUNT(*) FROM cases WHERE patient_id = u.id) AS n_cases FROM users u {where} "
                    f"ORDER BY u.role, u.name", args)
        return render_template("portal/admin_users.html", users=rows, error=error, role=role), (400 if error else 200)

    @bp.post("/users/<int:uid>/toggle")
    @auth.require("admin")
    def toggle(uid):
        u = db.user(uid)
        if u is None or u["id"] == g.user["id"]:
            abort(400)
        with db.tx() as c:
            c.execute("UPDATE users SET active = 1 - active WHERE id = ?", (uid,))
        db.audit(g.user["id"], "user_" + ("disabled" if u["active"] else "enabled"), detail={"user": uid}, ip=d.ip())
        return redirect(url_for("admin.users"))

    @bp.get("/audit")
    @auth.require("admin")
    def audit():
        page, offset = page_args(request)
        cat, who = request.args.get("cat", "").strip(), request.args.get("user", "").strip()
        cond, args = [], []
        if cat in audit_text.CATEGORIES:
            acts = [a for a in audit_text.EVENTS if audit_text.category(a) == cat]
            cond.append(f"a.action IN ({','.join('?' * len(acts))})")
            args += acts
        else:                                            # default: everything except routine image/report views
            cond.append("a.action NOT IN ('image_viewed', 'report_viewed')")
        if who:
            cond.append("u.name LIKE ?")
            args.append(f"%{who}%")
        rows = db.q(f"SELECT a.*, u.name AS user_name, u.role AS user_role, pu.name AS patient_name FROM audit a "
                    f"LEFT JOIN users u ON u.id = a.user_id LEFT JOIN cases c ON c.id = a.case_id "
                    f"LEFT JOIN users pu ON pu.id = c.patient_id WHERE {' AND '.join(cond)} "
                    f"ORDER BY a.ts DESC LIMIT ? OFFSET ?", tuple(args) + (PAGE_SIZE + 1, offset))
        zone = scheduling.tz(db)
        today = scheduling.local(time.time(), zone).date()
        days: list[tuple[str, list]] = []
        for r in rows[:PAGE_SIZE]:
            dday = scheduling.local(r["ts"], zone).date()
            name = "Today" if dday == today else "Yesterday" if (today - dday).days == 1 else dday.strftime("%A %d %B %Y")
            if not days or days[-1][0] != name:
                days.append((name, []))
            days[-1][1].append({"r": r, "label": audit_text.label(r["action"]), "cat": audit_text.category(r["action"]),
                                "text": audit_text.describe(r["action"], r["detail"])})
        return render_template("portal/admin_audit.html", days=days, more=len(rows) > PAGE_SIZE, page=page,
                               categories=audit_text.CATEGORIES, cat=cat, who=who)

    @bp.get("/system")
    @auth.require("admin")
    def system():
        data = d.data_dir
        blob_bytes = sum(p.stat().st_size for p in (data / "blobs").rglob("*") if p.is_file()) if (data / "blobs").exists() else 0
        db_bytes = sum(p.stat().st_size for p in data.glob("portal.db*"))
        disk = shutil.disk_usage(data)
        roles = {r["role"]: r["n"] for r in db.q("SELECT role, COUNT(*) AS n FROM users GROUP BY role")}
        last_login = db.one("SELECT MAX(ts) AS t FROM audit WHERE action = 'login'")["t"]
        out = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) AS n FROM outbox GROUP BY status")}
        info = {"model_version": d.model_info.get("version", "—"), "precision": d.model_info.get("precision") or "—",
                "classes": ", ".join(CLASS_TEXT.get(c, c) for c in d.model_info.get("class_names", [])) or "—",
                "llm": "Online" if d.assistant.llm.available() else "Offline", "uptime": ago(STARTED),
                "cases": db.one("SELECT COUNT(*) AS n FROM cases")["n"], "reports": db.one("SELECT COUNT(*) AS n FROM reports")["n"],
                "appointments": db.one("SELECT COUNT(*) AS n FROM appointments")["n"], "roles": roles,
                "blob": human_bytes(blob_bytes), "db": human_bytes(db_bytes), "disk_free": human_bytes(disk.free),
                "disk_total": human_bytes(disk.total), "disk_used_pct": round(100 * (disk.total - disk.free) / disk.total),
                "last_login": last_login, "host": socket.gethostname(), "mail": mailer.status(), "outbox": out}
        return render_template("portal/admin_system.html", info=info)

    @bp.get("/outbox")
    @auth.require("admin")
    def outbox():
        page, offset = page_args(request)
        status = request.args.get("status", "")
        where, args = ("WHERE status = ?", (status,)) if status in ("queued", "sent", "failed", "disabled") else ("", ())
        rows = db.q(f"SELECT id, to_email, subject, status, created, next_try, attempts, last_error, sent_at FROM outbox {where} "
                    f"ORDER BY id DESC LIMIT ? OFFSET ?", args + (PAGE_SIZE + 1, offset))
        counts = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) AS n FROM outbox GROUP BY status")}
        return render_template("portal/admin_outbox.html", rows=rows[:PAGE_SIZE], more=len(rows) > PAGE_SIZE, page=page,
                               status=status, counts=counts, mail=mailer.status())

    @bp.post("/outbox/<int:oid>/retry")
    @auth.require("admin")
    def outbox_retry(oid):
        mailer.retry(db, oid)
        db.audit(g.user["id"], "email_retry", detail={"outbox": oid}, ip=d.ip())
        flash("Queued for another attempt; it will be sent within 5 minutes.")
        return redirect(url_for("admin.outbox"))

    @bp.post("/outbox/test")
    @auth.require("admin")
    def outbox_test():
        me = g.user
        with db.tx() as c:
            c.execute("INSERT INTO outbox (user_id,to_email,subject,text,created,next_try) VALUES (?,?,?,?,?,?)",
                      (me["id"], me["email"], f"Test e-mail · {db.setting('clinic_name')}",
                       "This is a test message from the portal. E-mail delivery works.", time.time(), time.time()))
        res = mailer.flush(db)
        db.audit(me["id"], "email_test", detail=res, ip=d.ip())
        flash(f"Test e-mail sent to {me['email']}." if res["sent"] else
              "The test e-mail could not be sent yet; see the outbox for the error.", None if res["sent"] else "error")
        return redirect(url_for("admin.outbox"))

    @bp.route("/settings", methods=["GET", "POST"])
    @auth.require("admin")
    def settings():
        errors = {}
        if request.method == "POST":
            values = {k: request.form.get(k, "") for k, *_ in SETTING_FORM}
            errors = {k: e for k, v in values.items() if (e := validate(k, v))}
            if not errors:
                changed = {k: v.strip() for k, v in values.items() if v.strip() != db.setting(k)}
                for k, v in changed.items():
                    db.set_setting(k, v)
                db.audit(g.user["id"], "settings_updated", detail={"changed": sorted(changed)}, ip=d.ip())
                flash("Settings saved.")
                return redirect(url_for("admin.settings"))
            current = values
        else:
            current = db.settings()
        return render_template("portal/admin_settings.html", fields=SETTING_FORM, current=current, errors=errors,
                               defaults=DEFAULT_SETTINGS, mail=mailer.status()), (400 if errors else 200)

    return bp

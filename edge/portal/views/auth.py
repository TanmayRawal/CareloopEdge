"""Sign-in (with optional two-step code for staff), sign-out, patient registration and the account page."""

from __future__ import annotations

import time

from flask import Blueprint, abort, flash, g, redirect, render_template, request, session, url_for

from edge.portal import auth, totp
from edge.portal.common import safe_next
from edge.portal.db import CONSENT_VERSION

HOME = {"patient": "patient.dashboard", "doctor": "doctor.worklist", "admin": "care.followup"}
PENDING_2FA_S = 300


def blueprint(d) -> Blueprint:
    bp = Blueprint("auth", __name__)
    db = d.db

    @bp.get("/")
    def home():
        if g.user is None:               # the device home is the front door: Clinician / Patient entry points
            return redirect(url_for("device.home"))
        return redirect(url_for(HOME[g.user["role"]]))

    @bp.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("portal/login.html", expired=g.get("expired"))
        if not d.login_limiter.hit(d.ip() or "?"):
            return render_template("portal/login.html", error="Too many attempts. Wait a minute."), 429
        user, err = auth.check_login(db, request.form.get("email", ""), request.form.get("password", ""))
        if user is None:
            return render_template("portal/login.html", error=err, email=request.form.get("email", "")), 401
        nxt = safe_next(request.args.get("next", "")) or ""
        if user["totp_on"]:
            session.clear()
            session.update(pending_uid=int(user["id"]), pending_t=time.time(), pending_next=nxt)
            return redirect(url_for("auth.login_code"))
        auth.start_session(user)
        db.audit(user["id"], "login", ip=d.ip())
        return redirect(nxt or url_for("auth.home"))

    @bp.route("/login/code", methods=["GET", "POST"])
    def login_code():
        uid = session.get("pending_uid")
        if uid is None or time.time() - session.get("pending_t", 0) > PENDING_2FA_S:
            session.clear()
            return redirect(url_for("auth.login"))
        user = db.user(uid)
        if user is None or not user["active"] or not user["totp_on"]:
            session.clear()
            return redirect(url_for("auth.login"))
        if request.method == "GET":
            return render_template("portal/login_code.html")
        if not d.login_limiter.hit(d.ip() or "?"):
            return render_template("portal/login_code.html", error="Too many attempts. Wait a minute."), 429
        code = request.form.get("code", "").strip()
        ok = False
        if request.form.get("recovery"):
            left = totp.use_recovery_code(user["recovery_hashes"], code)
            if left is not None:
                with db.tx() as c:
                    c.execute("UPDATE users SET recovery_hashes = ? WHERE id = ?", (left, uid))
                db.audit(uid, "login_recovery_code", ip=d.ip())
                ok = True
        else:
            step = totp.match(d.store.unseal("totp", user["totp_secret"]), code, user["totp_last_step"])
            if step is not None:
                with db.tx() as c:   # the step is consumed: the same code cannot be used again
                    c.execute("UPDATE users SET totp_last_step = ? WHERE id = ? AND COALESCE(totp_last_step, -1) < ?",
                              (step, uid, step))
                ok = True
        if not ok:
            db.audit(uid, "login_code_failed", ip=d.ip())
            return render_template("portal/login_code.html", error="That code is not valid. Try the newest code."), 401
        nxt = session.get("pending_next") or ""
        auth.start_session(user)
        db.audit(uid, "login", detail={"two_step": True}, ip=d.ip())
        return redirect(nxt or url_for("auth.home"))

    @bp.post("/logout")
    def logout():
        if g.user is not None:
            db.audit(g.user["id"], "logout", ip=d.ip())
        session.clear()
        return redirect(url_for("auth.login"))

    @bp.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("portal/register.html")
        f = request.form
        email, name, pw = f.get("email", "").strip().lower(), f.get("name", "").strip(), f.get("password", "")
        err = None
        if not d.register_limiter.hit(d.ip() or "?"):
            err = "Too many registrations from this device. Try later."
        elif not name or "@" not in email or len(email) > 200:
            err = "Please enter your name and a valid e-mail address."
        elif pw != f.get("password2", ""):
            err = "Passwords do not match."
        elif f.get("consent") != "yes":
            err = "Please confirm the consent statement."
        else:
            err = auth.password_problem(pw, email)
        if err is None and db.user_by_email(email) is not None:
            err = "An account with this e-mail already exists."
        if err:
            return render_template("portal/register.html", error=err, form=f), 400
        uid = db.create_user(email, name, "patient", auth.hash_password(pw),
                             {"dob": f.get("dob") or None, "sex": f.get("sex") or None, "phone": f.get("phone") or None,
                              "district": (f.get("district") or "").strip()[:80] or None})
        db.audit(uid, "registered", ip=d.ip())
        auth.start_session(db.user(uid))
        return redirect(url_for("patient.dashboard"))

    # ------------------------------------------------------------------ account
    def consents(uid: int) -> list:
        return db.q("SELECT * FROM consents WHERE user_id = ? ORDER BY granted_at DESC", (uid,))

    def set_email_consent(me, on: bool) -> None:
        now = time.time()
        with db.tx() as c:
            c.execute("UPDATE users SET notify_email = ? WHERE id = ?", (int(on), me["id"]))
            if me["role"] != "patient":
                return
            active = c.execute("SELECT id FROM consents WHERE user_id = ? AND purpose = 'email_reminders' AND withdrawn_at IS NULL",
                               (me["id"],)).fetchone()
            if on and not active:
                c.execute("INSERT INTO consents (user_id,purpose,version,granted_at,recorded_by) VALUES (?,?,?,?,?)",
                          (me["id"], "email_reminders", CONSENT_VERSION, now, me["id"]))
            elif not on and active:
                c.execute("UPDATE consents SET withdrawn_at = ? WHERE id = ?", (now, active["id"]))

    def render_account(error=None, setup=None, codes=None):
        me = g.user
        req = db.one("SELECT * FROM data_requests WHERE user_id = ? AND status = 'open'", (me["id"],))
        return render_template("portal/account.html", profile=db.profile(me["id"]), error=error, consents=consents(me["id"]),
                               setup=setup, codes=codes, deletion=req,
                               recovery_left=totp.remaining_recovery(me["recovery_hashes"]) if me["totp_on"] else 0,
                               must_2fa=me["role"] != "patient" and db.setting("require_staff_2fa") == "1" and not me["totp_on"]), \
            (400 if error else 200)

    @bp.route("/account", methods=["GET", "POST"])
    @auth.require("patient", "doctor", "admin")
    def account():
        me = g.user
        if request.method == "GET":
            return render_account()
        f = request.form
        form = f.get("form")
        if form == "password":
            error = None
            if not auth.verify_password(me, f.get("current", "")):
                error = "Current password is incorrect."
            elif f.get("new", "") != f.get("new2", ""):
                error = "New passwords do not match."
            else:
                error = auth.password_problem(f.get("new", ""), me["email"] or "")
            if error:
                return render_account(error)
            with db.tx() as c:
                c.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (auth.hash_password(f["new"]), me["id"]))
            db.audit(me["id"], "password_changed", ip=d.ip())
            flash("Your password was changed.")
        elif form == "prefs":
            set_email_consent(me, bool(f.get("notify_email")))
            if me["role"] == "patient":
                with db.tx() as c:
                    c.execute("UPDATE patient_profiles SET phone = ?, district = ? WHERE user_id = ?",
                              ((f.get("phone") or "")[:30] or None, (f.get("district") or "").strip()[:80] or None, me["id"]))
            db.audit(me["id"], "preferences_updated", detail={"email_reminders": bool(f.get("notify_email"))}, ip=d.ip())
            flash("Your preferences were saved.")
        elif form == "deletion" and me["role"] == "patient":
            if not db.one("SELECT 1 FROM data_requests WHERE user_id = ? AND status = 'open'", (me["id"],)):
                with db.tx() as c:
                    c.execute("INSERT INTO data_requests (user_id, kind, note, created) VALUES (?,?,?,?)",
                              (me["id"], "deletion", (f.get("note") or "").strip()[:500] or None, time.time()))
                db.audit(me["id"], "deletion_requested", ip=d.ip())
            flash("Your request was sent to the department. They will contact you before anything is deleted.")
        elif form == "2fa_start" and me["role"] != "patient":
            secret = totp.new_secret()
            with db.tx() as c:
                c.execute("UPDATE users SET totp_secret = ?, totp_on = 0, totp_last_step = NULL WHERE id = ?",
                          (d.store.seal("totp", secret), me["id"]))
            return render_account(setup={"key": totp.pretty(secret),
                                         "uri": totp.otpauth_uri(secret, me["email"] or str(me["id"]), "CareLoop-Edge")})
        elif form == "2fa_confirm" and me["role"] != "patient" and me["totp_secret"] and not me["totp_on"]:
            secret = d.store.unseal("totp", me["totp_secret"])
            step = totp.match(secret, f.get("code", ""), None)
            if step is None:
                return render_account("That code is not valid. Enter the 6-digit code shown in the app now.",
                                      setup={"key": totp.pretty(secret), "uri": totp.otpauth_uri(
                                          secret, me["email"] or str(me["id"]), "CareLoop-Edge")})
            codes, hashes = totp.new_recovery_codes()
            with db.tx() as c:
                c.execute("UPDATE users SET totp_on = 1, totp_last_step = ?, recovery_hashes = ? WHERE id = ?",
                          (step, hashes, me["id"]))
            db.audit(me["id"], "two_step_enabled", ip=d.ip())
            g.user = db.user(me["id"])
            flash("Two-step sign-in is on.")
            return render_account(codes=codes)
        elif form == "2fa_off" and me["role"] != "patient":
            if not auth.verify_password(me, f.get("password", "")):
                return render_account("Password is incorrect.")
            with db.tx() as c:
                c.execute("UPDATE users SET totp_on = 0, totp_secret = NULL, totp_last_step = NULL, recovery_hashes = NULL "
                          "WHERE id = ?", (me["id"],))
            db.audit(me["id"], "two_step_disabled", ip=d.ip())
            flash("Two-step sign-in is off.")
        else:
            abort(400)
        return redirect(url_for("auth.account"))

    return bp

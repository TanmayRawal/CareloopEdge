"""Sign-in, sign-out, patient registration and the account page (all roles)."""

from __future__ import annotations

from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for

from edge.portal import auth
from edge.portal.common import safe_next

HOME = {"patient": "patient.dashboard", "doctor": "doctor.worklist", "admin": "admin.users"}


def blueprint(d) -> Blueprint:
    bp = Blueprint("auth", __name__)
    db = d.db

    @bp.get("/")
    def home():
        if g.user is None:
            return redirect(url_for("auth.login"))
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
        auth.start_session(user)
        db.audit(user["id"], "login", ip=d.ip())
        return redirect(safe_next(request.args.get("next", "")) or url_for("auth.home"))

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
                             {"dob": f.get("dob") or None, "sex": f.get("sex") or None, "phone": f.get("phone") or None})
        db.audit(uid, "registered", ip=d.ip())
        auth.start_session(db.user(uid))
        return redirect(url_for("patient.dashboard"))

    @bp.route("/account", methods=["GET", "POST"])
    @auth.require("patient", "doctor", "admin")
    def account():
        me = g.user
        profile = db.one("SELECT * FROM patient_profiles WHERE user_id = ?", (me["id"],))
        error = None
        if request.method == "POST":
            f = request.form
            if f.get("form") == "password":
                if not auth.verify_password(me, f.get("current", "")):
                    error = "Current password is incorrect."
                elif f.get("new", "") != f.get("new2", ""):
                    error = "New passwords do not match."
                else:
                    error = auth.password_problem(f.get("new", ""), me["email"])
                if error is None:
                    with db.tx() as c:
                        c.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (auth.hash_password(f["new"]), me["id"]))
                    db.audit(me["id"], "password_changed", ip=d.ip())
                    flash("Your password was changed.")
                    return redirect(url_for("auth.account"))
            else:
                with db.tx() as c:
                    c.execute("UPDATE users SET notify_email = ? WHERE id = ?", (1 if f.get("notify_email") else 0, me["id"]))
                    if me["role"] == "patient":
                        c.execute("UPDATE patient_profiles SET phone = ? WHERE user_id = ?", ((f.get("phone") or "")[:30] or None, me["id"]))
                db.audit(me["id"], "preferences_updated", ip=d.ip())
                flash("Your preferences were saved.")
                return redirect(url_for("auth.account"))
        return render_template("portal/account.html", profile=profile, error=error), (400 if error else 200)

    return bp

"""Pages reached from e-mail links without signing in: appointment replies, account invitations, password reset.

Every link carries a single-use token (edge.portal.tokens). Opening a link never changes anything: the page shows a
button and the change happens on POST (CSRF-protected), so mail scanners that prefetch links cannot confirm,
cancel or reset anything.
"""

from __future__ import annotations

from flask import Blueprint, abort, flash, g, redirect, render_template, request, session, url_for

from edge.portal import auth, notify, outreach, scheduling, tokens


def blueprint(d) -> Blueprint:
    bp = Blueprint("public", __name__)
    db = d.db

    def limited():
        if not d.token_limiter.hit(d.ip() or "?"):
            abort(429)

    # ------------------------------------------------------------------ appointment replies
    @bp.route("/r/<raw>", methods=["GET", "POST"])
    def reply(raw):
        limited()
        purpose = {"yes": "appt_yes", "no": "appt_no"}.get(request.args.get("a", ""))
        if purpose is None:
            abort(404)
        if request.method == "GET":
            t = tokens.peek(db, purpose, raw)
            appt = scheduling.appointment(db, t["ref_id"]) if t else None
            if appt is None or appt["patient_id"] != t["user_id"]:
                return render_template("portal/public_message.html", title="This link has expired",
                                       message="It may already have been used. Sign in to the patient portal to see your "
                                               "appointments."), 410
            return render_template("portal/public_reply.html", appt=appt, answer=request.args["a"], raw=raw)
        res = outreach.appointment_reply(db, purpose, raw)
        if res is None:
            return render_template("portal/public_message.html", title="This link has expired",
                                   message="It may already have been used. Sign in to the patient portal to see your "
                                           "appointments."), 410
        db.audit(res["appt"]["patient_id"], f"appointment_reply_{request.args['a']}", res["appt"]["case_id"],
                 {"appointment": res["appt"]["id"], "result": res["result"]}, d.ip())
        msg = {"confirmed": ("Thank you", "Your attendance is confirmed. We look forward to seeing you."),
               "cancelled": ("Appointment cancelled", "Thank you for letting us know. The time was released. Please book a new "
                                                      "time in the patient portal; our team may also call you."),
               "declined": ("Thank you for letting us know", "Our team will contact you to arrange another time."),
               "inactive": ("No action needed", "This appointment is no longer active.")}[res["result"]]
        return render_template("portal/public_message.html", title=msg[0], message=msg[1], ok=True)

    # ------------------------------------------------------------------ invitations and password reset
    def set_password_page(purpose: str, raw: str):
        limited()
        t = tokens.peek(db, purpose, raw)
        user = db.user(t["user_id"]) if t else None
        if user is None or not user["active"]:
            return render_template("portal/public_message.html", title="This link has expired",
                                   message="Ask the department for a new link, or use “Forgot password” on the sign-in page."), 410
        error = None
        if request.method == "POST":
            pw = request.form.get("password", "")
            if pw != request.form.get("password2", ""):
                error = "Passwords do not match."
            else:
                error = auth.password_problem(pw, user["email"] or "")
            if error is None:
                if tokens.consume(db, purpose, raw) is None:
                    abort(410)
                with db.tx() as c:
                    # the link arrived by e-mail, so the address is confirmed too
                    c.execute("UPDATE users SET pw_hash = ?, failed_logins = 0, locked_until = NULL, email_verified = 1 "
                              "WHERE id = ?",
                              (auth.hash_password(pw), user["id"]))
                db.audit(user["id"], "invite_accepted" if purpose == "invite" else "password_reset", ip=d.ip())
                session.clear()
                flash("Your password was set. Please sign in.")
                return redirect(url_for("auth.login"))
        return render_template("portal/public_password.html", user=user, invite=purpose == "invite", error=error), \
            (400 if error else 200)

    @bp.get("/verify/<raw>")
    def verify(raw):
        limited()
        t = tokens.consume(db, "verify", raw)
        if t is None:
            return render_template("portal/public_message.html", title="This link has expired",
                                   message="Sign in and use “Send a new link” on your Account page."), 410
        with db.tx() as c:
            c.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (t["user_id"],))
        db.audit(t["user_id"], "email_verified", ip=d.ip())
        return render_template("portal/public_message.html", title="E-mail address confirmed",
                               message="Thank you. You will now receive messages about your reports and appointments "
                                       "at this address.", ok=True)

    @bp.route("/welcome/<raw>", methods=["GET", "POST"])
    def welcome(raw):
        return set_password_page("invite", raw)

    @bp.route("/reset/<raw>", methods=["GET", "POST"])
    def reset(raw):
        return set_password_page("reset", raw)

    @bp.route("/forgot", methods=["GET", "POST"])
    def forgot():
        if g.get("user") is not None:
            return redirect(url_for("auth.home"))
        if request.method == "GET":
            return render_template("portal/public_forgot.html")
        if not d.reset_limiter.hit(d.ip() or "?"):
            return render_template("portal/public_forgot.html", error="Too many requests. Please try again later."), 429
        user = db.user_by_email(request.form.get("email", ""))
        if user is not None and user["active"] and user["pw_hash"]:
            raw = tokens.issue(db, "reset", user["id"])
            notify.notify(db, user, "password_reset", {}, None, in_app=False, force_email=True,
                          actions=[("Choose a new password", f"/reset/{raw}")])
            db.audit(user["id"], "password_reset_requested", ip=d.ip())
        # same answer whether or not the account exists
        return render_template("portal/public_message.html", title="Check your e-mail",
                               message="If an account exists for that address, we sent a link to reset the password. "
                                       "The link works once and expires in 1 hour.", ok=True)

    return bp

"""Radiologist area: worklist, case review and reporting, consultations, availability, analytics."""

from __future__ import annotations

import datetime as dt
import json
import time

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from edge.portal import auth, careplan, notify, outreach, scheduling, services
from edge.portal.common import PAGE_SIZE, fmt_iso, page_args
from edge.portal.db import DIAGNOSES, STATUSES

# Latest AI result per case in one query (instead of one query per worklist row).
LATEST_AI = "LEFT JOIN ai_results ai ON ai.id = (SELECT id FROM ai_results WHERE case_id = c.id ORDER BY created DESC, id DESC LIMIT 1)"


def blueprint(d) -> Blueprint:
    bp = Blueprint("doctor", __name__, url_prefix="/doctor")
    db, ctx = d.db, d.ctx

    def status_counts() -> dict:
        counts = dict.fromkeys(STATUSES, 0)
        counts.update({r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) AS n FROM cases GROUP BY status")})
        return counts

    @bp.get("/")
    @auth.require("doctor")
    def worklist():
        status = request.args.get("status", "open")
        search = request.args.get("q", "").strip()[:80]
        page, offset = page_args(request)
        where = {"open": "c.status IN ('awaiting_review','in_review')", "all": "1=1"}.get(
            status, "c.status = ?" if status in STATUSES else "1=0")
        args = (status,) if status in STATUSES else ()
        if search:
            where += " AND (u.name LIKE ? OR c.id LIKE ? OR printf('NE-%05d', u.id) LIKE ?)"
            args += (f"%{search}%", f"{search.lower()}%", f"%{search.upper()}%")
        order = "c.updated DESC" if status == "finalized" else "(c.status = 'finalized'), c.priority DESC, c.created"
        rows = db.q(f"SELECT c.*, u.name AS patient_name, dr.name AS doctor_name, p.dob AS patient_dob, p.sex AS patient_sex, "
                    f"ai.result_json AS ai_json FROM cases c JOIN users u ON u.id = c.patient_id "
                    f"LEFT JOIN users dr ON dr.id = c.doctor_id LEFT JOIN patient_profiles p ON p.user_id = c.patient_id "
                    f"{LATEST_AI} WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?", args + (PAGE_SIZE + 1, offset))
        more = len(rows) > PAGE_SIZE
        s = db.settings()
        limit_n, limit_u = float(s["unread_hours"]) * 3600, float(s["unread_hours_urgent"]) * 3600
        now = time.time()
        items = []
        for r in rows[:PAGE_SIZE]:
            lim = limit_u if r["priority"] >= 2 else limit_n
            overdue = r["status"] == "awaiting_review" and now - r["created"] > lim
            items.append({"case": r, "ai": json.loads(r["ai_json"]) if r["ai_json"] else None, "overdue": overdue})
        return render_template("portal/doctor_worklist.html", items=items, status=status, counts=status_counts(),
                               q=search, page=page, more=more)

    @bp.get("/case/<cid>")
    @auth.require("doctor")
    def case_review(cid):
        c = d.get_case_or_404(cid)
        services.open_for_review(ctx, c, g.user, d.ip())
        c = db.case(cid)
        ai = db.latest_ai(cid)
        db.audit(g.user["id"], "case_viewed", cid, ip=d.ip())
        patient = db.user(c["patient_id"])
        profile = db.one("SELECT * FROM patient_profiles WHERE user_id = ?", (c["patient_id"],))
        history = db.q("SELECT id, created, status FROM cases WHERE patient_id = ? AND id != ? ORDER BY created DESC LIMIT 5",
                       (c["patient_id"], cid))
        appt = scheduling.active_for_case(db, cid)
        return render_template("portal/doctor_case.html", case=c, ai=ai, r=ai["result"] if ai else None,
                               patient=patient, profile=profile, reports=db.reports(cid), diagnoses=DIAGNOSES,
                               history=history,
                               appt=scheduling.appointment(db, appt["id"]) if appt else None,
                               no_shows=scheduling.no_show_count(db, c["patient_id"]),
                               plan=careplan.active(db, c["patient_id"]))

    @bp.post("/case/<cid>/sign")
    @auth.require("doctor")
    def sign(cid):
        c = d.get_case_or_404(cid)
        f = request.form
        if not auth.verify_password(g.user, f.get("password", "")):
            flash("Signature failed: password incorrect.", "error")
            return redirect(url_for("doctor.case_review", cid=cid))
        agrees = {"yes": True, "no": False}.get(f.get("agrees"))
        try:
            v = services.sign_report(ctx, c, g.user, f.get("diagnosis", ""), agrees, f.get("findings", ""),
                                     f.get("recommendation", ""), f.get("amendment_reason") or None, d.ip())
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("doctor.case_review", cid=cid))
        msg = f"Report version {v} signed and released to the patient."
        kind = f.get("fu_kind", "")
        if kind and careplan.active(db, c["patient_id"]) is None:
            try:
                msg += " " + start_follow_up(c, kind, f)
            except careplan.PlanError as e:
                flash(f"Report signed, but the follow-up was not saved: {e}", "error")
        flash(msg)
        return redirect(url_for("doctor.worklist"))

    def start_follow_up(c, kind: str, f) -> str:
        """Approved care plan from the sign panel; a referral is drafted and signed with the same signature."""
        if kind == "referral" and not f.get("ref_facility", "").strip():
            raise careplan.PlanError("enter the facility to refer to (you can add it from the patient record).")
        zone = scheduling.tz(db)
        days = max(1, min(400, int(f.get("fu_days", "14") or 14)))
        due = (scheduling.local(time.time(), zone).date() + dt.timedelta(days=days)).isoformat()
        note = (f.get("recommendation") or "").strip()[:500]
        pid = careplan.create(db, c["patient_id"], kind, due, note, g.user, case_id=c["id"])
        db.audit(g.user["id"], "plan_created", c["id"], {"plan": pid, "kind": kind, "due": due}, d.ip())
        if kind != "referral":
            outreach.announce_plan(db, careplan.get(db, pid))
            return f"Follow-up {careplan.PLAN_TEXT[kind].lower()} due {fmt_iso(due)}; the patient was informed."
        rid = careplan.create_referral(db, c["patient_id"], f.get("ref_facility", ""), f.get("ref_specialty", ""),
                                       f.get("ref_urgency", "routine"), note or (f.get("findings") or "")[:1500], g.user,
                                       case_id=c["id"], plan_id=pid)
        careplan.sign_referral(db, d.store, careplan.referral(db, rid), g.user, db.setting("clinic_name"))
        db.audit(g.user["id"], "referral_signed", c["id"], {"referral": rid}, d.ip())
        notify.notify(db, db.user(c["patient_id"]), "referral_ready", {"facility": f.get("ref_facility", "")},
                      f"/patient/referrals/{rid}", dedupe=f"referral:{rid}")
        return "Referral letter signed and shared with the patient."

    @bp.post("/case/<cid>/reupload")
    @auth.require("doctor")
    def reupload(cid):
        c = d.get_case_or_404(cid)
        try:
            services.request_reupload(ctx, c, g.user, request.form.get("message", "").strip() or
                                      "Please upload a clearer MRI image.", d.ip())
            flash("The patient was asked for a new image.")
        except ValueError as e:
            flash(str(e), "error")
        return redirect(url_for("doctor.worklist"))

    # ------------------------------------------------------------------ consultations
    def my_appt(aid: int):
        a = scheduling.appointment(db, aid)
        if a is None or a["doctor_id"] != g.user["id"]:
            abort(404)
        return a

    @bp.get("/schedule")
    @auth.require("doctor")
    def schedule():
        zone = scheduling.tz(db)
        today = scheduling.local(time.time(), zone).date()
        try:
            day = dt.date.fromisoformat(request.args.get("day", "")) if request.args.get("day") else today
        except ValueError:
            day = today
        view = request.args.get("view", "day")
        span = 7 if view == "week" else 1
        start = dt.datetime.combine(day, dt.time(0), zone).timestamp()
        rows = db.q("SELECT a.*, p.name AS patient_name, p.email AS patient_email, pp.phone AS patient_phone, "
                    "(SELECT COUNT(*) FROM appointments x WHERE x.patient_id = a.patient_id AND x.status = 'no_show') AS no_shows "
                    "FROM appointments a JOIN users p ON p.id = a.patient_id LEFT JOIN patient_profiles pp ON pp.user_id = a.patient_id "
                    "WHERE a.doctor_id = ? AND a.start_ts >= ? AND a.start_ts < ? ORDER BY a.start_ts",
                    (g.user["id"], start, start + span * 86400 + 3600))
        unmarked = db.q("SELECT a.*, p.name AS patient_name FROM appointments a JOIN users p ON p.id = a.patient_id "
                        "WHERE a.doctor_id = ? AND a.status IN ('booked','arrived') AND a.end_ts < ? ORDER BY a.start_ts",
                        (g.user["id"], time.time()))
        by_day: dict = {}
        for r in rows:
            by_day.setdefault(scheduling.local(r["start_ts"], zone).date(), []).append(r)
        return render_template("portal/doctor_schedule.html", day=day, today=today, view=view, by_day=by_day,
                               unmarked=unmarked, now=time.time(), has_hours=bool(scheduling.hours(db, g.user["id"])),
                               prev_day=day - dt.timedelta(days=span), next_day=day + dt.timedelta(days=span),
                               span_days=[day + dt.timedelta(days=i) for i in range(span)])

    @bp.post("/appointments/<int:aid>/mark")
    @auth.require("doctor")
    def mark(aid):
        a = my_appt(aid)
        status = request.form.get("status", "")
        try:
            scheduling.mark(db, a, status, g.user)
        except scheduling.BookingError as e:
            flash(str(e), "error")
        else:
            db.audit(g.user["id"], f"appointment_{status}", a["case_id"], {"appointment": aid}, d.ip())
            if status == "no_show":
                notify.appointment_event(db, aid, "no_show")
            elif status == "completed" and careplan.complete_matching(db, a["patient_id"], ("consultation",),
                                                                       "Consultation completed"):
                db.audit(g.user["id"], "plan_completed", a["case_id"], {"kind": "consultation"}, d.ip())
            flash({"arrived": "Checked in.", "completed": "Consultation completed.",
                   "no_show": "Marked as did not attend. The patient was invited to rebook."}[status])
        return redirect(request.referrer if (request.referrer or "").find("/doctor/") >= 0 else url_for("doctor.schedule"))

    @bp.post("/appointments/<int:aid>/cancel")
    @auth.require("doctor")
    def cancel(aid):
        a = my_appt(aid)
        try:
            scheduling.cancel(db, a, g.user)
        except scheduling.BookingError as e:
            flash(str(e), "error")
        else:
            db.audit(g.user["id"], "appointment_cancelled", a["case_id"], {"appointment": aid, "by": "doctor"}, d.ip())
            notify.appointment_event(db, aid, "cancelled")
            flash("Appointment cancelled; the patient was notified.")
        return redirect(url_for("doctor.schedule"))

    @bp.route("/availability", methods=["GET", "POST"])
    @auth.require("doctor")
    def availability():
        error = None
        if request.method == "POST":
            f = request.form
            try:
                if f.get("form") == "hours":
                    rows = []
                    for wd in range(7):
                        if not f.get(f"on_{wd}"):
                            continue
                        for part in ("a", "b"):
                            st, en = f.get(f"start_{wd}_{part}", ""), f.get(f"end_{wd}_{part}", "")
                            if not st or not en:
                                continue
                            sm = int(st[:2]) * 60 + int(st[3:5])
                            em = int(en[:2]) * 60 + int(en[3:5])
                            rows.append({"weekday": wd, "start_min": sm, "end_min": em, "slot_min": int(f.get("slot_min", 20)),
                                         "room": f.get("room", "")})
                    scheduling.set_hours(db, g.user["id"], rows)
                    db.audit(g.user["id"], "hours_updated", detail={"blocks": len(rows)}, ip=d.ip())
                    flash("Working hours saved.")
                elif f.get("form") == "off_add":
                    day = dt.date.fromisoformat(f.get("day", ""))
                    with db.tx() as c:
                        c.execute("INSERT OR REPLACE INTO doctor_days_off (doctor_id, day, note) VALUES (?, ?, ?)",
                                  (g.user["id"], day.isoformat(), (f.get("note") or "")[:100] or None))
                    flash(f"{day:%d %b %Y} marked as unavailable.")
                elif f.get("form") == "off_del":
                    with db.tx() as c:
                        c.execute("DELETE FROM doctor_days_off WHERE doctor_id = ? AND day = ?", (g.user["id"], f.get("day", "")))
                return redirect(url_for("doctor.availability"))
            except (ValueError, scheduling.BookingError) as e:
                error = str(e) if isinstance(e, scheduling.BookingError) else "Please check the times and dates."
        hours = scheduling.hours(db, g.user["id"])
        by_wd: dict[int, list] = {}
        for h in hours:
            by_wd.setdefault(h["weekday"], []).append(h)
        today = scheduling.local(time.time(), scheduling.tz(db)).date()
        return render_template("portal/doctor_availability.html", by_wd=by_wd, weekdays=scheduling.WEEKDAYS,
                               slot_min=hours[0]["slot_min"] if hours else 20, room=(hours[0]["room"] if hours else "") or "",
                               days_off=scheduling.days_off(db, g.user["id"], today), error=error,
                               upcoming=len(scheduling.free_slots(db, g.user["id"], days=14))), (400 if error else 200)

    # ------------------------------------------------------------------ analytics
    @bp.get("/stats")
    @auth.require("doctor", "admin")
    def stats():
        from edge.portal import analytics
        period = request.args.get("period", "90")
        if period not in analytics.PERIODS:
            period = "90"
        a = analytics.compute(db, analytics.PERIODS[period])
        return render_template("portal/doctor_stats.html", a=a, period=period, periods=analytics.PERIODS,
                               classes=analytics.CLASSES)

    return bp

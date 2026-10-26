"""Patient area: studies, upload, signed reports, consultation booking."""

from __future__ import annotations

import datetime as dt
import time

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from edge.portal import auth, notify, scheduling, services
from edge.portal.db import APPT_ACTIVE


def blueprint(d) -> Blueprint:
    bp = Blueprint("patient", __name__, url_prefix="/patient")
    db, ctx = d.db, d.ctx

    def my_appt(aid: int):
        a = scheduling.appointment(db, aid)
        if a is None or a["patient_id"] != g.user["id"]:
            abort(404)
        return a

    def upcoming():
        return db.q(f"SELECT a.*, d.name AS doctor_name FROM appointments a JOIN users d ON d.id = a.doctor_id "
                    f"WHERE a.patient_id = ? AND a.status IN {APPT_ACTIVE} AND a.end_ts > ? ORDER BY a.start_ts",
                    (g.user["id"], time.time()))

    @bp.get("/")
    @auth.require("patient")
    def dashboard():
        cases = db.q("SELECT c.*, (SELECT id FROM appointments a WHERE a.case_id = c.id AND a.status IN ('booked','arrived') "
                     "LIMIT 1) AS appt_id FROM cases c WHERE patient_id = ? ORDER BY created DESC", (g.user["id"],))
        plan = db.one("SELECT * FROM care_plans WHERE patient_id = ? AND status = 'approved' ORDER BY due_date LIMIT 1",
                      (g.user["id"],))
        ref = db.one("SELECT * FROM referrals WHERE patient_id = ? AND status IN ('signed','sent','appointment_made') "
                     "ORDER BY signed_at DESC LIMIT 1", (g.user["id"],))
        up = upcoming()
        # care journey: 0 scan sent, 1 report ready, 2 consultation, 3 follow-up
        if not cases:
            stage = -1
        elif not any(c["status"] == "finalized" for c in cases):
            stage = 0
        elif up or db.one("SELECT 1 FROM appointments WHERE patient_id = ? AND status IN ('arrived','completed')", (g.user["id"],)):
            stage = 3 if db.one("SELECT 1 FROM appointments WHERE patient_id = ? AND status = 'completed'", (g.user["id"],)) else 2
        else:
            stage = 1
        return render_template("portal/patient_dashboard.html", cases=cases, upcoming=up, plan=plan, ref=ref, stage=stage)

    # ------------------------------------------------------------------ documents, referrals, data export
    @bp.route("/documents", methods=["GET", "POST"])
    @auth.require("patient")
    def records():
        if request.method == "POST":
            from edge.portal.views.care import store_document
            f = request.files.get("file")
            try:
                did = store_document(db, d.store, g.user["id"], g.user, request.form.get("kind", "other"),
                                     request.form.get("title", ""), f.filename if f else "", f.read() if f else b"")
            except ValueError as e:
                flash(str(e), "error")
            else:
                db.audit(g.user["id"], "document_added", detail={"patient": g.user["id"], "document": did}, ip=d.ip())
                flash("Document added. Your care team can now see it.")
            return redirect(url_for("patient.records"))
        from edge.portal.views.care import DOC_TEXT
        docs = db.q("SELECT * FROM documents WHERE patient_id = ? ORDER BY created DESC", (g.user["id"],))
        refs = db.q("SELECT r.*, s.name AS signed_by_name FROM referrals r LEFT JOIN users s ON s.id = r.signed_by "
                    "WHERE r.patient_id = ? AND r.status != 'draft' ORDER BY r.signed_at DESC", (g.user["id"],))
        return render_template("portal/patient_records.html", docs=docs, refs=refs, doc_text=DOC_TEXT,
                               doc_kinds=("prior_report", "biopsy", "referral_letter", "other"))

    @bp.get("/documents/<int:did>")
    @auth.require("patient")
    def document(did):
        from edge.portal.views.care import send_document
        doc = db.one("SELECT * FROM documents WHERE id = ? AND patient_id = ?", (did, g.user["id"]))
        if doc is None:
            abort(404)
        return send_document(d, doc)

    @bp.get("/referrals/<int:rid>")
    @auth.require("patient")
    def referral(rid):
        from edge.portal import careplan
        from edge.portal.views.care import send_letter
        ref = careplan.referral(db, rid)
        if ref is None or ref["patient_id"] != g.user["id"]:
            abort(404)
        db.audit(g.user["id"], "referral_viewed", detail={"referral": rid}, ip=d.ip())
        return send_letter(d, ref)

    @bp.get("/export")
    @auth.require("patient")
    def export():
        if not d.export_limiter.hit(f"u{g.user['id']}"):
            abort(429)
        from edge.portal import dataexport
        data = dataexport.build(db, d.store, g.user["id"])
        db.audit(g.user["id"], "data_exported", ip=d.ip())
        resp = d.app.response_class(data, mimetype="application/zip")
        resp.headers["Content-Disposition"] = f'attachment; filename="my-health-records-{dt.date.today().isoformat()}.zip"'
        return resp

    @bp.route("/upload", methods=["GET", "POST"])
    @auth.require("patient")
    def upload():
        if not auth.patient_intake_here(db):              # scans come in at the clinic's device, not from home
            return render_template("portal/patient_upload.html"), (403 if request.method == "POST" else 200)
        if request.method == "GET":
            return render_template("portal/patient_upload.html")
        try:
            name, data = services.combine_files([(f.filename, f.read()) for f in request.files.getlist("scan")])
            cid = services.submit_scan(ctx, g.user, data, name,
                                       request.form.get("note", ""), request.form.get("consent") == "yes", d.ip())
        except services.UploadError as e:
            return render_template("portal/patient_upload.html", error=str(e)), 400
        flash("Your scan was received. A radiologist will review it; your report will appear here.")
        return redirect(url_for("patient.case", cid=cid))

    @bp.route("/case/<cid>", methods=["GET", "POST"])
    @auth.require("patient")
    def case(cid):
        c = d.get_case_or_404(cid)
        if request.method == "POST":                        # re-upload requested by the doctor
            if not auth.patient_intake_here(db):
                abort(403)
            try:
                name, data = services.combine_files([(f.filename, f.read()) for f in request.files.getlist("scan")])
                services.resubmit_scan(ctx, g.user, c, data, name, d.ip())
                flash("Thank you — the new image was sent to your radiologist.")
            except services.UploadError as e:
                flash(str(e), "error")
            return redirect(url_for("patient.case", cid=cid))
        attempt = (db.latest_ai(cid) or {}).get("result", {}).get("attempt", 1)
        reports = db.reports(cid)
        profile = db.one("SELECT * FROM patient_profiles WHERE user_id = ?", (c["patient_id"],))
        appt = scheduling.active_for_case(db, cid)
        doc = None
        if reports:
            r = reports[0]
            ai = db.one("SELECT model_version FROM ai_results WHERE id = ?", (r["ai_result_id"],)) if r["ai_result_id"] else None
            doc = services.report_doc(db, c, r, db.user(r["doctor_id"]), r["version"], r["signed_at"],
                                      ai["model_version"] if ai else None)
        return render_template("portal/patient_case.html", case=c, reports=reports, profile=profile, doc=doc,
                               report=reports[0] if reports else None, attempt=attempt,
                               appt=scheduling.appointment(db, appt["id"]) if appt else None)

    # ------------------------------------------------------------------ appointments
    def slot_page(c, reschedule_of=None, error=None):
        doctor_id = scheduling.reporting_doctor(db, c)
        doctor = db.user(doctor_id) if doctor_id else None
        slots = scheduling.free_slots(db, doctor_id) if doctor_id else []
        days: dict[dt.date, list] = {}
        for s in slots:
            days.setdefault(s["day"], []).append(s)
        day_list = sorted(days)
        try:
            sel = dt.date.fromisoformat(request.args.get("day", "")) if request.args.get("day") else None
        except ValueError:
            sel = None
        if sel not in days:
            sel = day_list[0] if day_list else None
        return render_template("portal/patient_book.html", case=c, doctor=doctor, days=day_list, sel=sel,
                               slots=days.get(sel, []), reschedule_of=reschedule_of, error=error), (400 if error else 200)

    @bp.route("/book/<cid>", methods=["GET", "POST"])
    @auth.require("patient")
    def book(cid):
        c = d.get_case_or_404(cid)
        existing = scheduling.active_for_case(db, cid)
        if existing and request.method == "GET":
            return redirect(url_for("patient.reschedule", aid=existing["id"]))
        if request.method == "POST":
            if not d.booking_limiter.hit(f"u{g.user['id']}"):
                return slot_page(c, error="Too many booking attempts. Please try again later.")
            try:
                aid = scheduling.book(db, g.user, c, float(request.form.get("start_ts", "0")), request.form.get("reason", ""))
            except (scheduling.BookingError, ValueError) as e:
                return slot_page(c, error=str(e))
            db.audit(g.user["id"], "appointment_booked", cid, {"appointment": aid}, d.ip())
            notify.appointment_event(db, aid, "booked")
            return redirect(url_for("patient.appointment_done", aid=aid))
        if c["status"] != "finalized":
            flash("A consultation can be booked once your report is ready.", "error")
            return redirect(url_for("patient.case", cid=cid))
        return slot_page(c)

    @bp.route("/appointments/<int:aid>/reschedule", methods=["GET", "POST"])
    @auth.require("patient")
    def reschedule(aid):
        a = my_appt(aid)
        c = d.get_case_or_404(a["case_id"])
        if request.method == "POST":
            if not d.booking_limiter.hit(f"u{g.user['id']}"):
                return slot_page(c, a, error="Too many booking attempts. Please try again later.")
            try:
                new = scheduling.reschedule(db, a, g.user, float(request.form.get("start_ts", "0")))
            except (scheduling.BookingError, ValueError) as e:
                return slot_page(c, a, error=str(e))
            db.audit(g.user["id"], "appointment_rescheduled", a["case_id"], {"from": aid, "to": new}, d.ip())
            notify.appointment_event(db, new, "rescheduled")
            return redirect(url_for("patient.appointment_done", aid=new))
        return slot_page(c, a)

    @bp.get("/appointments/<int:aid>/confirmed")
    @auth.require("patient")
    def appointment_done(aid):
        return render_template("portal/patient_booked.html", a=my_appt(aid))

    @bp.post("/appointments/<int:aid>/cancel")
    @auth.require("patient")
    def cancel(aid):
        a = my_appt(aid)
        try:
            scheduling.cancel(db, a, g.user)
        except scheduling.BookingError as e:
            flash(str(e), "error")
            return redirect(url_for("patient.appointments"))
        db.audit(g.user["id"], "appointment_cancelled", a["case_id"], {"appointment": aid}, d.ip())
        notify.appointment_event(db, aid, "cancelled")
        flash("Your appointment was cancelled.")
        return redirect(url_for("patient.appointments"))

    @bp.get("/appointments/<int:aid>.ics")
    @auth.require("patient")
    def ics(aid):
        a = my_appt(aid)
        resp = d.app.response_class(notify.ics(db, a, notify.appt_uid(db, a)), mimetype="text/calendar")
        resp.headers["Content-Disposition"] = f'attachment; filename="appointment-{aid}.ics"'
        return resp

    @bp.get("/appointments")
    @auth.require("patient")
    def appointments():
        rows = db.q("SELECT a.*, d.name AS doctor_name FROM appointments a JOIN users d ON d.id = a.doctor_id "
                    "WHERE a.patient_id = ? ORDER BY a.start_ts DESC", (g.user["id"],))
        now = time.time()
        up = [r for r in rows if r["status"] in APPT_ACTIVE and r["end_ts"] > now][::-1]
        past = [r for r in rows if r not in up]
        cutoff = float(db.setting("cancel_cutoff_hours")) * 3600
        bookable = db.q("SELECT c.id, c.created FROM cases c WHERE c.patient_id = ? AND c.status = 'finalized' AND NOT EXISTS "
                        "(SELECT 1 FROM appointments a WHERE a.case_id = c.id AND a.status IN ('booked','arrived')) "
                        "ORDER BY c.created DESC", (g.user["id"],))
        return render_template("portal/patient_appointments.html", upcoming=up, past=past, now=now, cutoff=cutoff,
                               bookable=bookable)

    return bp

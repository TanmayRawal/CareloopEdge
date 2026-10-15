"""Care coordination (CareLoop): follow-up worklist, patient record with a unified timeline, walk-in registration,
plan approvals, referrals and prior-record documents.

Who does what
  admin   acts as the care coordinator: works the follow-up worklist, logs contacts, sends reminders (approved
          plans only), proposes plans, registers walk-in patients, books consultations for patients, tracks
          referrals, uploads prior records. Never sees AI output.
  doctor  approves or rejects proposed plans, creates plans and signs referrals, sees the patient record.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import time

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from edge.portal import auth, careplan, notify, outreach, risk, scheduling, services, tokens
from edge.portal.common import PAGE_SIZE, fmt_iso, page_args
from edge.portal.db import CONTACT_CHANNELS, CONTACT_OUTCOMES, DOC_KINDS, PLAN_KINDS

DOC_EXT = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
           ".dcm": "application/dicom", ".dicom": "application/dicom"}
DOC_TEXT = {"prior_report": "Prior report", "biopsy": "Biopsy / pathology", "referral_letter": "Referral letter",
            "dicom": "DICOM image", "other": "Other document"}
CONTACT_STATUS_TEXT = {"ok": "Reachable", "unreachable": "Unreachable", "wrong_number": "Wrong number"}
QUEUES = (("all", "All"), ("overdue", "Overdue"), ("soon", "Due soon"), ("approval", "Awaiting approval"),
          ("referral", "Referral open"), ("missed", "Not reached"), ("noshow", "Missed visits"))
MAX_DOC = 20 * 1024 * 1024


def doc_blob(doc_id: int) -> tuple[str, str]:
    return "documents", f"doc_{doc_id}"


def store_document(db, store, patient_id: int, by_user, kind: str, title: str, filename: str, data: bytes) -> int:
    """Validate and store an encrypted prior-record document (never analysed by the AI)."""
    name = (filename or "").strip()
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if ext not in DOC_EXT:
        raise ValueError("Use a PDF, JPG, PNG or DICOM file.")
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > MAX_DOC:
        raise ValueError("File too large (max 20 MB).")
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        raise ValueError("This does not look like a PDF file.")
    kind = kind if kind in DOC_KINDS else "other"
    with db.tx() as c:
        did = c.execute("INSERT INTO documents (patient_id,uploaded_by,kind,title,filename,mime,size,sha256,created) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (patient_id, by_user["id"], kind, (title or "").strip()[:120] or DOC_TEXT[kind], name[:150],
                         DOC_EXT[ext], len(data), hashlib.sha256(data).hexdigest(), time.time())).lastrowid
    try:
        store.put(*doc_blob(did), data)
    except Exception:
        with db.tx() as c:
            c.execute("DELETE FROM documents WHERE id = ?", (did,))
        raise
    return int(did)


def send_document(d, doc):
    resp = d.app.response_class(d.store.get(*doc_blob(doc["id"])), mimetype=doc["mime"])
    safe = "".join(ch for ch in doc["filename"] if ch.isalnum() or ch in "._- ")[:100] or f"document-{doc['id']}"
    resp.headers["Content-Disposition"] = f'attachment; filename="{safe}"'
    resp.headers["Content-Security-Policy"] = "default-src 'none'; sandbox"
    return resp


def send_letter(d, ref):
    if ref is None or ref["status"] == "draft" or not d.store.exists(*careplan.letter_kind(ref["id"])):
        abort(404)
    resp = d.app.response_class(d.store.get(*careplan.letter_kind(ref["id"])), mimetype="text/html")
    resp.headers["Content-Security-Policy"] = ("default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; "
                                               "base-uri 'none'; form-action 'none'")
    return resp


def blueprint(d) -> Blueprint:
    bp = Blueprint("care", __name__, url_prefix="/care")
    db = d.db

    def patient_or_404(uid: int):
        p = db.user(uid)
        if p is None or p["role"] != "patient":
            abort(404)
        return p

    def plan_or_404(pid: int):
        p = careplan.get(db, pid)
        if p is None:
            abort(404)
        return p

    def today():
        return scheduling.local(time.time(), scheduling.tz(db)).date()

    def back(uid: int | None = None):
        nxt = request.form.get("next", "")
        if nxt in ("worklist", "approvals", "referrals"):
            return redirect(url_for({"worklist": "care.followup", "approvals": "care.approvals",
                                     "referrals": "care.referrals"}[nxt], **({"q": request.form.get("q")} if request.form.get("q") else {})))
        return redirect(url_for("care.patient", uid=uid) if uid else url_for("care.followup"))

    # ------------------------------------------------------------------ follow-up worklist
    @bp.get("/")
    @auth.require("admin")
    def followup():
        queue = request.args.get("q", "all")
        scored = risk.scores(db)
        ids = list(scored)
        rows = []
        if ids:
            ph = ",".join("?" * len(ids))
            people = {r["id"]: r for r in db.q(
                f"SELECT u.id, u.name, u.email, p.phone, p.district, p.contact_status FROM users u "
                f"LEFT JOIN patient_profiles p ON p.user_id = u.id WHERE u.id IN ({ph})", ids)}
            plans = {}
            for p in db.q(f"SELECT * FROM care_plans WHERE status IN ('proposed','approved') AND patient_id IN ({ph}) "
                          f"ORDER BY due_date DESC", ids):
                plans[p["patient_id"]] = p                       # earliest due wins (rows are newest-due first)
            refs = {r["patient_id"]: r for r in db.q(
                f"SELECT * FROM referrals WHERE status IN ('signed','sent','appointment_made') AND patient_id IN ({ph})", ids)}
            last = {r["patient_id"]: r for r in db.q(
                f"SELECT c.* FROM contacts c WHERE c.id IN (SELECT MAX(id) FROM contacts WHERE patient_id IN ({ph}) "
                f"GROUP BY patient_id)", ids)}
            t = today()
            for pid, s in scored.items():
                plan = plans.get(pid)
                rows.append({"p": people[pid], "risk": s, "sig": s["signals"], "plan": plan, "ref": refs.get(pid),
                             "last": last.get(pid), "due": careplan.due_info(plan, t) if plan else None})
        match = {"all": lambda r: True,
                 "overdue": lambda r: r["sig"]["overdue_days"] > 0,
                 "soon": lambda r: r["sig"]["due_soon"],
                 "approval": lambda r: r["plan"] is not None and r["plan"]["status"] == "proposed",
                 "referral": lambda r: r["ref"] is not None,
                 "missed": lambda r: r["sig"]["failed_contacts"] > 0 or r["sig"]["contact_status"] != "ok",
                 "noshow": lambda r: r["sig"]["no_shows"] > 0}
        counts = {k: sum(1 for r in rows if match[k](r)) for k, _ in QUEUES}
        if queue not in match:
            queue = "all"
        shown = sorted((r for r in rows if match[queue](r)), key=lambda r: (-r["risk"]["score"], r["p"]["name"]))
        return render_template("portal/care_followup.html", rows=shown, counts=counts, queues=QUEUES, queue=queue,
                               channels=CONTACT_CHANNELS, outcomes=CONTACT_OUTCOMES, today=today())

    @bp.post("/patients/<int:uid>/contact")
    @auth.require("admin")
    def contact(uid):
        patient_or_404(uid)
        f = request.form
        plan = careplan.active(db, uid)
        try:
            nxt = f.get("next_action") or None
            if nxt:
                dt.date.fromisoformat(nxt)
            outreach.log_contact(db, uid, g.user, f.get("channel", ""), f.get("outcome", ""), f.get("note", ""),
                                 plan["id"] if plan else None, nxt)
        except (outreach.OutreachError, ValueError) as e:
            flash(str(e) if isinstance(e, outreach.OutreachError) else "Please check the follow-up date.", "error")
            return back(uid)
        db.audit(g.user["id"], "contact_logged", detail={"patient": uid, "outcome": f.get("outcome")}, ip=d.ip())
        flash("Contact recorded.")
        return back(uid)

    @bp.post("/plans/<int:pid>/remind")
    @auth.require("admin")
    def remind(pid):
        plan = plan_or_404(pid)
        try:
            emailed = outreach.send_reminder(db, plan, g.user, request.form.get("message", ""))
        except outreach.OutreachError as e:
            flash(str(e), "error")
            return back(plan["patient_id"])
        db.audit(g.user["id"], "followup_reminder_sent", detail={"patient": plan["patient_id"], "plan": pid,
                                                                  "email": emailed}, ip=d.ip())
        flash("Reminder sent by e-mail and in the portal." if emailed else
              "Reminder posted in the patient's portal (no e-mail on file or e-mail reminders are off).")
        return back(plan["patient_id"])

    @bp.post("/plans/<int:pid>/close")
    @auth.require("admin", "doctor")
    def close_plan(pid):
        plan = plan_or_404(pid)
        status = request.form.get("status", "")
        try:
            careplan.close(db, plan, status, request.form.get("reason", ""), g.user)
        except careplan.PlanError as e:
            flash(str(e), "error")
            return back(plan["patient_id"])
        db.audit(g.user["id"], f"plan_{status}", detail={"patient": plan["patient_id"], "plan": pid}, ip=d.ip())
        flash({"completed": "Follow-up marked as completed.", "lost": "Marked as lost to follow-up.",
               "cancelled": "Follow-up plan closed."}.get(status, "Saved."))
        return back(plan["patient_id"])

    @bp.post("/patients/<int:uid>/plan")
    @auth.require("admin", "doctor")
    def new_plan(uid):
        patient_or_404(uid)
        f = request.form
        if careplan.active(db, uid):
            flash("This patient already has an open follow-up plan. Close it first.", "error")
            return back(uid)
        case = db.one("SELECT id FROM cases WHERE patient_id = ? AND status = 'finalized' ORDER BY created DESC LIMIT 1", (uid,))
        try:
            pid = careplan.create(db, uid, f.get("kind", ""), f.get("due", ""), f.get("note", ""), g.user,
                                  case_id=case["id"] if case else None)
        except careplan.PlanError as e:
            flash(str(e), "error")
            return back(uid)
        plan = careplan.get(db, pid)
        db.audit(g.user["id"], "plan_created" if g.user["role"] == "doctor" else "plan_proposed",
                 detail={"patient": uid, "plan": pid, "kind": plan["kind"]}, ip=d.ip())
        if plan["status"] == "proposed":
            ctx = {"patient": plan["patient_name"], "step": careplan.PLAN_TEXT[plan["kind"]].lower(), "due": plan["due_date"]}
            for doc in db.q("SELECT * FROM users WHERE role = 'doctor' AND active = 1"):
                notify.notify(db, doc, "doc_plan_proposed", ctx, "/care/approvals", email=False)
            flash("Plan proposed. Outreach can start once a doctor approves it.")
        else:
            outreach.announce_plan(db, plan)
            flash("Follow-up plan created; the patient was informed.")
        return back(uid)

    # ------------------------------------------------------------------ approvals (doctor)
    @bp.get("/approvals")
    @auth.require("doctor")
    def approvals():
        rows = db.q("SELECT cp.*, u.name AS patient_name, pb.name AS proposed_by_name FROM care_plans cp "
                    "JOIN users u ON u.id = cp.patient_id LEFT JOIN users pb ON pb.id = cp.proposed_by "
                    "WHERE cp.status = 'proposed' ORDER BY cp.proposed_at")
        return render_template("portal/care_approvals.html", rows=rows)

    @bp.post("/plans/<int:pid>/approve")
    @auth.require("doctor")
    def approve(pid):
        plan = plan_or_404(pid)
        try:
            if request.form.get("decision") == "reject":
                careplan.close(db, plan, "cancelled", request.form.get("note") or "Not approved by the doctor", g.user)
                db.audit(g.user["id"], "plan_rejected", detail={"patient": plan["patient_id"], "plan": pid}, ip=d.ip())
                flash("Proposal declined.")
            else:
                careplan.approve(db, plan, g.user, request.form.get("due") or None, request.form.get("note"))
                db.audit(g.user["id"], "plan_approved", detail={"patient": plan["patient_id"], "plan": pid}, ip=d.ip())
                outreach.announce_plan(db, careplan.get(db, pid))
                flash(f"Follow-up approved for {plan['patient_name']}; the patient was informed.")
        except careplan.PlanError as e:
            flash(str(e), "error")
        return back(plan["patient_id"])

    # ------------------------------------------------------------------ patients
    @bp.get("/patients")
    @auth.require("admin", "doctor")
    def patients():
        search = request.args.get("q", "").strip()[:80]
        page, offset = page_args(request)
        where, args = "u.role = 'patient'", []
        if search:
            where += " AND (u.name LIKE ? OR printf('NE-%05d', u.id) LIKE ? OR p.phone LIKE ? OR u.email LIKE ?)"
            args += [f"%{search}%", f"%{search.upper()}%", f"%{search}%", f"%{search}%"]
        rows = db.q(f"SELECT u.*, p.phone, p.district, p.dob, p.sex, p.contact_status, "
                    f"(SELECT COUNT(*) FROM cases c WHERE c.patient_id = u.id) AS n_cases, "
                    f"(SELECT kind || '|' || status || '|' || due_date FROM care_plans cp WHERE cp.patient_id = u.id "
                    f" AND cp.status IN ('proposed','approved') ORDER BY due_date LIMIT 1) AS plan "
                    f"FROM users u LEFT JOIN patient_profiles p ON p.user_id = u.id WHERE {where} "
                    f"ORDER BY u.created DESC LIMIT ? OFFSET ?", tuple(args) + (PAGE_SIZE + 1, offset))
        ids = [r["id"] for r in rows[:PAGE_SIZE]]
        return render_template("portal/care_patients.html", rows=rows[:PAGE_SIZE], more=len(rows) > PAGE_SIZE, page=page,
                               q=search, risk=risk.scores(db) if ids else {})

    @bp.route("/patients/new", methods=["GET", "POST"])
    @auth.require("admin")
    def register():
        if request.method == "GET":
            return render_template("portal/care_register.html", form={})
        f = request.form
        name, email = f.get("name", "").strip(), f.get("email", "").strip().lower()
        err = None
        if not name or not (f.get("phone", "").strip() or email):
            err = "Enter the patient's name and a phone number or e-mail address."
        elif email and ("@" not in email or len(email) > 200):
            err = "Please check the e-mail address."
        elif email and db.user_by_email(email):
            err = "A patient with this e-mail address is already registered."
        elif f.get("consent") != "yes":
            err = "Record the patient's consent to continue."
        else:
            try:
                float(f.get("distance_km") or 0)
            except ValueError:
                err = "Distance must be a number of kilometres."
        if err:
            return render_template("portal/care_register.html", form=f, error=err), 400
        profile = {k: (f.get(k) or "").strip()[:200] or None for k in ("dob", "sex", "phone", "address", "district", "alt_phone")}
        profile["distance_km"] = f.get("distance_km") or None
        uid = db.create_user(email or None, name, "patient", None, profile, created_by=g.user["id"],
                             email_consent=bool(email) and f.get("email_consent") == "yes")
        db.audit(g.user["id"], "patient_registered", detail={"patient": uid, "walk_in": True}, ip=d.ip())
        msg = f"{name} registered."
        scan = request.files.get("scan")
        if scan and scan.filename:
            try:
                cid = services.submit_scan(d.ctx, db.user(uid), scan.read(), scan.filename, f.get("note", ""), True, d.ip())
                msg += " The scan was sent to the radiologists' worklist."
                db.audit(g.user["id"], "scan_uploaded_for_patient", cid, {"patient": uid}, d.ip())
            except services.UploadError as e:
                flash(f"Patient registered, but the scan was not accepted: {e}", "error")
        if email and f.get("invite") == "yes":
            send_invite(db.user(uid))
            msg += " An invitation e-mail was sent."
        flash(msg)
        return redirect(url_for("care.patient", uid=uid))

    def send_invite(p) -> None:
        raw = tokens.issue(db, "invite", p["id"])
        notify.notify(db, p, "invite", {"clinic": db.setting("clinic_name")}, None, in_app=False, force_email=True,
                      actions=[("Set your password", f"/welcome/{raw}")])
        db.audit(g.user["id"], "invite_sent", detail={"patient": p["id"]}, ip=d.ip())

    @bp.post("/patients/<int:uid>/invite")
    @auth.require("admin")
    def invite(uid):
        p = patient_or_404(uid)
        if not p["email"]:
            flash("Add an e-mail address first.", "error")
        else:
            send_invite(p)
            flash(f"Invitation sent to {p['email']}.")
        return back(uid)

    @bp.post("/patients/<int:uid>/profile")
    @auth.require("admin")
    def profile(uid):
        p = patient_or_404(uid)
        f = request.form
        email = f.get("email", "").strip().lower() or None
        if email and ("@" not in email or (db.user_by_email(email) and db.user_by_email(email)["id"] != uid)):
            flash("That e-mail address is not valid or belongs to another account.", "error")
            return back(uid)
        try:
            dist = float(f["distance_km"]) if f.get("distance_km") else None
        except ValueError:
            flash("Distance must be a number of kilometres.", "error")
            return back(uid)
        cs = f.get("contact_status") if f.get("contact_status") in CONTACT_STATUS_TEXT else "ok"
        with db.tx() as c:
            c.execute("UPDATE users SET email = ? WHERE id = ?", (email if p["pw_hash"] is None or email else p["email"], uid))
            c.execute("UPDATE patient_profiles SET phone=?, alt_phone=?, address=?, district=?, distance_km=?, contact_status=? "
                      "WHERE user_id=?", (*((f.get(k) or "").strip()[:200] or None for k in ("phone", "alt_phone", "address", "district")),
                                          dist, cs, uid))
        db.audit(g.user["id"], "patient_updated", detail={"patient": uid}, ip=d.ip())
        flash("Contact details saved.")
        return back(uid)

    @bp.get("/patients/<int:uid>")
    @auth.require("admin", "doctor")
    def patient(uid):
        p = patient_or_404(uid)
        prof = db.profile(uid)
        db.audit(g.user["id"], "patient_record_viewed", detail={"patient": uid}, ip=d.ip())
        cases = db.q("SELECT * FROM cases WHERE patient_id = ? ORDER BY created DESC", (uid,))
        plans = db.q("SELECT cp.*, pb.name AS proposed_by_name, ab.name AS approved_by_name FROM care_plans cp "
                     "LEFT JOIN users pb ON pb.id = cp.proposed_by LEFT JOIN users ab ON ab.id = cp.approved_by "
                     "WHERE cp.patient_id = ? ORDER BY cp.proposed_at DESC", (uid,))
        refs = db.q("SELECT r.*, s.name AS signed_by_name FROM referrals r LEFT JOIN users s ON s.id = r.signed_by "
                    "WHERE r.patient_id = ? ORDER BY r.created DESC", (uid,))
        appts = db.q("SELECT a.*, dr.name AS doctor_name FROM appointments a JOIN users dr ON dr.id = a.doctor_id "
                     "WHERE a.patient_id = ? ORDER BY a.start_ts DESC", (uid,))
        reports = db.q("SELECT r.case_id, r.version, r.signed_at, u.name AS doctor_name FROM reports r "
                       "JOIN users u ON u.id = r.doctor_id JOIN cases c ON c.id = r.case_id WHERE c.patient_id = ?", (uid,))
        docs = db.q("SELECT dc.*, u.name AS by_name FROM documents dc LEFT JOIN users u ON u.id = dc.uploaded_by "
                    "WHERE dc.patient_id = ? ORDER BY dc.created DESC", (uid,))
        cons = outreach.contacts(db, uid)
        ev = [(p["created"], "account", "Registered", "Walk-in registration" if prof and prof["created_by"] else "Self-registered online", None)]
        for c in cases:
            ev.append((c["created"], "clinical", "MRI scan received", "", c["id"]))
        for r in reports:
            ev.append((r["signed_at"], "clinical", "Report signed" if r["version"] == 1 else f"Report amended (v{r['version']})",
                       r["doctor_name"], r["case_id"]))
        for pl in plans:
            what = f"{careplan.PLAN_TEXT[pl['kind']]} due {fmt_iso(pl['due_date'])}"
            ev.append((pl["proposed_at"], "plan", "Follow-up planned" if pl["approved_at"] == pl["proposed_at"] else "Follow-up proposed",
                       f"{what} · {pl['proposed_by_name'] or ''}", None))
            if pl["approved_at"] and pl["approved_at"] != pl["proposed_at"]:
                ev.append((pl["approved_at"], "plan", "Follow-up approved", f"{what} · {pl['approved_by_name'] or ''}", None))
            if pl["closed_at"]:
                ev.append((pl["closed_at"], "alert" if pl["status"] == "lost" else "plan", careplan.PLAN_STATUS_TEXT[pl["status"]],
                           pl["close_reason"] or what, None))
        for r in refs:
            ev.append((r["signed_at"] or r["created"], "plan", "Referral signed" if r["signed_at"] else "Referral drafted",
                       f"{r['facility']} · {careplan.REFERRAL_TEXT[r['status']]}", None))
        for a in appts:
            ev.append((a["start_ts"], "alert" if a["status"] == "no_show" else "booking",
                       f"Consultation · {scheduling_text(a['status'])}", f"{a['doctor_name']} · {a['room'] or ''}", None))
        for c in cons:
            ev.append((c["created"], "alert" if c["outcome"] in ("no_answer", "wrong_number") else "contact",
                       outreach.OUTCOME_TEXT[c["outcome"]], f"{outreach.CHANNEL_TEXT[c['channel']]}"
                       f"{' · ' + c['note'] if c['note'] else ''}{' · ' + c['by_name'] if c['by_name'] else ''}", None))
        for dc in docs:
            ev.append((dc["created"], "account", "Document added", f"{DOC_TEXT[dc['kind']]} · {dc['title']}", None))
        ev.sort(key=lambda e: e[0], reverse=True)
        now = time.time()
        bookable = db.one("SELECT c.id FROM cases c WHERE c.patient_id = ? AND c.status = 'finalized' AND NOT EXISTS "
                          "(SELECT 1 FROM appointments a WHERE a.case_id = c.id AND a.status IN ('booked','arrived')) "
                          "ORDER BY c.created DESC LIMIT 1", (uid,))
        return render_template("portal/care_patient.html", p=p, prof=prof, events=ev, plans=plans, refs=refs, docs=docs,
                               plan=careplan.active(db, uid), risk=risk.scores(db, [uid], now).get(uid),
                               upcoming=[a for a in appts if a["status"] in ("booked", "arrived") and a["end_ts"] > now],
                               bookable=bookable, channels=CONTACT_CHANNELS, outcomes=CONTACT_OUTCOMES,
                               plan_kinds=PLAN_KINDS, doc_kinds=DOC_KINDS, doc_text=DOC_TEXT,
                               contact_status_text=CONTACT_STATUS_TEXT, today=today(),
                               default_due=(today() + dt.timedelta(days=14)).isoformat(),
                               urgency_text=careplan.URGENCY_TEXT)

    def scheduling_text(status: str) -> str:
        from edge.portal.common import APPT_TEXT
        return APPT_TEXT[status].lower()

    # ------------------------------------------------------------------ booking on behalf of a patient
    @bp.route("/patients/<int:uid>/book", methods=["GET", "POST"])
    @auth.require("admin")
    def book(uid):
        p = patient_or_404(uid)
        c = db.one("SELECT * FROM cases WHERE patient_id = ? AND status = 'finalized' AND NOT EXISTS "
                   "(SELECT 1 FROM appointments a WHERE a.case_id = cases.id AND a.status IN ('booked','arrived')) "
                   "ORDER BY created DESC LIMIT 1", (uid,))
        if c is None:
            flash("There is no reported study to book a consultation for.", "error")
            return redirect(url_for("care.patient", uid=uid))
        error = None
        if request.method == "POST":
            try:
                aid = scheduling.book(db, p, c, float(request.form.get("start_ts", "0")), request.form.get("reason", ""))
            except (scheduling.BookingError, ValueError) as e:
                error = str(e)
            else:
                db.audit(g.user["id"], "appointment_booked", c["id"], {"appointment": aid, "by": "coordinator"}, d.ip())
                notify.appointment_event(db, aid, "booked")
                outreach.log_contact(db, uid, g.user, request.form.get("channel") if request.form.get("channel") in CONTACT_CHANNELS
                                     else "phone", "reached_confirmed", "Consultation booked")
                flash("Consultation booked. The patient and the doctor were notified.")
                return redirect(url_for("care.patient", uid=uid))
        doctor_id = scheduling.reporting_doctor(db, c)
        slots = scheduling.free_slots(db, doctor_id) if doctor_id else []
        days: dict = {}
        for s in slots:
            days.setdefault(s["day"], []).append(s)
        day_list = sorted(days)
        try:
            sel = dt.date.fromisoformat(request.args.get("day", "")) if request.args.get("day") else None
        except ValueError:
            sel = None
        if sel not in days:
            sel = day_list[0] if day_list else None
        return render_template("portal/patient_book.html", case=c, doctor=db.user(doctor_id) if doctor_id else None,
                               days=day_list, sel=sel, slots=days.get(sel, []), reschedule_of=None, error=error,
                               staff_for=p), (400 if error else 200)

    # ------------------------------------------------------------------ referrals
    @bp.get("/referrals")
    @auth.require("admin", "doctor")
    def referrals():
        status = request.args.get("status", "open")
        cond = {"open": "r.status IN ('draft','signed','sent','appointment_made')", "all": "1=1"}.get(
            status, "r.status = ?" if status in careplan.REFERRAL_TEXT else "1=0")
        args = (status,) if status in careplan.REFERRAL_TEXT else ()
        rows = db.q(f"SELECT r.*, u.name AS patient_name, s.name AS signed_by_name FROM referrals r "
                    f"JOIN users u ON u.id = r.patient_id LEFT JOIN users s ON s.id = r.signed_by WHERE {cond} "
                    f"ORDER BY CASE r.urgency WHEN 'urgent' THEN 0 WHEN 'soon' THEN 1 ELSE 2 END, r.updated DESC", args)
        counts = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) AS n FROM referrals GROUP BY status")}
        stale = time.time() - float(db.setting("referral_stale_days")) * 86400
        return render_template("portal/care_referrals.html", rows=rows, status=status, counts=counts, stale=stale, now=time.time(),
                               next_states=careplan.REFERRAL_NEXT)

    @bp.post("/referrals/<int:rid>/status")
    @auth.require("admin")
    def referral_status(rid):
        ref = careplan.referral(db, rid) or abort(404)
        try:
            careplan.set_referral_status(db, ref, request.form.get("status", ""), request.form.get("note", ""))
        except careplan.PlanError as e:
            flash(str(e), "error")
        else:
            db.audit(g.user["id"], "referral_updated", detail={"patient": ref["patient_id"], "referral": rid,
                                                                "status": request.form.get("status")}, ip=d.ip())
            flash("Referral updated.")
        return back(ref["patient_id"])

    @bp.post("/patients/<int:uid>/referral")
    @auth.require("doctor")
    def new_referral(uid):
        patient_or_404(uid)
        f = request.form
        if not auth.verify_password(g.user, f.get("password", "")):
            flash("Signature failed: password incorrect.", "error")
            return back(uid)
        plan = careplan.active(db, uid)
        try:
            rid = careplan.create_referral(db, uid, f.get("facility", ""), f.get("specialty", ""), f.get("urgency", "routine"),
                                           f.get("reason", ""), g.user, plan_id=plan["id"] if plan and plan["kind"] == "referral" else None)
            careplan.sign_referral(db, d.store, careplan.referral(db, rid), g.user, db.setting("clinic_name"))
        except careplan.PlanError as e:
            flash(str(e), "error")
            return back(uid)
        db.audit(g.user["id"], "referral_signed", detail={"patient": uid, "referral": rid}, ip=d.ip())
        notify.notify(db, db.user(uid), "referral_ready", {"facility": f.get("facility", "")}, f"/patient/referrals/{rid}",
                      dedupe=f"referral:{rid}")
        flash("Referral signed and shared with the patient.")
        return back(uid)

    @bp.get("/referrals/<int:rid>/letter")
    @auth.require("admin", "doctor")
    def letter(rid):
        ref = careplan.referral(db, rid)
        db.audit(g.user["id"], "referral_viewed", detail={"referral": rid}, ip=d.ip())
        return send_letter(d, ref)

    # ------------------------------------------------------------------ documents
    @bp.post("/patients/<int:uid>/documents")
    @auth.require("admin", "doctor")
    def upload_document(uid):
        patient_or_404(uid)
        f = request.files.get("file")
        try:
            did = store_document(db, d.store, uid, g.user, request.form.get("kind", "other"), request.form.get("title", ""),
                                 f.filename if f else "", f.read() if f else b"")
        except ValueError as e:
            flash(str(e), "error")
            return back(uid)
        db.audit(g.user["id"], "document_added", detail={"patient": uid, "document": did}, ip=d.ip())
        flash("Document added to the patient record.")
        return back(uid)

    @bp.get("/documents/<int:did>")
    @auth.require("admin", "doctor")
    def document(did):
        doc = db.one("SELECT * FROM documents WHERE id = ?", (did,)) or abort(404)
        db.audit(g.user["id"], "document_viewed", detail={"patient": doc["patient_id"], "document": did}, ip=d.ip())
        return send_document(d, doc)

    return bp

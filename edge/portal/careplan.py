"""Follow-up care plans and referrals (CareLoop): what happens after a report, and who makes sure it happens.

A care plan is the patient's next step (consultation / repeat scan / referral / other) with a due date.
  * A doctor creates plans already approved (usually while signing the report).
  * The administrator (care coordinator) may propose a plan; it stays `proposed` until a doctor approves it.
  * Outreach to the patient is allowed only on an approved plan (outreach.send_reminder enforces this).
  * Plans complete automatically: a consultation when an appointment for the patient is completed, a repeat scan
    when the patient uploads a new scan, a referral when the referral is marked attended / closed.
A referral is a signed letter to another facility. The doctor signs it (an encrypted, printable snapshot, no AI
content); the administrator then tracks it: sent -> appointment made -> attended -> closed.
`now` is injectable for tests.
"""

from __future__ import annotations

import datetime as dt
import time

from flask import render_template

from edge.portal.db import PLAN_KINDS, PLAN_OPEN

PLAN_TEXT = {"consultation": "Consultation", "repeat_scan": "Repeat MRI", "referral": "Referral", "other": "Follow-up"}
PLAN_STATUS_TEXT = {"proposed": "Awaiting approval", "approved": "Active", "completed": "Completed", "lost": "Lost to follow-up",
                    "cancelled": "Cancelled"}
REFERRAL_TEXT = {"draft": "Draft", "signed": "Signed", "sent": "Sent", "appointment_made": "Appointment made",
                 "attended": "Attended", "closed": "Closed"}
URGENCY_TEXT = {"routine": "Routine", "soon": "Soon (within 2 weeks)", "urgent": "Urgent"}
# coordinator status updates allowed from each state (signing is the doctor's)
REFERRAL_NEXT = {"signed": ("sent", "closed"), "sent": ("appointment_made", "attended", "closed"),
                 "appointment_made": ("attended", "closed"), "attended": ("closed",)}


class PlanError(ValueError):
    pass


def _date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat((s or "").strip())
    except ValueError:
        raise PlanError("Please choose a valid due date.") from None


# ---------------------------------------------------------------------------------------------- plans
def create(db, patient_id: int, kind: str, due: str, note: str, by_user, case_id: str | None = None,
           now: float | None = None) -> int:
    now = time.time() if now is None else now
    if kind not in PLAN_KINDS:
        raise PlanError("Choose the type of follow-up.")
    due_d = _date(due)
    p = db.user(patient_id)
    if p is None or p["role"] != "patient":
        raise PlanError("Patient not found.")
    if case_id and (db.case(case_id) is None or db.case(case_id)["patient_id"] != patient_id):
        raise PlanError("Study not found.")
    by_doctor = by_user["role"] == "doctor"
    with db.tx() as c:
        cur = c.execute("INSERT INTO care_plans (patient_id,case_id,kind,due_date,note,status,proposed_by,proposed_at,"
                        "approved_by,approved_at,updated) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (patient_id, case_id, kind, due_d.isoformat(), (note or "").strip()[:500] or None,
                         "approved" if by_doctor else "proposed", by_user["id"], now,
                         by_user["id"] if by_doctor else None, now if by_doctor else None, now))
    return int(cur.lastrowid)


def get(db, plan_id: int):
    return db.one("SELECT cp.*, u.name AS patient_name, pb.name AS proposed_by_name, ab.name AS approved_by_name "
                  "FROM care_plans cp JOIN users u ON u.id = cp.patient_id LEFT JOIN users pb ON pb.id = cp.proposed_by "
                  "LEFT JOIN users ab ON ab.id = cp.approved_by WHERE cp.id = ?", (plan_id,))


def active(db, patient_id: int):
    """The patient's current open plan (earliest due), or None."""
    return db.one(f"SELECT * FROM care_plans WHERE patient_id = ? AND status IN {PLAN_OPEN} ORDER BY due_date, id LIMIT 1",
                  (patient_id,))


def approve(db, plan, doctor, due: str | None = None, note: str | None = None, now: float | None = None) -> None:
    now = time.time() if now is None else now
    if doctor["role"] != "doctor":
        raise PlanError("Only a doctor can approve a follow-up plan.")
    if plan["status"] != "proposed":
        raise PlanError("This plan is not waiting for approval.")
    due_s = _date(due).isoformat() if due else plan["due_date"]
    with db.tx() as c:
        c.execute("UPDATE care_plans SET status='approved', approved_by=?, approved_at=?, due_date=?, note=COALESCE(?, note), "
                  "updated=? WHERE id=?", (doctor["id"], now, due_s, (note or "").strip()[:500] or None, now, plan["id"]))


def close(db, plan, status: str, reason: str, by_user, now: float | None = None) -> None:
    """completed | lost | cancelled (a rejected proposal is 'cancelled')."""
    now = time.time() if now is None else now
    if status not in ("completed", "lost", "cancelled"):
        raise PlanError("Unknown status.")
    if plan["status"] not in PLAN_OPEN:
        raise PlanError("This plan is already closed.")
    with db.tx() as c:
        c.execute("UPDATE care_plans SET status=?, closed_at=?, close_reason=?, updated=? WHERE id=?",
                  (status, now, (reason or "").strip()[:300] or None, now, plan["id"]))


def complete_matching(db, patient_id: int, kinds: tuple[str, ...], reason: str, now: float | None = None) -> int:
    """Close the patient's approved plans of these kinds (an appointment was completed, a new scan arrived ...)."""
    now = time.time() if now is None else now
    ph = ",".join("?" * len(kinds))
    with db.tx() as c:
        cur = c.execute(f"UPDATE care_plans SET status='completed', closed_at=?, close_reason=?, updated=? "
                        f"WHERE patient_id=? AND status='approved' AND kind IN ({ph})", (now, reason, now, patient_id, *kinds))
    return cur.rowcount


def due_info(plan, today: dt.date) -> dict:
    due = dt.date.fromisoformat(plan["due_date"])
    days = (due - today).days
    return {"due": due, "days": days, "overdue": days < 0 and plan["status"] == "approved"}


# ------------------------------------------------------------------------------------------ referrals
def create_referral(db, patient_id: int, facility: str, specialty: str, urgency: str, reason: str, by_user,
                    case_id: str | None = None, plan_id: int | None = None, now: float | None = None) -> int:
    now = time.time() if now is None else now
    if not (facility or "").strip() or not (reason or "").strip():
        raise PlanError("Facility and reason are required.")
    if urgency not in URGENCY_TEXT:
        urgency = "routine"
    with db.tx() as c:
        cur = c.execute("INSERT INTO referrals (patient_id,case_id,plan_id,facility,specialty,urgency,reason,status,created_by,"
                        "created,updated) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (patient_id, case_id, plan_id, facility.strip()[:150], (specialty or "").strip()[:80] or None,
                         urgency, reason.strip()[:1500], "draft", by_user["id"], now, now))
    return int(cur.lastrowid)


def referral(db, rid: int):
    return db.one("SELECT r.*, u.name AS patient_name, s.name AS signed_by_name FROM referrals r "
                  "JOIN users u ON u.id = r.patient_id LEFT JOIN users s ON s.id = r.signed_by WHERE r.id = ?", (rid,))


def letter_kind(rid: int) -> tuple[str, str]:
    """Blob location of a signed referral letter (patient-scoped folder)."""
    return "referrals", f"letter_{rid}"


def sign_referral(db, store, ref, doctor, clinic: str, now: float | None = None) -> None:
    """Render the letter (patient details, reason, facility, doctor; no AI output) and store it encrypted."""
    now = time.time() if now is None else now
    if doctor["role"] != "doctor":
        raise PlanError("Only a doctor can sign a referral.")
    if ref["status"] != "draft":
        raise PlanError("This referral is already signed.")
    patient = db.user(ref["patient_id"])
    html = render_template("portal/referral_letter.html", ref=ref, patient=patient, profile=db.profile(ref["patient_id"]),
                           doctor=doctor, clinic=clinic, urgency=URGENCY_TEXT[ref["urgency"]],
                           signed=time.strftime("%d %b %Y", time.localtime(now)))
    sha = store.put(*letter_kind(ref["id"]), html.encode("utf-8"))
    with db.tx() as c:
        c.execute("UPDATE referrals SET status='signed', signed_by=?, signed_at=?, letter_sha256=?, updated=? WHERE id=?",
                  (doctor["id"], now, sha, now, ref["id"]))


def set_referral_status(db, ref, status: str, note: str = "", now: float | None = None) -> None:
    now = time.time() if now is None else now
    if status not in REFERRAL_NEXT.get(ref["status"], ()):
        raise PlanError("This change is not allowed.")
    with db.tx() as c:
        c.execute("UPDATE referrals SET status=?, status_note=COALESCE(?, status_note), updated=? WHERE id=?",
                  (status, (note or "").strip()[:300] or None, now, ref["id"]))
    if status in ("attended", "closed") and ref["plan_id"]:
        p = db.one("SELECT * FROM care_plans WHERE id = ?", (ref["plan_id"],))
        if p and p["status"] == "approved":
            close(db, p, "completed", "Referral " + REFERRAL_TEXT[status].lower(), None, now)

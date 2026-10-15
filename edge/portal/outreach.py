"""Patient outreach for follow-up: the contact log and reminders sent by the care coordinator (the administrator).

Rules
  * Every reminder is triggered by a person (send_reminder); nothing here runs on a timer.
  * Only for an APPROVED care plan (a doctor approved the follow-up) and only if the patient has not withdrawn
    consent for e-mail reminders. The in-app notification is always created.
  * Every attempt is written to the contact log, which also feeds the risk score.
  * Appointment reminder e-mails carry "I'll be there" / "I can't make it" links (single-use tokens): the reply
    is logged as a contact; "can't make it" cancels the slot when still allowed and alerts the team.
"""

from __future__ import annotations

import time

from edge.portal import careplan, notify, scheduling, tokens
from edge.portal.db import CONTACT_CHANNELS, CONTACT_OUTCOMES

OUTCOME_TEXT = {"reached_confirmed": "Reached · will come", "reached_declined": "Reached · declined",
                "no_answer": "No answer", "wrong_number": "Wrong number", "message_sent": "Reminder sent"}
CHANNEL_TEXT = {"phone": "Phone call", "in_person": "In person", "email": "E-mail", "portal": "Portal"}


class OutreachError(ValueError):
    pass


def log_contact(db, patient_id: int, by_user, channel: str, outcome: str, note: str = "", plan_id: int | None = None,
                next_action: str | None = None, now: float | None = None) -> int:
    now = time.time() if now is None else now
    if channel not in CONTACT_CHANNELS or outcome not in CONTACT_OUTCOMES:
        raise OutreachError("Choose how you contacted the patient and the outcome.")
    with db.tx() as c:
        cur = c.execute("INSERT INTO contacts (patient_id,plan_id,by_user,channel,outcome,note,next_action_date,created) "
                        "VALUES (?,?,?,?,?,?,?,?)", (patient_id, plan_id, by_user["id"] if by_user else None, channel,
                                                     outcome, (note or "").strip()[:500] or None, next_action or None, now))
        # keep the contact status in step with what the coordinator just learned
        if outcome == "wrong_number":
            c.execute("UPDATE patient_profiles SET contact_status = 'wrong_number' WHERE user_id = ?", (patient_id,))
        elif outcome in ("reached_confirmed", "reached_declined"):
            c.execute("UPDATE patient_profiles SET contact_status = 'ok' WHERE user_id = ?", (patient_id,))
    return int(cur.lastrowid)


def contacts(db, patient_id: int, limit: int = 50) -> list:
    return db.q("SELECT c.*, u.name AS by_name FROM contacts c LEFT JOIN users u ON u.id = c.by_user "
                "WHERE c.patient_id = ? ORDER BY c.created DESC LIMIT ?", (patient_id, limit))


def send_reminder(db, plan, by_user, message: str = "", now: float | None = None) -> bool:
    """Human-triggered follow-up reminder. Returns True if an e-mail was queued (False: in-app only)."""
    now = time.time() if now is None else now
    if plan["status"] != "approved":
        raise OutreachError("A doctor must approve this follow-up plan before the patient is contacted.")
    patient = db.user(plan["patient_id"])
    due = careplan.due_info(plan, scheduling.local(now, scheduling.tz(db)).date())["due"]
    link = "/patient/"
    if plan["kind"] == "consultation" and plan["case_id"]:
        link = f"/patient/book/{plan['case_id']}"
    ctx = {"step": careplan.PLAN_TEXT[plan["kind"]].lower(), "due": due.strftime("%d %b %Y"),
           "message": (message or "").strip()[:500] or "Please sign in to the patient portal or contact the department."}
    notify.notify(db, patient, "followup_reminder", ctx, link, dedupe=f"followup:{plan['id']}:{int(now)}", now=now)
    emailed = bool(patient["email"] and patient["active"] and patient["notify_email"])
    log_contact(db, patient["id"], by_user, "email" if emailed else "portal", "message_sent",
                (message or "").strip()[:200], plan["id"], now=now)
    return emailed


def reminder_actions(db, appt, now: float | None = None) -> list[tuple[str, str]]:
    """Buttons for an appointment reminder e-mail."""
    yes = tokens.issue(db, "appt_yes", appt["patient_id"], appt["id"], now)
    no = tokens.issue(db, "appt_no", appt["patient_id"], appt["id"], now)
    return [("I'll be there", f"/r/{yes}?a=yes"), ("I can't make it", f"/r/{no}?a=no")]


def appointment_reply(db, purpose: str, raw: str, now: float | None = None) -> dict | None:
    """Apply a confirm / decline link. None if the link is invalid, used or expired."""
    now = time.time() if now is None else now
    t = tokens.consume(db, purpose, raw, now)
    if t is None:
        return None
    a = scheduling.appointment(db, t["ref_id"])
    if a is None or a["patient_id"] != t["user_id"]:
        return None
    patient = db.user(a["patient_id"])
    if a["status"] != "booked":
        return {"appt": a, "result": "inactive"}
    if purpose == "appt_yes":
        log_contact(db, patient["id"], None, "email", "reached_confirmed", "Confirmed by e-mail link", now=now)
        return {"appt": a, "result": "confirmed"}
    log_contact(db, patient["id"], None, "email", "reached_declined", "Declined by e-mail link", now=now)
    try:
        scheduling.cancel(db, a, patient, now)
        cancelled = True
    except scheduling.BookingError:                      # inside the cutoff: the team handles it
        cancelled = False
    if cancelled:
        notify.appointment_event(db, a["id"], "cancelled", now)
    ctx = notify.appt_ctx(db, a) | {"extra": " The slot was released." if cancelled else
                                    " It is too late to release the slot automatically; please contact the patient."}
    for admin in db.q("SELECT * FROM users WHERE role = 'admin' AND active = 1"):
        notify.notify(db, admin, "admin_appt_declined", ctx, f"/care/patients/{a['patient_id']}",
                      dedupe=f"declined:{a['id']}", now=now)
    return {"appt": a, "result": "cancelled" if cancelled else "declined"}

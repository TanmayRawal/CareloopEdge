"""In-app notifications + e-mail outbox.

Every message becomes an in-app notification and (if the user allows e-mail) a queued outbox row that
`mailer.flush` delivers. Rule: e-mails never contain clinical content — no diagnosis, findings or AI output —
only "your report is ready", appointment times and links; the details stay behind the portal sign-in.
`dedupe` makes a message idempotent: the same key is never queued twice.
"""

from __future__ import annotations

import datetime as dt
import html
import sqlite3
import time
import uuid

from edge.portal import scheduling

# kind -> (subject, body) templates; {placeholders} are filled from `ctx`. Plain language, no clinical content.
TEMPLATES = {
    "report_ready": ("Your report is ready", "Your radiologist has reviewed your MRI scan and your report is now available "
                     "in the patient portal.\n\nYou can also book a follow-up consultation there."),
    "reupload": ("New image requested", "Your radiologist has asked for a new image for your scan.\n\nMessage: {message}"),
    "appt_booked": ("Appointment confirmed: {when}", "Your consultation with {doctor} is confirmed.\n\nWhen: {when}\n"
                    "Where: {room}\n\nPlease arrive 10 minutes early. You can reschedule or cancel up to {cutoff} hours before."),
    "appt_rescheduled": ("Appointment moved to {when}", "Your consultation with {doctor} has been moved.\n\nNew time: {when}\n"
                         "Where: {room}"),
    "appt_cancelled": ("Appointment cancelled", "Your consultation with {doctor} on {when} has been cancelled.\n\n"
                       "You can book a new time in the patient portal."),
    "appt_reminder": ("Reminder: appointment {rel}", "This is a reminder of your consultation with {doctor}.\n\nWhen: {when}\n"
                      "Where: {room}\n\nNeed to change it? Use the patient portal (up to {cutoff} hours before)."),
    "appt_no_show": ("We missed you today", "You were booked for a consultation with {doctor} on {when}, but we did not "
                     "see you.\n\nPlease book a new appointment in the patient portal at a time that suits you."),
    "doc_booking": ("New appointment: {patient}, {when}", "{patient} booked a consultation with you.\n\nWhen: {when}\nWhere: {room}"),
    "doc_cancel": ("Appointment cancelled: {patient}, {when}", "{patient} cancelled the consultation on {when}."),
    "doc_unread": ("{studies} waiting for review", "{studies} have been waiting longer than the department target "
                   "({hours} hours). The oldest arrived {oldest}.\n\nOpen the worklist to review them."),
    "doc_unsigned": ("Report not signed after {hours} hours", "A study you opened {age} ago has not been signed yet (patient {patient})."),
    "admin_escalation": ("{studies} unread for over {hours} hours", "{studies} have not been opened by any radiologist "
                         "for more than {hours} hours."),
    "doc_agenda": ("Today: {consultations}", "Your consultations today:\n\n{agenda}"),
    "doc_unmarked": ("Please record attendance: {patient}", "The consultation with {patient} at {when} has ended but attendance "
                     "was not recorded. Please mark it as completed or no-show."),
    # CareLoop follow-up
    "followup_planned": ("Your next step: {step} by {due}", "Your doctor has planned the next step in your care: {step}, "
                         "by {due}.\n\n{action}"),
    "followup_reminder": ("Your next step: {step}", "Your care team would like to remind you about the next step in your "
                          "care: {step}, due by {due}.\n\n{message}"),
    "referral_ready": ("Your referral letter is ready", "Your doctor has signed a referral to {facility}. You can view and "
                       "print the letter in the patient portal and take it to your appointment there."),
    "invite": ("Set up your patient portal account", "{clinic} has registered you for follow-up care. Set a password to see "
               "your reports, appointments and next steps online.\n\nThis link works once and expires in 7 days."),
    "password_reset": ("Reset your password", "We received a request to reset the password for your account.\n\nThe link "
                       "works once and expires in 1 hour. If you did not ask for this, you can ignore this e-mail."),
    "doc_approvals": ("{plans} waiting for your approval", "The care coordinator proposed {plans} that need a doctor's "
                      "approval. Patients are contacted only after a doctor approves."),
    "doc_plan_proposed": ("Follow-up plan to approve: {patient}", "A {step} for {patient} (due {due}) was proposed and needs "
                          "a doctor's approval."),
    "admin_digest": ("Follow-up today: {patients} need attention", "{summary}\n\nOpen the follow-up list to contact them."),
    "admin_appt_declined": ("Patient cannot attend: {patient}", "{patient} replied that they cannot attend the consultation "
                            "on {when} with {doctor}.{extra}"),
}


def plural(n: int, one: str, many: str | None = None) -> str:
    """plural(1, 'study', 'studies') -> '1 study'; plural(3, 'study', 'studies') -> '3 studies'."""
    return f"{n} {one if n == 1 else (many or one + 's')}"


def fmt_when(db, ts: float) -> str:
    return scheduling.local(ts, scheduling.tz(db)).strftime("%a %d %b %Y, %H:%M")


def ics(db, appt, uid: str, method: str = "REQUEST") -> str:
    """RFC 5545 calendar invite (UTC times)."""
    f = lambda ts: dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # noqa: E731
    esc = lambda s: (s or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")  # noqa: E731
    status = "CANCELLED" if method == "CANCEL" else "CONFIRMED"
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CareLoop-Edge//Portal//EN", f"METHOD:{method}",
             "BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{f(time.time())}", f"DTSTART:{f(appt['start_ts'])}",
             f"DTEND:{f(appt['end_ts'])}", f"SUMMARY:{esc('Consultation · ' + db.setting('clinic_name'))}",
             f"LOCATION:{esc(appt['room'] or db.setting('clinic_name'))}", f"STATUS:{status}",
             f"SEQUENCE:{int(appt['updated'])}", "END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines) + "\r\n"


def _html(db, title: str, body: str, link: str | None, actions: list | None = None) -> str:
    base = db.setting("base_url").rstrip("/")
    paras = "".join(f"<p style=\"margin:0 0 12px\">{html.escape(p).replace(chr(10), '<br>')}</p>" for p in body.split("\n\n"))
    btn = ("display:inline-block;margin:0 8px 8px 0;padding:10px 18px;border-radius:6px;text-decoration:none;font-weight:600;"
           "{}")
    styles = ("background:#0b5c7a;color:#fff", "background:#fff;color:#0b5c7a;border:1px solid #0b5c7a")
    items = actions or ([("Open the portal", link)] if link else [])
    button = ("<p style=\"margin:20px 0 8px\">" + "".join(
        f"<a href=\"{html.escape(base + path)}\" style=\"{btn.format(styles[min(i, 1)])}\">{html.escape(label)}</a>"
        for i, (label, path) in enumerate(items)) + "</p>") if items else ""
    clinic = html.escape(db.setting("clinic_name"))
    return (f"<!doctype html><html><body style=\"margin:0;background:#f3f5f8;font:15px/1.55 Segoe UI,Arial,sans-serif;color:#0f172a\">"
            f"<div style=\"max-width:560px;margin:24px auto;background:#fff;border:1px solid #e2e8f0;border-radius:8px\">"
            f"<div style=\"padding:16px 24px;border-bottom:3px solid #0b5c7a;font-weight:700\">{clinic}</div>"
            f"<div style=\"padding:20px 24px\"><h2 style=\"margin:0 0 14px;font-size:18px\">{html.escape(title)}</h2>{paras}{button}</div>"
            f"<div style=\"padding:12px 24px;color:#64748b;font-size:12px;border-top:1px solid #e2e8f0\">This message was sent by "
            f"{clinic}. It contains no medical details; sign in to the portal to see your records.</div></div></body></html>")


def notify(db, user, kind: str, ctx: dict | None = None, link: str | None = None, email: bool = True,
           dedupe: str | None = None, ics_text: str | None = None, now: float | None = None,
           actions: list | None = None, in_app: bool = True, force_email: bool = False) -> bool:
    """Create the in-app notification and queue the e-mail. Returns False if `dedupe` was already used.
    actions: [(label, path)] buttons for the e-mail only (they may carry single-use tokens, so they never go in-app).
    force_email: account e-mails (invite, password reset) that are sent even if reminders are switched off."""
    now = time.time() if now is None else now
    if dedupe and not db.once(f"notify:{dedupe}:{user['id']}", now):
        return False
    subject_t, body_t = TEMPLATES[kind]
    ctx = {k: ("" if v is None else v) for k, v in (ctx or {}).items()}
    subject, body = subject_t.format(**ctx), body_t.format(**ctx)
    if in_app:
        with db.tx() as c:
            c.execute("INSERT INTO notifications (user_id,kind,title,body,link,created) VALUES (?,?,?,?,?,?)",
                      (user["id"], kind, subject, body, link, now))
    if email and user["email"] and user["active"] and (user["notify_email"] or force_email):
        base = db.setting("base_url").rstrip("/")
        lines = [f"{label}: {base + path}" for label, path in (actions or ([("Open the portal", link)] if link else []))]
        text = f"{body}\n\n" + "\n".join(lines) + f"\n\n— {db.setting('clinic_name')}\n"
        try:
            with db.tx() as c:
                c.execute("INSERT INTO outbox (user_id,to_email,subject,text,html,ics,created,next_try,dedupe_key) "
                          "VALUES (?,?,?,?,?,?,?,?,?)",
                          (user["id"], user["email"], f"{subject} · {db.setting('clinic_name')}", text,
                           _html(db, subject, body, link, actions), ics_text, now, now,
                           f"{dedupe}:{user['id']}" if dedupe else None))
        except sqlite3.IntegrityError:
            pass
    return True


def unread(db, user_id: int, limit: int = 8) -> tuple[int, list]:
    n = db.one("SELECT COUNT(*) AS n FROM notifications WHERE user_id = ? AND read_at IS NULL", (user_id,))["n"]
    rows = db.q("SELECT * FROM notifications WHERE user_id = ? ORDER BY created DESC LIMIT ?", (user_id, limit))
    return n, rows


def mark_read(db, user_id: int, notif_id: int | None = None, now: float | None = None) -> None:
    now = time.time() if now is None else now
    with db.tx() as c:
        if notif_id is None:
            c.execute("UPDATE notifications SET read_at = ? WHERE user_id = ? AND read_at IS NULL", (now, user_id))
        else:
            c.execute("UPDATE notifications SET read_at = COALESCE(read_at, ?) WHERE user_id = ? AND id = ?",
                      (now, user_id, notif_id))


# ---------------------------------------------------------------------------------- appointment helpers
def appt_ctx(db, appt) -> dict:
    return {"when": fmt_when(db, appt["start_ts"]), "doctor": appt["doctor_name"], "patient": appt["patient_name"],
            "room": appt["room"] or db.setting("clinic_name"), "cutoff": db.setting("cancel_cutoff_hours")}


def appt_uid(db, appt) -> str:
    return f"appt-{appt['id']}-{uuid.uuid5(uuid.NAMESPACE_URL, db.setting('base_url'))}@neuroedge"


def appointment_event(db, appt_id: int, event: str, now: float | None = None) -> None:
    """booked | rescheduled | cancelled | no_show -> messages to patient (and doctor)."""
    a = scheduling.appointment(db, appt_id)
    patient, doctor = db.user(a["patient_id"]), db.user(a["doctor_id"])
    ctx = appt_ctx(db, a)
    if event in ("booked", "rescheduled"):
        notify(db, patient, "appt_booked" if event == "booked" else "appt_rescheduled", ctx, "/patient/appointments",
               ics_text=ics(db, a, appt_uid(db, a)), dedupe=f"appt:{event}:{a['id']}", now=now)
        notify(db, doctor, "doc_booking", ctx, "/doctor/schedule", dedupe=f"appt:{event}:{a['id']}", now=now)
    elif event == "cancelled":
        notify(db, patient, "appt_cancelled", ctx, "/patient/appointments", ics_text=ics(db, a, appt_uid(db, a), "CANCEL"),
               dedupe=f"appt:cancel:{a['id']}", now=now)
        notify(db, doctor, "doc_cancel", ctx, "/doctor/schedule", dedupe=f"appt:cancel:{a['id']}", now=now)
    elif event == "no_show":
        notify(db, patient, "appt_no_show", ctx, f"/patient/book/{a['case_id']}", dedupe=f"appt:noshow:{a['id']}", now=now)

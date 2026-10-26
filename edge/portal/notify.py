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

# kind -> (subject, body, details). {placeholders} are filled from `ctx` (a missing one is left empty); details are
# (label, value) rows shown as a table in the e-mail, and a row whose value is empty is left out. Plain language, no
# clinical content. Kinds starting with doc_ / admin_ go to staff; the others to patients.
TEMPLATES = {
    "report_ready": (
        "Your MRI report is available",
        "Your {exam} examination received on {received} has been reported by {doctor}. The signed report is now "
        "available in your patient portal.\n\n"
        "We recommend that you book a consultation to go through the findings with your doctor. Please do not make "
        "decisions about your treatment on the basis of the report alone.",
        [("Examination", "{exam}"), ("Accession no.", "{accession}"), ("Reporting radiologist", "{doctor}")]),
    "reupload": (
        "Additional images needed for your MRI",
        "Your radiologist needs additional images before your report can be completed.\n\n"
        "Message from your radiologist: {message}\n\n"
        "Please submit the new images {upload_where}. Your study stays in the queue and will be reported as soon as "
        "they arrive.", []),
    "appt_booked": (
        "Appointment confirmed: {when}",
        "We are pleased to confirm your consultation appointment. A calendar invitation is attached.\n\n"
        "Please arrive 10 minutes early and bring a photo ID, any previous scans and reports, and a list of your "
        "current medicines. If you cannot attend, please reschedule or cancel at least {cutoff} hours in advance so "
        "that the time can be offered to another patient.",
        [("Date and time", "{when}"), ("Doctor", "{doctor}"), ("Location", "{room}")]),
    "appt_rescheduled": (
        "Appointment rescheduled: {when}",
        "Your consultation has been moved to a new time, shown below. The previous appointment time is no longer "
        "valid. An updated calendar invitation is attached.",
        [("New date and time", "{when}"), ("Doctor", "{doctor}"), ("Location", "{room}")]),
    "appt_cancelled": (
        "Appointment cancelled",
        "Your consultation shown below has been cancelled.\n\n"
        "If you did not ask for this, or would like a new appointment, please book another time in the patient portal "
        "or contact the department.",
        [("Cancelled appointment", "{when}"), ("Doctor", "{doctor}")]),
    "appt_reminder": (
        "Reminder: your appointment {rel}",
        "This is a reminder of your upcoming consultation.\n\n"
        "Please arrive 10 minutes early and bring a photo ID and any previous scans or reports. If you can no longer "
        "attend, please let us know at least {cutoff} hours before, using the buttons below or the patient portal.",
        [("Date and time", "{when}"), ("Doctor", "{doctor}"), ("Location", "{room}")]),
    "appt_no_show": (
        "Missed appointment on {when}",
        "Our records show that you were not able to attend your consultation with {doctor} on {when}.\n\n"
        "It is important that your results are discussed with a doctor. Please book a new appointment at a time that "
        "suits you, or contact the department if you need help with travel or timing.", []),
    "doc_booking": (
        "New consultation: {patient}, {when}",
        "A consultation has been booked with you.",
        [("Patient", "{patient}"), ("Patient ID", "{mrn}"), ("Date and time", "{when}"), ("Location", "{room}")]),
    "doc_cancel": (
        "Consultation cancelled: {patient}, {when}",
        "The consultation below has been cancelled. The time is available for booking again.",
        [("Patient", "{patient}"), ("Patient ID", "{mrn}"), ("Date and time", "{when}")]),
    "doc_unread": (
        "Reporting backlog: {studies} awaiting review",
        "{studies} have been awaiting review for longer than the department target of {hours} hours. The oldest "
        "arrived {oldest}.\n\nPlease review them on the worklist, starting with studies marked urgent.", []),
    "doc_unsigned": (
        "Report not signed: patient {patient}",
        "A study you opened {age} ago has not been signed yet (patient {patient}). The department target for signing "
        "is {hours} hours. Please complete and sign the report so that it can be released to the patient.", []),
    "admin_escalation": (
        "Escalation: {studies} unread for over {hours} hours",
        "{studies} have not been opened by any radiologist for more than {hours} hours. Please arrange reporting cover "
        "so that patients' results are not delayed.", []),
    "doc_agenda": (
        "Today's consultations: {consultations}",
        "Your consultations for today:\n\n{agenda}", []),
    "doc_unmarked": (
        "Attendance not recorded: {patient}",
        "The consultation with {patient} at {when} has ended, but attendance has not been recorded. Please mark it as "
        "completed or did not attend, so that the patient's follow-up can continue.", []),
    # CareLoop follow-up
    "followup_planned": (
        "Your next step: {step} by {due}",
        "Following your recent report, your doctor has planned the next step in your care.\n\n{action}",
        [("Next step", "{step}"), ("Due by", "{due}")]),
    "followup_reminder": (
        "Your next step: {step}",
        "This is a reminder from your care team about the next step in your care.\n\n{message}",
        [("Next step", "{step}"), ("Due by", "{due}")]),
    "referral_ready": (
        "Your referral letter is ready",
        "Your doctor has signed a referral letter to {facility}. Please view or print it from the patient portal and "
        "take it to your appointment there, together with your MRI report and any previous scans.",
        [("Referred to", "{facility}")]),
    "invite": (
        "Set up your patient portal account",
        "{clinic} has registered you for follow-up care. Please set a password to see your reports, appointments and "
        "next steps online.\n\nFor your security, the link below works once and expires in 7 days. If you were not "
        "expecting this e-mail, please contact the department.", []),
    "verify_email": (
        "Please confirm your e-mail address",
        "Thank you for registering with {clinic}. Please confirm that this is your e-mail address, so that we can "
        "tell you when your report is ready and send your appointment details.\n\nThe link below works once and "
        "expires in 7 days. If you did not register, you can ignore this e-mail and no further messages will be sent.",
        []),
    "password_reset": (
        "Reset your password",
        "We received a request to reset the password for your account.\n\nFor your security, the link below works once "
        "and expires in 1 hour. If you did not ask for this, no action is needed and your password stays the same.", []),
    "doc_approvals": (
        "{plans} awaiting your approval",
        "The care coordinator has proposed {plans} that need a doctor's approval. Patients are contacted only after a "
        "doctor approves.", []),
    "doc_plan_proposed": (
        "Follow-up plan to approve: {patient}",
        "A follow-up plan has been proposed and needs a doctor's approval before the patient is contacted.",
        [("Patient", "{patient}"), ("Proposed step", "{step}"), ("Due", "{due}")]),
    "admin_digest": (
        "Follow-up today: {patients} need attention",
        "{summary}\n\nPlease open the follow-up list to contact them.", []),
    "admin_appt_declined": (
        "Patient cannot attend: {patient}",
        "{patient} has replied that they cannot attend the consultation on {when} with {doctor}.{extra}", []),
}
# The e-mail's main button (when the caller gives a link but no buttons of its own).
LINK_LABEL = {"verify_email": "Confirm my e-mail address", "report_ready": "View your report", "reupload": "View the request", "referral_ready": "View the referral letter",
              "appt_booked": "Manage your appointment", "appt_rescheduled": "Manage your appointment",
              "appt_cancelled": "Book a new time", "appt_reminder": "Manage your appointment",
              "appt_no_show": "Book a new appointment", "followup_planned": "View your next step",
              "followup_reminder": "View your next step"}
# Patient messages about results or next steps carry safety-netting advice (when not to wait for the appointment).
SAFETY = {"report_ready", "followup_planned", "followup_reminder", "appt_no_show", "reupload"}
SAFETY_TEXT = ("If you develop a sudden or severe headache, a seizure, new weakness or numbness, difficulty speaking, "
               "confusion or loss of vision, do not wait for your appointment: go to the nearest emergency department "
               "or call 108.")


# Set by the web server (edge.serve) to send queued e-mail right away instead of waiting for the 5-minute timer.
ON_QUEUED = None


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


class _Blank(dict):
    def __missing__(self, key):                       # a placeholder the caller did not fill stays empty
        return ""


def _message(db, user, kind: str, ctx: dict) -> dict:
    """The parts of one e-mail: greeting, body, detail rows, safety advice, sign-off and footer (plain text)."""
    subject_t, body_t, details_t = TEMPLATES[kind]
    staff = kind.startswith(("doc_", "admin_"))
    clinic = db.setting("clinic_name")
    contact = [x for x in (db.setting("clinic_address"), f"Tel. {db.setting('clinic_phone')}" if db.setting("clinic_phone")
                           else "") if x]
    if staff:
        footer = (f"Automated message from CareLoop-Edge at {clinic}. It contains patient information: do not forward it "
                  "outside the care team.")
    else:
        footer = (f"This is an automated message from {clinic}; replies to this address are not read. Medical results "
                  "are never sent by e-mail: sign in to the patient portal to see your records. This message is "
                  "confidential and meant only for the person it is addressed to; if you received it by mistake, please "
                  f"delete it and let {clinic} know. You can change your e-mail preferences under Account in the portal.")
    return {
        "subject": subject_t.format_map(ctx), "body": body_t.format_map(ctx),
        "greeting": f"Dear {user['name']}," if user["name"] else "Hello,",
        "details": [(label, v) for label, t in details_t if (v := t.format_map(ctx).strip())],
        "safety": SAFETY_TEXT if kind in SAFETY else "",
        "signoff": ["Kind regards,", clinic, *contact], "footer": footer,
    }


def _text(m: dict, links: list) -> str:
    parts = [m["greeting"], m["body"]]
    if m["details"]:
        parts.append("\n".join(f"{label}: {v}" for label, v in m["details"]))
    if links:
        parts.append("\n".join(f"{label}: {url}" for label, url in links))
    if m["safety"]:
        parts.append(m["safety"])
    parts += ["\n".join(m["signoff"]), "--\n" + m["footer"]]
    return "\n\n".join(parts) + "\n"


def _html(m: dict, links: list) -> str:
    e = html.escape
    accent, ink, muted, line = "#0b4f6c", "#15181c", "#5c636b", "#dfe2e6"
    paras = "".join(f'<p style="margin:0 0 14px">{e(p).replace(chr(10), "<br>")}</p>' for p in m["body"].split("\n\n"))
    details = ("<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" style=\"width:100%;border-collapse:collapse;"
               f"margin:4px 0 18px;border:1px solid {line}\">" + "".join(
                   f'<tr><td style="padding:8px 12px;border-bottom:1px solid {line};background:#f6f7f8;color:{muted};'
                   f'font-size:13px;width:38%">{e(label)}</td><td style="padding:8px 12px;border-bottom:1px solid {line};'
                   f'font-weight:600">{e(v)}</td></tr>' for label, v in m["details"]) + "</table>") if m["details"] else ""
    btn = "display:inline-block;margin:0 8px 8px 0;padding:11px 20px;border-radius:4px;text-decoration:none;font-weight:600;"
    styles = (f"background:{accent};color:#ffffff;border:1px solid {accent}", f"background:#ffffff;color:{accent};border:1px solid {accent}")
    buttons = ('<p style="margin:6px 0 18px">' + "".join(
        f'<a href="{e(url)}" style="{btn}{styles[min(i, 1)]}">{e(label)}</a>' for i, (label, url) in enumerate(links))
        + "</p>") if links else ""
    safety = (f'<p style="margin:0 0 18px;padding:10px 14px;background:#fbf0dc;border-left:3px solid #9a5800;'
              f'font-size:13.5px">{e(m["safety"])}</p>') if m["safety"] else ""
    signoff = "<br>".join(f"<b>{e(x)}</b>" if i == 1 else e(x) for i, x in enumerate(m["signoff"]))
    clinic, contact = m["signoff"][1], " · ".join(m["signoff"][2:])
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f'<title>{e(m["subject"])}</title></head>'
            f'<body style="margin:0;padding:0;background:#eef0f2;font:15px/1.6 Segoe UI,Helvetica,Arial,sans-serif;color:{ink}">'
            f'<span style="display:none;max-height:0;overflow:hidden">{e(m["body"].split(chr(10))[0][:140])}</span>'
            f'<div style="max-width:600px;margin:24px auto;background:#ffffff;border:1px solid {line}">'
            f'<div style="padding:18px 28px;border-bottom:3px solid {accent}">'
            f'<div style="font-weight:700;font-size:16px;letter-spacing:.02em">{e(clinic)}</div>'
            f'<div style="color:{muted};font-size:12.5px">Department of Radiology{(" · " + e(contact)) if contact else ""}</div></div>'
            f'<div style="padding:24px 28px 8px"><h1 style="margin:0 0 18px;font-size:19px;line-height:1.35">{e(m["subject"])}</h1>'
            f'<p style="margin:0 0 14px">{e(m["greeting"])}</p>{paras}{details}{buttons}{safety}'
            f'<p style="margin:0 0 20px">{signoff}</p></div>'
            f'<div style="padding:14px 28px;border-top:1px solid {line};color:{muted};font-size:12px;line-height:1.5">'
            f'{e(m["footer"])}</div></div></body></html>')


def notify(db, user, kind: str, ctx: dict | None = None, link: str | None = None, email: bool = True,
           dedupe: str | None = None, ics_text: str | None = None, now: float | None = None,
           actions: list | None = None, in_app: bool = True, force_email: bool = False) -> bool:
    """Create the in-app notification and queue the e-mail. Returns False if `dedupe` was already used.
    actions: [(label, path)] buttons for the e-mail only (they may carry single-use tokens, so they never go in-app).
    force_email: account e-mails (invite, password reset) that are sent even if reminders are switched off."""
    now = time.time() if now is None else now
    if dedupe and not db.once(f"notify:{dedupe}:{user['id']}", now):
        return False
    clinic = db.setting("clinic_name")
    ctx = _Blank({"clinic": clinic,
                  "upload_where": f"at the CareLoop screen at {clinic}" if db.setting("upload_device_only") == "1"
                  else "in the patient portal"} | {k: ("" if v is None else v) for k, v in (ctx or {}).items()})
    m = _message(db, user, kind, ctx)
    if in_app:
        with db.tx() as c:
            c.execute("INSERT INTO notifications (user_id,kind,title,body,link,created) VALUES (?,?,?,?,?,?)",
                      (user["id"], kind, m["subject"], m["body"], link, now))
    # unconfirmed address (self-registration): only the confirmation and password e-mails go out
    confirmed = user["email_verified"] or kind in ("verify_email", "password_reset", "invite")
    if email and user["email"] and user["active"] and confirmed and (user["notify_email"] or force_email):
        base = db.setting("base_url").rstrip("/")
        default = LINK_LABEL.get(kind, "Open CareLoop" if kind.startswith(("doc_", "admin_")) else "Open the patient portal")
        links = [(label, base + path) for label, path in (actions or ([(default, link)] if link else []))]
        try:
            with db.tx() as c:
                c.execute("INSERT INTO outbox (user_id,to_email,subject,text,html,ics,created,next_try,dedupe_key) "
                          "VALUES (?,?,?,?,?,?,?,?,?)",
                          (user["id"], user["email"], f"{m['subject']} – {clinic}", _text(m, links), _html(m, links),
                           ics_text, now, now, f"{dedupe}:{user['id']}" if dedupe else None))
        except sqlite3.IntegrityError:
            pass
        else:
            if ON_QUEUED:
                ON_QUEUED(db)
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
            "mrn": f"NE-{int(appt['patient_id']):05d}",
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

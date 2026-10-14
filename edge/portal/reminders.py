"""Reminder rules, evaluated every few minutes by `neuroedge-reminders.timer` (manage.py run-reminders).

Each rule is idempotent through a dedupe key, so running the job twice (or after downtime) never sends a message
twice; repeating reminders use a time bucket in the key (e.g. one "studies waiting" digest per 24 h).
Thresholds come from the department settings. `now` is injectable for tests.
"""

from __future__ import annotations

import datetime as dt
import time

from edge.portal import notify, scheduling


def _hours(db, key: str) -> float:
    return float(db.setting(key))


def _ago(seconds: float) -> str:
    h = seconds / 3600
    return f"{h:.0f} h" if h < 48 else f"{h / 24:.0f} days"


def unread_studies(db, now: float) -> int:
    """Rule 1: studies not opened past the target -> one digest to every radiologist (per 24 h bucket);
    past the escalation threshold -> administrators too."""
    normal, urgent, esc = (_hours(db, k) * 3600 for k in ("unread_hours", "unread_hours_urgent", "escalate_hours"))
    rows = db.q("SELECT id, created, priority FROM cases WHERE status = 'awaiting_review'")
    late = [r for r in rows if now - r["created"] >= (urgent if r["priority"] >= 2 else normal)]
    if not late:
        return 0
    bucket = int(now // 86400)
    oldest = min(r["created"] for r in late)
    ctx = {"n": len(late), "hours": db.setting("unread_hours"), "oldest": notify.fmt_when(db, oldest)}
    sent = 0
    for d in db.q("SELECT * FROM users WHERE role = 'doctor' AND active = 1"):
        sent += notify.notify(db, d, "doc_unread", ctx, "/doctor/", dedupe=f"unread:{bucket}", now=now)
    very = [r for r in rows if now - r["created"] >= esc]
    if very:
        for a in db.q("SELECT * FROM users WHERE role = 'admin' AND active = 1"):
            sent += notify.notify(db, a, "admin_escalation", {"n": len(very), "hours": db.setting("escalate_hours")},
                                  "/doctor/stats", dedupe=f"escalate:{bucket}", now=now)
    return sent


def unsigned_studies(db, now: float) -> int:
    """Rule 2: opened (in review) but not signed past the threshold -> the doctor who opened it, once per study/day."""
    lim = _hours(db, "unsigned_hours") * 3600
    sent = 0
    for c in db.q("SELECT c.*, u.name AS patient_name FROM cases c JOIN users u ON u.id = c.patient_id "
                  "WHERE c.status = 'in_review' AND c.doctor_id IS NOT NULL"):
        if now - c["updated"] >= lim:
            d = db.user(c["doctor_id"])
            if d and d["active"]:
                sent += notify.notify(db, d, "doc_unsigned", {"hours": db.setting("unsigned_hours"),
                                                              "age": _ago(now - c["updated"]), "patient": c["patient_name"]},
                                      f"/doctor/case/{c['id']}", dedupe=f"unsigned:{c['id']}:{int(now // 86400)}", now=now)
    return sent


def appointment_reminders(db, now: float) -> int:
    """Rule 6: reminders to the patient N hours before (default 24 h and 2 h)."""
    sent = 0
    offsets = sorted((float(x) for x in db.setting("remind_before_hours").split(",") if x.strip()), reverse=True)
    if not offsets:
        return 0
    rows = db.q("SELECT id, start_ts FROM appointments WHERE status = 'booked' AND start_ts > ? AND start_ts <= ?",
                (now, now + offsets[0] * 3600))
    for r in rows:
        a = scheduling.appointment(db, r["id"])
        due = [h for h in offsets if a["start_ts"] - now <= h * 3600]
        if not due:
            continue
        h = min(due)                                     # only the nearest due reminder (no 24 h mail at T-2 h)
        if a["created"] > a["start_ts"] - h * 3600:      # booked inside this window: the confirmation suffices
            continue
        ctx = notify.appt_ctx(db, a) | {"rel": "tomorrow" if h >= 12 else f"in {h:g} hours"}
        sent += notify.notify(db, db.user(a["patient_id"]), "appt_reminder", ctx, "/patient/appointments",
                              dedupe=f"remind:{a['id']}:{h:g}", now=now)
    return sent


def daily_agenda(db, now: float) -> int:
    """Rule 7: each doctor's consultations for today, once per day after `agenda_hour` local time."""
    zone = scheduling.tz(db)
    t = scheduling.local(now, zone)
    hh, mm = (int(x) for x in db.setting("agenda_hour").split(":"))
    if (t.hour, t.minute) < (hh, mm):
        return 0
    start = dt.datetime.combine(t.date(), dt.time(0), zone).timestamp()
    sent = 0
    for d in db.q("SELECT * FROM users WHERE role = 'doctor' AND active = 1"):
        rows = db.q("SELECT a.start_ts, a.room, p.name FROM appointments a JOIN users p ON p.id = a.patient_id "
                    "WHERE a.doctor_id = ? AND a.status = 'booked' AND a.start_ts >= ? AND a.start_ts < ? ORDER BY a.start_ts",
                    (d["id"], max(start, now - 3600), start + 86400))
        if rows:
            agenda = "\n".join(f"{scheduling.local(r['start_ts'], zone):%H:%M}  {r['name']}  ({r['room'] or 'clinic'})" for r in rows)
            sent += notify.notify(db, d, "doc_agenda", {"n": len(rows), "agenda": agenda}, "/doctor/schedule",
                                  dedupe=f"agenda:{t.date().isoformat()}", now=now)
    return sent


def unmarked_attendance(db, now: float) -> int:
    """Rule 8: slot ended N minutes ago and attendance not recorded -> remind the doctor once."""
    lim = float(db.setting("unmarked_after_min")) * 60
    sent = 0
    for r in db.q("SELECT id FROM appointments WHERE status IN ('booked','arrived') AND end_ts <= ?", (now - lim,)):
        a = scheduling.appointment(db, r["id"])
        sent += notify.notify(db, db.user(a["doctor_id"]), "doc_unmarked", notify.appt_ctx(db, a), "/doctor/schedule",
                              dedupe=f"unmarked:{a['id']}", now=now)
    return sent


RULES = (unread_studies, unsigned_studies, appointment_reminders, daily_agenda, unmarked_attendance)


def run_once(db, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    return {rule.__name__: rule(db, now) for rule in RULES}

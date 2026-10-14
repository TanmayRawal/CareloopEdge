"""Consultation scheduling: doctor working hours -> bookable slots -> appointments.

All times are stored as UNIX timestamps (UTC). Working hours and days off are defined in the department's local
time zone (setting `timezone`), so slots stay correct across DST changes. `now` is injectable for tests.
Double booking is impossible at the database level: a partial UNIQUE index on (doctor_id, start_ts) covers the
active statuses, so two concurrent bookings of one slot cannot both commit.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
from zoneinfo import ZoneInfo

from edge.portal.db import APPT_ACTIVE

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class BookingError(ValueError):
    pass


def tz(db) -> dt.tzinfo:
    try:
        return ZoneInfo(db.setting("timezone"))
    except Exception:                                   # unknown zone / no tz database: fail safe to UTC
        return dt.timezone.utc


def local(ts: float, zone: dt.tzinfo) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, zone)


def hours(db, doctor_id: int) -> list:
    return db.q("SELECT * FROM doctor_hours WHERE doctor_id = ? ORDER BY weekday, start_min", (doctor_id,))


def set_hours(db, doctor_id: int, rows: list[dict]) -> None:
    """Replace a doctor's weekly hours. rows: {weekday, start_min, end_min, slot_min, room}."""
    for r in rows:
        if not (0 <= r["weekday"] <= 6 and 0 <= r["start_min"] < r["end_min"] <= 1440 and 5 <= r["slot_min"] <= 120):
            raise BookingError("Invalid working hours.")
    by_day: dict[int, list] = {}
    for r in rows:
        by_day.setdefault(r["weekday"], []).append((r["start_min"], r["end_min"]))
    for spans in by_day.values():
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise BookingError("Working hours on the same day overlap.")
    with db.tx() as c:
        c.execute("DELETE FROM doctor_hours WHERE doctor_id = ?", (doctor_id,))
        for r in rows:
            c.execute("INSERT INTO doctor_hours (doctor_id,weekday,start_min,end_min,slot_min,room) VALUES (?,?,?,?,?,?)",
                      (doctor_id, r["weekday"], r["start_min"], r["end_min"], r["slot_min"], (r.get("room") or "")[:60] or None))


def days_off(db, doctor_id: int, from_day: dt.date | None = None) -> list:
    return db.q("SELECT * FROM doctor_days_off WHERE doctor_id = ? AND day >= ? ORDER BY day",
                (doctor_id, (from_day or dt.date.min).isoformat()))


def free_slots(db, doctor_id: int, now: float | None = None, days: int | None = None) -> list[dict]:
    """Bookable slots from now + lead time to the booking horizon, grouped later by the caller."""
    now = time.time() if now is None else now
    zone = tz(db)
    s = db.settings()
    lead = float(s["booking_lead_hours"]) * 3600
    days = int(s["booking_horizon_days"]) if days is None else days
    wk = {}
    for h in hours(db, doctor_id):
        wk.setdefault(h["weekday"], []).append(h)
    if not wk:
        return []
    today = local(now, zone).date()
    off = {r["day"] for r in days_off(db, doctor_id, today)}
    taken = {round(r["start_ts"]) for r in db.q(
        f"SELECT start_ts FROM appointments WHERE doctor_id = ? AND start_ts >= ? AND status IN {APPT_ACTIVE}",
        (doctor_id, now))}
    out = []
    for i in range(days + 1):
        day = today + dt.timedelta(days=i)
        if day.isoformat() in off:
            continue
        for h in wk.get(day.weekday(), []):
            m = h["start_min"]
            while m + h["slot_min"] <= h["end_min"]:
                start = dt.datetime.combine(day, dt.time(m // 60, m % 60), zone)
                ts = start.timestamp()
                if ts >= now + lead and round(ts) not in taken:
                    out.append({"start_ts": ts, "end_ts": ts + h["slot_min"] * 60, "room": h["room"],
                                "day": day, "label": start.strftime("%H:%M")})
                m += h["slot_min"]
    return out


def reporting_doctor(db, case) -> int | None:
    r = db.one("SELECT doctor_id FROM reports WHERE case_id = ? ORDER BY version DESC LIMIT 1", (case["id"],))
    return r["doctor_id"] if r else None


def active_for_case(db, case_id: str):
    return db.one(f"SELECT * FROM appointments WHERE case_id = ? AND status IN {APPT_ACTIVE} ORDER BY start_ts LIMIT 1",
                  (case_id,))


def book(db, patient, case, start_ts: float, reason: str = "", now: float | None = None) -> int:
    now = time.time() if now is None else now
    if case is None or case["patient_id"] != patient["id"]:
        raise BookingError("Study not found.")
    if case["status"] != "finalized":
        raise BookingError("A consultation can be booked once your report is ready.")
    doctor_id = reporting_doctor(db, case)
    if doctor_id is None:
        raise BookingError("No reporting doctor for this study.")
    if active_for_case(db, case["id"]):
        raise BookingError("You already have an appointment for this study. Reschedule it instead.")
    slot = next((s for s in free_slots(db, doctor_id, now) if abs(s["start_ts"] - float(start_ts)) < 1), None)
    if slot is None:
        raise BookingError("That time is no longer available. Please choose another slot.")
    try:
        with db.tx() as c:
            cur = c.execute("INSERT INTO appointments (case_id,patient_id,doctor_id,start_ts,end_ts,room,reason,status,"
                            "created,updated) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (case["id"], patient["id"], doctor_id, slot["start_ts"], slot["end_ts"], slot["room"],
                             (reason or "").strip()[:500] or None, "booked", now, now))
    except sqlite3.IntegrityError:
        raise BookingError("That time was just taken. Please choose another slot.") from None
    return int(cur.lastrowid)


def appointment(db, appt_id: int):
    return db.one("SELECT a.*, p.name AS patient_name, p.email AS patient_email, d.name AS doctor_name "
                  "FROM appointments a JOIN users p ON p.id = a.patient_id JOIN users d ON d.id = a.doctor_id "
                  "WHERE a.id = ?", (appt_id,))


def cancel(db, appt, by_user, now: float | None = None) -> None:
    now = time.time() if now is None else now
    if appt["status"] != "booked":
        raise BookingError("Only booked appointments can be cancelled.")
    if by_user["role"] == "patient":
        if appt["patient_id"] != by_user["id"]:
            raise BookingError("Appointment not found.")
        cutoff = float(db.setting("cancel_cutoff_hours")) * 3600
        if appt["start_ts"] - now < cutoff:
            raise BookingError(f"Appointments can be changed up to {db.setting('cancel_cutoff_hours')} hours before. "
                               "Please contact the department.")
    elif by_user["role"] == "doctor" and appt["doctor_id"] != by_user["id"]:
        raise BookingError("Appointment not found.")
    with db.tx() as c:
        c.execute("UPDATE appointments SET status='cancelled', cancelled_by=?, updated=? WHERE id=?",
                  (by_user["id"], now, appt["id"]))


def reschedule(db, appt, patient, new_start_ts: float, now: float | None = None) -> int:
    """Cancel + book atomically enough: the new slot is validated first, then the old one is released."""
    now = time.time() if now is None else now
    case = db.case(appt["case_id"])
    cancel(db, appt, patient, now)
    try:
        return book(db, patient, case, new_start_ts, appt["reason"] or "", now)
    except BookingError:
        with db.tx() as c:                               # restore the original booking
            c.execute("UPDATE appointments SET status='booked', cancelled_by=NULL, updated=? WHERE id=?", (now, appt["id"]))
        raise


TRANSITIONS = {"arrived": ("booked",), "completed": ("arrived", "booked"), "no_show": ("booked", "arrived")}


def mark(db, appt, status: str, doctor, now: float | None = None) -> None:
    now = time.time() if now is None else now
    if appt["doctor_id"] != doctor["id"]:
        raise BookingError("Appointment not found.")
    if status not in TRANSITIONS or appt["status"] not in TRANSITIONS[status]:
        raise BookingError("This change is not allowed.")
    if now < appt["start_ts"] - 15 * 60:
        raise BookingError("Attendance can be recorded from 15 minutes before the appointment.")
    with db.tx() as c:
        c.execute("UPDATE appointments SET status=?, marked_by=?, marked_at=?, updated=? WHERE id=?",
                  (status, doctor["id"], now, now, appt["id"]))


def no_show_count(db, patient_id: int) -> int:
    return db.one("SELECT COUNT(*) AS n FROM appointments WHERE patient_id = ? AND status = 'no_show'", (patient_id,))["n"]

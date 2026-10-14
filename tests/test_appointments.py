"""Scheduling, booking rules, reminders, notifications and the e-mail outbox (clock injected, fake SMTP)."""

import datetime as dt
import smtplib
from zoneinfo import ZoneInfo

import pytest

from edge.portal import mailer, notify, reminders, scheduling
from edge.portal.db import DB
from edge.portal.scheduling import BookingError

IST = ZoneInfo("Asia/Kolkata")
MON_0900 = dt.datetime(2026, 10, 12, 9, 0, tzinfo=IST).timestamp()      # a Monday


def at(day_offset: int, h: int, m: int = 0) -> float:
    return dt.datetime.combine(dt.date(2026, 10, 12) + dt.timedelta(days=day_offset), dt.time(h, m), IST).timestamp()


@pytest.fixture()
def w(tmp_path):
    db = DB(tmp_path / "p.db")
    doc = db.user(db.create_user("doc@x.test", "Dr Rao", "doctor", "h"))
    doc2 = db.user(db.create_user("doc2@x.test", "Dr Two", "doctor", "h"))
    admin = db.user(db.create_user("adm@x.test", "Admin", "admin", "h"))
    pat = db.user(db.create_user("pat@x.test", "Pat One", "patient", "h"))
    other = db.user(db.create_user("oth@x.test", "Other", "patient", "h"))
    cid = db.create_case(pat["id"], "s.png", "0" * 64, 10, None)
    db.add_report(cid, doc["id"], "glioma", True, "Mass in the left frontal lobe.", "", None, "0" * 64)
    scheduling.set_hours(db, doc["id"], [{"weekday": d, "start_min": 600, "end_min": 720, "slot_min": 20, "room": "Room 3"}
                                          for d in range(5)])
    return {"db": db, "doc": doc, "doc2": doc2, "admin": admin, "pat": pat, "other": other, "cid": cid}


# ----------------------------------------------------------------------------------------- slots & booking
def test_free_slots_respect_hours_lead_time_days_off_and_bookings(w):
    db = w["db"]
    s = scheduling.free_slots(db, w["doc"]["id"], now=MON_0900, days=1)
    starts = [x["start_ts"] for x in s]
    assert at(0, 10, 0) not in starts                         # inside the 2 h lead time
    assert at(0, 11, 0) in starts and at(0, 11, 40) in starts and at(0, 12, 0) not in starts
    assert len([x for x in s if x["day"] == dt.date(2026, 10, 13)]) == 6
    with db.tx() as c:
        c.execute("INSERT INTO doctor_days_off (doctor_id, day) VALUES (?, ?)", (w["doc"]["id"], "2026-10-13"))
    assert not [x for x in scheduling.free_slots(db, w["doc"]["id"], now=MON_0900, days=1) if x["day"] == dt.date(2026, 10, 13)]
    scheduling.book(db, w["pat"], db.case(w["cid"]), at(0, 11, 20), now=MON_0900)
    assert at(0, 11, 20) not in [x["start_ts"] for x in scheduling.free_slots(db, w["doc"]["id"], now=MON_0900, days=0)]


def test_booking_rules(w):
    db, case = w["db"], w["db"].case(w["cid"])
    with pytest.raises(BookingError):
        scheduling.book(db, w["other"], case, at(1, 10), now=MON_0900)          # not their study
    with pytest.raises(BookingError):
        scheduling.book(db, w["pat"], case, at(1, 10, 5), now=MON_0900)         # not a slot
    with pytest.raises(BookingError):
        scheduling.book(db, w["pat"], case, at(5, 10), now=MON_0900)            # Saturday: no hours
    aid = scheduling.book(db, w["pat"], case, at(1, 10), "Discuss results", now=MON_0900)
    a = scheduling.appointment(db, aid)
    assert a["doctor_id"] == w["doc"]["id"] and a["room"] == "Room 3" and a["status"] == "booked"
    with pytest.raises(BookingError):
        scheduling.book(db, w["pat"], case, at(1, 10, 20), now=MON_0900)        # one active booking per study


def test_unique_index_blocks_double_booking_race(w):
    db = w["db"]
    import sqlite3
    with db.tx() as c:
        c.execute("INSERT INTO appointments (case_id,patient_id,doctor_id,start_ts,end_ts,status,created,updated) "
                  "VALUES (?,?,?,?,?,'booked',0,0)", (w["cid"], w["pat"]["id"], w["doc"]["id"], at(1, 10), at(1, 10, 20)))
    with pytest.raises(sqlite3.IntegrityError):
        with db.tx() as c:
            c.execute("INSERT INTO appointments (case_id,patient_id,doctor_id,start_ts,end_ts,status,created,updated) "
                      "VALUES (?,?,?,?,?,'booked',0,0)", (w["cid"], w["other"]["id"], w["doc"]["id"], at(1, 10), at(1, 10, 20)))


def test_booking_needs_a_finalized_study(w):
    db = w["db"]
    cid = db.create_case(w["pat"]["id"], "b.png", "1" * 64, 10, None)
    with pytest.raises(BookingError, match="report is ready"):
        scheduling.book(db, w["pat"], db.case(cid), at(1, 10), now=MON_0900)


def test_cancel_cutoff_reschedule_and_marking(w):
    db = w["db"]
    aid = scheduling.book(db, w["pat"], db.case(w["cid"]), at(1, 10), now=MON_0900)
    a = scheduling.appointment(db, aid)
    with pytest.raises(BookingError, match="hours before"):
        scheduling.cancel(db, a, w["pat"], now=at(1, 9))                        # inside the 2 h cutoff
    new = scheduling.reschedule(db, a, w["pat"], at(2, 11), now=MON_0900)
    assert scheduling.appointment(db, aid)["status"] == "cancelled"
    b = scheduling.appointment(db, new)
    assert b["start_ts"] == at(2, 11)
    with pytest.raises(BookingError):
        scheduling.mark(db, b, "arrived", w["doc2"], now=at(2, 11))             # not their appointment
    with pytest.raises(BookingError):
        scheduling.mark(db, b, "arrived", w["doc"], now=MON_0900)               # too early
    scheduling.mark(db, b, "arrived", w["doc"], now=at(2, 11, 2))
    scheduling.mark(db, scheduling.appointment(db, new), "completed", w["doc"], now=at(2, 11, 18))
    with pytest.raises(BookingError):
        scheduling.mark(db, scheduling.appointment(db, new), "no_show", w["doc"], now=at(2, 12))


def test_overlapping_hours_rejected(w):
    with pytest.raises(BookingError, match="overlap"):
        scheduling.set_hours(w["db"], w["doc"]["id"], [{"weekday": 0, "start_min": 600, "end_min": 700, "slot_min": 20},
                                                       {"weekday": 0, "start_min": 690, "end_min": 760, "slot_min": 20}])


# ----------------------------------------------------------------------------------------- notifications
def test_notifications_dedupe_and_no_clinical_content(w):
    db = w["db"]
    aid = scheduling.book(db, w["pat"], db.case(w["cid"]), at(1, 10), now=MON_0900)
    notify.appointment_event(db, aid, "booked", now=MON_0900)
    notify.appointment_event(db, aid, "booked", now=MON_0900)                   # duplicate event: ignored
    pat_mail = db.q("SELECT * FROM outbox WHERE user_id = ?", (w["pat"]["id"],))
    assert len(pat_mail) == 1 and pat_mail[0]["ics"].startswith("BEGIN:VCALENDAR")
    assert "Dr Rao" in pat_mail[0]["text"] and "Room 3" in pat_mail[0]["text"]
    notify.notify(db, w["pat"], "report_ready", {}, "/patient/case/x", dedupe="report:x")
    for m in db.q("SELECT * FROM outbox"):
        body = (m["subject"] + m["text"] + (m["html"] or "")).lower()
        assert "glioma" not in body and "frontal" not in body and "mass" not in body
    n, rows = notify.unread(db, w["pat"]["id"])
    assert n == 2
    notify.mark_read(db, w["pat"]["id"])
    assert notify.unread(db, w["pat"]["id"])[0] == 0


def test_email_opt_out_keeps_in_app(w):
    db = w["db"]
    with db.tx() as c:
        c.execute("UPDATE users SET notify_email = 0 WHERE id = ?", (w["pat"]["id"],))
    notify.notify(db, db.user(w["pat"]["id"]), "report_ready", {}, "/x")
    assert notify.unread(db, w["pat"]["id"])[0] == 1
    assert not db.q("SELECT * FROM outbox WHERE user_id = ?", (w["pat"]["id"],))


# ----------------------------------------------------------------------------------------- reminders
def test_appointment_reminders_fire_once_each(w):
    db = w["db"]
    aid = scheduling.book(db, w["pat"], db.case(w["cid"]), at(2, 10), now=MON_0900)   # Wed 10:00
    kinds = lambda: [r["kind"] for r in db.q("SELECT kind FROM notifications WHERE user_id = ? ORDER BY id",  # noqa: E731
                                            (w["pat"]["id"],))]
    reminders.run_once(db, now=at(0, 12))                                       # 46 h before: nothing
    assert kinds().count("appt_reminder") == 0
    reminders.run_once(db, now=at(1, 10, 30))                                   # 23.5 h before: the 24 h reminder
    reminders.run_once(db, now=at(1, 11))
    assert kinds().count("appt_reminder") == 1
    reminders.run_once(db, now=at(2, 8, 30))                                    # 1.5 h before: the 2 h reminder
    reminders.run_once(db, now=at(2, 9))
    assert kinds().count("appt_reminder") == 2
    reminders.run_once(db, now=at(2, 11, 30))                                   # ended 70 min ago, unmarked
    reminders.run_once(db, now=at(2, 12))
    doc_kinds = [r["kind"] for r in db.q("SELECT kind FROM notifications WHERE user_id = ?", (w["doc"]["id"],))]
    assert doc_kinds.count("doc_unmarked") == 1 and doc_kinds.count("doc_agenda") == 1
    scheduling.mark(db, scheduling.appointment(db, aid), "no_show", w["doc"], now=at(2, 12, 5))
    notify.appointment_event(db, aid, "no_show", now=at(2, 12, 5))
    assert "appt_no_show" in kinds()


def test_unread_study_digest_and_escalation(w):
    db = w["db"]
    cid = db.create_case(w["pat"]["id"], "n.png", "2" * 64, 10, None)
    created = db.case(cid)["created"]
    count = lambda uid, k: db.one("SELECT COUNT(*) AS n FROM notifications WHERE user_id = ? AND kind = ?", (uid, k))["n"]  # noqa: E731
    reminders.run_once(db, now=created + 3 * 3600)
    assert count(w["doc"]["id"], "doc_unread") == 0
    reminders.run_once(db, now=created + 5 * 3600)
    reminders.run_once(db, now=created + 5.5 * 3600)
    assert count(w["doc"]["id"], "doc_unread") == 1 and count(w["doc2"]["id"], "doc_unread") == 1
    assert count(w["admin"]["id"], "admin_escalation") == 0
    reminders.run_once(db, now=created + 49 * 3600)
    assert count(w["admin"]["id"], "admin_escalation") == 1
    assert count(w["doc"]["id"], "doc_unread") >= 2                             # repeated the next day


def test_unsigned_study_reminder(w):
    db = w["db"]
    cid = db.create_case(w["pat"]["id"], "u.png", "3" * 64, 10, None)
    db.set_case(cid, status="in_review", doctor_id=w["doc2"]["id"])
    t = db.case(cid)["updated"]
    reminders.run_once(db, now=t + 25 * 3600)
    assert db.one("SELECT COUNT(*) AS n FROM notifications WHERE user_id = ? AND kind = 'doc_unsigned'", (w["doc2"]["id"],))["n"] == 1


# ----------------------------------------------------------------------------------------- mailer
class FakeSMTP:
    sent, fail = [], None

    def __init__(self, host, port):
        if FakeSMTP.fail == "connect":
            raise OSError("connection refused")

    def login(self, u, p):
        pass

    def send_message(self, m):
        if FakeSMTP.fail == "send":
            raise smtplib.SMTPServerDisconnected("dropped")
        FakeSMTP.sent.append(m)

    def quit(self):
        pass


def test_mailer_sends_backs_off_and_gives_up(w):
    db = w["db"]
    notify.notify(db, w["pat"], "report_ready", {}, "/x")
    cfg = {"NEUROEDGE_SMTP_HOST": "smtp.test", "NEUROEDGE_SMTP_USER": "u", "NEUROEDGE_SMTP_PASSWORD": "p"}
    assert mailer.flush(db, cfg={})["deferred"] == 1                            # unconfigured: stays queued
    assert db.one("SELECT status, last_error FROM outbox")["last_error"] == "SMTP not configured"
    FakeSMTP.sent, FakeSMTP.fail = [], "connect"
    assert mailer.flush(db, cfg=cfg, smtp_factory=FakeSMTP)["deferred"] == 1
    row = db.one("SELECT * FROM outbox")
    assert row["status"] == "queued" and row["attempts"] == 1 and row["next_try"] > row["created"]
    assert mailer.flush(db, now=row["created"] + 1, cfg=cfg, smtp_factory=FakeSMTP)["deferred"] == 0   # not due yet
    FakeSMTP.fail = None
    assert mailer.flush(db, now=row["next_try"] + 1, cfg=cfg, smtp_factory=FakeSMTP)["sent"] == 1
    assert db.one("SELECT status FROM outbox")["status"] == "sent" and FakeSMTP.sent[0]["To"] == "pat@x.test"
    notify.notify(db, w["pat"], "reupload", {"message": "x"}, "/y")
    FakeSMTP.fail = "send"
    t = db.one("SELECT next_try FROM outbox WHERE status = 'queued'")["next_try"]
    for _ in range(mailer.MAX_ATTEMPTS):
        mailer.flush(db, now=t, cfg=cfg, smtp_factory=FakeSMTP)
        t += 10 ** 6
    assert db.one("SELECT status FROM outbox ORDER BY id DESC")["status"] == "failed"


def test_mailer_file_mode_writes_eml_with_calendar(w, tmp_path):
    db = w["db"]
    aid = scheduling.book(db, w["pat"], db.case(w["cid"]), at(1, 10), now=MON_0900)
    notify.appointment_event(db, aid, "booked", now=MON_0900)
    out = mailer.flush(db, now=MON_0900 + 1, cfg={"NEUROEDGE_SMTP": f"file:{tmp_path / 'mail'}"})
    assert out["sent"] == 2
    import email
    cals = [part.get_payload(decode=True).decode() for p in (tmp_path / "mail").glob("*.eml")
            for part in email.message_from_bytes(p.read_bytes()).walk() if part.get_content_type() == "text/calendar"]
    assert len(cals) == 1 and "BEGIN:VCALENDAR" in cals[0] and "DTSTART:20261013T043000Z" in cals[0]

"""Portal v2 over HTTP: every page renders for its role, booking flow, attendance, notifications, admin settings,
access control on the new routes, and static-asset caching."""

import re
import time

from test_portal import PW, csrf, env, login, register, upload  # noqa: F401  (env is a fixture)

from edge.portal import scheduling


def finalize(env, p, d):
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    tok = csrf(d, f"/doctor/case/{cid}")
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": tok, "diagnosis": "meningioma", "agrees": "yes",
                                             "findings": "Extra-axial mass.", "password": PW})
    assert env["db"].case(cid)["status"] == "finalized"
    return cid


def all_day_hours(env):
    doc = env["db"].user_by_email("doc@clinic.test")
    scheduling.set_hours(env["db"], doc["id"], [{"weekday": w, "start_min": 0, "end_min": 1440, "slot_min": 30, "room": "OPD 2"}
                                               for w in range(7)])
    return doc


def test_every_page_renders_for_its_role(env):
    p, d, a = (env["app"].test_client() for _ in range(3))
    cid = finalize(env, p, d)
    login(a, "admin@clinic.test")
    for client, paths in ((p, ["/patient/", "/patient/appointments", f"/patient/case/{cid}", f"/patient/book/{cid}",
                               "/patient/upload", "/account", "/notifications/"]),
                          (d, ["/doctor/", "/doctor/?status=finalized", "/doctor/schedule", "/doctor/schedule?view=week",
                               "/doctor/availability", "/doctor/stats", "/account", "/notifications/", f"/doctor/case/{cid}"]),
                          (a, ["/admin/users", "/admin/users?role=patient", "/admin/audit", "/admin/audit?cat=view", "/admin/audit?cat=clinical&user=Pat",
                               "/admin/system", "/admin/outbox", "/admin/settings", "/doctor/stats", "/account"])):
        for path in paths:
            r = client.get(path)
            assert r.status_code == 200, (path, r.status_code)


def test_patient_books_reschedules_cancels_and_gets_messages(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    cid = finalize(env, p, d)
    db = env["db"]
    assert any(n["kind"] == "report_ready" for n in db.q("SELECT kind FROM notifications"))
    page = p.get(f"/patient/case/{cid}").get_data(as_text=True)
    assert "Book a consultation" in page
    assert p.get(f"/patient/book/{cid}").status_code == 200                 # no hours yet: empty state, not an error
    doc = all_day_hours(env)
    page = p.get(f"/patient/book/{cid}").get_data(as_text=True)
    slots = re.findall(r'name="start_ts" value="([0-9.]+)"', page)
    assert slots
    tok = csrf(p, f"/patient/book/{cid}")
    r = p.post(f"/patient/book/{cid}", data={"csrf": tok, "start_ts": slots[0], "reason": "Discuss"})
    assert r.status_code == 302 and "/confirmed" in r.headers["Location"]
    a = db.one("SELECT * FROM appointments")
    assert a["status"] == "booked" and a["doctor_id"] == doc["id"] and a["room"] == "OPD 2"
    assert p.get(f"/patient/appointments/{a['id']}.ics").get_data(as_text=True).startswith("BEGIN:VCALENDAR")
    mails = db.q("SELECT * FROM outbox WHERE to_email = 'pat@x.test'")
    assert any(m["ics"] for m in mails)
    assert all("meningioma" not in (m["text"] + (m["html"] or "")).lower() for m in mails)
    # reschedule to a later slot, then cancel
    later = re.findall(r'name="start_ts" value="([0-9.]+)"', p.get(f"/patient/appointments/{a['id']}/reschedule").get_data(as_text=True))
    later = [s for s in later if float(s) > a["start_ts"] + 7200]
    p.post(f"/patient/appointments/{a['id']}/reschedule", data={"csrf": tok, "start_ts": later[0]})
    b = db.one("SELECT * FROM appointments WHERE status = 'booked'")
    assert b["id"] != a["id"] and db.one("SELECT status FROM appointments WHERE id = ?", (a["id"],))["status"] == "cancelled"
    p.post(f"/patient/appointments/{b['id']}/cancel", data={"csrf": tok})
    assert db.one("SELECT status FROM appointments WHERE id = ?", (b["id"],))["status"] == "cancelled"
    kinds = [n["kind"] for n in db.q("SELECT kind FROM notifications WHERE user_id = ?", (b["patient_id"],))]
    assert {"appt_booked", "appt_rescheduled", "appt_cancelled"} <= set(kinds)


def test_other_patient_cannot_touch_appointments(env):
    p, d, o = (env["app"].test_client() for _ in range(3))
    cid = finalize(env, p, d)
    all_day_hours(env)
    slot = re.findall(r'name="start_ts" value="([0-9.]+)"', p.get(f"/patient/book/{cid}").get_data(as_text=True))[0]
    p.post(f"/patient/book/{cid}", data={"csrf": csrf(p, f"/patient/book/{cid}"), "start_ts": slot})
    aid = env["db"].one("SELECT id FROM appointments")["id"]
    register(o, "other@x.test", "Other Person")
    tok = csrf(o, "/patient/")
    assert o.get(f"/patient/book/{cid}").status_code == 404
    assert o.get(f"/patient/appointments/{aid}.ics").status_code == 404
    assert o.post(f"/patient/appointments/{aid}/cancel", data={"csrf": tok}).status_code == 404
    assert o.get("/doctor/schedule").status_code in (302, 403)
    assert env["db"].one("SELECT status FROM appointments")["status"] == "booked"


def test_doctor_marks_attendance_and_no_show_notifies_patient(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    cid = finalize(env, p, d)
    db = env["db"]
    doc = db.user_by_email("doc@clinic.test")
    pat = db.user_by_email("pat@x.test")
    now = time.time()
    with db.tx() as c:                                  # an appointment that started 10 minutes ago
        c.execute("INSERT INTO appointments (case_id,patient_id,doctor_id,start_ts,end_ts,status,created,updated) "
                  "VALUES (?,?,?,?,?,'booked',?,?)", (cid, pat["id"], doc["id"], now - 600, now + 600, now - 86400, now))
    aid = db.one("SELECT id FROM appointments")["id"]
    page = d.get("/doctor/schedule").get_data(as_text=True)
    assert "Pat One" in page and "No-show" in page
    tok = csrf(d, "/doctor/schedule")
    d.post(f"/doctor/appointments/{aid}/mark", data={"csrf": tok, "status": "no_show"})
    assert db.one("SELECT status FROM appointments")["status"] == "no_show"
    assert db.one("SELECT COUNT(*) AS n FROM notifications WHERE user_id = ? AND kind = 'appt_no_show'", (pat["id"],))["n"] == 1
    assert "Book again" in p.get("/patient/appointments").get_data(as_text=True)


def test_notifications_bell_and_mark_read(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    finalize(env, p, d)
    page = p.get("/patient/").get_data(as_text=True)
    assert 'class="bell-n"' in page and "Your report is ready" in page
    p.post("/notifications/read", data={"csrf": csrf(p, "/patient/")})
    assert 'class="bell-n"' not in p.get("/patient/").get_data(as_text=True)


def test_admin_settings_validate_and_save(env):
    a = env["app"].test_client()
    login(a, "admin@clinic.test")
    tok = csrf(a, "/admin/settings")
    form = {k: v for k, v in env["db"].settings().items()}
    bad = form | {"csrf": tok, "timezone": "Mars/Base", "unread_hours": "-1"}
    r = a.post("/admin/settings", data=bad)
    assert r.status_code == 400 and env["db"].setting("timezone") == "Asia/Kolkata"
    a.post("/admin/settings", data=form | {"csrf": tok, "unread_hours": "6", "clinic_name": "City Hospital Radiology"})
    assert env["db"].setting("unread_hours") == "6"
    assert "City Hospital Radiology" in a.get("/admin/system").get_data(as_text=True)


def test_doctor_sets_availability_over_http(env):
    d = env["app"].test_client()
    login(d, "doc@clinic.test")
    tok = csrf(d, "/doctor/availability")
    d.post("/doctor/availability", data={"csrf": tok, "form": "hours", "on_0": "1", "start_0_a": "09:00", "end_0_a": "12:00",
                                         "start_0_b": "14:00", "end_0_b": "16:00", "slot_min": "30", "room": "OPD 1"})
    rows = env["db"].q("SELECT * FROM doctor_hours")
    assert [(r["start_min"], r["end_min"]) for r in rows] == [(540, 720), (840, 960)] and rows[0]["room"] == "OPD 1"
    r = d.post("/doctor/availability", data={"csrf": tok, "form": "hours", "on_1": "1", "start_1_a": "10:00", "end_1_a": "09:00",
                                             "slot_min": "30"})
    assert r.status_code == 400 and len(env["db"].q("SELECT * FROM doctor_hours")) == 2   # invalid: nothing replaced
    d.post("/doctor/availability", data={"csrf": tok, "form": "off_add", "day": "2030-01-02", "note": "Leave"})
    assert env["db"].one("SELECT day FROM doctor_days_off")["day"] == "2030-01-02"


def test_static_assets_versioned_and_cached(env):
    c = env["app"].test_client()
    page = c.get("/login").get_data(as_text=True)
    css = re.search(r'href="(/static/portal\.css\?v=[0-9a-f]+)"', page).group(1)
    r = c.get(css)
    assert r.status_code == 200 and "immutable" in r.headers["Cache-Control"]
    assert c.get("/login").headers["Cache-Control"] == "no-store"

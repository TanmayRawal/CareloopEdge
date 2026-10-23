"""Analytics: care-loop closure, distance and district breakdowns, consultations, outreach and reading figures are
computed from recorded events with the right denominators; the page renders for doctors and administrators."""

import datetime as dt
import time

import pytest

pytest.importorskip("flask")

from test_portal import PW, csrf, env, login, register, upload  # noqa: E402,F401  (env is a fixture)

from edge.portal import analytics  # noqa: E402

DAY = 86400


def patient(db, name, km=None, district=None, status="ok"):
    uid = db.create_user(None, name, "patient", None, {"distance_km": km, "district": district})
    db.conn.execute("UPDATE patient_profiles SET contact_status = ? WHERE user_id = ?", (status, uid))
    return uid


def plan(db, pid, status, due, proposed, closed=None, kind="consultation", case_id=None, approved=True):
    db.conn.execute("INSERT INTO care_plans (patient_id, case_id, kind, due_date, status, proposed_at, approved_at, "
                    "closed_at, updated) VALUES (?,?,?,?,?,?,?,?,?)",
                    (pid, case_id, kind, due.isoformat(), status, proposed, proposed + 60 if approved else None, closed,
                     proposed))


def test_care_loop_figures(env):
    db, now = env["db"], time.time()
    today = dt.date.today()
    near, far, gone = patient(db, "Near", 5, "Alpha"), patient(db, "Far", 120, "Beta"), patient(db, "Gone", 70, "Beta",
                                                                                              "wrong_number")
    # near: completed on time; far: completed late; gone: lost; plus one overdue open plan
    plan(db, near, "completed", today, now - 10 * DAY, now - 9 * DAY)
    plan(db, far, "completed", today - dt.timedelta(days=20), now - 30 * DAY, now - 2 * DAY, kind="repeat_scan")
    plan(db, gone, "lost", today - dt.timedelta(days=40), now - 50 * DAY, now - 1 * DAY, kind="referral")
    plan(db, near, "approved", today - dt.timedelta(days=5), now - 8 * DAY)
    # an appointment no-show that was rebooked, and one attended
    cid = db.create_case(far, "s.png", "0" * 64, 10, "")
    for st, start in (("no_show", now - 3 * DAY), ("completed", now - 1 * DAY)):
        db.conn.execute("INSERT INTO appointments (case_id, patient_id, doctor_id, start_ts, end_ts, status, created, updated) "
                        "VALUES (?, ?, 1, ?, ?, ?, ?, ?)", (cid, far, start, start + 1200, st, start - 2 * DAY, start))
    db.conn.execute("INSERT INTO appointments (case_id, patient_id, doctor_id, start_ts, end_ts, status, created, updated) "
                    "VALUES (?, ?, 1, ?, ?, 'booked', ?, ?)", (cid, far, now + 5 * DAY, now + 5 * DAY + 1200, now - 2 * DAY, now))
    for outcome in ("no_answer", "no_answer", "reached_confirmed"):
        db.conn.execute("INSERT INTO contacts (patient_id, channel, outcome, created) VALUES (?, 'phone', ?, ?)",
                        (gone, outcome, now - DAY))
    db.conn.commit()

    a = analytics.compute(db, 90, now=now)
    L = a["loop"]
    assert (L["closure"]["n"], L["closure"]["d"]) == (2, 3) and L["lost"]["n"] == 1
    assert (L["on_time"]["n"], L["on_time"]["d"]) == (1, 2)          # far's plan closed after its due date
    assert L["overdue"] == 1 and L["unreachable"] == 1
    assert (L["no_show"]["n"], L["no_show"]["d"]) == (1, 2)
    assert (a["consult"]["rebooked"]["n"], a["consult"]["rebooked"]["d"]) == (1, 1) and a["consult"]["upcoming"] == 1
    assert (a["outreach"]["reach"]["n"], a["outreach"]["reach"]["d"]) == (1, 3)
    bands = {r["band"]: r for r in a["distance"]}
    assert bands["< 10 km"]["completed"] == 1 and bands["< 10 km"]["open"] == 1
    assert bands["≥ 100 km"]["completed"] == 1 and bands["60–100 km"]["lost"] == 1
    beta = next(r for r in a["districts"] if r["district"] == "Beta")
    assert beta["patients"] == 2 and (beta["closure"]["n"], beta["closure"]["d"]) == (1, 2)
    kinds = {r["kind"]: r for r in a["by_kind"]}
    assert kinds["referral"]["lost"] == 1 and kinds["repeat_scan"]["completed"] == 1
    # the 30-day window drops the plan proposed 50 days ago from "by kind" but keeps closures inside the window
    assert "referral" not in {r["kind"] for r in analytics.compute(db, 30, now=now)["by_kind"]}


def test_reading_and_ai_figures_and_page(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": csrf(d, f"/doctor/case/{cid}"), "diagnosis": "glioma",
                                             "agrees": "no", "findings": "Mass.", "password": PW, "fu_kind": ""})
    a = analytics.compute(env["db"], 30)
    assert a["reading"]["received"] == 1 and a["reading"]["reported"] == 1
    assert a["reading"]["within_sla"]["n"] == 1 and a["reading"]["backlog"] == 0
    ai = a["ai"]
    assert (ai["agreement"]["n"], ai["agreement"]["d"]) == (0, 1)          # stub AI said meningioma, report says glioma
    assert ai["confusion"]["meningioma"]["glioma"] == 1 and ai["doctor_said"]["no"] == 1
    for period in ("30", "90", "365", "all", "bogus"):
        page = d.get(f"/doctor/stats?period={period}").get_data(as_text=True)
        assert "Care-loop funnel" in page and "Distance to care" in page and "AI suggestion vs" in page
    adm = env["app"].test_client()
    login(adm, "admin@clinic.test")
    assert adm.get("/doctor/stats").status_code == 200

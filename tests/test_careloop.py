"""CareLoop follow-up: risk score, care plans and the doctor-approval gate, referrals, contact log, e-mail replies,
walk-in registration, the patient timeline, documents, data export, the hash-chained audit log and the v2 -> v3
migration. Unit tests use an injected clock; HTTP tests reuse the stub-engine portal from test_portal."""

import datetime as dt
import io
import json
import re
import sqlite3
import time
import zipfile
from pathlib import Path

import pytest
from test_portal import PNG, PW, csrf, env, login, register, upload  # noqa: F401  (env is a fixture)
from test_portal_flows import all_day_hours, finalize

from edge.portal import careplan, outreach, reminders, risk, scheduling, tokens
from edge.portal.db import DB, MIGRATIONS


@pytest.fixture()
def w(tmp_path):
    db = DB(tmp_path / "p.db")
    doc = db.user(db.create_user("doc@x.test", "Dr Rao", "doctor", "h"))
    admin = db.user(db.create_user("adm@x.test", "Coordinator", "admin", "h"))
    pat = db.user(db.create_user("pat@x.test", "Pat One", "patient", "h", {"distance_km": 70}))
    walk = db.user(db.create_user(None, "Walk In", "patient", None, {"phone": "98000"}, created_by=admin["id"],
                                  email_consent=False))
    return {"db": db, "doc": doc, "admin": admin, "pat": pat, "walk": walk}


def today(db):
    return scheduling.local(time.time(), scheduling.tz(db)).date()


def due(db, days):
    return (today(db) + dt.timedelta(days=days)).isoformat()


# ------------------------------------------------------------------------------------------- risk engine
def test_risk_score_is_explained_and_ordered():
    calm = risk.score({"overdue_days": 0, "due_soon": False})
    hot = risk.score({"overdue_days": 30, "no_shows": 2, "failed_contacts": 3, "contact_status": "wrong_number",
                      "referral_stale": True, "distance_km": 80})
    assert calm["score"] == 0 and calm["band"] == "low"
    assert hot["score"] == 100 and hot["band"] == "high"
    labels = [t for t, _ in hot["factors"]]
    assert "Overdue by 30 days" in labels and "2 missed appointments" in labels and "Lives 80 km away" in labels
    booked = risk.score({"overdue_days": 10, "booked": True, "contacted": True})
    assert booked["score"] < risk.score({"overdue_days": 10, "contacted": True})["score"]


def test_risk_engine_never_reads_clinical_data():
    src = Path(risk.__file__).read_text(encoding="utf-8").split('"""', 2)[2]      # code only, not the docstring
    for word in ("ai_results", "reports", "final_diagnosis", "findings", "result_json", "prediction"):
        assert word not in src, word


def test_contact_log_feeds_the_risk_score(w):
    db = w["db"]
    careplan.create(db, w["pat"]["id"], "repeat_scan", due(db, -5), "", w["doc"])
    before = risk.scores(db)[w["pat"]["id"]]
    for _ in range(2):
        outreach.log_contact(db, w["pat"]["id"], w["admin"], "phone", "no_answer")
    after = risk.scores(db)[w["pat"]["id"]]
    assert after["signals"]["failed_contacts"] == 2 and after["score"] > before["score"]
    outreach.log_contact(db, w["pat"]["id"], w["admin"], "phone", "reached_confirmed")
    assert risk.scores(db)[w["pat"]["id"]]["signals"]["failed_contacts"] == 0      # streak resets on success
    outreach.log_contact(db, w["pat"]["id"], w["admin"], "phone", "wrong_number")
    assert db.profile(w["pat"]["id"])["contact_status"] == "wrong_number"


# ------------------------------------------------------------------------------------- plans and approval
def test_doctor_plans_start_approved_coordinator_plans_need_approval(w):
    db = w["db"]
    p1 = careplan.get(db, careplan.create(db, w["pat"]["id"], "consultation", due(db, 14), "", w["doc"]))
    assert p1["status"] == "approved" and p1["approved_by"] == w["doc"]["id"]
    careplan.close(db, p1, "cancelled", "", w["doc"])
    p2 = careplan.get(db, careplan.create(db, w["pat"]["id"], "consultation", due(db, 14), "", w["admin"]))
    assert p2["status"] == "proposed"
    with pytest.raises(outreach.OutreachError):                       # no outreach before a doctor approves
        outreach.send_reminder(db, p2, w["admin"])
    with pytest.raises(careplan.PlanError):                           # a coordinator cannot approve
        careplan.approve(db, p2, w["admin"])
    careplan.approve(db, p2, w["doc"])
    p2 = careplan.get(db, p2["id"])
    assert outreach.send_reminder(db, p2, w["admin"], "Please call us") is True      # e-mail queued
    assert db.one("SELECT COUNT(*) AS n FROM outbox WHERE to_email = 'pat@x.test'")["n"] == 1
    assert db.one("SELECT outcome FROM contacts WHERE patient_id = ?", (w["pat"]["id"],))["outcome"] == "message_sent"


def test_reminder_without_email_is_in_app_only(w):
    db = w["db"]
    p = careplan.get(db, careplan.create(db, w["walk"]["id"], "repeat_scan", due(db, 3), "", w["doc"]))
    assert outreach.send_reminder(db, p, w["admin"]) is False
    assert db.one("SELECT COUNT(*) AS n FROM outbox")["n"] == 0
    assert db.one("SELECT COUNT(*) AS n FROM notifications WHERE user_id = ?", (w["walk"]["id"],))["n"] == 1


def test_plans_complete_automatically_and_go_lost_after_silence(w):
    db = w["db"]
    careplan.create(db, w["pat"]["id"], "consultation", due(db, 7), "", w["doc"])
    assert careplan.complete_matching(db, w["pat"]["id"], ("consultation",), "done") == 1
    assert careplan.active(db, w["pat"]["id"]) is None
    pid = careplan.create(db, w["walk"]["id"], "repeat_scan", due(db, -70), "", w["doc"])
    reminders.lost_to_follow_up(db, time.time())
    assert careplan.get(db, pid)["status"] == "lost"


# ------------------------------------------------------------------------------------------------ referrals
def test_referral_transitions(w, env):
    db = w["db"]
    rid = careplan.create_referral(db, w["pat"]["id"], "RCC", "Neurosurgery", "urgent", "Mass", w["doc"])
    ref = careplan.referral(db, rid)
    with pytest.raises(careplan.PlanError):
        careplan.set_referral_status(db, ref, "sent")                 # not signed yet
    with env["app"].test_request_context():
        with pytest.raises(careplan.PlanError):
            careplan.sign_referral(db, env["store"], ref, w["admin"], "Clinic")      # only a doctor signs
        careplan.sign_referral(db, env["store"], ref, w["doc"], "Clinic")
    ref = careplan.referral(db, rid)
    assert ref["status"] == "signed" and env["store"].exists(*careplan.letter_kind(rid))
    careplan.set_referral_status(db, ref, "sent")
    with pytest.raises(careplan.PlanError):
        careplan.set_referral_status(db, careplan.referral(db, rid), "signed")      # no going back


# ------------------------------------------------------------------------------- tokens and e-mail replies
def test_tokens_are_single_use_and_expire(w):
    db = w["db"]
    raw = tokens.issue(db, "reset", w["pat"]["id"], now=1000)
    assert tokens.peek(db, "reset", raw, now=1001) is not None
    assert tokens.peek(db, "invite", raw, now=1001) is None            # wrong purpose
    assert tokens.consume(db, "reset", raw, now=1002) is not None
    assert tokens.consume(db, "reset", raw, now=1003) is None          # used
    old = tokens.issue(db, "reset", w["pat"]["id"], now=1000)
    assert tokens.peek(db, "reset", old, now=1000 + 3601) is None      # expired after 1 h
    a, b = tokens.issue(db, "invite", w["pat"]["id"], now=1000), tokens.issue(db, "invite", w["pat"]["id"], now=1001)
    assert tokens.peek(db, "invite", a, now=1002) is None and tokens.peek(db, "invite", b, now=1002)   # newest wins
    assert raw not in json.dumps([dict(r) for r in db.q("SELECT * FROM tokens")])                     # only hashes stored


def test_appointment_reminder_links_confirm_and_decline(w):
    db = w["db"]
    cid = db.create_case(w["pat"]["id"], "s.png", "0" * 64, 10, None)
    db.add_report(cid, w["doc"]["id"], "glioma", True, "Mass.", "", None, "0" * 64)
    scheduling.set_hours(db, w["doc"]["id"], [{"weekday": d, "start_min": 0, "end_min": 1440, "slot_min": 30, "room": "R"}
                                              for d in range(7)])
    slot = scheduling.free_slots(db, w["doc"]["id"], days=3)[-1]
    aid = scheduling.book(db, w["pat"], db.case(cid), slot["start_ts"])
    acts = outreach.reminder_actions(db, scheduling.appointment(db, aid))
    yes = acts[0][1].split("/r/")[1].split("?")[0]
    no = acts[1][1].split("/r/")[1].split("?")[0]
    assert outreach.appointment_reply(db, "appt_no", yes) is None          # a yes-token cannot cancel
    assert outreach.appointment_reply(db, "appt_yes", yes)["result"] == "confirmed"
    assert outreach.appointment_reply(db, "appt_yes", yes) is None         # single use
    assert outreach.appointment_reply(db, "appt_no", no)["result"] == "cancelled"
    assert scheduling.appointment(db, aid)["status"] == "cancelled"
    assert db.one("SELECT COUNT(*) AS n FROM notifications WHERE user_id = ? AND kind = 'admin_appt_declined'",
                  (w["admin"]["id"],))["n"] == 1


# ---------------------------------------------------------------------------------------------- audit chain
def test_audit_chain_detects_edits_and_deletions(w):
    db = w["db"]
    for i in range(5):
        db.audit(w["admin"]["id"], "login", detail={"i": i})
    assert db.verify_audit_chain() == {"ok": True, "n": 5, "bad_id": None}
    db.conn.execute("UPDATE audit SET detail = '{\"i\": 9}' WHERE id = 3")
    assert db.verify_audit_chain()["bad_id"] == 3
    db.conn.execute("UPDATE audit SET detail = '{\"i\": 2}' WHERE id = 3")
    assert db.verify_audit_chain()["ok"]
    db.conn.execute("DELETE FROM audit WHERE id = 2")
    assert db.verify_audit_chain() == {"ok": False, "n": 1, "bad_id": 3}


def test_migration_v2_to_v3_keeps_data_and_backfills_chain(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for sql in MIGRATIONS[:2]:
        for stmt in [s for s in sql.split(";") if s.strip()]:
            con.execute(stmt)
    con.execute("INSERT INTO schema_version VALUES (2)")
    con.execute("INSERT INTO users (id,email,name,role,pw_hash,created) VALUES (1,'p@x.test','Pat','patient','h',1)")
    con.execute("INSERT INTO patient_profiles (user_id, phone) VALUES (1, '123')")
    con.execute("INSERT INTO cases (id,patient_id,status,created,updated,scan_filename,scan_sha256,scan_size,consent) "
                "VALUES ('c1',1,'finalized',1,1,'s.png','0',1,1)")
    con.execute("INSERT INTO audit (ts,user_id,action) VALUES (1,1,'login')")
    con.execute("INSERT INTO audit (ts,user_id,action) VALUES (2,1,'logout')")
    con.commit()
    con.close()
    db = DB(path)
    assert db.one("SELECT version FROM schema_version")[0] == len(MIGRATIONS)
    assert db.user(1)["email"] == "p@x.test" and db.profile(1)["phone"] == "123" and db.case("c1")["patient_id"] == 1
    assert db.verify_audit_chain() == {"ok": True, "n": 2, "bad_id": None}
    assert {r["purpose"] for r in db.q("SELECT purpose FROM consents WHERE user_id = 1")} == {"care", "email_reminders"}
    db.create_user(None, "No Mail", "patient", None)                   # e-mail is optional after v3
    db.create_user(None, "No Mail 2", "patient", None)


# ------------------------------------------------------------------------------------------------ HTTP flows
def test_sign_with_follow_up_then_coordinator_closes_the_loop(env):
    p, d, a = (env["app"].test_client() for _ in range(3))
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    tok = csrf(d, f"/doctor/case/{cid}")
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": tok, "diagnosis": "meningioma", "agrees": "yes",
                                             "findings": "Extra-axial mass.", "recommendation": "Neurosurgery opinion.",
                                             "password": PW, "fu_kind": "referral", "fu_days": "14",
                                             "ref_facility": "Regional Cancer Centre", "ref_urgency": "soon"})
    db = env["db"]
    pat = db.user_by_email("pat@x.test")
    plan = careplan.active(db, pat["id"])
    assert plan["kind"] == "referral" and plan["status"] == "approved"
    ref = db.one("SELECT * FROM referrals WHERE patient_id = ?", (pat["id"],))
    assert ref["status"] == "signed"
    # the patient sees the next step and the letter, never the AI output
    dash = p.get("/patient/").get_data(as_text=True)
    assert "Regional Cancer Centre" in dash and "Your next step" in dash
    letter = p.get(f"/patient/referrals/{ref['id']}").get_data(as_text=True)
    assert "Referral letter" in letter and "Neurosurgery opinion." in letter and "Grad-CAM" not in letter
    # the coordinator works the worklist and tracks the referral to completion
    login(a, "admin@clinic.test")
    assert "Pat One" in a.get("/care/").get_data(as_text=True)
    tok = csrf(a, "/care/")
    a.post(f"/care/patients/{pat['id']}/contact", data={"csrf": tok, "channel": "phone", "outcome": "reached_confirmed"})
    a.post(f"/care/referrals/{ref['id']}/status", data={"csrf": tok, "status": "sent"})
    a.post(f"/care/referrals/{ref['id']}/status", data={"csrf": tok, "status": "attended"})
    assert careplan.get(db, plan["id"])["status"] == "completed"
    page = a.get(f"/care/patients/{pat['id']}").get_data(as_text=True)
    assert "Referral signed" in page and "Reached · will come" in page and "Completed" in page
    assert db.verify_audit_chain()["ok"]


def test_coordinator_proposal_needs_doctor_approval_over_http(env):
    p, d, a = (env["app"].test_client() for _ in range(3))
    cid = finalize(env, p, d)
    pat = env["db"].case(cid)["patient_id"]
    login(a, "admin@clinic.test")
    tok = csrf(a, "/care/patients")
    a.post(f"/care/patients/{pat}/plan", data={"csrf": tok, "kind": "consultation", "due": due(env["db"], 10)})
    plan = careplan.active(env["db"], pat)
    assert plan["status"] == "proposed"
    r = a.post(f"/care/plans/{plan['id']}/remind", data={"csrf": tok})
    assert careplan.get(env["db"], plan["id"])["status"] == "proposed"
    assert env["db"].one("SELECT COUNT(*) AS n FROM contacts")["n"] == 0                 # refused: nothing sent
    assert a.post(f"/care/plans/{plan['id']}/approve", data={"csrf": tok}).status_code == 403   # admin cannot approve
    assert "Pat One" in d.get("/care/approvals").get_data(as_text=True)
    d.post(f"/care/plans/{plan['id']}/approve", data={"csrf": csrf(d, "/care/approvals"), "decision": "approve"})
    assert careplan.get(env["db"], plan["id"])["status"] == "approved"
    a.post(f"/care/plans/{plan['id']}/remind", data={"csrf": tok, "message": "Please book"})
    assert env["db"].one("SELECT outcome FROM contacts")["outcome"] == "message_sent"
    assert r.status_code == 302


def test_walk_in_registration_invite_and_booking(env):
    a = env["app"].test_client()
    login(a, "admin@clinic.test")
    tok = csrf(a, "/care/patients/new")
    r = a.post("/care/patients/new", data={"csrf": tok, "name": "Sunita Devi", "phone": "98000 11111", "district": "Sitapur",
                                           "distance_km": "64", "email": "sunita@x.test", "consent": "yes",
                                           "email_consent": "yes", "invite": "yes",
                                           "scan": (io.BytesIO(PNG), "s.png")}, content_type="multipart/form-data")
    assert r.status_code == 302
    db = env["db"]
    u = db.user_by_email("sunita@x.test")
    assert u["pw_hash"] is None and db.profile(u["id"])["distance_km"] == 64
    assert db.one("SELECT COUNT(*) AS n FROM cases WHERE patient_id = ?", (u["id"],))["n"] == 1     # scan sent for review
    mail = db.one("SELECT * FROM outbox WHERE to_email = 'sunita@x.test'")
    link = re.search(r"/welcome/([\w-]+)", mail["text"]).group(1)
    anon = env["app"].test_client()
    assert login(anon, "sunita@x.test").status_code == 401                 # no password yet: cannot sign in
    page = anon.get(f"/welcome/{link}")
    assert page.status_code == 200 and "Set your password" in page.get_data(as_text=True)
    anon.post(f"/welcome/{link}", data={"csrf": csrf(anon, f"/welcome/{link}"), "password": PW, "password2": PW})
    assert anon.get(f"/welcome/{link}").status_code == 410                  # single use
    assert login(anon, "sunita@x.test").status_code == 302
    # booking on the patient's behalf after the report
    d = env["app"].test_client()
    login(d, "doc@clinic.test")
    cid = db.q("SELECT id FROM cases WHERE patient_id = ?", (u["id"],))[0]["id"]
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": csrf(d, f"/doctor/case/{cid}"), "diagnosis": "no_tumor", "agrees": "yes",
                                             "findings": "Normal.", "password": PW})
    all_day_hours(env)
    slot = scheduling.free_slots(db, db.user_by_email("doc@clinic.test")["id"], days=2)[0]
    a.post(f"/care/patients/{u['id']}/book", data={"csrf": tok, "start_ts": slot["start_ts"], "channel": "phone"})
    assert db.one("SELECT status FROM appointments WHERE patient_id = ?", (u["id"],))["status"] == "booked"


def test_forgot_password_does_not_reveal_accounts_and_resets_once(env):
    c = env["app"].test_client()
    register(c, "pat@x.test")
    anon = env["app"].test_client()
    a = anon.post("/forgot", data={"csrf": csrf(anon, "/forgot"), "email": "pat@x.test"}).get_data(as_text=True)
    b = anon.post("/forgot", data={"csrf": csrf(anon, "/forgot"), "email": "nobody@x.test"}).get_data(as_text=True)
    assert "Check your e-mail" in a and "Check your e-mail" in b
    link = re.search(r"/reset/([\w-]+)", env["db"].one("SELECT text FROM outbox WHERE subject LIKE 'Reset%'")["text"]).group(1)
    new = "another good pass 9"
    anon.post(f"/reset/{link}", data={"csrf": csrf(anon, f"/reset/{link}"), "password": new, "password2": new})
    assert login(env["app"].test_client(), "pat@x.test", new).status_code == 302
    assert anon.get(f"/reset/{link}").status_code == 410


def test_email_reply_pages_change_nothing_on_get(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    cid = finalize(env, p, d)
    all_day_hours(env)
    db = env["db"]
    pat = db.user_by_email("pat@x.test")
    slot = scheduling.free_slots(db, db.user_by_email("doc@clinic.test")["id"], days=3)[-1]
    aid = scheduling.book(db, pat, db.case(cid), slot["start_ts"])
    no = outreach.reminder_actions(db, scheduling.appointment(db, aid))[1][1]
    anon = env["app"].test_client()
    assert anon.get(no).status_code == 200 and scheduling.appointment(db, aid)["status"] == "booked"   # GET: no change
    anon.post(no, data={"csrf": csrf(anon, no)})
    assert scheduling.appointment(db, aid)["status"] == "cancelled"


def test_access_control_on_care_routes(env):
    p, d, a = (env["app"].test_client() for _ in range(3))
    cid = finalize(env, p, d)
    pat = env["db"].case(cid)["patient_id"]
    other = env["app"].test_client()
    register(other, "other@x.test", "Other Patient")
    for path in ("/care/", "/care/patients", f"/care/patients/{pat}", "/care/referrals", "/care/approvals"):
        assert p.get(path).status_code == 403, path
    login(a, "admin@clinic.test")
    assert a.get("/care/approvals").status_code == 403                    # approvals are the doctor's
    assert a.get(f"/doctor/case/{cid}").status_code == 403                # the coordinator never sees AI output
    page = a.get(f"/care/patients/{pat}").get_data(as_text=True)
    assert "meningioma" not in page.lower() and "Grad-CAM" not in page
    # documents: the patient sees only their own
    p.post("/patient/documents", data={"csrf": csrf(p, "/patient/documents"), "kind": "biopsy", "title": "Biopsy",
                                       "file": (io.BytesIO(b"%PDF-1.4 test"), "b.pdf")}, content_type="multipart/form-data")
    did = env["db"].one("SELECT id FROM documents")["id"]
    assert p.get(f"/patient/documents/{did}").data.startswith(b"%PDF")
    assert other.get(f"/patient/documents/{did}").status_code == 404
    assert a.get(f"/care/documents/{did}").status_code == 200
    r = p.post("/patient/documents", data={"csrf": csrf(p, "/patient/documents"), "kind": "other",
                                           "file": (io.BytesIO(b"MZ fake exe"), "x.pdf")}, content_type="multipart/form-data")
    assert env["db"].one("SELECT COUNT(*) AS n FROM documents")["n"] == 1 and r.status_code == 302     # rejected


def test_data_export_contains_only_the_owner(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    finalize(env, p, d)
    other = env["app"].test_client()
    register(other, "other@x.test", "Other Patient")
    upload(other)
    z = zipfile.ZipFile(io.BytesIO(p.get("/patient/export").data))
    rec = json.loads(z.read("record.json"))
    assert rec["patient"]["email"] == "pat@x.test" and len(rec["studies"]) == 1 and rec["reports"][0]["findings"]
    assert any(n.startswith("reports/") for n in z.namelist()) and any(n.startswith("scans/") for n in z.namelist())
    blob = b"".join(z.read(n) for n in z.namelist())
    assert b"other@x.test" not in blob and b"Other Patient" not in blob and b"result_json" not in blob
    assert env["db"].one("SELECT COUNT(*) AS n FROM audit WHERE action = 'data_exported'")["n"] == 1


def test_email_reminder_consent_is_recorded(env):
    p = env["app"].test_client()
    register(p)
    pat = env["db"].user_by_email("pat@x.test")
    p.post("/account", data={"csrf": csrf(p, "/account"), "form": "prefs"})            # untick e-mail
    c = env["db"].one("SELECT * FROM consents WHERE user_id = ? AND purpose = 'email_reminders'", (pat["id"],))
    assert c["withdrawn_at"] is not None and env["db"].user(pat["id"])["notify_email"] == 0
    p.post("/account", data={"csrf": csrf(p, "/account"), "form": "deletion", "note": "moving"})
    a = env["app"].test_client()
    login(a, "admin@clinic.test")
    assert "Account deletion requests" in a.get("/admin/users").get_data(as_text=True)


def test_every_new_page_renders_for_its_role(env):
    p, d, a = (env["app"].test_client() for _ in range(3))
    cid = finalize(env, p, d)
    pat = env["db"].case(cid)["patient_id"]
    login(a, "admin@clinic.test")
    for c, paths in ((a, ["/care/", "/care/?q=overdue", "/care/?q=missed", "/care/patients", "/care/patients?q=Pat",
                          "/care/patients/new", f"/care/patients/{pat}", "/care/referrals", "/care/referrals?status=all",
                          "/admin/users", "/admin/audit", "/admin/system", "/admin/settings", "/account"]),
                     (d, ["/care/approvals", "/care/patients", f"/care/patients/{pat}", "/care/referrals", "/account"]),
                     (p, ["/patient/", "/patient/documents", "/account"])):
        for path in paths:
            r = c.get(path)
            assert r.status_code == 200, (path, r.status_code)
    anon = env["app"].test_client()
    for path in ("/forgot", "/login"):
        assert anon.get(path).status_code == 200
    assert anon.get("/r/not-a-token?a=yes").status_code == 410
    assert anon.get("/welcome/not-a-token").status_code == 410


def test_patient_is_told_about_the_follow_up_the_doctor_decides(env):
    p, d, a = (env["app"].test_client() for _ in range(3))
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": csrf(d, f"/doctor/case/{cid}"), "diagnosis": "meningioma", "agrees": "yes",
                                             "findings": "Small mass.", "password": PW, "fu_kind": "repeat_scan", "fu_days": "7"})
    db = env["db"]
    mail = db.one("SELECT * FROM outbox WHERE to_email = 'pat@x.test' AND subject LIKE 'Your next step%'")
    assert mail and "repeat mri" in mail["subject"].lower() and "CareLoop screen" in mail["text"]
    assert "meningioma" not in (mail["text"] + mail["html"]).lower()             # no clinical content in e-mail
    # a coordinator proposal is announced only once a doctor approves it
    pat = db.user_by_email("pat@x.test")["id"]
    careplan.close(db, careplan.active(db, pat), "completed", "", None)
    login(a, "admin@clinic.test")
    a.post(f"/care/patients/{pat}/plan", data={"csrf": csrf(a, "/care/patients"), "kind": "consultation", "due": due(db, 10)})
    n = db.one("SELECT COUNT(*) AS n FROM outbox WHERE subject LIKE 'Your next step: consultation%'")["n"]
    assert n == 0
    plan = careplan.active(db, pat)
    d.post(f"/care/plans/{plan['id']}/approve", data={"csrf": csrf(d, "/care/approvals"), "decision": "approve"})
    assert db.one("SELECT COUNT(*) AS n FROM outbox WHERE subject LIKE 'Your next step: consultation%'")["n"] == 1

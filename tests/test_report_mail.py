"""Structured radiology report (indication, technique, comparison, findings, impression, recommendation; reporter's
credentials) and the e-mail layout (greeting, detail table, safety advice, sign-off, confidentiality footer)."""

import pytest

pytest.importorskip("flask")

from test_portal import PW, csrf, env, login, register, upload  # noqa: E402,F401  (env is a fixture)


def sign(env, d, cid, **extra):
    data = {"csrf": csrf(d, f"/doctor/case/{cid}"), "diagnosis": "meningioma", "agrees": "yes",
            "findings": "Lesion: right parafalcine extra-axial mass, 32 x 28 x 30 mm.", "password": PW} | extra
    return d.post(f"/doctor/case/{cid}/sign", data=data)


def test_report_has_the_standard_sections_and_signature(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p, "rp@x.test", "Asha Verma")
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    form = d.get(f"/doctor/case/{cid}").get_data(as_text=True)
    assert "Single two-dimensional MR image" in form and "None available." in form    # technique / comparison defaults
    assert 'data-tpl="Lesion:' in form                                                  # structured findings template
    d.post("/account", data={"csrf": csrf(d, "/account"), "form": "signature",
                             "credentials": "MD (Radiodiagnosis), Consultant Radiologist", "reg_no": "RMC 12345"})
    env["db"].set_setting("clinic_phone", "+91 141 000 0000")
    sign(env, d, cid, indication="Headache for 3 months.", impression="1. Extra-axial mass, most consistent with meningioma.",
         recommendation="Neurosurgical consultation.", technique="MRI brain, single image.", comparison="None available.")
    snap = p.get(f"/case/{cid}/report/1").get_data(as_text=True)
    order = [snap.index(h) for h in ("Clinical indication", "Technique", "Comparison", "Findings", "Impression",
                                     "Recommendation")]
    assert order == sorted(order)                                                        # ACR section order
    for text in ("Headache for 3 months.", "1. Extra-axial mass", "MD (Radiodiagnosis)", "Reg. no. RMC 12345",
                 "Tel. +91 141 000 0000", "Final", "decision support", "Asha Verma"):
        assert text in snap
    assert "script" not in snap.lower()
    view = p.get(f"/patient/case/{cid}").get_data(as_text=True)                           # same document in the portal
    assert "Clinical indication" in view and "Reg. no. RMC 12345" in view
    r = env["db"].reports(cid)[0]
    assert r["indication"] == "Headache for 3 months." and r["impression"].startswith("1.")


def test_report_email_is_structured_and_carries_no_clinical_content(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p, "rm@x.test", "Ravi Kumar")
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    sign(env, d, cid)
    m = env["db"].one("SELECT * FROM outbox WHERE to_email = 'rm@x.test' AND subject LIKE 'Your MRI report%'")
    text, html = m["text"], m["html"]
    assert text.startswith("Dear Ravi Kumar,")
    assert f"Accession no.: {cid[:8].upper()}" in text and "Reporting radiologist:" in text
    assert "call 108" in text and "Kind regards," in text and "never sent by e-mail" in text
    assert "<table" in html and "Dear Ravi Kumar," in html
    assert all(w not in (text + html).lower() for w in ("meningioma", "parafalcine", "32 x 28"))


def test_staff_email_has_its_own_footer(env):
    from edge.portal import notify
    doc = env["db"].user_by_email("doc@clinic.test")
    notify.notify(env["db"], doc, "doc_unread", {"studies": "2 studies", "hours": 4, "oldest": "5 hours ago"}, "/doctor/")
    m = env["db"].one("SELECT * FROM outbox WHERE subject LIKE 'Reporting backlog%'")
    assert "do not forward it outside the care team" in m["text"] and "call 108" not in m["text"]


def test_referral_letter_is_a_formal_letter_with_the_imaging_summary(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p, "rf@x.test", "Meena Joshi")
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    d.post("/account", data={"csrf": csrf(d, "/account"), "form": "signature",
                             "credentials": "MD (Radiodiagnosis)", "reg_no": "RMC 12345"})
    sign(env, d, cid, impression="1. Extra-axial mass, most consistent with meningioma.", fu_kind="referral",
         ref_facility="SMS Hospital, Jaipur", ref_specialty="Neurosurgery", ref_urgency="soon")
    rid = env["db"].one("SELECT id FROM referrals")["id"]
    letter = p.get(f"/patient/referrals/{rid}").get_data(as_text=True)
    for text in ("Dear Colleague,", "The Consultant, Neurosurgery", "SMS Hospital, Jaipur", "Re: Meena Joshi",
                 "Imaging summary", "1. Extra-axial mass", f"accession {cid[:8].upper()}", "Yours sincerely,",
                 "Reg. no. RMC 12345", "within the next two weeks"):
        assert text in letter
    assert "script" not in letter.lower()


def test_no_email_until_the_address_is_confirmed(env):
    import re
    db = env["db"]
    p = env["app"].test_client()
    register(p, "new@x.test", "Nita Rao", confirm=False)
    user = db.user_by_email("new@x.test")
    assert not user["email_verified"]
    from edge.portal import notify
    notify.notify(db, user, "report_ready", {}, "/patient/")
    subjects = [r["subject"] for r in db.q("SELECT subject FROM outbox WHERE to_email = 'new@x.test'")]
    assert len(subjects) == 1 and subjects[0].startswith("Please confirm")           # only the confirmation
    link = re.search(r"(/verify/[\w-]+)", db.one("SELECT text FROM outbox WHERE to_email = 'new@x.test'")["text"]).group(1)
    assert "E-mail address confirmed" in env["app"].test_client().get(link).get_data(as_text=True)
    assert env["app"].test_client().get(link).status_code == 410                      # works once
    notify.notify(db, db.user(user["id"]), "report_ready", {}, "/patient/")
    assert db.one("SELECT COUNT(*) AS n FROM outbox WHERE to_email = 'new@x.test'")["n"] == 2


def test_password_change_signs_out_other_sessions(env):
    a, b = env["app"].test_client(), env["app"].test_client()
    register(a, "pw@x.test", "Pia Sen")
    login(b, "pw@x.test")                                                            # second device
    new = "another long passphrase 42"
    a.post("/account", data={"csrf": csrf(a, "/account"), "form": "password", "current": PW, "new": new, "new2": new})
    assert a.get("/patient/").status_code == 200                                     # this browser stays signed in
    assert b.get("/patient/").status_code == 302                                     # the other one is signed out


def test_signed_reports_cannot_be_edited_in_the_database(env):
    import sqlite3
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p, "im@x.test", "Imran Ali")
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    sign(env, d, cid)
    for sql in ("UPDATE reports SET findings = 'x'", "DELETE FROM reports"):
        with pytest.raises(sqlite3.IntegrityError):
            env["db"].conn.execute(sql)


def test_gzip_bomb_is_refused_without_inflating_it(monkeypatch):
    import gzip
    from neuroedge import volume
    monkeypatch.setattr(volume, "MAX_NIFTI_BYTES", 1 << 20)
    bomb = gzip.compress(b"\0" * (8 << 20))
    with pytest.raises(ValueError, match="too large"):
        volume.load_nifti(bomb)

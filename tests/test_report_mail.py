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

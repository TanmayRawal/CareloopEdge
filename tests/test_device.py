"""Device home page (the Pi's own screen): open without login, never shows patient data, and the kiosk cookie turns on
the "device home" button in the portal pages."""

import re

import pytest

pytest.importorskip("flask")

from test_portal import env, login, register, upload  # noqa: E402,F401  (env is a fixture)


def test_device_page_is_public_and_has_no_patient_data(env):
    p, k = env["app"].test_client(), env["app"].test_client()
    register(p, "pat@x.test", "Zebulon Quixote")
    upload(p)
    r = k.get("/device")
    page = r.get_data(as_text=True)
    assert r.status_code == 200 and "no-store" in r.headers["Cache-Control"]
    assert "Clinician" in page and "Patient" in page                              # the two entry points
    assert "Zebulon" not in page and "Quixote" not in page and "pat@x.test" not in page
    assert "Meningioma" not in page and "NE-0" not in page                          # no results, no record numbers
    # no internal model names or provenance notices on the device screen (those are on Admin > System)
    assert "test-1" not in page and "Development model" not in page and "signature" not in page
    assert 'id="brain"' in page
    src = re.search(r'data-src="([^"]+)"', page).group(1)
    data = k.get(src).data                                          # 3D brain point cloud: int8 x y z nx ny nz
    assert len(data) % 6 == 0 and len(data) // 6 > 5000


def test_device_page_names_a_verified_release(env):
    env["app"].extensions["portal"]["ctx"].model_version = "0.3.0"
    from edge.portal.views import device
    info = {"version": "0.3.0", "verified": True, "whole_study": True}

    class D:
        model_info, db, data_dir = info, env["db"], env["tmp"]
    s = device.status(D)
    assert s["model"] == "Model 0.3.0" and s["verified"] and not s["issues"] and s["ready"]
    D.model_info = {"version": "mri_v2vol_seed0_int8", "verified": False}
    assert device.status(D)["model"] is None and device.status(D)["ready"]     # a dev build is not a problem to show


def body_classes(page: str) -> set[str]:
    return set(re.search(r'<body class="([^"]*)"', page).group(1).split())


def test_kiosk_cookie_adds_device_home_button(env):
    c = env["app"].test_client()
    login(c, "doc@clinic.test")
    assert "kiosk" not in body_classes(c.get("/doctor/").get_data(as_text=True))
    r = c.get("/device?kiosk=1")
    assert "ne_kiosk=1" in r.headers["Set-Cookie"] and "HttpOnly" in r.headers["Set-Cookie"]
    page = c.get("/doctor/").get_data(as_text=True)
    assert {"kiosk", "theme-dark"} <= body_classes(page) and 'href="/device"' in page
    c.get("/device?kiosk=0")
    assert "kiosk" not in body_classes(c.get("/doctor/").get_data(as_text=True))


def test_front_door_is_the_device_page(env):
    c = env["app"].test_client()
    assert c.get("/").headers["Location"].endswith("/device")              # signed out: device home
    login(c, "doc@clinic.test")
    assert c.get("/").headers["Location"].endswith("/doctor/")             # signed in: straight to work


def test_kiosk_signs_out_sooner_and_returns_home(env):
    import time as _t
    c = env["app"].test_client()
    c.get("/device?kiosk=1")
    login(c, "doc@clinic.test")
    with c.session_transaction() as s:                       # 6 minutes without activity
        s["t_seen"] = _t.time() - 6 * 60
    assert c.get("/doctor/").status_code in (302, 401, 403)   # the kiosk session has closed
    login(c, "doc@clinic.test")
    from test_portal import csrf
    r = c.post("/logout", data={"csrf": csrf(c, "/doctor/")})
    assert r.headers["Location"].endswith("/device")          # signing out on the kiosk shows the device home
    d = env["app"].test_client()                              # a normal browser: 6 minutes idle is fine
    login(d, "doc@clinic.test")
    with d.session_transaction() as s:
        s["t_seen"] = _t.time() - 6 * 60
    assert d.get("/doctor/").status_code == 200


def test_patient_names_are_initials_on_the_kiosk_only(env):
    from edge.portal.common import initials
    assert initials("Meera Joshi") == "M. J." and initials("Dr. A. Rao") == "D. A. R." and initials("") == "—"
    p = env["app"].test_client()
    register(p, "zq@x.test", "Zebulon Quixote")
    upload(p)
    doc = env["app"].test_client()
    login(doc, "doc@clinic.test")
    assert "Zebulon Quixote" in doc.get("/doctor/").get_data(as_text=True)
    doc.get("/device?kiosk=1")
    page = doc.get("/doctor/").get_data(as_text=True)
    assert "Zebulon" not in page and "Z. Q." in page


def test_kiosk_browser_is_locked_down_by_policy():
    from pathlib import Path
    s = (Path(__file__).resolve().parents[1] / "edge/install_pi.sh").read_text(encoding="utf-8")
    for key in ('"URLBlocklist": ["*"]', '"DeveloperToolsAvailability": 2', '"DownloadRestrictions": 3',
                '"PrintingEnabled": false', '"URLAllowlist": ["https://localhost", "about:blank"]'):
        assert key in s


def test_patients_upload_only_at_the_device(env):
    phone = env["app"].test_client()
    phone.environ_base["REMOTE_ADDR"] = "10.0.0.7"                  # the patient's phone on the clinic Wi-Fi
    assert "Accounts are created at the clinic" in phone.get("/register").get_data(as_text=True)
    assert register(phone, "ph@x.test", "Phone Patient").status_code == 403
    assert env["db"].user_by_email("ph@x.test") is None
    assert register(env["app"].test_client(), "ph@x.test", "Phone Patient").status_code == 302   # at the device
    login(phone, "ph@x.test")                                      # then signs in from the phone
    page = phone.get("/patient/upload").get_data(as_text=True)
    assert "Scans are uploaded at the clinic" in page and 'id="upload-form"' not in page
    assert 'href="/patient/upload"' not in phone.get("/patient/").get_data(as_text=True)
    assert upload(phone).status_code == 403
    from edge.portal import auth                                   # Funnel visitors also reach Caddy from loopback,
    with env["app"].test_request_context("/", base_url="https://x.ts.net", environ_base={"REMOTE_ADDR": "127.0.0.1"}):   # but under the public host name
        assert not auth.on_device()
    with env["app"].test_request_context("/", base_url="https://localhost", environ_base={"REMOTE_ADDR": "127.0.0.1"}):
        assert auth.on_device()
    kiosk = env["app"].test_client()                               # the device's own screen (https://localhost)
    login(kiosk, "ph@x.test")
    assert upload(kiosk).status_code == 302
    assert phone.get("/patient/").status_code == 200               # reports and booking stay open on the phone
    admin = env["app"].test_client()
    login(admin, "admin@clinic.test")
    env["db"].set_setting("upload_device_only", "0")               # department setting: upload from anywhere
    assert upload(phone).status_code == 302
    assert "Only at the clinic device" in admin.get("/admin/settings").get_data(as_text=True)

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
    assert "test-1" in page and "Clinician sign-in" in page                       # model version, entry points
    assert "Zebulon" not in page and "Quixote" not in page and "pat@x.test" not in page
    assert "Meningioma" not in page and "NE-0" not in page                          # no results, no record numbers
    assert "unsigned dev bundle" in page                                             # stub model: not signature-verified


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

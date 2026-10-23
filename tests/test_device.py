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
    assert 'id="brain"' in page and "Ready" in page
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

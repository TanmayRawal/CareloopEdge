"""Two-step sign-in for staff: RFC 6238 test vectors, replay protection, recovery codes, the login flow and the
'require two-step for staff' setting."""

import hashlib
import re

from test_portal import PW, csrf, env, login  # noqa: F401  (env is a fixture)

from edge.portal import totp

RFC_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"          # base32("12345678901234567890")


def test_rfc6238_sha1_vectors():
    for t, code in ((59, "94287082"), (1111111109, "07081804"), (1111111111, "14050471"), (1234567890, "89005924"),
                    (2000000000, "69279037"), (20000000000, "65353130")):
        assert totp.code_at(RFC_SECRET, t // 30, digits=8, digest=hashlib.sha1) == code


def test_match_window_and_replay():
    s = totp.new_secret()
    now = 1_800_000_000
    code = totp.code_at(s, now // 30)
    step = totp.match(s, code, None, now)
    assert step == now // 30
    assert totp.match(s, code, step, now) is None                         # replay of the same code refused
    assert totp.match(s, totp.code_at(s, now // 30 - 1), None, now) is not None     # 30 s clock drift is fine
    assert totp.match(s, totp.code_at(s, now // 30 - 3), None, now) is None         # 90 s old: refused
    assert totp.match(s, "12345", None, now) is None


def test_recovery_codes_are_single_use():
    codes, stored = totp.new_recovery_codes(3)
    assert totp.remaining_recovery(stored) == 3
    left = totp.use_recovery_code(stored, codes[1])
    assert left and totp.remaining_recovery(left) == 2
    assert totp.use_recovery_code(left, codes[1]) is None
    assert totp.use_recovery_code(left, "nope-nope") is None


def setup_2fa(c, env):
    page = c.post("/account", data={"csrf": csrf(c, "/account"), "form": "2fa_start"}).get_data(as_text=True)
    key = re.search(r'class="mono key">([A-Z2-7 ]+)<', page).group(1).replace(" ", "")
    import time
    c.post("/account", data={"csrf": csrf(c, "/account"), "form": "2fa_confirm", "code": totp.code_at(key, int(time.time()) // 30)})
    return key


def test_staff_login_requires_the_code_once_enabled(env):
    import time
    c = env["app"].test_client()
    login(c, "doc@clinic.test")
    key = setup_2fa(c, env)
    u = env["db"].user_by_email("doc@clinic.test")
    assert u["totp_on"] == 1 and key not in (u["totp_secret"] or "")             # stored encrypted
    c2 = env["app"].test_client()
    r = login(c2, "doc@clinic.test")
    assert r.headers["Location"].endswith("/login/code")
    assert c2.get("/doctor/").status_code == 302                               # not signed in yet
    r = c2.post("/login/code", data={"csrf": csrf(c2, "/login/code"), "code": "000000"})
    assert r.status_code == 401
    # the code used at setup is consumed; wait-free: use the next step's code, which is inside the window
    nxt = totp.code_at(key, int(time.time()) // 30 + 1)
    r = c2.post("/login/code", data={"csrf": csrf(c2, "/login/code"), "code": nxt})
    assert r.status_code == 302 and c2.get("/doctor/").status_code == 200
    c3 = env["app"].test_client()
    login(c3, "doc@clinic.test")
    assert c3.post("/login/code", data={"csrf": csrf(c3, "/login/code"), "code": nxt}).status_code == 401   # replay


def test_required_two_step_redirects_staff_to_setup(env):
    env["db"].set_setting("require_staff_2fa", "1")
    c = env["app"].test_client()
    login(c, "admin@clinic.test")
    r = c.get("/care/")
    assert r.status_code == 302 and r.headers["Location"].endswith("/account")
    assert c.get("/account").status_code == 200
    p = env["app"].test_client()
    from test_portal import register
    register(p)
    assert p.get("/patient/").status_code == 200                               # patients are not affected

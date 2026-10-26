"""Clinical portal: roles, object-level access, CSRF, lockout, encrypted storage, workflow, report versioning,
backup/restore, and that patients never receive AI output. A stub engine replaces the model."""

import io
import re
import sqlite3

import pytest

pytest.importorskip("flask")
pytest.importorskip("argon2")

from edge.portal import auth  # noqa: E402
from edge.portal.app import create_app  # noqa: E402
from edge.portal.db import DB  # noqa: E402
from edge.portal.storage import BlobStore, generate_key  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"scan-bytes" * 20
PW = "correct horse 42"


class StubEngine:
    def __init__(self, status="ok"):
        self.status = status

    def analyze(self, data, filename=None):
        if data.startswith(b"BAD"):
            raise ValueError("not an image")
        ok = self.status == "ok"
        return {"status": self.status, "reject_reasons": [] if ok else ["out_of_distribution"],
                "prediction": "meningioma" if ok else None, "raw_argmax": "meningioma",
                "probabilities": {"glioma": .05, "meningioma": .9, "no_tumor": .02, "pituitary": .03},
                "prediction_set": ["meningioma"], "conformal_alpha": 0.05,
                "ood": {"mahalanobis_score": 0.1, "threshold": 1.5, "is_ood": not ok, "warning": False},
                "explanation": {"explained_class": "meningioma", "runner_up": "glioma", "evidence_inside_lesion": 0.4,
                                "logit_decomposition_error": 1e-7},
                "lesion": {"present": True, "image_side": "image-left", "image_vertical": "upper",
                           "area_fraction_of_image": 0.02},
                "flags": [] if ok else ["out_of_distribution"], "needs_expert_review": not ok,
                "timings_ms": {}, "_state": {},
                "overlays": {k: b"PNG-" + k.encode() for k in ("original", "cam", "lesion", "cam_lesion")}}


class NoLLM:
    def available(self):
        return False


@pytest.fixture()
def env(tmp_path):
    from edge.agent.orchestrator import Assistant
    key = tmp_path / "portal.key"
    generate_key(key)
    app = create_app(StubEngine(), {"version": "test-1"}, tmp_path / "data", key, assistant=Assistant(llm=NoLLM()))
    app.testing = True
    db = app.extensions["portal"]["db"]
    db.create_user("doc@clinic.test", "Dr Test", "doctor", auth.hash_password(PW))
    db.create_user("admin@clinic.test", "Admin", "admin", auth.hash_password(PW))
    return {"app": app, "db": db, "store": app.extensions["portal"]["store"], "tmp": tmp_path}


def csrf(c, path="/login"):
    return re.search(r'name="csrf-token" content="([^"]+)"|name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(0).split('"')[-2]


def login(c, email, pw=PW):
    return c.post("/login", data={"csrf": csrf(c), "email": email, "password": pw})


def register(c, email="pat@x.test", name="Pat One", confirm=True):
    """Sign up; confirm=True then follows the e-mail confirmation link, as the patient would."""
    r = c.post("/register", data={"csrf": csrf(c, "/register"), "name": name, "email": email, "password": PW,
                                  "password2": PW, "consent": "yes"})
    mail = c.application.extensions["portal"]["db"].one(
        "SELECT text FROM outbox WHERE to_email = ? AND subject LIKE 'Please confirm%' ORDER BY id DESC", (email.lower(),))
    if confirm and mail:
        c.get(re.search(r"(/verify/[\w-]+)", mail["text"]).group(1))
    return r


def upload(c, data=PNG, consent="yes"):
    return c.post("/patient/upload", data={"csrf": csrf(c, "/patient/upload"), "scan": (io.BytesIO(data), "s.png"),
                                           "consent": consent}, content_type="multipart/form-data")


# --------------------------------------------------------------------------------------- auth
def test_csrf_required_everywhere(env):
    c = env["app"].test_client()
    assert c.post("/login", data={"email": "doc@clinic.test", "password": PW}).status_code == 400


def test_lockout_after_failures(env):
    c = env["app"].test_client()
    for _ in range(5):
        assert login(c, "doc@clinic.test", "wrong password!!").status_code == 401
    r = login(c, "doc@clinic.test")            # correct password, but locked
    assert r.status_code == 401 and "Too many failed attempts" in r.get_data(as_text=True)


def test_passwords_hashed_and_policy(env):
    row = env["db"].user_by_email("doc@clinic.test")
    assert row["pw_hash"].startswith("$argon2") and PW not in row["pw_hash"]
    assert auth.password_problem("short") and auth.password_problem("aaaaaaaaaaaa")


# ----------------------------------------------------------------------------- roles + objects
def test_roles_and_object_level_access(env):
    a, b, d = (env["app"].test_client() for _ in range(3))
    register(a, "a@x.test", "Pat A")
    register(b, "b@x.test", "Pat B")
    assert upload(a).status_code == 302
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    assert a.get(f"/patient/case/{cid}").status_code == 200
    assert b.get(f"/patient/case/{cid}").status_code == 404           # other patient's case: not found
    assert b.get(f"/case/{cid}/img/original_1.png").status_code == 404
    assert a.get("/doctor/").status_code == 403                       # patients cannot reach doctor pages
    assert a.get(f"/doctor/case/{cid}").status_code == 403
    assert a.get("/admin/users").status_code == 403
    login(d, "doc@clinic.test")
    assert d.get("/patient/").status_code == 403
    assert d.get(f"/doctor/case/{cid}").status_code == 200


def test_patient_never_receives_ai_output(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    for kind in ("cam_1", "lesion_1", "cam_lesion_1"):
        assert p.get(f"/case/{cid}/img/{kind}.png").status_code == 404
    assert p.get(f"/case/{cid}/img/original_1.png").status_code == 200
    page = p.get(f"/patient/case/{cid}").get_data(as_text=True)
    assert "90.0" not in page and "Grad-CAM" not in page and "prediction set" not in page.lower()
    # after the doctor signs, the patient page shows the doctor's diagnosis and still no AI
    login(d, "doc@clinic.test")
    tok = csrf(d, f"/doctor/case/{cid}")
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": tok, "diagnosis": "glioma", "agrees": "no",
                                             "findings": "Intra-axial enhancing mass.", "password": PW})
    page = p.get(f"/patient/case/{cid}").get_data(as_text=True)
    assert "Glioma" in page and "Intra-axial enhancing mass." in page
    assert "Meningioma" not in page and "90.0" not in page                # the AI's (different) suggestion is absent


# --------------------------------------------------------------------------------- workflow
def test_full_workflow_and_report_versions(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    assert env["db"].case(cid)["status"] == "awaiting_review"
    login(d, "doc@clinic.test")
    assert "Pat One" in d.get("/doctor/").get_data(as_text=True)
    tok = csrf(d, f"/doctor/case/{cid}")
    assert env["db"].case(cid)["status"] == "in_review"
    r = d.post(f"/doctor/case/{cid}/sign", data={"csrf": tok, "diagnosis": "meningioma", "agrees": "yes",
                                                 "findings": "Extra-axial mass.", "password": "wrong password!"})
    assert env["db"].case(cid)["status"] == "in_review"                  # wrong signing password: not signed
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": tok, "diagnosis": "meningioma", "agrees": "yes",
                                             "findings": "Extra-axial mass.", "password": PW})
    assert env["db"].case(cid)["status"] == "finalized"
    v1 = env["db"].reports(cid)[0]
    snap1 = p.get(f"/case/{cid}/report/1").get_data(as_text=True)
    assert "Extra-axial mass." in snap1 and "script" not in snap1.lower()
    # amendment without reason is refused; with reason creates v2 and keeps v1 intact
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": tok, "diagnosis": "meningioma", "agrees": "yes",
                                             "findings": "Amended text.", "password": PW})
    assert len(env["db"].reports(cid)) == 1
    d.post(f"/doctor/case/{cid}/sign", data={"csrf": tok, "diagnosis": "meningioma", "agrees": "yes",
                                             "findings": "Amended text.", "amendment_reason": "typo", "password": PW})
    reps = env["db"].reports(cid)
    assert [r["version"] for r in reps] == [2, 1] and reps[1]["findings"] == v1["findings"]
    assert p.get(f"/case/{cid}/report/1").get_data(as_text=True) == snap1
    actions = {r["action"] for r in env["db"].q("SELECT action FROM audit")}
    assert {"case_submitted", "review_started", "report_signed", "report_amended", "login"} <= actions


def test_reupload_flow(env):
    p, d = env["app"].test_client(), env["app"].test_client()
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    login(d, "doc@clinic.test")
    tok = csrf(d, f"/doctor/case/{cid}")
    d.post(f"/doctor/case/{cid}/reupload", data={"csrf": tok, "message": "Please send an axial slice."})
    assert env["db"].case(cid)["status"] == "reupload_requested"
    page = p.get(f"/patient/case/{cid}").get_data(as_text=True)
    assert "Please send an axial slice." in page
    p.post(f"/patient/case/{cid}", data={"csrf": csrf(p, f"/patient/case/{cid}"), "scan": (io.BytesIO(PNG + b"2"), "s2.png")},
           content_type="multipart/form-data")
    assert env["db"].case(cid)["status"] == "awaiting_review"
    assert env["db"].latest_ai(cid)["result"]["attempt"] == 2
    assert env["store"].exists(cid, "scan_1") and env["store"].exists(cid, "scan_2")   # history kept


def test_upload_validation(env):
    p = env["app"].test_client()
    register(p)
    assert upload(p, consent="no").status_code == 400
    assert upload(p, data=b"BAD not an image").status_code == 400
    assert env["db"].q("SELECT COUNT(*) AS n FROM cases")[0]["n"] == 0           # failed uploads leave nothing
    assert not any((env["tmp"] / "data" / "blobs").iterdir())


# ------------------------------------------------------------------------ storage + database
def test_encrypted_storage_roundtrip_and_tamper(env):
    p = env["app"].test_client()
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    raw = (env["tmp"] / "data" / "blobs" / cid / "scan_1.enc").read_bytes()
    assert PNG not in raw and b"scan-bytes" not in raw                   # ciphertext at rest
    assert env["store"].get(cid, "scan_1") == PNG                         # byte-identical read-back
    import hashlib
    assert env["db"].case(cid)["scan_sha256"] == hashlib.sha256(PNG).hexdigest()
    f = env["tmp"] / "data" / "blobs" / cid / "scan_1.enc"
    b = bytearray(raw); b[20] ^= 1; f.write_bytes(bytes(b))
    from cryptography.exceptions import InvalidTag
    with pytest.raises(InvalidTag):
        env["store"].get(cid, "scan_1")
    other = BlobStore(env["tmp"] / "x", generate_key_bytes())
    with pytest.raises(Exception):
        other.get(cid, "scan_1")


def generate_key_bytes():
    import os
    return os.urandom(32)


def test_backup_restore_and_restart(env, tmp_path):
    p = env["app"].test_client()
    register(p)
    upload(p)
    cid = env["db"].q("SELECT id FROM cases")[0]["id"]
    env["db"].backup(tmp_path / "bk" / "portal.db")
    copy = sqlite3.connect(tmp_path / "bk" / "portal.db")
    assert copy.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 1
    copy.close()
    again = DB(env["tmp"] / "data" / "portal.db")                         # survives a "restart"
    assert again.case(cid)["status"] == "awaiting_review"
    from edge.portal.db import MIGRATIONS
    assert again.one("SELECT version FROM schema_version")["version"] == len(MIGRATIONS)


def test_admin_creates_doctor_and_disables_user(env):
    a = env["app"].test_client()
    login(a, "admin@clinic.test")
    tok = csrf(a, "/admin/users")
    a.post("/admin/users", data={"csrf": tok, "name": "Dr New", "email": "new@clinic.test", "role": "doctor", "password": PW})
    u = env["db"].user_by_email("new@clinic.test")
    assert u["role"] == "doctor"
    a.post(f"/admin/users/{u['id']}/toggle", data={"csrf": tok})
    n = env["app"].test_client()
    assert "disabled" in login(n, "new@clinic.test").get_data(as_text=True)


# --------------------------------------------------------------------------------------- HTTPS / reverse proxy
def test_tls_name_allowlist():
    import socket
    from edge.portal.app import tls_name_allowed
    assert tls_name_allowed("10.219.120.61") and tls_name_allowed("192.168.1.5") and tls_name_allowed("fe80::1")
    assert tls_name_allowed("localhost") and tls_name_allowed(f"{socket.gethostname()}.local")
    assert not tls_name_allowed("8.8.8.8") and not tls_name_allowed("evil.example.com")
    assert not tls_name_allowed("other-device.local")


def test_tls_ask_only_from_loopback_without_proxy_headers(env):
    c = env["app"].test_client()
    assert c.get("/_tls_ask?domain=10.0.0.7").status_code == 200
    assert c.get("/_tls_ask?domain=example.com").status_code == 403
    assert c.get("/_tls_ask?domain=10.0.0.7", environ_base={"REMOTE_ADDR": "10.0.0.9"}).status_code == 404
    assert c.get("/_tls_ask?domain=10.0.0.7", headers={"X-Forwarded-For": "10.0.0.9"}).status_code == 404


def test_behind_proxy_secure_cookie_and_client_ip(tmp_path):
    from edge.agent.orchestrator import Assistant
    key = tmp_path / "portal.key"
    generate_key(key)
    app = create_app(StubEngine(), {"version": "t"}, tmp_path / "data", key, assistant=Assistant(llm=NoLLM()),
                     behind_proxy=True)
    db = app.extensions["portal"]["db"]
    db.create_user("doc@clinic.test", "Dr Test", "doctor", auth.hash_password(PW))
    c = app.test_client()
    hdr = {"X-Forwarded-For": "10.0.0.42", "X-Forwarded-Proto": "https"}
    page = c.get("/login", headers=hdr, base_url="https://localhost")
    tok = re.search(r'name="csrf" value="([^"]+)"', page.get_data(as_text=True)).group(1)
    r = c.post("/login", data={"csrf": tok, "email": "doc@clinic.test", "password": PW}, headers=hdr,
               base_url="https://localhost")
    assert r.status_code == 302
    cookie = r.headers.get("Set-Cookie", "")
    assert cookie.startswith("__Host-ne_session=") and "Secure" in cookie and "HttpOnly" in cookie
    ips = [row["ip"] for row in db.conn.execute("SELECT ip FROM audit WHERE action = 'login'")]
    assert ips == ["10.0.0.42"]


def test_login_redirect_rejects_offsite_targets():
    from edge.portal.app import safe_next
    assert safe_next("/doctor/case/abc") == "/doctor/case/abc"
    for bad in ("//evil.example", "/\\evil.example", "/x\\y", "https://evil.example", "evil", "/a\r\nSet-Cookie:x", ""):
        assert safe_next(bad) is None, bad


def test_decompression_bomb_rejected_from_header():
    import struct
    import zlib
    from neuroedge.preprocess import _read_raster, raster_dims
    ihdr = struct.pack(">IIBBBBB", 60000, 60000, 8, 0, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
           + b"\x00" * 64)
    assert raster_dims(png) == (60000, 60000)
    with pytest.raises(ValueError, match="too large"):
        _read_raster(png)


def test_raster_dims_real_formats():
    import cv2
    import numpy as np
    from neuroedge.preprocess import raster_dims
    img = np.zeros((123, 77), np.uint8)
    for ext in (".png", ".jpg", ".bmp"):
        ok, buf = cv2.imencode(ext, img)
        assert raster_dims(buf.tobytes()) == (123, 77), ext

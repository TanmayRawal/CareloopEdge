"""End-to-end: publish (sign + encrypt) -> server -> enroll -> pull -> decrypt + verify on the device."""

import hashlib
import io
import json
import urllib.error

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("model_signing")
from fastapi.testclient import TestClient  # noqa: E402

from edge.sync import Device  # noqa: E402
from neuroedge.bundle import BUNDLE_FILES, generate_signing_key  # noqa: E402
from server.app import create_app  # noqa: E402
from server.publish import build_payload  # noqa: E402

ADMIN = "admin-token-for-tests-0123456789"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_opener(client: TestClient):
    """Route urllib requests made by edge.sync into the in-process FastAPI app."""
    def opener(req, timeout=None):
        path = req.full_url.split("://", 1)[1].split("/", 1)[1]
        r = client.request(req.get_method(), "/" + path, headers=dict(req.header_items()), content=req.data)
        if r.status_code >= 400:
            raise urllib.error.HTTPError(req.full_url, r.status_code, r.text, {}, None)
        return _Resp(r.content)
    return opener


@pytest.fixture()
def world(tmp_path):
    app = create_app(str(tmp_path / "server"), hashlib.sha256(ADMIN.encode()).hexdigest())
    client = TestClient(app)
    keys = tmp_path / "keys"
    keys.mkdir()
    generate_signing_key(keys / "priv.pem", keys / "pub.pem")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    files = {f: f"content of {f}".encode() * 50 for f in BUNDLE_FILES}
    files["bundle.json"] = json.dumps({"task": "mri", "class_names": ["a", "b"], "preprocess_fingerprint": "x"}).encode()
    for f, d in files.items():
        (bundle / f).write_bytes(d)
    return {"client": client, "keys": keys, "bundle": bundle, "files": files, "tmp": tmp_path}


def _publish(w, version="1.0.0", priv=None):
    import base64
    pub = base64.b64decode(w["client"].get("/v1/server-key").json()["x25519_public_b64"])
    payload = build_payload(w["bundle"], "mri", version, priv or w["keys"] / "priv.pem", pub)
    return w["client"].post(f"/v1/admin/models/mri/{version}", json=payload, headers={"Authorization": f"Bearer {ADMIN}"})


def _enroll(w, name="pi"):
    tok = w["client"].post("/v1/admin/enrollment-tokens", headers={"Authorization": f"Bearer {ADMIN}"}).json()["token"]
    dev = Device(w["tmp"] / f"device-{name}", opener=make_opener(w["client"]))
    dev_id = dev.enroll("http://registry.test", tok, name, w["keys"] / "pub.pem")
    return dev, dev_id, tok


def test_full_roundtrip(world):
    assert _publish(world).status_code == 200
    dev, _, _ = _enroll(world)
    assert "installed" in dev.pull("mri")
    files, version = dev.load_bundle("mri")
    assert version == "1.0.0" and files == world["files"]
    cards = world["client"].get("/v1/models").json()
    assert cards[0]["name"] == "mri" and "blob" not in json.dumps(cards).lower()


def test_versions_immutable_and_admin_required(world):
    assert _publish(world).status_code == 200
    assert _publish(world).status_code == 409
    r = world["client"].post("/v1/admin/enrollment-tokens", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_enrollment_token_single_use(world):
    _, _, tok = _enroll(world)
    with pytest.raises(urllib.error.HTTPError) as e:
        Device(world["tmp"] / "d2", opener=make_opener(world["client"])).enroll(
            "http://registry.test", tok, "pi2", world["keys"] / "pub.pem")
    assert e.value.code == 403


def test_tampered_ciphertext_rejected(world):
    _publish(world)
    dev, _, _ = _enroll(world)
    dev.pull("mri")
    blob = dev.home / "models" / "mri" / "1.0.0" / "blob.enc"
    data = bytearray(blob.read_bytes())
    data[100] ^= 1
    blob.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="no verifiable model bundle"):
        dev.load_bundle("mri")


def test_wrong_publisher_key_rejected(world):
    other = world["tmp"] / "other"
    other.mkdir()
    generate_signing_key(other / "priv.pem", other / "pub.pem")
    _publish(world, priv=other / "priv.pem")          # signed by an attacker key
    dev, _, _ = _enroll(world)                        # device pins the genuine key
    with pytest.raises(Exception):
        dev.pull("mri")
    assert not (dev.home / "models" / "mri" / "1.0.0").exists()


def test_revoked_device_blocked(world):
    _publish(world)
    dev, dev_id, _ = _enroll(world)
    world["client"].post(f"/v1/admin/devices/{dev_id}/revoke", headers={"Authorization": f"Bearer {ADMIN}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        dev.pull("mri")
    assert e.value.code == 403


def test_other_device_cannot_use_wrapped_key(world):
    _publish(world)
    dev_a, _, _ = _enroll(world, "a")
    dev_b, _, _ = _enroll(world, "b")
    dev_a.pull("mri")
    # copy A's encrypted model + A's wrapped key into B: B's private key cannot unwrap it
    import shutil
    shutil.copytree(dev_a.home / "models", dev_b.home / "models")
    with pytest.raises(RuntimeError):
        dev_b.load_bundle("mri")


def test_nan_metric_published_as_null_and_raw_nan_rejected(world):
    import base64
    meta = json.loads(world["files"]["bundle.json"])
    meta["metrics"] = {"external": {"macro_auroc": float("nan"), "acc": 0.9}}
    (world["bundle"] / "bundle.json").write_bytes(json.dumps(meta).encode())
    assert _publish(world).status_code == 200                      # publisher maps NaN -> null
    dev, _, _ = _enroll(world)
    assert "installed" in dev.pull("mri")                          # /latest serves standard JSON
    card = world["client"].get("/v1/models").json()[0]["card"]
    assert card["metrics"]["external"]["macro_auroc"] is None
    pub = base64.b64decode(world["client"].get("/v1/server-key").json()["x25519_public_b64"])
    raw = build_payload(world["bundle"], "mri", "2.0.0", world["keys"] / "priv.pem", pub)
    raw["manifest"]["card"]["metrics"] = {"x": float("nan")}       # a non-compliant client
    r = world["client"].post("/v1/admin/models/mri/2.0.0", content=json.dumps(raw).encode(),
                             headers={"Authorization": f"Bearer {ADMIN}", "Content-Type": "application/json"})
    assert r.status_code == 422


def test_model_revocation_falls_back_and_blocks_download(world):
    assert _publish(world, "1.0.0").status_code == 200
    assert _publish(world, "1.1.0").status_code == 200
    dev, _, _ = _enroll(world)
    h = {"Authorization": f"Bearer {ADMIN}"}
    assert world["client"].post("/v1/admin/models/mri/1.1.0/revoke").status_code in (401, 403)   # admin only
    assert world["client"].post("/v1/admin/models/mri/1.1.0/revoke", headers=h).status_code == 200
    assert "1.0.0 installed" in dev.pull("mri")                  # latest skips the revoked version
    import urllib.error
    with pytest.raises(urllib.error.HTTPError):
        dev._http("/v1/models/mri/1.1.0/blob", raw=True)          # and it cannot be downloaded


def test_admin_hash_read_from_data_volume(tmp_path, monkeypatch):
    monkeypatch.delenv("NE_ADMIN_TOKEN_SHA256", raising=False)
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "admin.env").write_text(f"NE_ADMIN_TOKEN_SHA256={hashlib.sha256(ADMIN.encode()).hexdigest()}\n")
    c = TestClient(create_app(str(tmp_path / "srv")))
    assert c.post("/v1/admin/enrollment-tokens", headers={"Authorization": f"Bearer {ADMIN}"}).status_code == 200
    assert c.post("/v1/admin/enrollment-tokens", headers={"Authorization": "Bearer wrong"}).status_code in (401, 403)

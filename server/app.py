"""NeuroEdge model registry (FastAPI). Run behind Caddy for TLS (see docker-compose.yml).

Public
  GET  /health
  GET  /v1/models                       model cards (metrics, intended use, limitations) — no weights
  GET  /v1/server-key                   server X25519 public key (publishers wrap data keys to it)
Devices (Authorization: Bearer nek_<device_id>_<secret>)
  POST /v1/devices/enroll               one-time enrollment token + device X25519 public key -> API key
  GET  /v1/models/{name}/latest         manifest + OMS signature + data key re-wrapped for THIS device
  GET  /v1/models/{name}/{version}/blob AES-256-GCM ciphertext (useless without the device's key)
Admin (Authorization: Bearer <admin token>; only its SHA-256 is configured on the server)
  POST /v1/admin/enrollment-tokens      issue a single-use, 24 h enrollment token
  POST /v1/admin/models/{name}/{version}  publish (immutable versions)
  POST /v1/admin/models/{name}/{version}/revoke  withdraw a version (no longer served as latest or downloadable)
  POST /v1/admin/devices/{id}/revoke    revoke a device
  GET  /v1/admin/devices | /v1/admin/audit

The server never sees model plaintext: publishers upload ciphertext + a data key wrapped to the server
key; the server only re-wraps that key for authenticated, non-revoked devices.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import re
import secrets
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from neuroedge.bundle import context
from neuroedge.security import WrappedKey, unwrap_key, wrap_key
from server.store import Store, sha256_hex

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ph = PasswordHasher()


class EnrollReq(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    name: str = Field(min_length=1, max_length=64)
    public_key_b64: str


class PublishReq(BaseModel):
    manifest: dict
    signature_b64: str
    wrapped_key: dict
    blob_b64: str


# --------------------------------------------------------------------------- rate limit
class RateLimiter:
    """Sliding-window limiter per (bucket, client). In-memory: one server process."""

    def __init__(self):
        self.events: dict[tuple[str, str], deque] = defaultdict(deque)

    def check(self, bucket: str, client: str, limit: int, window_s: float = 60.0) -> None:
        q = self.events[(bucket, client)]
        now = time.monotonic()
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, "rate limit exceeded")
        q.append(now)


def create_app(data_dir: str | None = None, admin_token_sha256: str | None = None) -> FastAPI:
    store = Store(data_dir or os.environ.get("NE_DATA", "server_data"))
    store.ensure_server_key()
    admin_hash = admin_token_sha256 or os.environ.get("NE_ADMIN_TOKEN_SHA256", "")
    env_file = store.root / "admin.env"            # written by `manage init` inside the data volume
    if not admin_hash and env_file.exists():
        admin_hash = dict(line.split("=", 1) for line in env_file.read_text().split() if "=" in line).get(
            "NE_ADMIN_TOKEN_SHA256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", admin_hash):
        raise RuntimeError("NE_ADMIN_TOKEN_SHA256 must be set (run: python -m server.manage init)")
    limiter = RateLimiter()
    app = FastAPI(title="NeuroEdge model registry", version="1.0", docs_url=None, redoc_url=None)

    def client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def bearer(authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        return authorization[7:].strip()

    def require_admin(request: Request, authorization: str | None = Header(None)) -> None:
        ip = client_ip(request)
        limiter.check("admin", ip, 30)
        if not hmac.compare_digest(sha256_hex(bearer(authorization).encode()), admin_hash):
            store.audit("anonymous", "admin_auth_failed", ip=ip)
            limiter.check("auth_fail", ip, 10)
            raise HTTPException(401, "invalid admin token")

    def require_device(request: Request, authorization: str | None = Header(None)) -> dict:
        ip = client_ip(request)
        limiter.check("device", ip, 120)
        m = re.fullmatch(r"nek_([0-9a-f]{32})_([A-Za-z0-9_-]{20,})", bearer(authorization))
        dev = store.get_device(m.group(1)) if m else None
        ok = False
        if dev is not None:
            try:
                ok = _ph.verify(dev["secret_hash"], m.group(2))
            except VerifyMismatchError:
                ok = False
        if not ok:
            store.audit("anonymous", "device_auth_failed", ip=ip)
            limiter.check("auth_fail", ip, 10)
            raise HTTPException(401, "invalid device credentials")
        if dev["revoked"]:
            store.audit(dev["id"], "revoked_device_attempt", ip=ip)
            raise HTTPException(403, "device revoked")
        store.touch_device(dev["id"])
        return dev

    # ------------------------------------------------------------------- public
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models(request: Request):
        limiter.check("public", client_ip(request), 60)
        return store.list_models()

    @app.get("/v1/server-key")
    def server_key():
        return {"x25519_public_b64": base64.b64encode(store.server_public_key()).decode()}

    # ------------------------------------------------------------------- devices
    @app.post("/v1/devices/enroll")
    def enroll(req: EnrollReq, request: Request):
        ip = client_ip(request)
        limiter.check("enroll", ip, 10)
        try:
            pub = base64.b64decode(req.public_key_b64, validate=True)
        except ValueError:
            raise HTTPException(400, "bad public key encoding")
        if len(pub) != 32:
            raise HTTPException(400, "public key must be 32 raw bytes (X25519)")
        device_id = secrets.token_hex(16)
        if not store.consume_enrollment_token(req.token, device_id):
            store.audit("anonymous", "enroll_rejected", ip=ip)
            raise HTTPException(403, "invalid, used or expired enrollment token")
        secret = secrets.token_urlsafe(32)
        store.add_device(device_id, req.name, pub, _ph.hash(secret))
        store.audit(device_id, "enrolled", {"name": req.name}, ip)
        return {"device_id": device_id, "api_key": f"nek_{device_id}_{secret}"}   # shown once

    @app.get("/v1/models/{name}/latest")
    def latest(name: str, request: Request, dev: dict = Depends(require_device)):
        m = store.latest_model(name)
        if m is None:
            raise HTTPException(404, "no such model")
        ctx = context(name, m["version"])
        data_key = unwrap_key(WrappedKey.from_dict(m["wrapped_key"]), store.server_private_key(), ctx)
        rewrapped = wrap_key(data_key, dev["public_key"], ctx)
        store.audit(dev["id"], "manifest", {"model": name, "version": m["version"]}, client_ip(request))
        return {"name": name, "version": m["version"], "manifest": m["manifest"],
                "signature_b64": base64.b64encode(m["signature"]).decode(),
                "wrapped_key": rewrapped.to_dict(), "blob_sha256": m["blob_sha256"],
                "blob_url": f"/v1/models/{name}/{m['version']}/blob"}

    @app.get("/v1/models/{name}/{version}/blob")
    def blob(name: str, version: str, request: Request, dev: dict = Depends(require_device)):
        m = store.get_model(name, version)
        if m is None or m["revoked"]:
            raise HTTPException(404, "no such model version")
        store.audit(dev["id"], "download", {"model": name, "version": version}, client_ip(request))
        return FileResponse(store.blob_path(name, version), media_type="application/octet-stream")

    # ------------------------------------------------------------------- admin
    @app.post("/v1/admin/enrollment-tokens", dependencies=[Depends(require_admin)])
    def new_token(request: Request):
        token = store.new_enrollment_token()
        store.audit("admin", "enrollment_token_issued", ip=client_ip(request))
        return {"token": token, "expires_in_s": 24 * 3600}

    @app.post("/v1/admin/models/{name}/{version}", dependencies=[Depends(require_admin)])
    def publish(name: str, version: str, req: PublishReq, request: Request):
        if not (NAME_RE.fullmatch(name) and NAME_RE.fullmatch(version)):
            raise HTTPException(400, "invalid name/version")
        if store.model_exists(name, version):
            raise HTTPException(409, "version already published (versions are immutable)")
        try:   # a manifest the server could not serve back as standard JSON must never be stored
            json.dumps(req.manifest, allow_nan=False)
        except ValueError:
            raise HTTPException(422, "manifest contains NaN/Infinity; publish undefined values as null")
        blob = base64.b64decode(req.blob_b64)
        if sha256_hex(blob) != req.manifest.get("blob_sha256"):
            raise HTTPException(400, "blob hash does not match manifest")
        try:   # the wrapped key must be unwrappable by the server for this exact name/version
            unwrap_key(WrappedKey.from_dict(req.wrapped_key), store.server_private_key(), context(name, version))
        except Exception:
            raise HTTPException(400, "wrapped key is not addressed to this server / name / version")
        store.add_model(name, version, req.manifest, base64.b64decode(req.signature_b64), req.wrapped_key, blob)
        store.audit("admin", "published", {"model": name, "version": version}, client_ip(request))
        return {"published": f"{name}/{version}", "blob_sha256": sha256_hex(blob)}

    @app.post("/v1/admin/devices/{device_id}/revoke", dependencies=[Depends(require_admin)])
    def revoke(device_id: str, request: Request):
        if not store.revoke_device(device_id):
            raise HTTPException(404, "no such device")
        store.audit("admin", "device_revoked", {"device": device_id}, client_ip(request))
        return {"revoked": device_id}

    @app.post("/v1/admin/models/{name}/{version}/revoke", dependencies=[Depends(require_admin)])
    def revoke_model(name: str, version: str, request: Request):
        """Withdraw a version: devices stop receiving it as latest and cannot download it; the record stays (audit)."""
        if not store.revoke_model(name, version):
            raise HTTPException(404, "no such model version")
        store.audit("admin", "model_revoked", {"model": name, "version": version}, client_ip(request))
        return {"revoked": f"{name}/{version}"}

    @app.get("/v1/admin/devices", dependencies=[Depends(require_admin)])
    def devices():
        return store.list_devices()

    @app.get("/v1/admin/audit", dependencies=[Depends(require_admin)])
    def audit(limit: int = 200):
        rows = store.q("SELECT ts, actor, action, detail, ip FROM audit ORDER BY ts DESC LIMIT ?", (min(limit, 1000),))
        return [dict(zip(("ts", "actor", "action", "detail", "ip"), r)) for r in rows]

    return app
    # run with:  uvicorn --factory server.app:create_app --host 0.0.0.0 --port 8000

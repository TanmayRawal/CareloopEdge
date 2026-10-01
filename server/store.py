"""Persistent state for the model registry: SQLite metadata + encrypted blobs on disk.

Layout under NE_DATA (default ./server_data):
  registry.sqlite3                 devices, enrollment tokens, model versions, audit log
  blobs/<name>/<version>.enc       AES-256-GCM ciphertext of the packed bundle (never plaintext)
  secrets/server_x25519.key        server private key (raw 32 B, 0600) — unwraps publisher-wrapped keys
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from neuroedge.security import generate_device_keypair

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, public_key BLOB NOT NULL, secret_hash TEXT NOT NULL,
  created REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, last_seen REAL);
CREATE TABLE IF NOT EXISTS enrollment_tokens (
  token_sha256 TEXT PRIMARY KEY, created REAL NOT NULL, expires REAL NOT NULL, used_by TEXT);
CREATE TABLE IF NOT EXISTS models (
  name TEXT NOT NULL, version TEXT NOT NULL, created REAL NOT NULL, manifest TEXT NOT NULL,
  signature BLOB NOT NULL, wrapped_key TEXT NOT NULL, blob_sha256 TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (name, version));
CREATE TABLE IF NOT EXISTS audit (
  ts REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT, ip TEXT);
"""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        (self.root / "secrets").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "registry.sqlite3", check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ server key
    @property
    def server_key_path(self) -> Path:
        return self.root / "secrets" / "server_x25519.key"

    def ensure_server_key(self) -> None:
        if not self.server_key_path.exists():
            priv, _ = generate_device_keypair()
            self.server_key_path.write_bytes(priv)
            try:
                os.chmod(self.server_key_path, 0o600)
            except OSError:
                pass

    def server_private_key(self) -> bytes:
        return self.server_key_path.read_bytes()

    def server_public_key(self) -> bytes:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

        k = X25519PrivateKey.from_private_bytes(self.server_private_key())
        return k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    # ------------------------------------------------------------------ helpers
    def q(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    def audit(self, actor: str, action: str, detail: dict | None = None, ip: str | None = None) -> None:
        self.q("INSERT INTO audit VALUES (?,?,?,?,?)", (time.time(), actor, action, json.dumps(detail or {}), ip))

    # ------------------------------------------------------------------ enrollment
    def new_enrollment_token(self, ttl_s: int = 24 * 3600) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self.q("INSERT INTO enrollment_tokens VALUES (?,?,?,NULL)", (sha256_hex(token.encode()), now, now + ttl_s))
        return token

    def consume_enrollment_token(self, token: str, device_id: str) -> bool:
        h = sha256_hex(token.encode())
        with self._lock:
            row = self.db.execute("SELECT expires, used_by FROM enrollment_tokens WHERE token_sha256=?", (h,)).fetchone()
            if row is None or row[1] is not None or row[0] < time.time():
                return False
            self.db.execute("UPDATE enrollment_tokens SET used_by=? WHERE token_sha256=?", (device_id, h))
            return True

    # ------------------------------------------------------------------ devices
    def add_device(self, device_id: str, name: str, public_key: bytes, secret_hash: str) -> None:
        self.q("INSERT INTO devices (id,name,public_key,secret_hash,created) VALUES (?,?,?,?,?)",
               (device_id, name, public_key, secret_hash, time.time()))

    def get_device(self, device_id: str) -> dict | None:
        r = self.q("SELECT id,name,public_key,secret_hash,revoked FROM devices WHERE id=?", (device_id,))
        if not r:
            return None
        return dict(zip(("id", "name", "public_key", "secret_hash", "revoked"), r[0]))

    def touch_device(self, device_id: str) -> None:
        self.q("UPDATE devices SET last_seen=? WHERE id=?", (time.time(), device_id))

    def revoke_device(self, device_id: str) -> bool:
        with self._lock:
            return self.db.execute("UPDATE devices SET revoked=1 WHERE id=?", (device_id,)).rowcount == 1

    def list_devices(self) -> list[dict]:
        rows = self.q("SELECT id,name,created,revoked,last_seen FROM devices ORDER BY created")
        return [dict(zip(("id", "name", "created", "revoked", "last_seen"), r)) for r in rows]

    # ------------------------------------------------------------------ models
    def blob_path(self, name: str, version: str) -> Path:
        return self.root / "blobs" / name / f"{version}.enc"

    def add_model(self, name: str, version: str, manifest: dict, signature: bytes, wrapped_key: dict, blob: bytes) -> None:
        p = self.blob_path(name, version)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, p)
        self.q("INSERT INTO models VALUES (?,?,?,?,?,?,?,0)", (name, version, time.time(), json.dumps(manifest),
                                                              signature, json.dumps(wrapped_key), sha256_hex(blob)))

    def revoke_model(self, name: str, version: str) -> bool:
        with self._lock:
            return self.db.execute("UPDATE models SET revoked=1 WHERE name=? AND version=?", (name, version)).rowcount == 1

    def model_exists(self, name: str, version: str) -> bool:
        return bool(self.q("SELECT 1 FROM models WHERE name=? AND version=?", (name, version)))

    def latest_model(self, name: str) -> dict | None:
        r = self.q("SELECT version,manifest,signature,wrapped_key,blob_sha256 FROM models "
                   "WHERE name=? AND revoked=0 ORDER BY created DESC LIMIT 1", (name,))
        if not r:
            return None
        v, man, sig, wk, h = r[0]
        return {"name": name, "version": v, "manifest": json.loads(man), "signature": sig,
                "wrapped_key": json.loads(wk), "blob_sha256": h}

    def get_model(self, name: str, version: str) -> dict | None:
        r = self.q("SELECT manifest,blob_sha256,revoked FROM models WHERE name=? AND version=?", (name, version))
        return None if not r else {"manifest": json.loads(r[0][0]), "blob_sha256": r[0][1], "revoked": r[0][2]}

    def list_models(self) -> list[dict]:
        rows = self.q("SELECT name, version, manifest, created FROM models WHERE revoked=0 ORDER BY created")
        latest: dict[str, dict] = {}
        for name, version, man, created in rows:
            latest[name] = {"name": name, "latest_version": version, "card": json.loads(man).get("card", {}),
                            "published": created}
        return list(latest.values())

"""Two-step sign-in for staff: time-based one-time codes (RFC 6238, SHA-1, 6 digits, 30 s) from any authenticator
app (Google Authenticator, Microsoft Authenticator, ...). Standard library only.

* The shared secret is stored encrypted with the portal key (BlobStore.seal), never in clear.
* A code is accepted for the current step +/- 1 (clock drift) and only once (the last used step is stored, so a
  code seen over someone's shoulder cannot be replayed).
* 8 single-use recovery codes, stored as argon2 hashes, for a lost phone.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import struct
import time
import urllib.parse

from edge.portal import auth

STEP, DIGITS, WINDOW = 30, 6, 1


def new_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _key(secret_b32: str) -> bytes:
    s = secret_b32.replace(" ", "").upper()
    return base64.b32decode(s + "=" * (-len(s) % 8))


def code_at(secret_b32: str, step: int, digits: int = DIGITS, digest=hashlib.sha1) -> str:
    mac = hmac.new(_key(secret_b32), struct.pack(">Q", step), digest).digest()
    off = mac[-1] & 0x0F
    val = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return f"{val:0{digits}d}"


def match(secret_b32: str, code: str, last_step: int | None, now: float | None = None) -> int | None:
    """The matching step (to store as last used), or None. Rejects reuse of a step <= last_step."""
    code = "".join(ch for ch in (code or "") if ch.isdigit())
    if len(code) != DIGITS:
        return None
    step = int((time.time() if now is None else now) // STEP)
    for s in range(step - WINDOW, step + WINDOW + 1):
        if (last_step is None or s > last_step) and hmac.compare_digest(code_at(secret_b32, s), code):
            return s
    return None


def pretty(secret_b32: str) -> str:
    return " ".join(secret_b32[i:i + 4] for i in range(0, len(secret_b32), 4))


def otpauth_uri(secret_b32: str, account: str, issuer: str) -> str:
    label = urllib.parse.quote(f"{issuer}:{account}")
    q = urllib.parse.urlencode({"secret": secret_b32, "issuer": issuer, "digits": DIGITS, "period": STEP})
    return f"otpauth://totp/{label}?{q}"


def new_recovery_codes(n: int = 8) -> tuple[list[str], str]:
    """(codes to show once, JSON of their hashes to store)."""
    codes = [f"{secrets.token_hex(2)}-{secrets.token_hex(2)}" for _ in range(n)]
    return codes, json.dumps([auth.hash_password(c) for c in codes])


def use_recovery_code(stored_json: str | None, code: str) -> str | None:
    """New JSON with the used code removed, or None if the code is not valid."""
    hashes = json.loads(stored_json or "[]")
    code = (code or "").strip().lower()
    for i, h in enumerate(hashes):
        if h and auth.verify_password({"pw_hash": h}, code):
            hashes[i] = None
            return json.dumps(hashes)
    return None


def remaining_recovery(stored_json: str | None) -> int:
    return sum(1 for h in json.loads(stored_json or "[]") if h)

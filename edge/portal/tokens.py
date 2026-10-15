"""Single-use, expiring tokens for links sent by e-mail: account invite, password reset, "I'll be there" /
"I can't make it" on appointment reminders. Only the SHA-256 of a token is stored, so a copy of the database does
not contain usable links. A token is consumed atomically (UPDATE ... WHERE used_at IS NULL), so it works once.
"""

from __future__ import annotations

import hashlib
import secrets
import time

TTL = {"invite": 7 * 86400, "reset": 3600, "appt_yes": 30 * 86400, "appt_no": 30 * 86400}


def _h(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def issue(db, purpose: str, user_id: int, ref_id: int | None = None, now: float | None = None) -> str:
    now = time.time() if now is None else now
    raw = secrets.token_urlsafe(32)
    with db.tx() as c:
        if purpose in ("invite", "reset"):               # a new link replaces older unused ones
            c.execute("UPDATE tokens SET used_at = ? WHERE user_id = ? AND purpose = ? AND used_at IS NULL",
                      (now, user_id, purpose))
        c.execute("INSERT INTO tokens (purpose,token_hash,user_id,ref_id,created,expires) VALUES (?,?,?,?,?,?)",
                  (purpose, _h(raw), user_id, ref_id, now, now + TTL[purpose]))
    return raw


def peek(db, purpose: str, raw: str, now: float | None = None):
    """The valid (unused, unexpired) token row, without consuming it."""
    now = time.time() if now is None else now
    if not raw or len(raw) > 100:
        return None
    return db.one("SELECT * FROM tokens WHERE token_hash = ? AND purpose = ? AND used_at IS NULL AND expires > ?",
                  (_h(raw), purpose, now))


def consume(db, purpose: str, raw: str, now: float | None = None):
    now = time.time() if now is None else now
    row = peek(db, purpose, raw, now)
    if row is None:
        return None
    with db.tx() as c:
        if c.execute("UPDATE tokens SET used_at = ? WHERE id = ? AND used_at IS NULL", (now, row["id"])).rowcount != 1:
            return None
    return row

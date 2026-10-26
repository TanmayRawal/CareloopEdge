"""Authentication and authorisation for the portal.

* Passwords: argon2id (argon2-cffi defaults), never stored or logged in clear. Policy: >= 10 characters.
* Brute force: per-account lockout (5 failures -> 15 min) + per-IP rate limit on the login endpoint.
* Sessions: Flask signed cookie, HttpOnly + SameSite=Strict, cleared on login (no fixation),
  30-minute idle timeout, 12-hour absolute lifetime.
* Authorisation: every view declares its roles with @require(...); object-level checks (a patient may
  only touch their own cases) live in the views via can_view_case().
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from functools import wraps

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from flask import abort, g, redirect, request, session, url_for

_ph = PasswordHasher()
MIN_PASSWORD = 10
MAX_FAILURES, LOCK_SECONDS = 5, 15 * 60
IDLE_SECONDS, ABSOLUTE_SECONDS = 30 * 60, 12 * 3600
KIOSK_IDLE_SECONDS = 5 * 60          # the device's own, shared screen: a forgotten session closes sooner


def on_kiosk() -> bool:
    """This browser is the device's shared screen (cookie set by /device?kiosk=1)."""
    from flask import request
    return request.cookies.get("ne_kiosk") == "1"


def on_device() -> bool:
    """The request comes from the device's own screen: a client on this machine that asked for localhost. Unlike the
    kiosk cookie this cannot be set from elsewhere: LAN computers have their own address, and Tailscale Funnel
    visitors (who also reach Caddy from loopback) carry their address in X-Forwarded-For and the public host name."""
    from flask import request
    host = request.host if request.host.endswith("]") else request.host.rsplit(":", 1)[0]
    return request.remote_addr in ("127.0.0.1", "::1") and host.strip("[]") in ("localhost", "127.0.0.1", "::1")


def patient_intake_here(db) -> bool:
    """Patients may sign up and upload scans from this browser (department setting: only at the device, or anywhere)."""
    return db.setting("upload_device_only") != "1" or on_device()
# A real hash of a random string: verifying against it for unknown e-mails keeps response time constant,
# so the login form does not reveal which accounts exist.
_DUMMY = _ph.hash("not-a-real-password-#")


def hash_password(pw: str) -> str:
    return _ph.hash(pw)


def password_problem(pw: str, email: str = "") -> str | None:
    if len(pw) < MIN_PASSWORD:
        return f"Password must be at least {MIN_PASSWORD} characters."
    if email and pw.lower() == email.lower():
        return "Password must not be your e-mail address."
    if len(set(pw)) < 4:
        return "Password is too simple."
    return None


class RateLimiter:
    def __init__(self, limit: int, window_s: float):
        self.limit, self.window = limit, window_s
        self.events: dict[str, deque] = defaultdict(deque)

    def hit(self, key: str) -> bool:
        """Record an event; False if the key exceeded the limit within the window."""
        now = time.monotonic()
        if len(self.events) > 5000:                      # drop keys whose events have all expired
            for k in [k for k, q in self.events.items() if not q or now - q[-1] > self.window]:
                del self.events[k]
        q = self.events[key]
        while q and now - q[0] > self.window:
            q.popleft()
        q.append(now)
        return len(q) <= self.limit


def check_login(db, email: str, password: str) -> tuple[object | None, str | None]:
    """(user row, None) on success, (None, message) otherwise. Updates lockout counters."""
    user = db.user_by_email(email)
    if user is None or not user["pw_hash"]:            # unknown, or a walk-in patient who has not accepted the invite
        try:
            _ph.verify(_DUMMY, password)
        except (VerificationError, InvalidHashError):
            pass
        return None, "Incorrect e-mail or password."
    if not user["active"]:
        return None, "This account is disabled. Contact the clinic administrator."
    if user["locked_until"] and user["locked_until"] > time.time():
        mins = int((user["locked_until"] - time.time()) // 60) + 1
        return None, f"Too many failed attempts. Try again in {mins} minute(s)."
    try:
        _ph.verify(user["pw_hash"], password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        fails = user["failed_logins"] + 1
        locked = time.time() + LOCK_SECONDS if fails >= MAX_FAILURES else None
        with db.tx() as c:
            c.execute("UPDATE users SET failed_logins = ?, locked_until = ? WHERE id = ?",
                      (0 if locked else fails, locked, user["id"]))
        db.audit(user["id"], "login_failed" + ("_locked" if locked else ""), ip=request.remote_addr)
        return None, "Incorrect e-mail or password."
    if _ph.check_needs_rehash(user["pw_hash"]):
        with db.tx() as c:
            c.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (_ph.hash(password), user["id"]))
    with db.tx() as c:
        c.execute("UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = ?", (user["id"],))
    return user, None


def verify_password(user, password: str) -> bool:
    try:
        return _ph.verify(user["pw_hash"], password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def start_session(user) -> None:
    session.clear()                                  # new session on login: no fixation, fresh CSRF token
    now = time.time()
    session.update(uid=int(user["id"]), role=user["role"], t_login=now, t_seen=now, pwv=pw_version(user))


def pw_version(user) -> str:
    """Changes whenever the password changes: sessions opened with an older password are no longer accepted."""
    return hashlib.sha256((user["pw_hash"] or "").encode("utf-8")).hexdigest()[:16]


def load_user(db) -> None:
    """before_request: populate g.user, enforcing idle/absolute timeouts and account state."""
    g.user = None
    uid = session.get("uid")
    if uid is None:
        return
    now = time.time()
    idle = KIOSK_IDLE_SECONDS if on_kiosk() else IDLE_SECONDS
    if now - session.get("t_seen", 0) > idle or now - session.get("t_login", 0) > ABSOLUTE_SECONDS:
        session.clear()
        g.expired = True
        return
    user = db.user(uid)
    if (user is None or not user["active"] or user["role"] != session.get("role")
            or session.get("pwv") != pw_version(user)):
        session.clear()
        return
    session["t_seen"] = now
    g.user = user


def require(*roles: str):
    def deco(view):
        @wraps(view)
        def wrapped(*a, **kw):
            if g.get("user") is None:
                return redirect(url_for("auth.login", next=request.path))
            if g.user["role"] not in roles:
                abort(403)
            return view(*a, **kw)
        return wrapped
    return deco


def can_view_case(user, case) -> bool:
    """Object-level rule: patients see only their own cases; doctors and admins see the clinic's cases."""
    if case is None or user is None:
        return False
    return user["role"] in ("doctor", "admin") or case["patient_id"] == user["id"]

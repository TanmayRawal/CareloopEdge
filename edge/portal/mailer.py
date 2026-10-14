"""E-mail delivery for the outbox (SMTP with STARTTLS/SSL, retry with exponential backoff).

Configuration (never stored in the database or editable from the web UI):
  /etc/neuroedge/smtp.env  (root:neuroedge 640, written by `neuroedge-portal smtp-config`)
    NEUROEDGE_SMTP_HOST=smtp.gmail.com   NEUROEDGE_SMTP_PORT=587   NEUROEDGE_SMTP_SECURITY=starttls|ssl|none
    NEUROEDGE_SMTP_USER=...              NEUROEDGE_SMTP_PASSWORD=...   NEUROEDGE_SMTP_FROM="Clinic <noreply@...>"
  or NEUROEDGE_SMTP=file:<dir>   development/demo: write .eml files instead of sending.
Without configuration rows stay queued ("SMTP not configured") and are sent once it is configured.
"""

from __future__ import annotations

import os
import smtplib
import socket
import ssl
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

SMTP_ENV_FILE = Path(os.environ.get("NEUROEDGE_SMTP_ENV", "/etc/neuroedge/smtp.env"))
MAX_ATTEMPTS = 10
BACKOFF = [60, 120, 300, 600, 1800, 3600, 7200, 14400, 21600]       # seconds; then 6 h


def load_config() -> dict:
    cfg = {}
    try:
        for line in SMTP_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    for k, v in os.environ.items():                     # environment overrides the file (dev / tests)
        if k.startswith("NEUROEDGE_SMTP"):
            cfg[k] = v
    return cfg


def status(cfg: dict | None = None) -> dict:
    cfg = load_config() if cfg is None else cfg
    if cfg.get("NEUROEDGE_SMTP", "").startswith("file:"):
        return {"configured": True, "mode": "file", "detail": "Demo mode · e-mails saved as files", "path": cfg["NEUROEDGE_SMTP"][5:]}
    if cfg.get("NEUROEDGE_SMTP_HOST"):
        return {"configured": True, "mode": "smtp",
                "detail": f"{cfg['NEUROEDGE_SMTP_HOST']}:{cfg.get('NEUROEDGE_SMTP_PORT', '587')} as "
                          f"{cfg.get('NEUROEDGE_SMTP_USER') or 'anonymous'}"}
    return {"configured": False, "mode": None, "detail": "SMTP not configured"}


def build_message(row, cfg: dict) -> EmailMessage:
    m = EmailMessage()
    sender = cfg.get("NEUROEDGE_SMTP_FROM") or cfg.get("NEUROEDGE_SMTP_USER") or "noreply@neuroedge.local"
    m["From"], m["To"], m["Subject"] = sender, row["to_email"], row["subject"]
    m["Date"], m["Message-ID"] = formatdate(localtime=True), make_msgid(domain="neuroedge.local")
    m["Auto-Submitted"] = "auto-generated"
    m.set_content(row["text"])
    if row["html"]:
        m.add_alternative(row["html"], subtype="html")
    if row["ics"]:
        m.add_attachment(row["ics"].encode("utf-8"), maintype="text", subtype="calendar",
                         filename="appointment.ics", params={"method": "REQUEST"})
    return m


class Transport:
    """One SMTP connection for a whole flush (opened lazily)."""

    def __init__(self, cfg: dict, smtp_factory=None):
        self.cfg, self.factory, self.conn = cfg, smtp_factory, None

    def send(self, msg: EmailMessage) -> None:
        mode = self.cfg.get("NEUROEDGE_SMTP", "")
        if mode.startswith("file:"):
            d = Path(mode[5:])
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{int(time.time() * 1000)}-{abs(hash(msg['Message-ID'])) % 10**8}.eml").write_bytes(bytes(msg))
            return
        if self.conn is None:
            host = self.cfg["NEUROEDGE_SMTP_HOST"]
            port = int(self.cfg.get("NEUROEDGE_SMTP_PORT", "587"))
            sec = self.cfg.get("NEUROEDGE_SMTP_SECURITY", "starttls").lower()
            ctx = ssl.create_default_context()
            if self.factory:
                self.conn = self.factory(host, port)
            elif sec == "ssl":
                self.conn = smtplib.SMTP_SSL(host, port, timeout=15, context=ctx)
            else:
                self.conn = smtplib.SMTP(host, port, timeout=15)
                if sec == "starttls":
                    self.conn.starttls(context=ctx)
            if self.cfg.get("NEUROEDGE_SMTP_USER"):
                self.conn.login(self.cfg["NEUROEDGE_SMTP_USER"], self.cfg.get("NEUROEDGE_SMTP_PASSWORD", ""))
        self.conn.send_message(msg)

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.quit()
            except Exception:
                pass


def flush(db, now: float | None = None, limit: int = 50, cfg: dict | None = None, smtp_factory=None) -> dict:
    """Send due outbox rows. Returns counts {sent, failed, deferred}."""
    now = time.time() if now is None else now
    cfg = load_config() if cfg is None else cfg
    st = status(cfg)
    counts = {"sent": 0, "failed": 0, "deferred": 0}
    rows = db.q("SELECT * FROM outbox WHERE status = 'queued' AND next_try <= ? ORDER BY next_try LIMIT ?", (now, limit))
    if not rows:
        return counts
    if not st["configured"]:
        with db.tx() as c:
            c.execute("UPDATE outbox SET last_error = 'SMTP not configured' WHERE status = 'queued' AND last_error IS NULL")
        counts["deferred"] = len(rows)
        return counts
    tr = Transport(cfg, smtp_factory)
    try:
        for row in rows:
            try:
                tr.send(build_message(row, cfg))
                with db.tx() as c:
                    c.execute("UPDATE outbox SET status='sent', sent_at=?, attempts=attempts+1, last_error=NULL WHERE id=?",
                              (now, row["id"]))
                counts["sent"] += 1
            except (smtplib.SMTPException, OSError, socket.timeout, KeyError, ValueError) as e:
                attempts = row["attempts"] + 1
                failed = attempts >= MAX_ATTEMPTS or isinstance(e, smtplib.SMTPRecipientsRefused)
                delay = BACKOFF[min(attempts - 1, len(BACKOFF) - 1)]
                with db.tx() as c:
                    c.execute("UPDATE outbox SET status=?, attempts=?, next_try=?, last_error=? WHERE id=?",
                              ("failed" if failed else "queued", attempts, now + delay, f"{type(e).__name__}: {e}"[:300],
                               row["id"]))
                counts["failed" if failed else "deferred"] += 1
                if isinstance(e, (smtplib.SMTPAuthenticationError, OSError)) and not isinstance(e, smtplib.SMTPRecipientsRefused):
                    tr.close()
                    tr = Transport(cfg, smtp_factory)    # connection-level problem: do not hammer the server
                    break
    finally:
        tr.close()
    return counts


def retry(db, outbox_id: int, now: float | None = None) -> None:
    with db.tx() as c:
        c.execute("UPDATE outbox SET status='queued', next_try=?, attempts=0, last_error=NULL WHERE id=? AND status IN ('failed','queued')",
                  (time.time() if now is None else now, outbox_id))

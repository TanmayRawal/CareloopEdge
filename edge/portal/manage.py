"""Portal administration (run on the Pi as the service user; see docs/PI_DEPLOY.md).

  python -m edge.portal.manage init        --data /var/lib/neuroedge/portal --key /etc/neuroedge/portal.key
  python -m edge.portal.manage create-user --data ... --role doctor --email dr@clinic.org --name "Dr A. Rao"
          (password read interactively, or from stdin with --password-stdin; never from the command line)
  python -m edge.portal.manage backup      --data ... --dest /media/usb/neuroedge-backups
  python -m edge.portal.manage restore     --data ... --src /media/usb/neuroedge-backups/<timestamp>   (portal stopped)
  python -m edge.portal.manage run-reminders --data ...       reminder rules + send due e-mails (neuroedge-reminders.timer)
  python -m edge.portal.manage smtp-config  [--file /etc/neuroedge/smtp.env]   interactive; password via getpass
  python -m edge.portal.manage smtp-test    --data ... --to you@hospital.org
  python -m edge.portal.manage set-setting  --data ... KEY VALUE   (e.g. base_url https://rpi.local)
  python -m edge.portal.manage verify-audit --data ...       check the tamper-evident audit chain (exit 1 if altered)
  python -m edge.portal.manage reset-2fa    --data ... --email dr@clinic.org   (staff member lost their phone)

Backups contain the SQLite database (consistent online snapshot) and the encrypted blobs. They are only
readable with the key file, which is NOT included: keep a copy of the key somewhere safe and separate.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
import time
from pathlib import Path

from edge.portal.auth import hash_password, password_problem
from edge.portal.db import DB
from edge.portal.storage import generate_key


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--key", required=True, type=Path)
    p = sub.add_parser("create-user")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--role", required=True, choices=["doctor", "admin"])
    p.add_argument("--email", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--password-stdin", action="store_true")
    p = sub.add_parser("backup")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--dest", required=True, type=Path)
    p = sub.add_parser("restore")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--src", required=True, type=Path)
    p = sub.add_parser("run-reminders")
    p.add_argument("--data", required=True, type=Path)
    p = sub.add_parser("smtp-config")
    p.add_argument("--data", type=Path)                          # accepted for symmetry with the other commands; unused
    p.add_argument("--file", type=Path, default=Path(os.environ.get("NEUROEDGE_SMTP_ENV", "/etc/neuroedge/smtp.env")))
    p = sub.add_parser("smtp-test")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--to", required=True)
    p = sub.add_parser("set-setting")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("key")
    p.add_argument("value")
    p = sub.add_parser("verify-audit")
    p.add_argument("--data", required=True, type=Path)
    p = sub.add_parser("reset-2fa")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--email", required=True)
    a = ap.parse_args()

    if a.cmd == "init":
        if a.key.exists():
            print(f"key exists: {a.key} (kept)")
        else:
            generate_key(a.key)
            print(f"created key {a.key} - back it up separately; without it stored scans/reports are unreadable")
        DB(a.data / "portal.db")
        print(f"database ready: {a.data / 'portal.db'}")
    elif a.cmd == "create-user":
        pw = sys.stdin.readline().rstrip("\n") if a.password_stdin else getpass.getpass("Password: ")
        if not a.password_stdin and pw != getpass.getpass("Repeat password: "):
            sys.exit("passwords do not match")
        if (err := password_problem(pw, a.email)):
            sys.exit(err)
        db = DB(a.data / "portal.db")
        if db.user_by_email(a.email):
            sys.exit("a user with this e-mail exists")
        uid = db.create_user(a.email, a.name, a.role, hash_password(pw))
        db.audit(None, "user_created_cli", detail={"user": uid, "role": a.role})
        print(f"created {a.role} {a.name} <{a.email}> (id {uid})")
    elif a.cmd == "backup":
        out = a.dest / time.strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True)
        DB(a.data / "portal.db").backup(out / "portal.db")
        if (a.data / "blobs").exists():
            shutil.copytree(a.data / "blobs", out / "blobs")
        n = sum(1 for _ in (out / "blobs").rglob("*.enc")) if (out / "blobs").exists() else 0
        print(f"backup written to {out} (database + {n} encrypted files; key NOT included)")
    elif a.cmd == "restore":
        if not (a.src / "portal.db").exists():
            sys.exit(f"{a.src} is not a backup")
        a.data.mkdir(parents=True, exist_ok=True)
        for f in ("portal.db", "portal.db-wal", "portal.db-shm"):
            (a.data / f).unlink(missing_ok=True)
        shutil.copy2(a.src / "portal.db", a.data / "portal.db")
        if (a.data / "blobs").exists():
            shutil.rmtree(a.data / "blobs")
        if (a.src / "blobs").exists():
            shutil.copytree(a.src / "blobs", a.data / "blobs")
        print(f"restored {a.src} -> {a.data}")
    elif a.cmd == "run-reminders":
        from edge.portal import mailer, reminders
        db = DB(a.data / "portal.db")
        out = {"rules": reminders.run_once(db), "mail": mailer.flush(db)}
        print(json.dumps(out))
    elif a.cmd == "smtp-config":
        print("Outgoing e-mail (SMTP). Gmail: host smtp.gmail.com, port 587, starttls, and an App Password "
              "(Google account -> Security -> App passwords), not your normal password.")
        host = input("SMTP host: ").strip()
        port = input("Port [587]: ").strip() or "587"
        sec = (input("Security starttls/ssl/none [starttls]: ").strip() or "starttls").lower()
        user = input("Username (e-mail address): ").strip()
        pw = getpass.getpass("Password / app password (hidden): ")
        sender = input(f"From address [{user}]: ").strip() or user
        if sec not in ("starttls", "ssl", "none") or not host or not port.isdigit():
            sys.exit("invalid settings")
        lines = [f"NEUROEDGE_SMTP_HOST={host}", f"NEUROEDGE_SMTP_PORT={port}", f"NEUROEDGE_SMTP_SECURITY={sec}",
                 f"NEUROEDGE_SMTP_USER={user}", f"NEUROEDGE_SMTP_PASSWORD={pw}", f'NEUROEDGE_SMTP_FROM="{sender}"']
        a.file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(a.file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("# written by neuroedge-portal smtp-config\n" + "\n".join(lines) + "\n")
        print(f"saved {a.file} (mode 640). Test it: neuroedge-portal smtp-test --to <your address>")
    elif a.cmd == "smtp-test":
        from edge.portal import mailer
        db = DB(a.data / "portal.db")
        with db.tx() as c:
            c.execute("INSERT INTO outbox (to_email,subject,text,created,next_try) VALUES (?,?,?,?,?)",
                      (a.to, f"Test e-mail · {db.setting('clinic_name')}", "E-mail delivery from the portal works.",
                       time.time(), time.time()))
        print(json.dumps({"status": mailer.status(), "result": mailer.flush(db)}))
        row = db.one("SELECT status, last_error FROM outbox ORDER BY id DESC LIMIT 1")
        print(f"last message: {row['status']}{' — ' + row['last_error'] if row['last_error'] else ''}")
    elif a.cmd == "set-setting":
        db = DB(a.data / "portal.db")
        db.set_setting(a.key, a.value)
        print(f"{a.key} = {db.setting(a.key)}")
    elif a.cmd == "verify-audit":
        res = DB(a.data / "portal.db").verify_audit_chain()
        print(json.dumps(res))
        if not res["ok"]:
            sys.exit(f"audit log altered at entry {res['bad_id']}")
    elif a.cmd == "reset-2fa":
        db = DB(a.data / "portal.db")
        u = db.user_by_email(a.email)
        if u is None or u["role"] == "patient":
            sys.exit("no staff account with that e-mail")
        with db.tx() as c:
            c.execute("UPDATE users SET totp_on = 0, totp_secret = NULL, totp_last_step = NULL, recovery_hashes = NULL "
                      "WHERE id = ?", (u["id"],))
        db.audit(None, "two_step_disabled", detail={"user": u["id"], "by": "console"})
        print(f"two-step sign-in reset for {u['name']}; they can set it up again under Account")


if __name__ == "__main__":
    main()

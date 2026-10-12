"""Portal administration (run on the Pi as the service user; see docs/PI_DEPLOY.md).

  python -m edge.portal.manage init        --data /var/lib/neuroedge/portal --key /etc/neuroedge/portal.key
  python -m edge.portal.manage create-user --data ... --role doctor --email dr@clinic.org --name "Dr A. Rao"
          (password read interactively, or from stdin with --password-stdin; never from the command line)
  python -m edge.portal.manage backup      --data ... --dest /media/usb/neuroedge-backups
  python -m edge.portal.manage restore     --data ... --src /media/usb/neuroedge-backups/<timestamp>   (portal stopped)

Backups contain the SQLite database (consistent online snapshot) and the encrypted blobs. They are only
readable with the key file, which is NOT included: keep a copy of the key somewhere safe and separate.
"""

from __future__ import annotations

import argparse
import getpass
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


if __name__ == "__main__":
    main()

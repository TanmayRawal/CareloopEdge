"""Start the CareLoop-Edge clinical portal.

Production (Pi, systemd + gunicorn): model decrypted in RAM from the enrolled device's verified bundle
  gunicorn -w 1 --threads 4 -b 127.0.0.1:8080 "edge.serve:wsgi()"      (Caddy serves HTTPS on :443 in front)
  env: NEUROEDGE_PORTAL_DATA (default /var/lib/neuroedge/portal), NEUROEDGE_PORTAL_KEY
       (default /etc/neuroedge/portal.key), NEUROEDGE_SECRET (session signing), NEUROEDGE_MODEL,
       NEUROEDGE_BEHIND_PROXY=1 (trust one reverse proxy: client IP + HTTPS-only cookies)
Development (PC): plaintext bundle directory, local data dir; --init-dev creates the key, the database and three
test accounts (random passwords, written to <portal-data>/DEV_ACCOUNTS.txt) on first start
  python -m edge.serve --bundle-dir <DATA_ROOT>/bundles/mri_v1_0.2.0_int8 --portal-data <DATA_ROOT>/portal_dev --init-dev
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set before OpenCV is imported: caps any image the decoder will allocate (decompression bombs).
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(4096 * 4096))

from edge.engine import Engine  # noqa: E402
from edge.portal.app import create_app  # noqa: E402

PI_DATA, PI_KEY = "/var/lib/neuroedge/portal", "/etc/neuroedge/portal.key"


def build(bundle_dir: str | None = None, model: str = "mri", threads: int = 4,
          data_dir: str | None = None, key_file: str | None = None):
    if bundle_dir:
        engine = Engine.from_dir(bundle_dir, threads=threads)
        version = Path(bundle_dir).name
    else:
        from edge.sync import Device

        files, version = Device().load_bundle(model)      # decrypt + verify in memory, rollback on failure
        engine = Engine(files, threads=threads)
    meta = engine.meta
    info = {"class_names": meta["class_names"], "precision": meta.get("precision"), "version": version,
            "metrics": meta.get("metrics", {})}
    data_dir = data_dir or os.environ.get("NEUROEDGE_PORTAL_DATA", PI_DATA)
    key_file = key_file or os.environ.get("NEUROEDGE_PORTAL_KEY", PI_KEY)
    return create_app(engine, info, data_dir, key_file, secret_key=os.environ.get("NEUROEDGE_SECRET"),
                      behind_proxy=os.environ.get("NEUROEDGE_BEHIND_PROXY") == "1")


def seed_dev_accounts(app, data_dir: Path) -> None:
    """First start of a development portal: one admin, doctor and patient with random passwords. Never used on the Pi
    (there, accounts are created with `neuroedge-portal create-user`, passwords typed by a person)."""
    import secrets

    from edge.portal.auth import hash_password
    db = app.extensions["portal"]["db"]
    if db.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
        return
    rows = []
    for role, email, name, profile in (("admin", "admin@neuroedge.test", "Clinic Admin", None),
                                       ("doctor", "doctor@neuroedge.test", "Dr. Asha Rao", None),
                                       ("patient", "patient@neuroedge.test", "Rahul Verma",
                                        {"dob": None, "sex": None, "phone": None})):
        pw = secrets.token_urlsafe(12)
        uid = db.create_user(email, name, role, hash_password(pw), profile)
        if role == "doctor":                             # demo consultation hours: Mon-Sat 09-13 and 14-17
            from edge.portal import scheduling
            scheduling.set_hours(db, uid, [{"weekday": w, "start_min": a, "end_min": b, "slot_min": 20,
                                            "room": "Radiology OPD, Room 3"}
                                           for w in range(6) for a, b in ((540, 780), (840, 1020))])
        rows.append(f"{role:<8} {email:<28} {pw}")
    (data_dir / "DEV_ACCOUNTS.txt").write_text(
        "# NeuroEdge DEV portal test accounts (local only; delete the folder to regenerate). Not real users.\n\n"
        + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"created test accounts: {data_dir / 'DEV_ACCOUNTS.txt'}")


DEMO_PATIENTS = [  # name, dob, sex, phone, district, km, plan kind, due offset (days), plan status, contact outcomes
    ("Sunita Devi", "1968-04-12", "Female", "98xxxx1201", "Barabanki", 64, "repeat_scan", -21, "approved", ["no_answer", "no_answer"]),
    ("Mohammed Irfan", "1975-09-30", "Male", "98xxxx4410", "Lucknow", 8, "consultation", 3, "approved", []),
    ("Kavita Sharma", "1990-01-05", "Female", "97xxxx7781", "Sitapur", 88, "referral", -9, "approved", ["wrong_number"]),
    ("Ramesh Yadav", "1959-11-17", "Male", "99xxxx2093", "Unnao", 52, "consultation", 10, "proposed", []),
    ("Anjali Gupta", "1983-06-22", "Female", "96xxxx3307", "Lucknow", 5, "repeat_scan", 45, "approved", ["reached_confirmed"]),
]


def seed_dev_followup(app) -> None:
    """Development only: walk-in demo patients with follow-up plans in different states, so the follow-up worklist,
    risk ranking, approvals and referrals have something to show. Runs once (when no care plan exists yet)."""
    import datetime as dt
    import time

    from edge.portal import careplan, outreach, scheduling
    db = app.extensions["portal"]["db"]
    store = app.extensions["portal"]["store"]
    if db.one("SELECT COUNT(*) AS n FROM care_plans")["n"]:
        return
    doctor = db.one("SELECT * FROM users WHERE role = 'doctor' ORDER BY id LIMIT 1")
    admin = db.one("SELECT * FROM users WHERE role = 'admin' ORDER BY id LIMIT 1")
    if doctor is None or admin is None:
        return
    today = scheduling.local(time.time(), scheduling.tz(db)).date()
    for name, dob, sex, phone, district, km, kind, offset, status, outcomes in DEMO_PATIENTS:
        uid = db.create_user(None, name, "patient", None, {"dob": dob, "sex": sex, "phone": phone, "district": district,
                                                           "distance_km": km}, created_by=admin["id"], email_consent=False)
        with db.tx() as c:                               # registered at the desk some weeks ago
            c.execute("UPDATE users SET created = ? WHERE id = ?", (time.time() - 40 * 86400, uid))
        due =(today + dt.timedelta(days=offset)).isoformat()
        pid = careplan.create(db, uid, kind, due, "", doctor if status == "approved" else admin)
        if kind == "referral":
            with app.test_request_context():
                rid = careplan.create_referral(db, uid, "Regional Cancer Centre, Lucknow", "Neurosurgery", "soon",
                                               "Suspected meningioma on MRI; neurosurgical opinion requested.", doctor,
                                               plan_id=pid, now=time.time() - 20 * 86400)
                careplan.sign_referral(db, store, careplan.referral(db, rid), doctor, db.setting("clinic_name"),
                                       now=time.time() - 20 * 86400)
            careplan.set_referral_status(db, careplan.referral(db, rid), "sent")
        for i, outcome in enumerate(outcomes):
            outreach.log_contact(db, uid, admin, "phone", outcome, "", pid, now=time.time() - (len(outcomes) - i) * 86400)
    print("seeded demo follow-up patients")


def start_dev_reminders(app, every_s: int = 60) -> None:
    """Development only: run the reminder rules + e-mail flush in a background thread (on the Pi this is
    neuroedge-reminders.timer, a separate process)."""
    import threading
    import time

    from edge.portal import mailer, reminders
    db = app.extensions["portal"]["db"]

    def loop():
        while True:
            try:
                reminders.run_once(db)
                mailer.flush(db)
            except Exception as e:                       # keep the demo server alive
                print(f"reminders: {e}")
            time.sleep(every_s)
    threading.Thread(target=loop, name="dev-reminders", daemon=True).start()


def wsgi():
    return build(os.environ.get("NEUROEDGE_BUNDLE_DIR"), os.environ.get("NEUROEDGE_MODEL", "mri"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir")
    ap.add_argument("--model", default="mri")
    ap.add_argument("--portal-data")
    ap.add_argument("--portal-key", help="default: <portal-data>/portal.key with --init-dev, else /etc/neuroedge/portal.key")
    ap.add_argument("--init-dev", action="store_true", help="create the key + database if missing (development)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    key = args.portal_key or (str(Path(args.portal_data) / "portal.key") if args.init_dev and args.portal_data else None)
    if args.init_dev:
        import hashlib

        from edge.portal.storage import generate_key
        if key and not Path(key).exists():
            generate_key(Path(key))
        if args.portal_data:
            # Development / demo conveniences: e-mails become .eml files, sessions survive restarts.
            smtp_env = Path(args.portal_data) / "smtp.env"
            os.environ.setdefault("NEUROEDGE_SMTP_ENV", str(smtp_env))
            if not (smtp_env.exists() and "NEUROEDGE_SMTP_HOST=" in smtp_env.read_text(encoding="utf-8")):
                # no real mail account configured: save e-mails as .eml files instead of sending
                os.environ.setdefault("NEUROEDGE_SMTP", f"file:{Path(args.portal_data) / 'mail'}")
            os.environ.setdefault("NEUROEDGE_SECRET", hashlib.sha256(Path(key).read_bytes() + b"session").hexdigest())
    app = build(args.bundle_dir, args.model, args.threads, args.portal_data, key)
    if args.init_dev and args.portal_data:
        seed_dev_accounts(app, Path(args.portal_data))
        seed_dev_followup(app)
        start_dev_reminders(app)
    print(json.dumps({"serving": f"http://{args.host}:{args.port}", "bundle": args.bundle_dir or args.model}))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

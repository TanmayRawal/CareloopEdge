"""Start the NeuroEdge clinical portal.

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
        db.create_user(email, name, role, hash_password(pw), profile)
        rows.append(f"{role:<8} {email:<28} {pw}")
    (data_dir / "DEV_ACCOUNTS.txt").write_text(
        "# NeuroEdge DEV portal test accounts (local only; delete the folder to regenerate). Not real users.\n\n"
        + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"created test accounts: {data_dir / 'DEV_ACCOUNTS.txt'}")


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
        from edge.portal.storage import generate_key
        if key and not Path(key).exists():
            generate_key(Path(key))
    app = build(args.bundle_dir, args.model, args.threads, args.portal_data, key)
    if args.init_dev and args.portal_data:
        seed_dev_accounts(app, Path(args.portal_data))
    print(json.dumps({"serving": f"http://{args.host}:{args.port}", "bundle": args.bundle_dir or args.model}))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

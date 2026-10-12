"""Start the NeuroEdge clinical portal.

Production (Pi, systemd + gunicorn): model decrypted in RAM from the enrolled device's verified bundle
  gunicorn -w 1 --threads 4 -b 0.0.0.0:8080 "edge.serve:wsgi()"
  env: NEUROEDGE_PORTAL_DATA (default /var/lib/neuroedge/portal), NEUROEDGE_PORTAL_KEY
       (default /etc/neuroedge/portal.key), NEUROEDGE_SECRET (session signing), NEUROEDGE_MODEL
Development (PC): plaintext bundle directory, local data dir; --init-dev creates the key + database once
  python -m edge.serve --bundle-dir <DATA_ROOT>/bundles/mri_v0_int8 --portal-data <DATA_ROOT>/portal_dev --init-dev
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from edge.engine import Engine
from edge.portal.app import create_app

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
    return create_app(engine, info, data_dir, key_file, secret_key=os.environ.get("NEUROEDGE_SECRET"))


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
    print(json.dumps({"serving": f"http://{args.host}:{args.port}", "bundle": args.bundle_dir or args.model}))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

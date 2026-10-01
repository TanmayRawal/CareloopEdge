"""Start the NeuroEdge web app.

Production (Pi): model decrypted in RAM from the enrolled device's verified bundle
  python -m edge.serve --model mri --host 0.0.0.0 --port 8080
  (or under gunicorn:  gunicorn -w 1 --threads 4 -b 0.0.0.0:8080 "edge.serve:wsgi()")
Development: plaintext bundle directory produced by training/make_bundle.py
  python -m edge.serve --bundle-dir C:/Users/Tanmay/neuroedge_data/bundles/mri_v0_int8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from edge.engine import Engine
from edge.webapp.app import create_app


def build(bundle_dir: str | None = None, model: str = "mri", threads: int = 4):
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
    return create_app(engine, info, secret_key=os.environ.get("NEUROEDGE_SECRET"))


def wsgi():
    return build(os.environ.get("NEUROEDGE_BUNDLE_DIR"), os.environ.get("NEUROEDGE_MODEL", "mri"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir")
    ap.add_argument("--model", default="mri")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    app = build(args.bundle_dir, args.model, args.threads)
    print(json.dumps({"serving": f"http://{args.host}:{args.port}", "bundle": args.bundle_dir or args.model}))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

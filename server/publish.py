"""Publish a model bundle to the registry (run on the TRAINING PC, where the signing key lives).

  python -m server.publish --bundle <bundle_dir> --name mri --version 1.0.0 \
      --server https://registry.example.org --signing-key ~/neuroedge_secrets/signing/signing_priv.pem

Steps (the server never receives plaintext or the signing key):
  1. OpenSSF Model Signing over the plaintext bundle directory (EC P-256, offline key)
  2. deterministic tar  ->  AES-256-GCM with a fresh data key, AAD = name/version
  3. data key wrapped to the server's X25519 public key (fetched over TLS; pin it with --server-key-b64)
  4. upload ciphertext + manifest + signature + wrapped key (admin token from env NE_ADMIN_TOKEN)
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path

from neuroedge.bundle import BUNDLE_FILES, OPTIONAL_FILES, context, pack, sign_dir
from neuroedge.security import encrypt_bundle, sha256_hex, wrap_key


def _finite(o):
    """JSON has no NaN/Infinity. An undefined metric (e.g. AUROC on a subset missing a class) is published as null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def build_payload(bundle_dir: Path, name: str, version: str, signing_key: Path, server_pub: bytes) -> dict:
    files = {f: (bundle_dir / f).read_bytes() for f in BUNDLE_FILES}
    files |= {f: (bundle_dir / f).read_bytes() for f in OPTIONAL_FILES if (bundle_dir / f).exists()}
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "bundle"          # sign exactly the shipped files, nothing else
        clean.mkdir()
        for f, data in files.items():
            (clean / f).write_bytes(data)
        sig_path = Path(tmp) / "bundle.sig"
        sign_dir(clean, signing_key, sig_path)
        signature = sig_path.read_bytes()
    ctx = context(name, version)
    data_key, blob = encrypt_bundle(pack(files), ctx)
    wrapped = wrap_key(data_key, server_pub, ctx)
    meta = json.loads(files["bundle.json"])
    manifest = {
        "name": name, "version": version, "blob_sha256": sha256_hex(blob),
        "file_sha256": {f: sha256_hex(d) for f, d in files.items()},
        "preprocess_fingerprint": meta.get("preprocess_fingerprint"),
        "card": {"task": meta.get("task"), "classes": meta.get("class_names"), "precision": meta.get("precision"),
                 "metrics": meta.get("metrics"), "quantization": meta.get("quantization", {}).get("acceptance"),
                 "intended_use": "Research and education only. Not a medical device; not for clinical decisions.",
                 "limitations": ["2D single-slice T1-weighted MRI only", "trained on public data (BRISC 2025)",
                                 "performance drops under scanner/population shift (see external metrics)"]},
    }
    return {"manifest": _finite(manifest), "signature_b64": base64.b64encode(signature).decode(),
            "wrapped_key": wrapped.to_dict(), "blob_b64": base64.b64encode(blob).decode()}


def _request(url: str, payload: dict | None = None, token: str | None = None) -> dict:
    data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--signing-key", required=True, type=Path)
    ap.add_argument("--server-key-b64", help="pinned server public key (recommended)")
    args = ap.parse_args()
    token = os.environ.get("NE_ADMIN_TOKEN")
    if not token:
        raise SystemExit("set NE_ADMIN_TOKEN in the environment")
    server_pub_b64 = args.server_key_b64 or _request(f"{args.server}/v1/server-key")["x25519_public_b64"]
    payload = build_payload(args.bundle, args.name, args.version, args.signing_key, base64.b64decode(server_pub_b64))
    print(_request(f"{args.server}/v1/admin/models/{args.name}/{args.version}", payload, token))


if __name__ == "__main__":
    main()

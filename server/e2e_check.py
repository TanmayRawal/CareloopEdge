"""End-to-end check of a LIVE registry deployment (Docker + Caddy TLS), from the publisher's and a device's side.

  NE_ADMIN_TOKEN=... python -m server.e2e_check --server https://localhost:8443 --ca keys/caddy_root.crt \
      --bundle <bundle_dir> --signing-key ~/neuroedge_secrets/signing/signing_priv.pem --signing-pub keys/signing_pub.pem --version 0.1.0

Publishes the bundle (unless --skip-publish), enrolls a throw-away device in a temp dir, pulls and verifies, and
runs the negative checks a real attacker would try. Prints one PASS/FAIL line per check; exit code 1 on any FAIL.
The test device is revoked at the end so it leaves no usable credential behind.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from edge.sync import Device
from neuroedge.bundle import BUNDLE_FILES, OPTIONAL_FILES
from neuroedge.security import sha256_hex

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    RESULTS.append((bool(ok), name, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def status_of(fn) -> int:
    try:
        fn()
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--ca", type=Path, help="CA bundle for the registry (Caddy internal root); default: system store")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--signing-key", type=Path)
    ap.add_argument("--signing-pub", required=True, type=Path)
    ap.add_argument("--version", required=True)
    ap.add_argument("--name", default="mri")
    ap.add_argument("--skip-publish", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("NE_ADMIN_TOKEN") or sys.exit("set NE_ADMIN_TOKEN")
    srv = args.server.rstrip("/")

    ctx = ssl.create_default_context(cafile=str(args.ca) if args.ca else None)

    def opener(req, timeout=60):
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)

    def call(path, method="GET", data=None, auth=None):
        req = urllib.request.Request(srv + path, method=method, data=json.dumps(data).encode() if data else None,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {auth}"} if auth else {})})
        with opener(req) as r:
            return json.loads(r.read())

    # ---------------------------------------------------------------- transport
    check(call("/health") == {"status": "ok"}, "health over verified TLS")
    try:
        urllib.request.urlopen(srv + "/health", timeout=10, context=ssl.create_default_context(cafile=None))
        check(args.ca is None, "client without the registry CA is refused")
    except urllib.error.URLError as e:
        check("CERTIFICATE_VERIFY_FAILED" in str(e), "client without the registry CA is refused")

    if args.ca:      # only now: the "untrusted client" check above must not see the registry CA
        os.environ["SSL_CERT_FILE"] = str(args.ca.resolve())

    # ---------------------------------------------------------------- publish
    if not args.skip_publish:
        from server.publish import build_payload
        pub = base64.b64decode(call("/v1/server-key")["x25519_public_b64"])
        payload = build_payload(args.bundle, args.name, args.version, args.signing_key, pub)
        r = call(f"/v1/admin/models/{args.name}/{args.version}", "POST", payload, auth=token)
        check(r.get("published") == f"{args.name}/{args.version}", "publish signed + encrypted bundle")
        check(status_of(lambda: call(f"/v1/admin/models/{args.name}/{args.version}", "POST", payload, auth=token)) == 409,
              "re-publishing the same version is refused (immutable)")
        check(status_of(lambda: call(f"/v1/admin/models/{args.name}/x9", "POST", payload, auth="wrong")) == 401,
              "publish with a wrong admin token is refused")
    cards = call("/v1/models")
    check(any(c["name"] == args.name for c in cards) and "blob_b64" not in json.dumps(cards),
          "public model card lists the model without any weights")

    # ---------------------------------------------------------------- device
    home = Path(tempfile.mkdtemp(prefix="ne_e2e_dev_"))
    try:
        tok = call("/v1/admin/enrollment-tokens", "POST", {}, auth=token)["token"]
        dev = Device(home / "dev", opener=opener)
        dev_id = dev.enroll(srv, tok, "e2e-check", args.signing_pub)
        check(bool(dev_id), "enroll with a one-time token")
        check(status_of(lambda: Device(home / "dev2", opener=opener).enroll(srv, tok, "replay", args.signing_pub)) in (400, 401, 403),
              "enrollment token cannot be reused")
        msg = dev.pull(args.name)
        check("installed" in msg or "already present" in msg, "pull: download + trial decrypt + signature check", msg)
        files, version = dev.load_bundle(args.name)
        local = {f: (args.bundle / f).read_bytes() for f in BUNDLE_FILES}
        local |= {f: (args.bundle / f).read_bytes() for f in OPTIONAL_FILES if (args.bundle / f).exists()}
        check(version == args.version or args.skip_publish, "device activated the published version", version)
        check({f: sha256_hex(d) for f, d in files.items()} == {f: sha256_hex(d) for f, d in local.items()},
              "decrypted files are byte-identical to the publisher's bundle", f"{len(files)} files")
        enc = next((home / "dev" / "models" / args.name).glob("*/blob.enc")).read_bytes()
        check(b"ustar" not in enc and b"bundle.json" not in enc, "on-device copy is ciphertext only")

        # attacker checks
        check(status_of(lambda: call(f"/v1/models/{args.name}/latest")) == 401, "latest without credentials -> 401")
        check(status_of(lambda: call(f"/v1/models/{args.name}/{version}/blob")) == 401, "blob without credentials -> 401")
        bad = json.loads((home / "dev" / "device.json").read_text())
        check(status_of(lambda: call(f"/v1/models/{args.name}/latest", auth=bad["api_key"][:-4] + "AAAA")) == 401,
              "forged device key -> 401")
        tampered = home / "tampered"
        shutil.copytree(home / "dev", tampered)
        blob = next((tampered / "models" / args.name).glob("*/blob.enc"))
        b = bytearray(blob.read_bytes())
        b[len(b) // 2] ^= 1
        blob.write_bytes(bytes(b))
        try:
            Device(tampered, opener=opener).load_bundle(args.name)
            check(False, "one flipped ciphertext bit is detected")
        except RuntimeError as e:
            check("InvalidTag" in str(e) or "mismatch" in str(e), "one flipped ciphertext bit is detected")
        other = home / "other-key"
        shutil.copytree(home / "dev", other)
        (other / "device.key").write_bytes(os.urandom(32))
        try:
            Device(other, opener=opener).load_bundle(args.name)
            check(False, "a different device key cannot decrypt the stored bundle")
        except RuntimeError:
            check(True, "a different device key cannot decrypt the stored bundle")

        # revocation
        call(f"/v1/admin/devices/{dev_id}/revoke", "POST", {}, auth=token)
        check(status_of(lambda: call(f"/v1/models/{args.name}/latest", auth=bad["api_key"])) == 403,
              "revoked device -> 403")
        audit = call("/v1/admin/audit?limit=200", auth=token)
        acts = {a["action"] for a in audit}
        check({"enrolled", "download", "device_revoked", "device_auth_failed"} <= acts, "actions are audit-logged",
              ", ".join(sorted(acts)))
    finally:
        shutil.rmtree(home, ignore_errors=True)

    n_fail = sum(1 for ok, *_ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

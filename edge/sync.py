"""Device-side model sync for the Raspberry Pi.

  python -m edge.sync enroll --server https://registry.example.org --token <one-time> --name pi5-lab \
         --signing-pub signing_pub.pem
  python -m edge.sync pull --model mri          # (systemd timer runs this daily)
  python -m edge.sync status

State (NEUROEDGE_HOME, default ~/.neuroedge; secrets chmod 600):
  device.key            X25519 private key — generated here, never leaves the device
  device.json           server URL, device id, API key
  signing_pub.pem       PINNED publisher key (copied at enrollment, never fetched from the server,
                        so a compromised server cannot substitute models)
  models/<name>/<ver>/  blob.enc (ciphertext) + meta.json (manifest, OMS signature, wrapped key)
  models/<name>/current active version
Only ciphertext is stored. load_bundle() decrypts in RAM, verifies the OMS signature and per-file
SHA-256s, and falls back to the previous version if the newest one fails.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from neuroedge.bundle import context, unpack, verify_files
from neuroedge.security import WrappedKey, decrypt_bundle, generate_device_keypair, sha256_hex, unwrap_key

HOME = Path(os.environ.get("NEUROEDGE_HOME", Path.home() / ".neuroedge"))


def _private(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class Device:
    def __init__(self, home: Path = HOME, opener=None):
        self.home = Path(home)
        self.opener = opener or urllib.request.urlopen     # injectable for tests

    # ------------------------------------------------------------------ http
    def _http(self, path: str, payload: dict | None = None, raw: bool = False, auth: bool = True):
        cfg = json.loads((self.home / "device.json").read_text()) if auth else {}
        server = cfg.get("server") if auth else payload.pop("_server")
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        req = urllib.request.Request(server.rstrip("/") + path, headers=headers, method="POST" if payload else "GET",
                                     data=json.dumps(payload).encode() if payload else None)
        with self.opener(req, timeout=300) as r:
            body = r.read()
        return body if raw else json.loads(body)

    # ------------------------------------------------------------------ enrollment
    def enroll(self, server: str, token: str, name: str, signing_pub: Path) -> str:
        self.home.mkdir(parents=True, exist_ok=True)
        priv, pub = generate_device_keypair()
        resp = self._http("/v1/devices/enroll", {"_server": server, "token": token, "name": name,
                                                 "public_key_b64": base64.b64encode(pub).decode()}, auth=False)
        _private(self.home / "device.key", priv)
        _private(self.home / "device.json", json.dumps({"server": server, "device_id": resp["device_id"],
                                                        "api_key": resp["api_key"]}).encode())
        shutil.copyfile(signing_pub, self.home / "signing_pub.pem")
        return resp["device_id"]

    # ------------------------------------------------------------------ pull
    def pull(self, name: str, keep: int = 2) -> str:
        latest = self._http(f"/v1/models/{name}/latest")
        version = latest["version"]
        mdir = self.home / "models" / name
        vdir = mdir / version
        if (vdir / "blob.enc").exists():
            return f"{name} {version} already present"
        blob = self._http(latest["blob_url"], raw=True)
        if sha256_hex(blob) != latest["blob_sha256"] or sha256_hex(blob) != latest["manifest"]["blob_sha256"]:
            raise ValueError("downloaded blob hash mismatch")
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "blob.enc").write_bytes(blob)
        meta = {k: latest[k] for k in ("name", "version", "manifest", "signature_b64", "wrapped_key")}
        (vdir / "meta.json").write_text(json.dumps(meta))
        try:
            self._open(vdir)                          # trial decrypt + verify BEFORE activating
        except Exception:
            shutil.rmtree(vdir, ignore_errors=True)
            raise
        history = self._history(name)
        (mdir / "current").write_text(version)
        (mdir / "history.json").write_text(json.dumps([version] + [v for v in history if v != version]))
        for old in self._history(name)[keep:]:
            shutil.rmtree(mdir / old, ignore_errors=True)
        return f"{name} {version} installed"

    def _history(self, name: str) -> list[str]:
        p = self.home / "models" / name / "history.json"
        return json.loads(p.read_text()) if p.exists() else []

    # ------------------------------------------------------------------ load
    def _open(self, vdir: Path) -> dict[str, bytes]:
        meta = json.loads((vdir / "meta.json").read_text())
        ctx = context(meta["name"], meta["version"])
        key = unwrap_key(WrappedKey.from_dict(meta["wrapped_key"]), (self.home / "device.key").read_bytes(), ctx)
        files = unpack(decrypt_bundle(key, (vdir / "blob.enc").read_bytes(), ctx))
        for f, h in meta["manifest"]["file_sha256"].items():
            if sha256_hex(files[f]) != h:
                raise ValueError(f"file hash mismatch: {f}")
        verify_files(files, base64.b64decode(meta["signature_b64"]), self.home / "signing_pub.pem")
        return files

    def load_bundle(self, name: str) -> tuple[dict[str, bytes], str]:
        """Decrypted, signature-verified bundle files (in memory) + version; rolls back on failure."""
        errors = []
        for version in self._history(name):
            try:
                return self._open(self.home / "models" / name / version), version
            except Exception as e:   # noqa: BLE001 — report every failed candidate
                errors.append(f"{version}: {type(e).__name__}: {e}")
        raise RuntimeError("no verifiable model bundle available: " + "; ".join(errors or ["none installed"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll")
    e.add_argument("--server", required=True)
    e.add_argument("--token", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--signing-pub", required=True, type=Path)
    p = sub.add_parser("pull")
    p.add_argument("--model", default="mri")
    s = sub.add_parser("status")
    s.add_argument("--model", default="mri")
    args = ap.parse_args()
    dev = Device()
    if args.cmd == "enroll":
        print("enrolled as", dev.enroll(args.server, args.token, args.name, args.signing_pub))
    elif args.cmd == "pull":
        print(dev.pull(args.model))
    else:
        try:
            _, v = dev.load_bundle(args.model)
            print(f"{args.model}: version {v} verified OK")
        except RuntimeError as err:
            print(err)
            sys.exit(1)


if __name__ == "__main__":
    main()

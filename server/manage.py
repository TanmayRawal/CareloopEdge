"""Server administration.

  python -m server.manage init --data server_data
      creates the data dir + server X25519 key and prints a NEW admin token ONCE; the server stores
      only its SHA-256 (put it in .env as NE_ADMIN_TOKEN_SHA256).
  python -m server.manage signing-key --out ~/neuroedge_secrets/signing   (outside any cloud-synced folder)
      creates the OFFLINE OpenSSF Model Signing keypair (EC P-256) used by publish.py. Copy only
      signing_pub.pem to devices; keep signing_priv.pem off the server.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
from pathlib import Path

from neuroedge.bundle import generate_signing_key
from server.store import Store


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--data", default="server_data")
    p_key = sub.add_parser("signing-key")
    p_key.add_argument("--out", default="keys")
    args = ap.parse_args()

    if args.cmd == "init":
        Store(args.data).ensure_server_key()
        token = secrets.token_urlsafe(32)
        env = Path(args.data) / "admin.env"
        env.write_text(f"NE_ADMIN_TOKEN_SHA256={hashlib.sha256(token.encode()).hexdigest()}\n")
        print(f"server initialised in {args.data}; admin token hash written to {env}")
        print("ADMIN TOKEN (shown once — store it in a password manager):")
        print(token)
    elif args.cmd == "signing-key":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        generate_signing_key(out / "signing_priv.pem", out / "signing_pub.pem")
        print(f"wrote {out / 'signing_priv.pem'} (keep OFFLINE) and {out / 'signing_pub.pem'} (copy to devices)")


if __name__ == "__main__":
    main()

"""Encrypted-at-rest file store for patient scans, explanation overlays and signed report snapshots.

Layout:  <root>/blobs/<case_id>/<kind>.enc      file = nonce (12 B) || AES-256-GCM(ciphertext + tag)
         AAD = "<case_id>/<kind>"  -> a blob cannot be swapped to another case or kind without detection.
The 32-byte key lives in a separate file (installer: /etc/neuroedge/portal.key, root:neuroedge 0640), so a
copy of the data directory alone is unreadable. Integrity: GCM authenticates every read; the database also
keeps the SHA-256 of the plaintext scan and of every report snapshot.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def generate_key(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite a key (all stored data depends on it)")
    path.write_bytes(AESGCM.generate_key(bit_length=256))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class BlobStore:
    def __init__(self, root: Path, key: bytes):
        if len(key) != 32:
            raise ValueError("portal key must be 32 bytes (AES-256)")
        self.root = Path(root) / "blobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._aead = AESGCM(key)

    @classmethod
    def from_key_file(cls, root: Path, key_file: Path) -> "BlobStore":
        return cls(root, Path(key_file).read_bytes())

    def _path(self, case_id: str, kind: str) -> Path:
        if not (_SAFE.match(case_id) and _SAFE.match(kind)):
            raise ValueError("invalid blob name")
        return self.root / case_id / f"{kind}.enc"

    def put(self, case_id: str, kind: str, data: bytes) -> str:
        """Encrypt and store; returns the SHA-256 of the plaintext. Atomic (write temp, then rename)."""
        p = self._path(case_id, kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        nonce = os.urandom(12)
        blob = nonce + self._aead.encrypt(nonce, data, f"{case_id}/{kind}".encode())
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, p)
        return hashlib.sha256(data).hexdigest()

    def get(self, case_id: str, kind: str) -> bytes:
        """Decrypt; raises cryptography.exceptions.InvalidTag if the file was altered or moved."""
        blob = self._path(case_id, kind).read_bytes()
        return self._aead.decrypt(blob[:12], blob[12:], f"{case_id}/{kind}".encode())

    def exists(self, case_id: str, kind: str) -> bool:
        return self._path(case_id, kind).exists()

    def delete_case(self, case_id: str) -> int:
        d = self.root / case_id
        if not (_SAFE.match(case_id) and d.is_dir()):
            return 0
        n = 0
        for f in d.iterdir():
            f.unlink()
            n += 1
        d.rmdir()
        return n

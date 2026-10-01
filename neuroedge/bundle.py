"""Model-bundle packaging, signing and verification (shared by publisher, server and Pi).

Lifecycle
  publisher : files -> sign_dir (OpenSSF Model Signing, EC P-256 key kept offline) -> pack (deterministic
              tar) -> encrypt (AES-256-GCM, neuroedge.security) -> data key wrapped for the SERVER key
  server    : stores ciphertext; re-wraps the data key for each enrolled device on request
  Pi        : unwrap with device key -> decrypt in RAM -> unpack (strict) -> verify_files (pinned public
              key) -> bytes handed to the inference engine; plaintext is only ever materialised in a
              RAM-backed directory (/dev/shm) for the duration of signature verification.
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

BUNDLE_FILES = ("bundle.json", "model.onnx", "head.npz", "calibration.json", "ood.npz")
OPTIONAL_FILES = ("model_fp32.pt",)      # FP32 weights for on-demand LayerCAM ("Deep Explain")
MAX_FILE_BYTES = 512 * 1024 * 1024


def context(name: str, version: str) -> bytes:
    """AAD binding ciphertexts and wrapped keys to one model name + version."""
    return f"neuroedge-bundle|{name}|{version}".encode()


# ------------------------------------------------------------------------------ packing
def pack(files: dict[str, bytes]) -> bytes:
    """Deterministic tar (sorted names, zeroed metadata): same files -> same bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def unpack(blob: bytes, required: tuple[str, ...] = BUNDLE_FILES,
           optional: tuple[str, ...] = OPTIONAL_FILES) -> dict[str, bytes]:
    """Strict in-memory extraction: regular files with whitelisted flat names only."""
    allowed = set(required) | set(optional)
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        for m in tar.getmembers():
            if not m.isfile() or m.name not in allowed or "/" in m.name or "\\" in m.name:
                raise ValueError(f"unexpected bundle member: {m.name!r}")
            if m.size > MAX_FILE_BYTES or m.name in out:
                raise ValueError(f"invalid bundle member: {m.name!r}")
            out[m.name] = tar.extractfile(m).read()
    missing = set(required) - set(out)
    if missing:
        raise ValueError(f"bundle incomplete: {sorted(missing)}")
    return out


# ------------------------------------------------------------------------------ signing
def generate_signing_key(private_pem: Path, public_pem: Path) -> None:
    """EC P-256 keypair for OpenSSF Model Signing (keep the private key OFFLINE)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    private_pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption()))
    public_pem.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                         serialization.PublicFormat.SubjectPublicKeyInfo))
    try:
        os.chmod(private_pem, 0o600)
    except OSError:
        pass


def sign_dir(model_dir: Path, private_pem: Path, signature_path: Path) -> None:
    import model_signing

    model_signing.signing.Config().use_elliptic_key_signer(private_key=private_pem).sign(model_dir, signature_path)


def _ram_dir() -> Path:
    shm = Path("/dev/shm")
    base = shm if shm.is_dir() and os.access(shm, os.W_OK) else None   # Linux / Raspberry Pi OS
    return Path(tempfile.mkdtemp(prefix="neuroedge-verify-", dir=base))


def verify_files(files: dict[str, bytes], signature: bytes, public_pem: Path) -> None:
    """Verify an OMS signature over in-memory files. Raises on any mismatch.

    The files are written to a RAM-backed temp dir only for the duration of verification
    (the OMS library verifies paths), then securely removed.
    """
    import model_signing

    tmp = _ram_dir()
    try:
        model_dir = tmp / "bundle"
        model_dir.mkdir()
        for name, data in files.items():
            (model_dir / name).write_bytes(data)
        sig = tmp / "bundle.sig"
        sig.write_bytes(signature)
        model_signing.verifying.Config().use_elliptic_key_verifier(public_key=public_pem).verify(model_dir, sig)
    finally:
        for p in sorted(tmp.rglob("*"), reverse=True):
            if p.is_file():
                p.write_bytes(b"\0" * min(p.stat().st_size, 1 << 20))   # best-effort scrub
        shutil.rmtree(tmp, ignore_errors=True)

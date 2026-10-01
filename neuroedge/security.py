"""Model-bundle confidentiality: AES-256-GCM envelope encryption with per-device key wrapping.

    bundle bytes --AES-256-GCM(data_key, aad=context)--> ciphertext   (stored once on the server)
    data_key --X25519-ECDH + HKDF-SHA256 + AES-256-GCM--> wrapped key (one per enrolled device)

* GCM is authenticated: any modified byte makes decryption fail (tamper evidence).
* `context` (e.g. b"model=mri-resnet50;version=3") is bound as AAD to both layers, so a wrapped
  key or ciphertext cannot be replayed against another model/version.
* Each device keeps its X25519 private key locally; the server only ever sees public keys.
* Integrity/provenance signing uses the OpenSSF Model Signing (OMS) library in server/ and
  edge/sync.py; this module only provides confidentiality primitives.

Threat-model note: this protects the model in transit and at rest and restricts download to
enrolled devices. It cannot prevent extraction by someone with root access to an enrolled device.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_NONCE = 12
_WRAP_INFO = b"neuroedge/key-wrap/v1"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


# -------------------------------------------------------------------------- device keys
def generate_device_keypair() -> tuple[bytes, bytes]:
    """Return (private_raw32, public_raw32)."""
    priv = X25519PrivateKey.generate()
    priv_b = priv.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                serialization.NoEncryption())
    pub_b = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return priv_b, pub_b


# -------------------------------------------------------------------------- bundle layer
def encrypt_bundle(plaintext: bytes, context: bytes) -> tuple[bytes, bytes]:
    """Encrypt with a fresh random 256-bit data key. Returns (data_key, nonce || ciphertext||tag)."""
    data_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(_NONCE)
    return data_key, nonce + AESGCM(data_key).encrypt(nonce, plaintext, context)


def decrypt_bundle(data_key: bytes, blob: bytes, context: bytes) -> bytes:
    """Raises cryptography.exceptions.InvalidTag if the blob, key or context is wrong."""
    return AESGCM(data_key).decrypt(blob[:_NONCE], blob[_NONCE:], context)


# -------------------------------------------------------------------------- key wrapping
@dataclass(frozen=True)
class WrappedKey:
    ephemeral_public: str   # base64
    nonce: str              # base64
    ciphertext: str         # base64

    def to_dict(self) -> dict:
        return {"ephemeral_public": self.ephemeral_public, "nonce": self.nonce, "ciphertext": self.ciphertext}

    @classmethod
    def from_dict(cls, d: dict) -> "WrappedKey":
        return cls(d["ephemeral_public"], d["nonce"], d["ciphertext"])


def _kek(shared: bytes, eph_pub: bytes, dev_pub: bytes, context: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=eph_pub + dev_pub,
                info=_WRAP_INFO + b"|" + context).derive(shared)


def wrap_key(data_key: bytes, device_public: bytes, context: bytes) -> WrappedKey:
    eph = X25519PrivateKey.generate()
    eph_pub = eph.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    shared = eph.exchange(X25519PublicKey.from_public_bytes(device_public))
    nonce = os.urandom(_NONCE)
    ct = AESGCM(_kek(shared, eph_pub, device_public, context)).encrypt(nonce, data_key, context)
    return WrappedKey(_b64(eph_pub), _b64(nonce), _b64(ct))


def unwrap_key(wrapped: WrappedKey, device_private: bytes, context: bytes) -> bytes:
    priv = X25519PrivateKey.from_private_bytes(device_private)
    dev_pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    eph_pub = _unb64(wrapped.ephemeral_public)
    shared = priv.exchange(X25519PublicKey.from_public_bytes(eph_pub))
    kek = _kek(shared, eph_pub, dev_pub, context)
    return AESGCM(kek).decrypt(_unb64(wrapped.nonce), _unb64(wrapped.ciphertext), context)

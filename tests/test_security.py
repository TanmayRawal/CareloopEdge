import pytest
from cryptography.exceptions import InvalidTag

from neuroedge.security import (decrypt_bundle, encrypt_bundle, generate_device_keypair, unwrap_key,
                                wrap_key)

CTX = b"model=mri-resnet50;version=1"


def test_envelope_roundtrip():
    payload = b"onnx-model-bytes" * 1000
    key, blob = encrypt_bundle(payload, CTX)
    priv, pub = generate_device_keypair()
    wrapped = wrap_key(key, pub, CTX)
    assert decrypt_bundle(unwrap_key(wrapped, priv, CTX), blob, CTX) == payload


def test_tampered_ciphertext_rejected():
    key, blob = encrypt_bundle(b"secret model", CTX)
    tampered = bytearray(blob)
    tampered[20] ^= 0x01
    with pytest.raises(InvalidTag):
        decrypt_bundle(key, bytes(tampered), CTX)


def test_wrong_context_rejected():
    key, blob = encrypt_bundle(b"secret model", CTX)
    with pytest.raises(InvalidTag):
        decrypt_bundle(key, blob, b"model=mri-resnet50;version=2")


def test_other_device_cannot_unwrap():
    key, _ = encrypt_bundle(b"m", CTX)
    _, pub_a = generate_device_keypair()
    priv_b, _ = generate_device_keypair()
    with pytest.raises(InvalidTag):
        unwrap_key(wrap_key(key, pub_a, CTX), priv_b, CTX)

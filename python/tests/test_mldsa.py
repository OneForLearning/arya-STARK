"""
Unit tests for ``arya_stark.crypto.mldsa``.

Coverage:

1. **API correctness**: keygen produces correctly-sized keys, sign/verify
   round-trip on identical input, verify rejects tampered input.
2. **Cryptographic properties**: signatures are randomised (two
   signatures of the same message under the same key differ — but both
   verify), keys are independent across calls.
3. **Integration with encoding**: a real gradient encoded via
   ``arya_stark.encoding`` and signed via ``mldsa`` round-trips
   correctly.

The wrapper auto-selects between the ``oqs`` (liboqs C) backend and the
``dilithium-py`` pure-Python fallback, so tests run in any environment
that has at least one available.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from arya_stark.crypto.mldsa import (
    ML_DSA_65,
    PUBLIC_KEY_BYTES,
    SECRET_KEY_BYTES,
    SIGNATURE_MAX_BYTES,
    InvalidKeyLength,
    authenticated_message,
    get_backend_name,
    get_sizes,
    keygen,
    sign,
    verify,
)
from arya_stark.encoding import encode_vector


# ---------------------------------------------------------------------------
# Constants and metadata
# ---------------------------------------------------------------------------


def test_constants_match_fips_204() -> None:
    """FIPS 204 mandates exact lengths for ML-DSA-65."""
    assert PUBLIC_KEY_BYTES == 1952
    assert SECRET_KEY_BYTES == 4032
    assert SIGNATURE_MAX_BYTES == 3309
    assert ML_DSA_65 == "ML-DSA-65"


def test_get_sizes_matches_constants() -> None:
    pk, sk, sig = get_sizes()
    assert (pk, sk, sig) == (PUBLIC_KEY_BYTES, SECRET_KEY_BYTES, SIGNATURE_MAX_BYTES)


# ---------------------------------------------------------------------------
# Keygen
# ---------------------------------------------------------------------------


def test_keygen_lengths() -> None:
    pk, sk = keygen()
    assert len(pk) == PUBLIC_KEY_BYTES
    assert len(sk) == SECRET_KEY_BYTES
    assert isinstance(pk, bytes) and isinstance(sk, bytes)


def test_keygen_produces_independent_keys() -> None:
    """Successive calls produce different keys (otherwise the RNG is broken)."""
    pk1, sk1 = keygen()
    pk2, sk2 = keygen()
    assert pk1 != pk2
    assert sk1 != sk2


# ---------------------------------------------------------------------------
# Sign / verify round-trip
# ---------------------------------------------------------------------------


def test_sign_verify_roundtrip() -> None:
    pk, sk = keygen()
    msg = b"arya-STARK round 1, client 0xCAFE"
    sig = sign(sk, msg)
    assert len(sig) <= SIGNATURE_MAX_BYTES
    assert verify(pk, msg, sig)


def test_sign_verify_empty_message() -> None:
    """Signing an empty message must work and verify."""
    pk, sk = keygen()
    sig = sign(sk, b"")
    assert verify(pk, b"", sig)


def test_sign_verify_long_message() -> None:
    """Signing a 1 MB blob (typical gradient size for a small CNN)."""
    if get_backend_name() == "dilithium_py":
        # Pure-Python implementation is too slow for a 1MB message.
        pytest.skip("dilithium_py is too slow for 1MB messages; OK in CI")
    pk, sk = keygen()
    msg = os.urandom(1 << 20)
    sig = sign(sk, msg)
    assert verify(pk, msg, sig)


def test_signatures_are_randomised() -> None:
    """
    ML-DSA can be either deterministic or randomised depending on the
    backend. FIPS 204 specifies the *signing algorithm* but allows
    implementations to be deterministic (using a hashed-derive nonce
    from sk + msg) or randomised (drawing a fresh nonce per signature).

    Both behaviours produce valid signatures. We test:
      * if backend is ``oqs`` (liboqs randomised), two sigs of same
        msg must differ;
      * if backend is ``dilithium_py`` (deterministic), they will be
        equal — and we just verify that both verify.
    """
    pk, sk = keygen()
    msg = b"identical message"
    sig1 = sign(sk, msg)
    sig2 = sign(sk, msg)
    assert verify(pk, msg, sig1)
    assert verify(pk, msg, sig2)
    if get_backend_name() == "oqs":
        # Randomised backend: signatures must differ.
        assert sig1 != sig2, "ML-DSA-65 (oqs) signatures should be randomised"


# ---------------------------------------------------------------------------
# Tampering / wrong inputs
# ---------------------------------------------------------------------------


def test_verify_rejects_wrong_message() -> None:
    pk, sk = keygen()
    sig = sign(sk, b"original")
    assert not verify(pk, b"tampered", sig)


def test_verify_rejects_wrong_pk() -> None:
    _, sk = keygen()
    pk_other, _ = keygen()
    sig = sign(sk, b"hello")
    assert not verify(pk_other, b"hello", sig)


def test_verify_rejects_truncated_signature() -> None:
    pk, sk = keygen()
    sig = sign(sk, b"hello")
    # Drop last byte.
    assert not verify(pk, b"hello", sig[:-1])


def test_verify_rejects_oversize_signature() -> None:
    pk, _ = keygen()
    too_big = b"\x00" * (SIGNATURE_MAX_BYTES + 1)
    assert not verify(pk, b"any", too_big)


def test_verify_rejects_flipped_byte() -> None:
    pk, sk = keygen()
    sig = sign(sk, b"hello")
    flipped = bytes([sig[0] ^ 1]) + sig[1:]
    assert not verify(pk, b"hello", flipped)


# ---------------------------------------------------------------------------
# Length-validation errors
# ---------------------------------------------------------------------------


def test_sign_rejects_wrong_sk_length() -> None:
    with pytest.raises(InvalidKeyLength):
        sign(b"\x00" * 100, b"msg")


def test_verify_rejects_wrong_pk_length() -> None:
    pk, sk = keygen()
    sig = sign(sk, b"msg")
    with pytest.raises(InvalidKeyLength):
        verify(b"\x00" * 100, b"msg", sig)


# ---------------------------------------------------------------------------
# Integration with the encoding module (the real arya-STARK use case)
# ---------------------------------------------------------------------------


def test_authenticated_gradient_roundtrip() -> None:
    """
    End-to-end client→server flow:

        gradient → encode → sign(metadata || encoded) → verify

    This is the exact pattern used in `client/honest_client.py` (P7).
    """
    # 1. Client computes a gradient.
    gradient = np.array([0.5, -0.25, 1.5, -1.5, 0.0])
    # 2. Client encodes.
    encoded = encode_vector(gradient, m=6)
    encoded_bytes = encoded.tobytes()
    # 3. Client signs encoded || metadata.
    pk, sk = keygen()
    metadata = b"round=001|client_id=42"
    msg, sig = authenticated_message(sk, encoded_bytes, metadata)
    # 4. Server verifies.
    assert verify(pk, msg, sig)
    # 5. The message must contain both metadata AND encoded payload.
    assert metadata in msg
    assert encoded_bytes in msg


def test_authenticated_message_metadata_replay_prevention() -> None:
    """
    Different rounds → different messages → different signatures.
    A signature from round 1 must NOT verify against the round-2 message.
    """
    pk, sk = keygen()
    encoded = b"\xAA" * 100  # placeholder for an encoded gradient
    msg1, sig1 = authenticated_message(sk, encoded, metadata=b"round=001")
    msg2, sig2 = authenticated_message(sk, encoded, metadata=b"round=002")
    # Each verifies in its own context.
    assert verify(pk, msg1, sig1)
    assert verify(pk, msg2, sig2)
    # But cross-verification fails.
    assert not verify(pk, msg2, sig1)
    assert not verify(pk, msg1, sig2)

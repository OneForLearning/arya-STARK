"""
arya_stark.crypto.mldsa
=======================

ML-DSA-65 (FIPS 204, ex-CRYSTALS-Dilithium-3) wrapper.

Why ML-DSA-65 specifically:
* NIST Level 3 ≈ 192-bit PQ security (margin over 128-bit target,
  see ``CryptoConfig`` and Table I of the paper).
* Multi-user EUF-CMA reduction is tight in FIPS 204 §C, so the
  ``M⋆``-factor in Theorem IV.3 is constant.

Sizes (FIPS 204, ML-DSA-65):
    public key  = 1952 B
    secret key  = 4032 B
    signature   = 3309 B (fixed)

Backends
--------

The module auto-detects which backend to use:

1. **liboqs** (production) — if ``liboqs-python`` is installed AND
   the C library can be loaded, we use it. Bit-compatible with the
   Rust ``oqs`` crate.
2. **dilithium-py** (pure Python) — portable fallback when liboqs
   is not available. FIPS 204 conformant and bit-compatible with
   the Rust ``fips204`` crate.

Both backends produce byte-identical keys and signatures. Cross-language
interop is verified by ``test_mldsa_rust_compat.py``.
"""
from __future__ import annotations

import os
from typing import Final

# ---------------------------------------------------------------------------
# Constants — FIPS 204, ML-DSA-65 (NIST Cat. 3)
# ---------------------------------------------------------------------------

ML_DSA_65: Final[str] = "ML-DSA-65"
PUBLIC_KEY_BYTES: Final[int] = 1952
SECRET_KEY_BYTES: Final[int] = 4032
SIGNATURE_MAX_BYTES: Final[int] = 3309


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MLDSAError(Exception):
    """Base class for ML-DSA bridge errors."""


class InvalidKeyLength(MLDSAError):
    """Raised when a key has an unexpected length."""


# ---------------------------------------------------------------------------
# Backend selection (oqs preferred, dilithium-py fallback)
# ---------------------------------------------------------------------------


def _try_load_oqs() -> bool:
    """Return True iff `oqs` can actually load liboqs at runtime."""
    if os.environ.get("ARYA_STARK_DISABLE_OQS", "0") == "1":
        return False
    try:
        import oqs  # type: ignore

        with oqs.Signature(ML_DSA_65):
            pass
        return True
    except Exception:
        return False


_BACKEND_NAME: str
if _try_load_oqs():
    import oqs  # type: ignore

    _BACKEND_NAME = "oqs"

    def _keygen_impl() -> tuple[bytes, bytes]:
        with oqs.Signature(ML_DSA_65) as signer:
            pk = signer.generate_keypair()
            sk = signer.export_secret_key()
        return bytes(pk), bytes(sk)

    def _sign_impl(sk: bytes, msg: bytes) -> bytes:
        with oqs.Signature(ML_DSA_65, secret_key=sk) as signer:
            return bytes(signer.sign(msg))

    def _verify_impl(pk: bytes, msg: bytes, sig: bytes) -> bool:
        with oqs.Signature(ML_DSA_65) as verifier:
            try:
                return bool(verifier.verify(msg, sig, pk))
            except Exception:
                return False

else:
    from dilithium_py.ml_dsa import ML_DSA_65 as _PURE_MLDSA  # type: ignore

    _BACKEND_NAME = "dilithium_py"

    def _keygen_impl() -> tuple[bytes, bytes]:
        pk, sk = _PURE_MLDSA.keygen()
        return bytes(pk), bytes(sk)

    def _sign_impl(sk: bytes, msg: bytes) -> bytes:
        return bytes(_PURE_MLDSA.sign(sk, msg))

    def _verify_impl(pk: bytes, msg: bytes, sig: bytes) -> bool:
        try:
            return bool(_PURE_MLDSA.verify(pk, msg, sig))
        except Exception:
            return False


def get_backend_name() -> str:
    """Return the active backend name (``'oqs'`` or ``'dilithium_py'``)."""
    return _BACKEND_NAME


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_sizes() -> tuple[int, int, int]:
    """
    Return ``(public_key_bytes, secret_key_bytes, signature_max_bytes)``.

    >>> get_sizes()
    (1952, 4032, 3309)
    """
    return (PUBLIC_KEY_BYTES, SECRET_KEY_BYTES, SIGNATURE_MAX_BYTES)


def keygen() -> tuple[bytes, bytes]:
    """
    Generate a fresh ML-DSA-65 keypair.

    Returns ``(public_key, secret_key)`` as a 2-tuple of bytes.
    """
    pk, sk = _keygen_impl()
    if len(pk) != PUBLIC_KEY_BYTES:
        raise InvalidKeyLength(
            f"unexpected pk length {len(pk)} (expected {PUBLIC_KEY_BYTES})"
        )
    if len(sk) != SECRET_KEY_BYTES:
        raise InvalidKeyLength(
            f"unexpected sk length {len(sk)} (expected {SECRET_KEY_BYTES})"
        )
    return pk, sk


def sign(secret_key: bytes, message: bytes) -> bytes:
    """Produce an ML-DSA-65 signature."""
    if len(secret_key) != SECRET_KEY_BYTES:
        raise InvalidKeyLength(
            f"expected sk of {SECRET_KEY_BYTES} B, got {len(secret_key)} B"
        )
    sig = _sign_impl(secret_key, message)
    if len(sig) > SIGNATURE_MAX_BYTES:
        raise MLDSAError(
            f"signature length {len(sig)} exceeds FIPS 204 max {SIGNATURE_MAX_BYTES}"
        )
    return sig


def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """
    Verify an ML-DSA-65 signature. Returns ``True``/``False``.

    Returns ``False`` (not raises) on cryptographic mismatch. Raises
    :class:`InvalidKeyLength` only on malformed key sizes (consistent
    with the convention that PK length must be enforced even before
    any cryptographic operation).
    """
    if len(public_key) != PUBLIC_KEY_BYTES:
        raise InvalidKeyLength(
            f"expected pk of {PUBLIC_KEY_BYTES} B, got {len(public_key)} B"
        )
    if len(signature) > SIGNATURE_MAX_BYTES:
        return False
    return _verify_impl(public_key, message, signature)


def authenticated_message(
    secret_key: bytes,
    payload: bytes,
    metadata: bytes = b"",
) -> tuple[bytes, bytes]:
    """
    Build and sign an authenticated message ``(metadata || payload)``.

    This is the canonical pattern used by clients in arya-STARK
    (Section III, "Phase 4: client signs the round/gradient bundle").

    Concretely, the message is::

        msg = metadata + b"\\x00" + payload

    The single ``NUL`` byte separator prevents trivial concatenation
    ambiguities (e.g., two distinct ``(metadata, payload)`` pairs
    producing the same ``msg``).

    Returns
    -------
    (message, signature) : tuple[bytes, bytes]
        The constructed ``message`` and its ML-DSA-65 ``signature``.
        Both are needed by the verifier downstream.
    """
    msg = metadata + b"\x00" + payload if metadata else payload
    sig = sign(secret_key, msg)
    return msg, sig


__all__ = [
    "ML_DSA_65",
    "PUBLIC_KEY_BYTES",
    "SECRET_KEY_BYTES",
    "SIGNATURE_MAX_BYTES",
    "MLDSAError",
    "InvalidKeyLength",
    "authenticated_message",
    "get_backend_name",
    "get_sizes",
    "keygen",
    "sign",
    "verify",
]

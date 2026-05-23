"""
Tests for the Python ↔ Rust STARK bridge.

These tests exercise the full prove/verify pipeline through the
Rust binaries. They are skipped if the binaries are not built.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from arya_stark.client.stark_bridge import (
    StarkBridgeError,
    StarkProof,
    prove_dot_product,
    verify_dot_product,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = REPO_ROOT / "rust" / "target" / "release"


def _bin_available() -> bool:
    if os.environ.get("ARYA_STARK_PROVE_BIN") and os.environ.get("ARYA_STARK_VERIFY_BIN"):
        return True
    return (RELEASE_DIR / "prove").exists() and (RELEASE_DIR / "verify").exists()


pytestmark = pytest.mark.skipif(
    not _bin_available(),
    reason="STARK binaries not built (run cargo build --release -p stark-prover)",
)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_simple_dot_product_proves_and_verifies() -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [2, 4, 6, 8, 10, 12, 14, 16]
    expected_c = sum(ai * bi for ai, bi in zip(a, b))  # 408

    proof = prove_dot_product(a, b)
    assert isinstance(proof, StarkProof)
    assert proof.public_output == expected_c
    assert proof.n == 8
    assert proof.size_bytes >= 5_000, (
        f"proof too small ({proof.size_bytes} B) — possible 43-bytes regression"
    )

    assert verify_dot_product(proof, a, b, expected_c)


def test_proof_size_for_realistic_n() -> None:
    """For n=100, proof should be ≥ 25 KB."""
    rng = np.random.default_rng(seed=42)
    a = rng.integers(0, 1_000_000, size=100, dtype=np.uint64)
    b = rng.integers(0, 1_000_000, size=100, dtype=np.uint64)
    proof = prove_dot_product(a, b)
    assert proof.size_bytes >= 25 * 1024, (
        f"proof size {proof.size_bytes} B suspiciously small for n=100"
    )


# ---------------------------------------------------------------------------
# Soundness
# ---------------------------------------------------------------------------


def test_wrong_claimed_c_rejected() -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [2, 4, 6, 8, 10, 12, 14, 16]
    proof = prove_dot_product(a, b)
    correct_c = proof.public_output
    assert verify_dot_product(proof, a, b, correct_c)
    # Lying about c
    assert not verify_dot_product(proof, a, b, correct_c + 1)
    assert not verify_dot_product(proof, a, b, 0)


def test_modified_proof_rejected() -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [2, 4, 6, 8, 10, 12, 14, 16]
    proof = prove_dot_product(a, b)
    # Tamper with proof bytes deep inside
    bad_bytes = bytearray(proof.proof_bytes)
    idx = len(bad_bytes) // 2
    bad_bytes[idx] ^= 0xFF
    bad_proof = StarkProof(
        proof_bytes=bytes(bad_bytes),
        public_output=proof.public_output,
        n=proof.n,
    )
    # Either verify returns False, or raises StarkBridgeError on malformed bytes.
    try:
        result = verify_dot_product(bad_proof, a, b, proof.public_output)
        assert result is False
    except StarkBridgeError:
        # Proof corruption detected at deserialisation level — also acceptable.
        pass


def test_modified_inputs_rejected() -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [2, 4, 6, 8, 10, 12, 14, 16]
    proof = prove_dot_product(a, b)
    # Verify with different a
    a_bad = [1, 2, 3, 4, 5, 6, 7, 9]  # last value differs
    assert not verify_dot_product(proof, a_bad, b, proof.public_output)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(StarkBridgeError, match="same length"):
        prove_dot_product([1, 2, 3], [1, 2])


def test_numpy_arrays_accepted() -> None:
    """numpy.uint64 arrays should work just like Python lists."""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)
    b = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint64)
    proof = prove_dot_product(a, b)
    expected = sum(int(x) * int(y) for x, y in zip(a, b))  # 1440
    assert proof.public_output == expected
    assert verify_dot_product(proof, a, b, expected)

"""
Unit tests for ``arya_stark.encoding``.

Three layers of coverage:

1. **Hand-picked edge cases** — zeros, ones, negative ones, prime
   boundaries, denormals, smallest representable values.
2. **Property-based round-trips** — for thousands of random floats,
   verify ``decode(encode(x)) == quantise(x, m)`` (within 10^-m).
3. **Linearity invariant** (Proposition III.2) — sum-then-decode
   equals decode-then-sum, when admissible.

The cross-language equivalence with Rust is checked separately in
``test_encoding_rust_compat.py`` (P1 final step), which calls the
Rust binary and compares outputs byte-for-byte.
"""
from __future__ import annotations

import math
import random
from typing import List

import numpy as np
import pytest

from arya_stark.encoding import (
    GOLDILOCKS_PRIME,
    HALF_PRIME,
    aggregate_field_sum,
    decode_scalar,
    decode_vector,
    encode_scalar,
    encode_vector,
    is_admissible,
    max_admissible_value,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_goldilocks_prime() -> None:
    assert GOLDILOCKS_PRIME == (1 << 64) - (1 << 32) + 1
    assert GOLDILOCKS_PRIME == 0xFFFF_FFFF_0000_0001
    # p ≈ 2^64 - 2^32, just below 2^64
    assert (1 << 63) < GOLDILOCKS_PRIME < (1 << 64)


def test_half_prime() -> None:
    assert HALF_PRIME == GOLDILOCKS_PRIME // 2
    assert 2 * HALF_PRIME <= GOLDILOCKS_PRIME < 2 * HALF_PRIME + 2


# ---------------------------------------------------------------------------
# Hand-picked edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_zero_encodes_to_zero(m: int) -> None:
    assert encode_scalar(0.0, m) == 0
    assert decode_scalar(0, m) == 0.0


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_negative_one(m: int) -> None:
    """encode(-1.0) = p - 10^m"""
    assert encode_scalar(-1.0, m) == GOLDILOCKS_PRIME - 10**m
    assert decode_scalar(GOLDILOCKS_PRIME - 10**m, m) == -1.0


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_positive_one(m: int) -> None:
    assert encode_scalar(1.0, m) == 10**m
    assert decode_scalar(10**m, m) == 1.0


@pytest.mark.parametrize("m", [3, 6])
def test_quantisation_grid_above_step(m: int) -> None:
    """A value cleanly above the grid step encodes to +1.

    For floor semantics, the negative case is non-trivial:
    ``floor(-1.91) = -2``, NOT ``-1``. So the negative pair encodes
    to ``GOLDILOCKS_PRIME - 2``, not ``- 1``. We test both directions
    explicitly to lock in this semantic.
    """
    if m == 6:
        epsilon = 2.0 ** -19  # ≈ 1.91e-6, exactly representable
        assert encode_scalar(epsilon, 6) == 1
        # floor(-1.91) = -2 (NOT -1), so x_int = -2, field elt = p - 2.
        assert encode_scalar(-epsilon, 6) == GOLDILOCKS_PRIME - 2
    elif m == 3:
        epsilon = 2.0 ** -9  # ≈ 1.95e-3
        assert encode_scalar(epsilon, 3) == 1
        assert encode_scalar(-epsilon, 3) == GOLDILOCKS_PRIME - 2


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------


def test_admissibility_obvious() -> None:
    assert is_admissible(0.0, 6)
    assert is_admissible(1e3, 6)
    assert is_admissible(1e6, 6)
    # p / (2 · 10^6) ≈ 9.22 × 10^12 → 1e13 should be over the limit.
    assert not is_admissible(1e13, 6)
    assert not is_admissible(-1e13, 6)


def test_admissibility_at_boundary() -> None:
    bound = max_admissible_value(6)
    # Half the limit is safe.
    assert is_admissible(bound / 2, 6)
    # 2× the limit is unsafe.
    assert not is_admissible(2 * bound, 6)


def test_max_admissible_value_consistent() -> None:
    for m in range(1, 9):
        assert math.isclose(
            max_admissible_value(m),
            GOLDILOCKS_PRIME / (2.0 * (10**m)),
            rel_tol=1e-15,
        )


def test_inadmissible_raises() -> None:
    with pytest.raises(ValueError, match="not admissible"):
        encode_scalar(1e15, 6)


def test_non_finite_raises() -> None:
    with pytest.raises(ValueError):
        encode_scalar(float("inf"), 6)
    with pytest.raises(ValueError):
        encode_scalar(float("nan"), 6)


# ---------------------------------------------------------------------------
# Round-trip property: decode(encode(x)) ≈ quantise(x)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [3, 4, 5, 6, 7])
def test_round_trip_random_floats(m: int) -> None:
    """
    For 1000 random floats in the admissible range, encoding then
    decoding recovers x up to the quantisation error 10^-m.
    """
    rng = random.Random(42)
    bound = 0.95 * max_admissible_value(m)  # 5% margin to avoid boundary issues
    for _ in range(1000):
        x = rng.uniform(-bound, bound)
        e = encode_scalar(x, m)
        x_back = decode_scalar(e, m)
        assert abs(x - x_back) <= 1.0 / (10**m), (
            f"Round-trip exceeds quantisation: x={x}, x_back={x_back}, "
            f"err={abs(x - x_back)}, allowed=10^-{m}"
        )


def test_round_trip_realistic_gradient_range() -> None:
    """
    Gradients in FL typically have ‖g‖∞ ≤ 100. We stress-test this range
    extensively at m=6 (default).
    """
    rng = random.Random(123)
    for _ in range(5000):
        x = rng.uniform(-100.0, 100.0)
        e = encode_scalar(x, 6)
        x_back = decode_scalar(e, 6)
        assert abs(x - x_back) <= 1e-6


# ---------------------------------------------------------------------------
# Vector API
# ---------------------------------------------------------------------------


def test_encode_decode_vector_shape() -> None:
    g = np.array([0.0, 0.5, -0.5, 1.234, -1.234])
    e = encode_vector(g, 6)
    assert e.shape == g.shape
    assert e.dtype == np.uint64
    g_back = decode_vector(e, 6)
    np.testing.assert_allclose(g_back, g, atol=1e-6)


def test_encode_vector_handles_2d_input_raveled() -> None:
    g_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    e = encode_vector(g_2d, 6)
    assert e.shape == (4,)


def test_encode_empty_vector() -> None:
    e = encode_vector(np.array([], dtype=np.float64), 6)
    assert e.shape == (0,)
    assert e.dtype == np.uint64


# ---------------------------------------------------------------------------
# Linearity (Proposition III.2)
# ---------------------------------------------------------------------------


def test_linearity_two_clients() -> None:
    """For 2 admissible vectors, sum-then-decode = decode-then-sum."""
    g1 = np.array([1.5, -0.25, 3.14])
    g2 = np.array([0.5, 0.75, -2.71])
    e1 = encode_vector(g1, 6)
    e2 = encode_vector(g2, 6)
    e_sum = aggregate_field_sum(np.stack([e1, e2]))
    g_decoded_sum = decode_vector(e_sum, 6)
    g_real_sum = g1 + g2
    np.testing.assert_allclose(g_decoded_sum, g_real_sum, atol=2e-6)


def test_linearity_M_clients() -> None:
    """100 clients with bounded gradients."""
    rng = np.random.default_rng(seed=7)
    M, d = 100, 50
    grads = rng.uniform(-1.0, 1.0, size=(M, d))
    encoded = np.stack([encode_vector(g, 6) for g in grads])
    e_sum = aggregate_field_sum(encoded)
    g_decoded = decode_vector(e_sum, 6)
    g_real = grads.sum(axis=0)
    np.testing.assert_allclose(g_decoded, g_real, atol=1e-4)


def test_linearity_admissibility_constraint() -> None:
    """
    Aggregation breaks once Σ |x_int_i| ≥ p/2 (Section III-D, end).

    With M=10 and a 6-precision encoding, the per-client value must be
    bounded by p/(2·M·10^6) ≈ 9.22 × 10^11. Going beyond should not
    silently corrupt the result, but rather be detected at encode time.
    """
    M = 10
    safe_limit = GOLDILOCKS_PRIME / (2 * M * (10**6))
    # Each client at half of the safe per-client bound: aggregation OK.
    g = np.full(5, safe_limit / 2)
    encoded = np.stack([encode_vector(g, 6) for _ in range(M)])
    e_sum = aggregate_field_sum(encoded)
    g_decoded = decode_vector(e_sum, 6)
    np.testing.assert_allclose(g_decoded, M * g, rtol=1e-9)


# ---------------------------------------------------------------------------
# Determinism (cross-platform / cross-version stability)
# ---------------------------------------------------------------------------


def test_encoding_deterministic_under_repr() -> None:
    """
    Encoding via the Decimal-from-repr path is deterministic: encoding
    the same float twice always yields the same field element.
    """
    x = 0.1 + 0.2  # famous floating-point quirk
    e1 = encode_scalar(x, 6)
    e2 = encode_scalar(x, 6)
    assert e1 == e2


def test_known_test_vectors() -> None:
    """
    Reference vectors that the Rust port MUST match bit-for-bit.

    These are the values consumed by the cross-language compatibility
    test ``test_encoding_rust_compat.py``.

    Important: we restrict to values that are **exactly representable**
    in IEEE 754 binary64 (powers of 2 and small dyadic rationals), so
    that every reasonable interpretation of "encode" agrees. Decimals
    like ``1e-6`` or ``2e-6`` are NOT exactly representable and lead
    to platform-dependent floor results.
    """
    vectors: List[tuple[float, int, int]] = [
        # (x, m, expected_field_element) — all exactly representable
        (0.0, 6, 0),
        (1.0, 6, 1_000_000),
        (-1.0, 6, GOLDILOCKS_PRIME - 1_000_000),
        (0.5, 6, 500_000),                   # 1/2
        (-0.5, 6, GOLDILOCKS_PRIME - 500_000),
        (0.25, 6, 250_000),                  # 1/4
        (-0.25, 6, GOLDILOCKS_PRIME - 250_000),
        (0.125, 6, 125_000),                 # 1/8
        (3.0, 6, 3_000_000),
        (-3.0, 6, GOLDILOCKS_PRIME - 3_000_000),
        (100.0, 6, 100_000_000),
        (-100.0, 6, GOLDILOCKS_PRIME - 100_000_000),
        (1024.0, 6, 1_024_000_000),          # 2^10
        (-1024.0, 6, GOLDILOCKS_PRIME - 1_024_000_000),
    ]
    for x, m, expected in vectors:
        got = encode_scalar(x, m)
        assert got == expected, (
            f"Mismatch for ({x}, m={m}): got {got}, expected {expected}"
        )

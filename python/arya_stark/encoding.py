"""
arya_stark.encoding
===================

Bit-exact encoding `φ : ℝ → 𝔽_p` of real-valued gradients into the
Goldilocks finite field, mirroring exactly the Rust implementation in
``rust/stark-prover/src/encoding.rs``.

The two implementations MUST produce identical outputs on every input;
this is enforced by ``python/tests/test_encoding_roundtrip.py``.

Definitions (paper §III-D)
--------------------------

Given precision ``m ∈ ℕ`` and prime ``p`` (Goldilocks):

    x_int = floor(x · 10^m)              ∈ ℤ      (signed integer)
    x̃     = x_int mod p                  ∈ [0, p) (field element)

Decoding rule::

    if x̃ < p/2:  x_int = x̃
    else:        x_int = x̃ - p
    x^(m)        = x_int / 10^m

A value ``x ∈ ℝ`` is *admissible* (Definition III.1) iff
``|x_int| < p/2``, i.e. ``|x| < p/(2 · 10^m)``.

Notes on numerical correctness
------------------------------

* ``floor(x * 10^m)`` is computed in Python using a deterministic
  recipe based on ``Decimal`` for floats outside the safe ``int * int``
  domain, to ensure bit-exact agreement with the Rust path that uses
  ``i128`` arithmetic.
* The Goldilocks prime ``p = 2^64 − 2^32 + 1`` is just below ``2^64``,
  so ``p`` does not fit in a signed 64-bit integer; computations in
  Rust use ``i128`` / ``u128`` accordingly.
* A negative integer ``x_int`` is mapped to ``x_int + p`` (Python
  ``%`` already does this for ``int``; we re-derive it via explicit
  arithmetic in Rust).

Public API
----------

* :func:`encode_scalar`         — single ``float → 𝔽_p``.
* :func:`decode_scalar`         — single ``𝔽_p → float``.
* :func:`encode_vector`         — batched ``np.ndarray → np.ndarray[uint64]``.
* :func:`decode_vector`         — batched ``np.ndarray[uint64] → np.ndarray``.
* :func:`is_admissible`         — admissibility predicate.
* :func:`max_admissible_value`  — symbolic limit, ``p / (2·10^m)``.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Goldilocks prime ``p = 2^64 − 2^32 + 1``.
GOLDILOCKS_PRIME: Final[int] = (1 << 64) - (1 << 32) + 1

#: Half-prime threshold used for sign extension in decoding.
HALF_PRIME: Final[int] = GOLDILOCKS_PRIME // 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _floor_scaled(x: float, m: int) -> int:
    """
    Return ``floor(x · 10^m)`` as a Python int.

    Uses ``Decimal`` for guaranteed determinism with ``17`` significant
    digits in the source string. ``%.17g`` is the IEEE 754 spec-mandated
    minimum number of digits that uniquely identifies any binary64
    float, so this representation is bit-exact across Python and Rust
    (whereas ``repr(x)``'s shortest representation is platform-dependent
    for halfway values like ``664648604144431.25``).

    >>> _floor_scaled(1.234567, 6)
    1234567
    >>> _floor_scaled(-1.234567, 6)
    -1234567
    >>> _floor_scaled(1.2345674, 6)
    1234567
    >>> _floor_scaled(-1.2345674, 6)
    -1234568
    >>> _floor_scaled(0.0, 6)
    0
    >>> _floor_scaled(1e-12, 6)
    0
    """
    if not np.isfinite(x):
        raise ValueError(f"Cannot encode non-finite value {x!r}")
    # Use 17 significant digits (IEEE 754 spec) to guarantee unique
    # cross-language representation. ``%.17g`` is identical to Rust's
    # ``format!("{:.17e}", x)`` modulo trivial syntactic differences,
    # and both decode to the same ``Decimal``.
    d = Decimal(format(x, ".17g")) * (Decimal(10) ** m)
    return int(d.to_integral_value(rounding=ROUND_FLOOR))


def _signed_to_field(x_int: int) -> int:
    """
    Map a (possibly negative) Python int into ``[0, p)``.

    Python's ``%`` already returns a value in ``[0, p)`` for any int,
    so this is essentially ``x_int % p``. We name the helper to make
    the cross-language intent explicit.
    """
    return x_int % GOLDILOCKS_PRIME


def _field_to_signed(x_tilde: int) -> int:
    """Inverse of :func:`_signed_to_field`, for the admissible range."""
    if x_tilde < HALF_PRIME:
        return x_tilde
    return x_tilde - GOLDILOCKS_PRIME


# ---------------------------------------------------------------------------
# Public scalar API
# ---------------------------------------------------------------------------


def is_admissible(x: float, m: int) -> bool:
    """
    Return ``True`` iff ``|floor(x · 10^m)| < p/2``.

    >>> is_admissible(0.0, 6)
    True
    >>> is_admissible(1e6, 6)
    True
    >>> is_admissible(1e13, 6)
    False
    """
    if not np.isfinite(x):
        return False
    return abs(_floor_scaled(x, m)) < HALF_PRIME


def max_admissible_value(m: int) -> float:
    """
    Return the symbolic upper bound ``p/(2·10^m)`` as a float.

    >>> max_admissible_value(6) > 9e12
    True
    """
    return GOLDILOCKS_PRIME / (2.0 * (10**m))


def encode_scalar(x: float, m: int = 6) -> int:
    """
    Encode a scalar ``x ∈ ℝ`` into ``𝔽_p`` (returned as a Python ``int``
    in ``[0, p)``).

    Raises
    ------
    ValueError
        If ``x`` is not finite, or if ``x`` is outside the admissible
        range ``|x_int| < p/2``.

    Examples
    --------
    >>> encode_scalar(0.0, 6)
    0
    >>> encode_scalar(1.234567, 6)
    1234567
    >>> encode_scalar(-1.0, 6) == GOLDILOCKS_PRIME - 1_000_000
    True
    """
    x_int = _floor_scaled(x, m)
    if abs(x_int) >= HALF_PRIME:
        raise ValueError(
            f"Value {x!r} (x_int={x_int}) is not admissible at m={m}: "
            f"|x_int| must be < p/2 = {HALF_PRIME}"
        )
    return _signed_to_field(x_int)


def decode_scalar(x_tilde: int, m: int = 6) -> float:
    """
    Decode a field element ``x̃ ∈ [0, p)`` back to a real value.

    Inverse of :func:`encode_scalar` up to the quantisation error
    ``≤ 10^-m`` (Proposition III.1).

    Examples
    --------
    >>> decode_scalar(0, 6)
    0.0
    >>> decode_scalar(1_234_566, 6)
    1.234566
    >>> decode_scalar(GOLDILOCKS_PRIME - 1_000_000, 6) == -1.0
    True
    """
    if not 0 <= x_tilde < GOLDILOCKS_PRIME:
        raise ValueError(f"x̃={x_tilde} not in [0, p={GOLDILOCKS_PRIME})")
    x_int = _field_to_signed(x_tilde)
    # Use float-division since the result type is float; precision
    # loss here is at most 10^-m, by construction.
    return x_int / (10**m)


# ---------------------------------------------------------------------------
# Public vector API (gradient-shaped tensors)
# ---------------------------------------------------------------------------


def encode_vector(g: np.ndarray, m: int = 6) -> np.ndarray:
    """
    Encode a gradient vector ``g ∈ ℝ^d`` into ``𝔽_p^d``.

    Parameters
    ----------
    g
        1-D ``np.ndarray`` of any float dtype.
    m
        Decimal precision (default 6).

    Returns
    -------
    np.ndarray
        1-D ``np.ndarray`` of dtype ``np.uint64`` (each entry in ``[0, p)``).

    Raises
    ------
    ValueError
        If any entry is not admissible.

    Examples
    --------
    >>> g = np.array([0.0, 1.0, -1.0])
    >>> e = encode_vector(g, 6)
    >>> int(e[0]), int(e[1]), int(e[2]) == GOLDILOCKS_PRIME - 1_000_000
    (0, 1000000, True)
    """
    g = np.asarray(g, dtype=np.float64).ravel()
    out = np.empty(g.shape, dtype=np.uint64)
    for i, x in enumerate(g):
        out[i] = np.uint64(encode_scalar(float(x), m))
    return out


def decode_vector(g_tilde: np.ndarray, m: int = 6) -> np.ndarray:
    """
    Decode a vector of field elements back to a real-valued gradient.

    Examples
    --------
    >>> e = np.array([0, 1_000_000, GOLDILOCKS_PRIME - 1_000_000], dtype=np.uint64)
    >>> decode_vector(e, 6).tolist()
    [0.0, 1.0, -1.0]
    """
    g_tilde = np.asarray(g_tilde, dtype=np.uint64).ravel()
    out = np.empty(g_tilde.shape, dtype=np.float64)
    for i, xt in enumerate(g_tilde):
        out[i] = decode_scalar(int(xt), m)
    return out


# ---------------------------------------------------------------------------
# Aggregation linearity (Proposition III.2)
# ---------------------------------------------------------------------------


def aggregate_field_sum(encoded_grads: np.ndarray) -> np.ndarray:
    """
    Sum of ``M`` encoded gradients in ``𝔽_p``, coordinate-wise.

    Used by the server to verify the linearity property of the
    encoding (Proposition III.2): provided the sum stays admissible,
    decoding the field-sum equals the sum of decoded values.

    Parameters
    ----------
    encoded_grads
        2-D ``np.ndarray`` of shape ``(M, d)`` with dtype ``np.uint64``.

    Returns
    -------
    np.ndarray
        1-D ``np.ndarray`` of shape ``(d,)``, dtype ``np.uint64``.

    Notes
    -----
    The naive ``np.sum`` would overflow ``uint64`` for ``M ≥ 2``.
    We therefore reduce to Python ints (arbitrary precision) and
    take ``mod p`` per coordinate.
    """
    encoded_grads = np.asarray(encoded_grads, dtype=np.uint64)
    if encoded_grads.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {encoded_grads.shape}")
    M, d = encoded_grads.shape
    out = np.empty(d, dtype=np.uint64)
    for j in range(d):
        s = 0
        for i in range(M):
            s += int(encoded_grads[i, j])
        out[j] = np.uint64(s % GOLDILOCKS_PRIME)
    return out


__all__ = [
    "GOLDILOCKS_PRIME",
    "HALF_PRIME",
    "is_admissible",
    "max_admissible_value",
    "encode_scalar",
    "decode_scalar",
    "encode_vector",
    "decode_vector",
    "aggregate_field_sum",
]

//! Gradient encoding `φ : ℝ → 𝔽_p` — bit-exact mirror of
//! `python/arya_stark/encoding.py`.
//!
//! See the module docstring of the Python file for the mathematical
//! definitions; this comment focuses on the Rust-specific subtleties.
//!
//! Numerical determinism
//! ---------------------
//!
//! The Goldilocks prime `p = 2^64 − 2^32 + 1` is just below `2^64`,
//! so it does NOT fit in a signed 64-bit integer. All intermediate
//! computations therefore use `i128` / `u128`.
//!
//! `floor(x · 10^m)` is computed via the `f64` path:
//!
//!   x_int = (x * 10_f64.powi(m)).floor() as i128
//!
//! This matches Python's `Decimal(repr(x)) * 10^m` for every finite
//! `f64` `x` such that `|x| < p / (2 · 10^m)`, because both Python and
//! Rust use IEEE 754 `binary64` floats and `repr(float)` round-trips
//! exactly. The cross-language test
//! `python/tests/test_encoding_rust_compat.py` enforces this byte-for-byte.
//!
//! Mapping to the field
//! --------------------
//!
//! For `x_int ∈ ℤ`:
//!
//!   x̃ = ((x_int mod p) + p) mod p   ∈ [0, p)
//!
//! In Rust, `%` on signed integers can return a negative result; we
//! use `i128::rem_euclid` which yields a non-negative remainder,
//! exactly matching Python's `int %` semantics.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Goldilocks prime `p = 2^64 − 2^32 + 1`.
pub const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// Half-prime threshold used for sign extension in decoding.
pub const HALF_PRIME: u64 = GOLDILOCKS_PRIME / 2;

/// Default decimal precision `m`, matching `CryptoConfig.encoding_precision_m`
/// in `python/arya_stark/config.py`.
pub const DEFAULT_PRECISION: u32 = 6;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EncodingError {
    #[error("non-finite input value")]
    NonFinite,

    #[error("value not admissible: |x_int|={abs_x_int} ≥ p/2={half_prime}")]
    NotAdmissible { abs_x_int: u128, half_prime: u64 },

    #[error("field element {value} ≥ p={prime}")]
    OutOfField { value: u64, prime: u64 },
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// `floor(x · 10^m)` as a 128-bit signed integer, bit-exact with
/// Python's ``Decimal(format(x, '.17g')) * 10^m`` then ``floor``.
///
/// We use **17 significant digits** (the IEEE 754 spec-mandated
/// minimum that uniquely identifies any binary64 float) to guarantee
/// cross-language equivalence. The shortest round-trip
/// representation (``{x}`` in Rust, ``repr(x)`` in Python) can disagree
/// on halfway values like ``664648604144431.25`` because the two
/// languages may pick ``.2`` vs ``.3`` due to different banker's
/// rounding implementations.
///
/// Implementation:
///   1. Format ``x`` with 17 significant digits in scientific notation
///      via ``format!("{:.16e}", x)``: ``"6.6464860414443125e14"``.
///   2. Parse that into ``(mantissa_digits, exponent)`` and combine
///      with ``m`` for the final scaling, all in ``i128``.
#[inline]
fn floor_scaled(x: f64, m: u32) -> Result<i128, EncodingError> {
    if !x.is_finite() {
        return Err(EncodingError::NonFinite);
    }
    // ``{:.16e}`` produces 1 leading digit + 16 decimals = 17 significant digits.
    // For x = 0.0 it produces "0.0000000000000000e0".
    let s = format!("{x:.16e}");
    parse_decimal_floor_scaled(&s, m)
}

/// Internal: take a decimal string ``s`` (as produced by `{x}` for an
/// `f64`) and integer ``m``, return ``floor(x · 10^m)`` as `i128`.
fn parse_decimal_floor_scaled(s: &str, m: u32) -> Result<i128, EncodingError> {
    // Split off optional sign.
    let (sign, rest) = match s.as_bytes().first() {
        Some(b'-') => (-1_i128, &s[1..]),
        Some(b'+') => (1_i128, &s[1..]),
        _ => (1_i128, s),
    };

    // Split off optional exponent ("...eK" or "...EK").
    let (mantissa, exp): (&str, i32) = match rest.find(|c: char| c == 'e' || c == 'E') {
        Some(idx) => {
            let exp_str = &rest[idx + 1..];
            let exp_val: i32 = exp_str.parse().map_err(|_| EncodingError::NonFinite)?;
            (&rest[..idx], exp_val)
        }
        None => (rest, 0),
    };

    // Split mantissa on '.'.
    let (int_part, frac_part) = match mantissa.find('.') {
        Some(idx) => (&mantissa[..idx], &mantissa[idx + 1..]),
        None => (mantissa, ""),
    };

    let int_part = if int_part.is_empty() { "0" } else { int_part };

    // Combine all digits as a big integer in i128.
    // The decimal point is logically `frac_part.len()` positions to
    // the left; the exponent then shifts it by `exp`.
    //
    //   value = sign · digits · 10^(exp - frac_len)
    //
    // We want floor(value · 10^m) = sign · floor(digits · 10^(exp − frac_len + m)).
    //
    // Let `shift = exp - frac_len + m`.
    //
    //   shift ≥ 0 → multiply digits by 10^shift, no rounding.
    //   shift < 0 → divide digits by 10^(-shift); for negative `value`,
    //               we must use *floor* division (towards −∞), not
    //               truncation, to match Python's `Decimal(...).
    //               to_integral_value(rounding=ROUND_FLOOR)`.

    let digits_str: String = int_part
        .chars()
        .chain(frac_part.chars())
        .filter(|c| c.is_ascii_digit())
        .collect();

    if digits_str.is_empty() {
        // Should not happen for a finite float, but guard anyway.
        return Ok(0);
    }

    let mut digits: i128 = digits_str
        .parse()
        .map_err(|_| EncodingError::NonFinite)?;

    let frac_len: i32 = frac_part.len() as i32;
    let shift: i32 = exp - frac_len + m as i32;

    let result: i128 = if shift >= 0 {
        // Pure multiplication by 10^shift.
        let factor = pow10_i128(shift as u32)?;
        digits.checked_mul(factor).ok_or(EncodingError::NonFinite)?
    } else {
        // Floor division: for positive sign, classical floor is `div`
        // truncated; for negative sign, we have to round AWAY from 0.
        //
        // sign · floor(digits / 10^k) where k = -shift
        let k = (-shift) as u32;
        let divisor = pow10_i128(k)?;
        // Note: at this stage, `digits` is non-negative (we stripped sign).
        // So integer division towards 0 is truncation.
        // For sign = +1: floor(d / D) = d / D (truncation).
        // For sign = -1: we want floor(-d / D) = -(ceil(d / D))
        //                                       = -((d + D - 1) / D).
        if sign >= 0 {
            digits / divisor
        } else {
            // Compute ceil(digits / divisor) for non-negative digits.
            let q = digits / divisor;
            let r = digits % divisor;
            if r == 0 {
                q
            } else {
                q + 1
            }
        }
    };

    Ok(sign * result)
}

#[inline]
fn pow10_i128(exp: u32) -> Result<i128, EncodingError> {
    let mut result: i128 = 1;
    let ten: i128 = 10;
    for _ in 0..exp {
        result = result.checked_mul(ten).ok_or(EncodingError::NonFinite)?;
    }
    Ok(result)
}

/// Map a signed integer into `[0, p)`.
#[inline]
fn signed_to_field(x_int: i128) -> u64 {
    let p = GOLDILOCKS_PRIME as i128;
    let r = x_int.rem_euclid(p);
    debug_assert!(r >= 0 && r < p);
    r as u64
}

/// Inverse of [`signed_to_field`] for the admissible range.
#[inline]
fn field_to_signed(x_tilde: u64) -> i128 {
    if x_tilde < HALF_PRIME {
        x_tilde as i128
    } else {
        x_tilde as i128 - GOLDILOCKS_PRIME as i128
    }
}

// ---------------------------------------------------------------------------
// Public scalar API
// ---------------------------------------------------------------------------

/// Encode a real `x` into `𝔽_p`.
pub fn encode_scalar(x: f64, m: u32) -> Result<u64, EncodingError> {
    let x_int = floor_scaled(x, m)?;
    if x_int.unsigned_abs() >= HALF_PRIME as u128 {
        return Err(EncodingError::NotAdmissible {
            abs_x_int: x_int.unsigned_abs(),
            half_prime: HALF_PRIME,
        });
    }
    Ok(signed_to_field(x_int))
}

/// Decode a field element back to a real value.
pub fn decode_scalar(x_tilde: u64, m: u32) -> Result<f64, EncodingError> {
    if x_tilde >= GOLDILOCKS_PRIME {
        return Err(EncodingError::OutOfField {
            value: x_tilde,
            prime: GOLDILOCKS_PRIME,
        });
    }
    let x_int = field_to_signed(x_tilde);
    let scale = 10_f64.powi(m as i32);
    Ok(x_int as f64 / scale)
}

/// True iff `|floor(x · 10^m)| < p/2`.
pub fn is_admissible(x: f64, m: u32) -> bool {
    match floor_scaled(x, m) {
        Ok(x_int) => x_int.unsigned_abs() < HALF_PRIME as u128,
        Err(_) => false,
    }
}

/// Symbolic upper bound `p / (2 · 10^m)`.
pub fn max_admissible_value(m: u32) -> f64 {
    GOLDILOCKS_PRIME as f64 / (2.0 * 10_f64.powi(m as i32))
}

// ---------------------------------------------------------------------------
// Vector API
// ---------------------------------------------------------------------------

/// Encode a slice of reals into a `Vec<u64>` of field elements.
pub fn encode_vector(g: &[f64], m: u32) -> Result<Vec<u64>, EncodingError> {
    g.iter().map(|&x| encode_scalar(x, m)).collect()
}

/// Decode a slice of field elements back to a `Vec<f64>`.
pub fn decode_vector(g_tilde: &[u64], m: u32) -> Result<Vec<f64>, EncodingError> {
    g_tilde.iter().map(|&xt| decode_scalar(xt, m)).collect()
}

/// Coordinate-wise sum in `𝔽_p` of `M` encoded gradients of width `d`.
///
/// Matches Python's `aggregate_field_sum`. Reduces after each addition
/// in `u128` to avoid overflow.
pub fn aggregate_field_sum(encoded_grads: &[Vec<u64>]) -> Vec<u64> {
    if encoded_grads.is_empty() {
        return Vec::new();
    }
    let d = encoded_grads[0].len();
    let mut out = vec![0_u64; d];
    let p = GOLDILOCKS_PRIME as u128;
    for col in 0..d {
        let mut s: u128 = 0;
        for row in encoded_grads {
            s = (s + row[col] as u128) % p;
        }
        out[col] = s as u64;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldilocks_prime_value() {
        let expected = (1_u128 << 64) - (1_u128 << 32) + 1;
        assert_eq!(GOLDILOCKS_PRIME as u128, expected);
        assert_eq!(GOLDILOCKS_PRIME, 0xFFFF_FFFF_0000_0001);
    }

    #[test]
    fn half_prime_is_floor_division() {
        assert_eq!(HALF_PRIME, GOLDILOCKS_PRIME / 2);
    }

    #[test]
    fn zero_encodes_to_zero() {
        for m in [3, 4, 5, 6] {
            assert_eq!(encode_scalar(0.0, m).unwrap(), 0);
            assert_eq!(decode_scalar(0, m).unwrap(), 0.0);
        }
    }

    #[test]
    fn negative_one() {
        for m in [3, 4, 5, 6] {
            let expected = GOLDILOCKS_PRIME - 10_u64.pow(m);
            assert_eq!(encode_scalar(-1.0, m).unwrap(), expected);
            assert_eq!(decode_scalar(expected, m).unwrap(), -1.0);
        }
    }

    #[test]
    fn positive_one() {
        for m in [3, 4, 5, 6] {
            assert_eq!(encode_scalar(1.0, m).unwrap(), 10_u64.pow(m));
            assert_eq!(decode_scalar(10_u64.pow(m), m).unwrap(), 1.0);
        }
    }

    #[test]
    fn quantisation_grid_above_step() {
        // 2^-19 ≈ 1.91e-6. Exactly representable, above 1e-6.
        // floor(1.91) = 1, floor(-1.91) = -2 (NOT -1).
        let epsilon = 2.0_f64.powi(-19);
        assert_eq!(encode_scalar(epsilon, 6).unwrap(), 1);
        assert_eq!(encode_scalar(-epsilon, 6).unwrap(), GOLDILOCKS_PRIME - 2);
    }

    /// These vectors are the same as the Python `test_known_test_vectors`.
    /// Any mismatch indicates a Rust ↔ Python encoding divergence.
    /// All values are exactly representable in IEEE 754 binary64.
    #[test]
    fn known_test_vectors_match_python() {
        let vectors: &[(f64, u32, u64)] = &[
            (0.0, 6, 0),
            (1.0, 6, 1_000_000),
            (-1.0, 6, GOLDILOCKS_PRIME - 1_000_000),
            (0.5, 6, 500_000),
            (-0.5, 6, GOLDILOCKS_PRIME - 500_000),
            (0.25, 6, 250_000),
            (-0.25, 6, GOLDILOCKS_PRIME - 250_000),
            (0.125, 6, 125_000),
            (3.0, 6, 3_000_000),
            (-3.0, 6, GOLDILOCKS_PRIME - 3_000_000),
            (100.0, 6, 100_000_000),
            (-100.0, 6, GOLDILOCKS_PRIME - 100_000_000),
            (1024.0, 6, 1_024_000_000),
            (-1024.0, 6, GOLDILOCKS_PRIME - 1_024_000_000),
        ];
        for &(x, m, expected) in vectors {
            let got = encode_scalar(x, m).unwrap();
            assert_eq!(got, expected, "mismatch for x={x}, m={m}");
        }
    }

    #[test]
    fn admissibility_obvious() {
        assert!(is_admissible(0.0, 6));
        assert!(is_admissible(1e6, 6));
        assert!(!is_admissible(1e13, 6));
        assert!(!is_admissible(-1e13, 6));
    }

    #[test]
    fn admissibility_at_boundary() {
        let bound = max_admissible_value(6);
        assert!(is_admissible(bound / 2.0, 6));
        assert!(!is_admissible(2.0 * bound, 6));
    }

    #[test]
    fn inadmissible_returns_error() {
        let r = encode_scalar(1e15, 6);
        assert!(matches!(r, Err(EncodingError::NotAdmissible { .. })));
    }

    #[test]
    fn nan_inf_return_error() {
        assert!(matches!(
            encode_scalar(f64::NAN, 6),
            Err(EncodingError::NonFinite)
        ));
        assert!(matches!(
            encode_scalar(f64::INFINITY, 6),
            Err(EncodingError::NonFinite)
        ));
    }

    #[test]
    fn out_of_field_decode_returns_error() {
        let r = decode_scalar(GOLDILOCKS_PRIME, 6);
        assert!(matches!(r, Err(EncodingError::OutOfField { .. })));
    }

    #[test]
    fn round_trip_random_floats() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let next = |state: &mut u64| -> f64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = *state >> 11;
            let r01 = (bits as f64) * (1.0_f64 / (1_u64 << 53) as f64);
            (r01 - 0.5) * 200.0
        };
        for _ in 0..1000 {
            let x = next(&mut state);
            let e = encode_scalar(x, 6).unwrap();
            let x_back = decode_scalar(e, 6).unwrap();
            assert!(
                (x - x_back).abs() <= 1e-6,
                "round-trip exceeded quantisation: x={x}, x_back={x_back}"
            );
        }
    }

    #[test]
    fn encode_decode_vector_shape() {
        let g = vec![0.0, 0.5, -0.5, 1.234, -1.234];
        let e = encode_vector(&g, 6).unwrap();
        assert_eq!(e.len(), g.len());
        let g_back = decode_vector(&e, 6).unwrap();
        for (a, b) in g.iter().zip(g_back.iter()) {
            assert!((a - b).abs() <= 1e-6);
        }
    }

    #[test]
    fn encode_empty_vector() {
        let e = encode_vector(&[], 6).unwrap();
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn linearity_two_clients() {
        let g1 = vec![1.5, -0.25, 3.14];
        let g2 = vec![0.5, 0.75, -2.71];
        let e1 = encode_vector(&g1, 6).unwrap();
        let e2 = encode_vector(&g2, 6).unwrap();
        let e_sum = aggregate_field_sum(&[e1, e2]);
        let g_decoded = decode_vector(&e_sum, 6).unwrap();
        for ((a, b), c) in g1.iter().zip(g2.iter()).zip(g_decoded.iter()) {
            assert!((a + b - c).abs() < 2e-6);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn linearity_M_clients() {
        let m_clients = 100;
        let d = 50;
        let mut grads: Vec<Vec<f64>> = Vec::with_capacity(m_clients);
        let mut state: u64 = 7;
        for _ in 0..m_clients {
            let mut row = Vec::with_capacity(d);
            for _ in 0..d {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = state >> 11;
                let r01 = (bits as f64) * (1.0_f64 / (1_u64 << 53) as f64);
                row.push((r01 - 0.5) * 2.0);
            }
            grads.push(row);
        }
        let encoded: Vec<Vec<u64>> = grads
            .iter()
            .map(|g| encode_vector(g, 6).unwrap())
            .collect();
        let e_sum = aggregate_field_sum(&encoded);
        let g_decoded = decode_vector(&e_sum, 6).unwrap();
        let mut g_real = vec![0.0; d];
        for row in &grads {
            for (j, &v) in row.iter().enumerate() {
                g_real[j] += v;
            }
        }
        for (a, b) in g_real.iter().zip(g_decoded.iter()) {
            assert!(
                (a - b).abs() <= 1e-4,
                "linearity broken: real={a}, decoded={b}"
            );
        }
    }
}

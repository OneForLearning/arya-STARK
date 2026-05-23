//! `encoding-bench` — Cross-language consistency checker.
//!
//! Reads a JSON file produced by Python's
//! ``arya_stark.encoding`` (via ``tests/test_encoding_rust_compat.py``)
//! and verifies that re-encoding every vector in Rust yields the same
//! bytes.
//!
//! Exits with status 0 on full match, 1 on any mismatch.
//!
//! Input format (JSON):
//!
//! ```json
//! {
//!   "vectors": [
//!     {
//!       "x_bits": "0x3ff3c0ca428c59fb",
//!       "x_repr": "1.234567",
//!       "m": 6,
//!       "expected": 1234567
//!     },
//!     ...
//!   ]
//! }
//! ```
//!
//! Note: ``x_bits`` is the IEEE-754 binary64 representation of the
//! float, sent as the canonical reference. We reconstruct the f64 via
//! ``f64::from_bits`` to avoid any loss in JSON parsing.
//! ``x_repr`` is included as a human-readable hint only.

use std::fs::File;
use std::io::Read;
use std::process::ExitCode;

use serde::{Deserialize, Serialize};
use stark_prover::encoding::{decode_scalar, encode_scalar};

#[derive(Debug, Deserialize, Serialize)]
struct TestVector {
    /// Hex-encoded `u64` IEEE 754 bits of `x` (e.g. ``"0x3ff..."``).
    x_bits: String,
    /// Human-readable repr (informational only).
    #[serde(default)]
    x_repr: String,
    m: u32,
    expected: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct TestSuite {
    vectors: Vec<TestVector>,
}

fn parse_hex_u64(s: &str) -> Option<u64> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: encoding-bench <input.json>");
        return ExitCode::from(2);
    }
    let path = &args[1];

    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut buf = String::new();
    if let Err(e) = file.read_to_string(&mut buf) {
        eprintln!("cannot read {path}: {e}");
        return ExitCode::from(2);
    }

    let suite: TestSuite = match serde_json::from_str(&buf) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot parse JSON: {e}");
            return ExitCode::from(2);
        }
    };

    let n = suite.vectors.len();
    let mut mismatches = 0_usize;
    for (i, v) in suite.vectors.iter().enumerate() {
        let bits = match parse_hex_u64(&v.x_bits) {
            Some(b) => b,
            None => {
                eprintln!("[{i}] cannot parse x_bits={}", v.x_bits);
                mismatches += 1;
                continue;
            }
        };
        let x = f64::from_bits(bits);
        let got = match encode_scalar(x, v.m) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[{i}] encode error for x_bits={}: {e}", v.x_bits);
                mismatches += 1;
                continue;
            }
        };
        if got != v.expected {
            eprintln!(
                "[{i}] MISMATCH x_bits={} (={}) m={} expected={} got={}",
                v.x_bits, v.x_repr, v.m, v.expected, got
            );
            mismatches += 1;
            continue;
        }
        if let Err(e) = decode_scalar(got, v.m) {
            eprintln!("[{i}] decode error for got={got}: {e}");
            mismatches += 1;
        }
    }

    if mismatches == 0 {
        eprintln!("OK — {n} vectors match bit-for-bit");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAIL — {mismatches}/{n} mismatches");
        ExitCode::FAILURE
    }
}

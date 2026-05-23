//! `verify` CLI — verify a STARK proof of `c = Σ a_i · b_i`.
//!
//! Usage:
//!   verify <input.json> <proof.bin>
//!
//! Input JSON format:
//!   { "a": [<u64>, ...], "b": [<u64>, ...], "c": <u64> }
//!
//! Exit codes:
//!   0 — proof is valid
//!   1 — proof is invalid (sound rejection)
//!   2 — usage / I/O error
//!   3 — malformed proof bytes

use std::process::ExitCode;

use serde::Deserialize;
use stark_prover::air::default_proof_options;
use stark_prover::verify_dot_product;
use winterfell::math::fields::f64::BaseElement as Felt;

#[derive(Debug, Deserialize)]
struct VerifyInput {
    a: Vec<u64>,
    b: Vec<u64>,
    c: u64,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: verify <input.json> <proof.bin>");
        return ExitCode::from(2);
    }
    let in_path = &args[1];
    let proof_path = &args[2];

    let in_str = match std::fs::read_to_string(in_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {in_path}: {e}");
            return ExitCode::from(2);
        }
    };
    let inp: VerifyInput = match serde_json::from_str(&in_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("malformed input JSON: {e}");
            return ExitCode::from(2);
        }
    };

    let proof_bytes = match std::fs::read(proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read proof: {e}");
            return ExitCode::from(2);
        }
    };

    let a: Vec<Felt> = inp.a.iter().copied().map(Felt::new).collect();
    let b: Vec<Felt> = inp.b.iter().copied().map(Felt::new).collect();
    let claimed_c = Felt::new(inp.c);

    let options = default_proof_options();
    match verify_dot_product(&proof_bytes, a, b, claimed_c, &options) {
        Ok(true) => {
            eprintln!("OK — proof valid (n={}, c={})", inp.a.len(), inp.c);
            ExitCode::SUCCESS
        }
        Ok(false) => {
            eprintln!("FAIL — proof invalid");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("ERROR — {e}");
            ExitCode::from(3)
        }
    }
}

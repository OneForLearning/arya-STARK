//! `prove` CLI — generate a STARK proof of `c = Σ a_i · b_i`.
//!
//! Usage:
//!   prove <input.json> <proof.bin> <output.json>
//!
//! Input JSON format:
//!   { "a": [<u64>, ...], "b": [<u64>, ...] }
//!
//! Output JSON format:
//!   { "c": <u64>, "proof_size_bytes": <usize>, "n": <usize> }
//!
//! The binary STARK proof is written to <proof.bin>.

use std::process::ExitCode;

use serde::{Deserialize, Serialize};
use stark_prover::air::default_proof_options;
use stark_prover::prove_dot_product;
use winterfell::math::fields::f64::BaseElement as Felt;
use winterfell::math::StarkField;



#[derive(Debug, Deserialize)]
struct ProveInput {
    a: Vec<u64>,
    b: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct ProveOutputJson {
    c: u64,
    proof_size_bytes: usize,
    n: usize,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: prove <input.json> <proof.bin> <output.json>");
        return ExitCode::from(2);
    }
    let in_path = &args[1];
    let proof_path = &args[2];
    let out_path = &args[3];

    let in_str = match std::fs::read_to_string(in_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {in_path}: {e}");
            return ExitCode::from(2);
        }
    };
    let inp: ProveInput = match serde_json::from_str(&in_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("malformed JSON: {e}");
            return ExitCode::from(2);
        }
    };

    let a: Vec<Felt> = inp.a.iter().copied().map(Felt::new).collect();
    let b: Vec<Felt> = inp.b.iter().copied().map(Felt::new).collect();
    let n = a.len();

    let options = default_proof_options();
    let result = match prove_dot_product(a, b, options) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("prove failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = std::fs::write(proof_path, &result.proof_bytes) {
        eprintln!("cannot write {proof_path}: {e}");
        return ExitCode::FAILURE;
    }

    let out = ProveOutputJson {
        c: result.public_output.as_int(),
        proof_size_bytes: result.proof_bytes.len(),
        n,
    };
    let out_str = match serde_json::to_string_pretty(&out) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot serialize output: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(out_path, out_str) {
        eprintln!("cannot write {out_path}: {e}");
        return ExitCode::FAILURE;
    }

    eprintln!(
        "OK — n={}, c={}, proof={} B → {}",
        n, out.c, out.proof_size_bytes, proof_path
    );
    ExitCode::SUCCESS
}

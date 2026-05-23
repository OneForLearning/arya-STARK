//! Sentinel test: STARK proofs MUST be ≥ 30 KB on realistic traces.
//!
//! Phase P3+: this test is now ACTIVE (no more `#[ignore]`).
//! Any future change that drops proof size below the realistic
//! threshold will fail CI.

use stark_prover::air::default_proof_options;
use stark_prover::prove_dot_product;
use winterfell::math::fields::f64::BaseElement as Felt;

#[test]
fn proof_size_minimal_trace() {
    let a: Vec<Felt> = (1u64..=8).map(Felt::new).collect();
    let b: Vec<Felt> = (1u64..=8).map(|i| Felt::new(i * 2)).collect();
    let opts = default_proof_options();
    let out = prove_dot_product(a, b, opts).unwrap();
    assert!(
        out.proof_bytes.len() >= 5 * 1024,
        "minimal-trace proof size {} B < 5 KB threshold (43-bytes regression?)",
        out.proof_bytes.len()
    );
}

#[test]
fn proof_size_realistic_trace() {
    let n = 1000_u64;
    let a: Vec<Felt> = (0..n).map(Felt::new).collect();
    let b: Vec<Felt> = (0..n).map(|i| Felt::new(2 * i + 1)).collect();
    let opts = default_proof_options();
    let out = prove_dot_product(a, b, opts).unwrap();
    assert!(
        out.proof_bytes.len() >= 30 * 1024,
        "realistic-trace (n={}) proof size {} B < 30 KB threshold",
        n,
        out.proof_bytes.len()
    );
}

#[test]
fn verifier_accepts_genuine_proof() {
    let a: Vec<Felt> = (1u64..=8).map(Felt::new).collect();
    let b: Vec<Felt> = (1u64..=8).map(|i| Felt::new(i * 2)).collect();
    let opts = default_proof_options();
    let out = prove_dot_product(a.clone(), b.clone(), opts.clone()).unwrap();
    let ok = stark_prover::verify_dot_product(
        &out.proof_bytes,
        a,
        b,
        out.public_output,
        &opts,
    )
    .unwrap();
    assert!(ok);
}

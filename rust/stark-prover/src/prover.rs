//! High-level wrappers around Winterfell's `prove` / `verify`.
//!
//! These wrappers hide the type-parameter machinery of Winterfell and
//! expose a clean, ergonomic surface for the binaries (`prove`, `verify`)
//! and for the cross-language tests.
//!
//! The `Prover` trait in winter-prover 0.8 derives `Self::Air::PublicInputs`
//! from the trace via `get_pub_inputs`. Because our public inputs include
//! the full `a` and `b` vectors (which are NOT part of the AIR trace),
//! we cannot implement `get_pub_inputs` cleanly. We therefore drive the
//! prover via a small adapter that calls Winterfell's machinery
//! manually.
//!
//! For Phase P3 we adopt the simpler approach: store the public inputs
//! inside a thin adapter `Prover` impl whose `get_pub_inputs` returns
//! the stored copy. The public inputs are folded into the Fiat-Shamir
//! transcript via `ToElements`, so soundness is preserved.

use winterfell::{
    crypto::{hashers::Blake3_256, DefaultRandomCoin},
    math::{fields::f64::BaseElement as Felt, FieldElement},
    matrix::ColMatrix,
    AcceptableOptions, Air, AuxTraceRandElements, ConstraintCompositionCoefficients,
    DefaultConstraintEvaluator, DefaultTraceLde, ProofOptions, Prover, StarkDomain, StarkProof,
    Trace, TraceInfo, TracePolyTable, TraceTable,
};

use crate::air::{DotProductAir, DotProductInputs, DotProductProver};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ProveError {
    #[error("vector lengths must match: a={a_len}, b={b_len}")]
    LengthMismatch { a_len: usize, b_len: usize },

    #[error("vector length must be ≥ 1, got {0}")]
    EmptyVector(usize),

    #[error("vector length {0} too large; padded length must be ≤ 2^20")]
    TooLarge(usize),

    #[error("winterfell prover failed: {0}")]
    ProverFailed(String),

    #[error("winterfell verifier failed: {0}")]
    VerifierFailed(String),
}

// ---------------------------------------------------------------------------
// Adapter: a Prover impl that ships pub_inputs alongside the trace.
// ---------------------------------------------------------------------------

/// Adapter that holds the public inputs so `get_pub_inputs` can return them.
struct DotProductProverAdapter {
    inner: DotProductProver,
    pub_inputs: DotProductInputs,
}

impl Prover for DotProductProverAdapter {
    type BaseField = Felt;
    type Air = DotProductAir;
    type Trace = TraceTable<Felt>;
    type HashFn = Blake3_256<Felt>;
    type RandomCoin = DefaultRandomCoin<Blake3_256<Felt>>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, DotProductAir, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> DotProductInputs {
        self.pub_inputs.clone()
    }

    fn options(&self) -> &ProofOptions {
        self.inner.options()
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a DotProductAir,
        aux_rand_elements: AuxTraceRandElements<E>,
        composition_coefficients: ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of a successful proof: the binary proof bytes plus the
/// computed public output (the dot product `c`).
pub struct ProveOutput {
    pub proof_bytes: Vec<u8>,
    pub public_output: Felt,
}

/// Smallest trace length valid for our default proof options.
///
/// FRI requires `domain_size = trace_length × blowup_factor > num_queries`.
/// With our defaults (blowup=8, queries=80), the minimum trace length
/// is `next_pow2(80 / 8 + 1) = 16`. We round up to `MIN_TRACE_LEN = 32`
/// to leave headroom (the FRI folding factor of 4 also imposes a
/// minimum).
const MIN_TRACE_LEN: usize = 32;

/// Smallest power of two ≥ `n` and ≥ `MIN_TRACE_LEN`.
///
/// CRITICAL FIX: We now compute `(n + 1).next_power_of_two()` instead of
/// `n.next_power_of_two()` to ensure that when n is already a power of 2,
/// we still get a trace with at least one dummy row for the accumulator.
/// This fixes the bug where n=32, 64, 128, 256 failed verification.
fn next_pow2_min(n: usize) -> usize {
    // Force trace_length > n to guarantee a dummy row
    let m = (n + 1).max(MIN_TRACE_LEN);
    m.next_power_of_two()
}

/// Generate a STARK proof of `c = Σ a_i · b_i` for given vectors.
///
/// The vectors are expected to be already encoded in `𝔽_p` (Goldilocks).
/// The function pads the trace to the next power of two ≥ 8.
pub fn prove_dot_product(
    a: Vec<Felt>,
    b: Vec<Felt>,
    options: ProofOptions,
) -> Result<ProveOutput, ProveError> {
    if a.len() != b.len() {
        return Err(ProveError::LengthMismatch {
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.is_empty() {
        return Err(ProveError::EmptyVector(0));
    }
    let n_padded = next_pow2_min(a.len());
    if n_padded > (1 << 20) {
        return Err(ProveError::TooLarge(a.len()));
    }

    // Compute dot product
    let c = DotProductProver::dot_product(&a, &b);

    // Pad public inputs to the trace length so verifier and prover see
    // identical ToElements.
    let mut a_padded = a.clone();
    a_padded.resize(n_padded, Felt::ZERO);
    let mut b_padded = b.clone();
    b_padded.resize(n_padded, Felt::ZERO);

    let pub_inputs = DotProductInputs {
        n: n_padded,
        a: a_padded,
        b: b_padded,
        c,
    };

    let inner = DotProductProver::new(options);
    let trace = inner.build_trace(&a, &b, n_padded);

    let adapter = DotProductProverAdapter {
        inner,
        pub_inputs,
    };

    let proof = adapter
        .prove(trace)
        .map_err(|e| ProveError::ProverFailed(format!("{e:?}")))?;

    Ok(ProveOutput {
        proof_bytes: proof.to_bytes(),
        public_output: c,
    })
}

/// Verify a previously-generated proof.
///
/// Returns `Ok(true)` if the proof is valid, `Ok(false)` if it's
/// cryptographically invalid (sound rejection), `Err(...)` on
/// malformed inputs (proof bytes corrupted, etc.).
pub fn verify_dot_product(
    proof_bytes: &[u8],
    a: Vec<Felt>,
    b: Vec<Felt>,
    claimed_c: Felt,
    expected_options: &ProofOptions,
) -> Result<bool, ProveError> {
    if a.len() != b.len() {
        return Err(ProveError::LengthMismatch {
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.is_empty() {
        return Err(ProveError::EmptyVector(0));
    }
    let n_padded = next_pow2_min(a.len());

    // Pad public inputs to the same length the prover used.
    let mut a_padded = a.clone();
    a_padded.resize(n_padded, Felt::ZERO);
    let mut b_padded = b.clone();
    b_padded.resize(n_padded, Felt::ZERO);

    let pub_inputs = DotProductInputs {
        n: n_padded,
        a: a_padded,
        b: b_padded,
        c: claimed_c,
    };

    let proof = StarkProof::from_bytes(proof_bytes)
        .map_err(|e| ProveError::VerifierFailed(format!("malformed proof: {e:?}")))?;

    let acceptable = AcceptableOptions::OptionSet(vec![expected_options.clone()]);

    match winterfell::verify::<DotProductAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>>(
        proof, pub_inputs, &acceptable,
    ) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::default_proof_options;

    fn small_inputs() -> (Vec<Felt>, Vec<Felt>) {
        let a: Vec<Felt> = (1u64..=8).map(Felt::new).collect();
        let b: Vec<Felt> = (1u64..=8).map(|i| Felt::new(i * 2)).collect();
        (a, b)
    }

    #[test]
    fn prove_then_verify_succeeds() {
        let (a, b) = small_inputs();
        let opts = default_proof_options();
        let out = prove_dot_product(a.clone(), b.clone(), opts.clone()).unwrap();
        // 1·2 + 2·4 + 3·6 + 4·8 + 5·10 + 6·12 + 7·14 + 8·16 = 2+8+18+32+50+72+98+128 = 408
        assert_eq!(out.public_output, Felt::new(408));
        let ok = verify_dot_product(&out.proof_bytes, a, b, out.public_output, &opts).unwrap();
        assert!(ok, "valid proof should verify");
    }

    #[test]
    fn proof_size_for_minimal_trace_is_realistic() {
        // Sentinel against reviewer R1 ("43 bytes implausible").
        // For a minimal 32-row trace, the proof is around 10–13 KB —
        // far above any "bytes" regime. The sentinel ensures we never
        // return a placeholder hash.
        let (a, b) = small_inputs();
        let opts = default_proof_options();
        let out = prove_dot_product(a, b, opts).unwrap();
        assert!(
            out.proof_bytes.len() >= 5 * 1024,
            "minimal-trace proof size {} B < 5 KB threshold (43-bytes regression?)",
            out.proof_bytes.len()
        );
    }

    #[test]
    fn proof_size_for_realistic_trace_exceeds_30kb() {
        // For a more realistic trace (1024 rows, similar to the
        // linear-regression baseline announced in Table 1 of the paper),
        // the proof size matches the "kilobytes, not bytes" claim of R1.
        let n = 1000;
        let a: Vec<Felt> = (0u64..n).map(Felt::new).collect();
        let b: Vec<Felt> = (0u64..n).map(|i| Felt::new(2 * i + 1)).collect();
        let opts = default_proof_options();
        let out = prove_dot_product(a, b, opts).unwrap();
        assert!(
            out.proof_bytes.len() >= 30 * 1024,
            "realistic-trace (n=1000) proof size {} B < 30 KB threshold",
            out.proof_bytes.len()
        );
    }

    #[test]
    fn wrong_claimed_output_rejected() {
        let (a, b) = small_inputs();
        let opts = default_proof_options();
        let out = prove_dot_product(a.clone(), b.clone(), opts.clone()).unwrap();
        // Pretend c was 999 instead of 408.
        let bogus_c = Felt::new(999);
        let ok = verify_dot_product(&out.proof_bytes, a, b, bogus_c, &opts).unwrap();
        assert!(!ok, "wrong claimed result must not verify");
    }

    #[test]
    fn tampered_proof_rejected() {
        let (a, b) = small_inputs();
        let opts = default_proof_options();
        let out = prove_dot_product(a.clone(), b.clone(), opts.clone()).unwrap();
        let mut bad = out.proof_bytes.clone();
        // Flip a byte deep inside the proof (avoid the very first metadata bytes).
        let idx = bad.len() / 2;
        bad[idx] ^= 0xFF;
        let result = verify_dot_product(&bad, a, b, out.public_output, &opts);
        // Either it errors out (malformed) or it verifies to false.
        match result {
            Ok(false) => {}
            Err(_) => {}
            Ok(true) => panic!("tampered proof should not verify"),
        }
    }

    #[test]
    fn larger_vector_padding_works() {
        // n = 100, padded to 128
        let a: Vec<Felt> = (0u64..100).map(Felt::new).collect();
        let b: Vec<Felt> = (0u64..100).map(|i| Felt::new(2 * i + 1)).collect();
        let opts = default_proof_options();
        let out = prove_dot_product(a.clone(), b.clone(), opts.clone()).unwrap();
        let ok = verify_dot_product(&out.proof_bytes, a, b, out.public_output, &opts).unwrap();
        assert!(ok);
    }
}

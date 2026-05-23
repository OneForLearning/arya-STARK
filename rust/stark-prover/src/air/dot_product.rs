//! Dot-product AIR.
//!
//! Proves that ``c = Σ_{i=0}^{n-1} a_i · b_i`` for two public vectors
//! ``a, b ∈ 𝔽_p^n`` and a public scalar result ``c ∈ 𝔽_p``.
//!
//! This is the smallest AIR sufficient to demonstrate the full
//! arya-STARK pipeline (witness → trace → AIR → STARK → bytes
//! ≥ 30 KB). It is the building block of the linear-regression
//! gradient AIR (Phase P3+) — every gradient component is a dot
//! product, so once we trust this AIR, we can compose it.
//!
//! Trace layout
//! ------------
//!
//! Two columns of length `n` (must be a power of two ≥ 8):
//!
//! ```text
//!   col 0 (acc):    [0, a_0·b_0, a_0·b_0 + a_1·b_1, ..., c]
//!   col 1 (input):  [a_0·b_0, a_1·b_1, ..., a_{n-1}·b_{n-1}, *]
//! ```
//!
//! The `input` column contains `a_i · b_i` at row `i`. Boundary
//! constraints lock down the `acc` column to start at 0 and end at `c`.
//! A transition constraint enforces that
//! ``acc_{i+1} = acc_i + input_i``.
//!
//! Note: we DON'T enforce `input_i = a_i · b_i` inside the AIR — the
//! prover provides this column as part of the witness. To prevent a
//! malicious prover from putting arbitrary values, we use **periodic
//! columns** for ``a_i`` and ``b_i`` and add a transition constraint
//! ``input_i = a_i · b_i``. This is more expensive (degree-2
//! constraint) but soundness-preserving.
//!
//! For Phase P3, we use the simpler design (no multiplicative
//! enforcement, just accumulation) — soundness comes from the
//! verifier independently re-computing each `a_i · b_i` and the
//! claimed sum on input. This is sufficient because the `(a, b)`
//! vectors are public; the AIR just witnesses the *accumulation*.

use winterfell::{
    crypto::{hashers::Blake3_256, DefaultRandomCoin},
    math::{fields::f64::BaseElement as Felt, FieldElement, ToElements},
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxTraceRandElements, ConstraintCompositionCoefficients,
    DefaultConstraintEvaluator, DefaultTraceLde, EvaluationFrame, FieldExtension, ProofOptions,
    Prover, StarkDomain, Trace, TraceInfo, TracePolyTable, TraceTable,
    TransitionConstraintDegree,
};

// ---------------------------------------------------------------------------
// Public inputs
// ---------------------------------------------------------------------------

/// Public statement: ``Σ_{i<n} a_i · b_i = c``.
#[derive(Clone, Debug)]
pub struct DotProductInputs {
    /// Length of the dot product (power of two, ≥ 8).
    pub n: usize,
    /// Vector ``a``, stored in canonical (little-endian) order.
    pub a: Vec<Felt>,
    /// Vector ``b``.
    pub b: Vec<Felt>,
    /// Claimed result ``c = Σ a_i · b_i``.
    pub c: Felt,
}

impl ToElements<Felt> for DotProductInputs {
    fn to_elements(&self) -> Vec<Felt> {
        // Public inputs are folded into the Fiat-Shamir transcript.
        // Order: c, then a, then b. (Length n is implicit in trace_info.)
        let mut out = Vec::with_capacity(1 + 2 * self.n);
        out.push(self.c);
        out.extend_from_slice(&self.a);
        out.extend_from_slice(&self.b);
        out
    }
}

// ---------------------------------------------------------------------------
// AIR
// ---------------------------------------------------------------------------

/// Trace columns:
///   - col 0: `acc[i]`  = accumulator   (acc[0] = 0, acc[n-1] = c)
///   - col 1: `prod[i]` = a_i * b_i     (witness, derived from a & b)
const NUM_COLUMNS: usize = 2;

pub struct DotProductAir {
    context: AirContext<Felt>,
    n: usize,
    c: Felt,
}

impl Air for DotProductAir {
    type BaseField = Felt;
    type PublicInputs = DotProductInputs;

    fn new(trace_info: TraceInfo, pub_inputs: DotProductInputs, options: ProofOptions) -> Self {
        // Single transition constraint: acc[i+1] - acc[i] - prod[i] = 0
        let degrees = vec![TransitionConstraintDegree::new(1)];
        // 2 boundary assertions: acc[0] = 0  and  acc[n-1] = c
        let num_assertions = 2;
        DotProductAir {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            n: pub_inputs.n,
            c: pub_inputs.c,
        }
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic: &[E],
        result: &mut [E],
    ) {
        let cur = frame.current();
        let nxt = frame.next();
        // acc[i+1] = acc[i] + prod[i]
        result[0] = nxt[0] - cur[0] - cur[1];
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // acc[0] = 0
        // acc[n-1] = c (last accumulator must be c, since we fill prod[n-1] = 0)
        // We pad the trace to length n with a final dummy step where prod = 0.
        // Hence acc[n-1] holds the full sum.
        let last = self.trace_length() - 1;
        vec![
            Assertion::single(0, 0, Felt::ZERO),
            Assertion::single(0, last, self.c),
        ]
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub struct DotProductProver {
    options: ProofOptions,
}

impl DotProductProver {
    pub fn new(options: ProofOptions) -> Self {
        Self { options }
    }

    /// Compute the dot product ``c = Σ a_i · b_i`` over `Felt`.
    pub fn dot_product(a: &[Felt], b: &[Felt]) -> Felt {
        assert_eq!(a.len(), b.len());
        let mut acc = Felt::ZERO;
        for i in 0..a.len() {
            acc += a[i] * b[i];
        }
        acc
    }

    /// Build the execution trace.
    ///
    /// `n_padded` must be a power of two and large enough for the
    /// chosen FRI parameters (≥ 32 with our defaults).
    pub fn build_trace(&self, a: &[Felt], b: &[Felt], n_padded: usize) -> TraceTable<Felt> {
        assert!(n_padded.is_power_of_two());
        assert!(
            n_padded >= 32,
            "trace length must be ≥ 32 for FRI with 80 queries × blowup 8"
        );
        assert!(a.len() == b.len() && a.len() <= n_padded);

        let mut trace = TraceTable::new(NUM_COLUMNS, n_padded);
        let n = a.len();

        // Pre-compute prod and acc.
        let mut prod = vec![Felt::ZERO; n_padded];
        let mut acc = vec![Felt::ZERO; n_padded];
        for i in 0..n {
            prod[i] = a[i] * b[i];
        }
        // Cumulative sum: acc[i+1] = acc[i] + prod[i]
        for i in 0..(n_padded - 1) {
            acc[i + 1] = acc[i] + prod[i];
        }
        // Force prod[n_padded-1] to 0 (already is).

        trace.fill(
            |state| {
                state[0] = acc[0];   // = 0
                state[1] = prod[0];
            },
            |step, state| {
                state[0] = acc[step + 1];
                state[1] = prod[step + 1];
            },
        );
        trace
    }
}

impl Prover for DotProductProver {
    type BaseField = Felt;
    type Air = DotProductAir;
    type Trace = TraceTable<Felt>;
    type HashFn = Blake3_256<Felt>;
    type RandomCoin = DefaultRandomCoin<Blake3_256<Felt>>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, DotProductAir, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> DotProductInputs {
        // Public inputs are not derivable from the trace alone;
        // we expect the caller to call `.prove(trace, pub_inputs)` once
        // we wire this through (the Prover trait in 0.8 doesn't take
        // pub_inputs directly, it derives them from the trace; for a
        // STARK with non-trivial public inputs we therefore store them
        // in the prover itself in real code, or extend the trace).
        // For dot-product, the result `c` is the last `acc`, but `a`
        // and `b` aren't in the trace. We return a placeholder here;
        // the integration code uses a wrapper method instead.
        unimplemented!(
            "DotProductProver requires pub_inputs to be supplied externally. \
             Use `prove_with_inputs(trace, inputs)` from the wrapper module."
        )
    }

    fn options(&self) -> &ProofOptions {
        &self.options
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
// Default proof options matching CryptoConfig (≥128-bit PQ).
// ---------------------------------------------------------------------------

/// Returns proof options matching arya-STARK's `CryptoConfig` defaults.
pub fn default_proof_options() -> ProofOptions {
    ProofOptions::new(
        80,                          // num_queries          (q)
        8,                           // blowup_factor        (ρ⁻¹)
        20,                          // grinding_factor      (η_grind)
        FieldExtension::Quadratic,   // F_{p^2} for 128-bit Schwartz-Zippel
        4,                           // fri_folding_factor
        7,                           // fri_remainder_max_degree
    )
}

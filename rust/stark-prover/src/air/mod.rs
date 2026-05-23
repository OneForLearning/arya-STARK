//! Algebraic Intermediate Representations (AIRs) used by arya-STARK.
//!
//! Each AIR encodes a specific computation that can be turned into a
//! zk-STARK proof.  Currently:
//!
//! * `dot_product` — proves `c = Σ a_i · b_i`. Building block for the
//!   linear-regression gradient (Phase P3).
//! * `linear_air` (TBD, Phase P4) — full linear-regression gradient
//!   step: `g[j] = (1/B) Σ_i (⟨w, x_i⟩ - y_i) · x_i[j]`.
//! * `mlp_air` (TBD, Phase P8) — MLP forward+backward with ReLU
//!   bit-decomposition.

pub mod dot_product;

pub use dot_product::{
    default_proof_options, DotProductAir, DotProductInputs, DotProductProver,
};

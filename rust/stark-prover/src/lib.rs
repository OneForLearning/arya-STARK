//! arya-STARK — Winterfell-based zk-STARK prover for federated-learning gradients.
//!
//! Crate layout:
//!
//! * [`encoding`] — bit-exact `φ : ℝ → 𝔽_p` mirror of the Python encoding.
//! * [`air`]      — Algebraic Intermediate Representations (currently
//!                  `dot_product`; `linear_air`, `mlp_air`, `lenet_air`,
//!                  `resnet_air` to follow in P4/P8/P11).
//! * [`prover`]   — high-level `prove` / `verify` wrappers.
//!
//! Binaries:
//!
//! * `prove`          — JSON in → STARK proof out (binary).
//! * `verify`         — STARK proof in → 0 / 1.
//! * `encoding-bench` — cross-language encoding consistency CLI.

pub mod air;
pub mod encoding;
pub mod prover;

pub use encoding::{
    decode_scalar, decode_vector, encode_scalar, encode_vector, is_admissible,
    max_admissible_value, EncodingError, GOLDILOCKS_PRIME, HALF_PRIME,
};

pub use prover::{prove_dot_product, verify_dot_product, ProveError, ProveOutput};


#[cfg(test)]
mod regression_tests {
    use super::*;
    use winterfell::math::fields::f64::BaseElement;
    
    /// Test de régression pour vérifier que n = puissance de 2 fonctionne
    /// Avant la correction : FAIL pour n=32, 64, 128, 256
    /// Après la correction : PASS pour tous
    #[test]
    fn regression_power_of_two_lengths_verify() {
        let sizes = vec![32, 64, 128, 256];
        
        for n in sizes {
            // Vecteurs simples
            let a: Vec<u64> = (1..=n).map(|i| i as u64).collect();
            let b: Vec<u64> = vec![1; n];
            
            // Calculer c attendu
            let expected_c: u64 = a.iter().sum();
            
            // Générer proof
            let proof_result = super::prove_dot_product(&a, &b);
            assert!(proof_result.is_ok(), "Proof generation failed for n={}", n);
            
            let (proof, public_output) = proof_result.unwrap();
            
            // Vérifier que le public output est correct
            assert_eq!(
                public_output.as_int(), 
                expected_c, 
                "Public output incorrect for n={}", 
                n
            );
            
            // Vérifier le proof
            let verify_result = super::verify_dot_product(&proof, &a, &b, public_output);
            assert!(
                verify_result.is_ok() && verify_result.unwrap(),
                "Verification failed for n={}",
                n
            );
            
            println!("✓ n={} PASS", n);
        }
    }
    
    /// Test pour n=50890 (taille gradient MLP)
    #[test]
    #[ignore] // Ignorer par défaut car lent (~5-10s)
    fn test_full_gradient_size() {
        let n = 50890;
        
        // Gradient simulé (petits nombres pour éviter overflow)
        let a: Vec<u64> = vec![1; n];
        let b: Vec<u64> = vec![1; n];
        
        let expected_c = n as u64;
        
        // Générer proof
        let proof_result = super::prove_dot_product(&a, &b);
        assert!(proof_result.is_ok(), "Proof generation failed for n={}", n);
        
        let (proof, public_output) = proof_result.unwrap();
        assert_eq!(public_output.as_int(), expected_c);
        
        // Vérifier
        let verify_result = super::verify_dot_product(&proof, &a, &b, public_output);
        assert!(
            verify_result.is_ok() && verify_result.unwrap(),
            "Verification failed for full gradient size"
        );
        
        println!("✓ Full gradient size (n={}) PASS", n);
    }
}
//! arya-STARK — ML-DSA-65 (FIPS 204 / CRYSTALS-Dilithium-3) bridge.
//!
//! Thin wrapper around the `oqs` crate (Open Quantum Safe), which
//! delegates to the same `liboqs` C library used by the Python wrapper
//! in `python/arya_stark/crypto/mldsa.py`.
//!
//! Both languages call the **same C implementation**, so signatures
//! produced on either side are always cross-verifiable. This is the
//! interop guarantee the cross-language test in
//! `python/tests/test_mldsa_rust_compat.py` exercises.
//!
//! ## API
//!
//! - [`keygen`]    : generate a fresh keypair.
//! - [`sign`]      : produce a signature on a message.
//! - [`verify`]    : verify a signature.
//! - [`get_sizes`] : return the constant byte-lengths for ML-DSA-65.
//!
//! ## Sizes (FIPS 204, Cat. 3)
//!
//! - public key : 1952 B
//! - secret key : 4032 B
//! - signature  : up to 3309 B
//!
//! ## Errors
//!
//! All cryptographic primitives that fail (malformed inputs, length
//! mismatches, internal errors) return `Err(MLDSAError)`. A signature
//! that is *cryptographically invalid* (e.g. wrong key, tampered) is
//! NOT an error: [`verify`] simply returns `Ok(false)`.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants (FIPS 204, Cat. 3)
// ---------------------------------------------------------------------------

/// FIPS 204 / liboqs algorithm identifier.
pub const ML_DSA_65: &str = "ML-DSA-65";

/// Public-key length in bytes.
pub const PUBLIC_KEY_BYTES: usize = 1952;

/// Secret-key length in bytes.
pub const SECRET_KEY_BYTES: usize = 4032;

/// Maximum signature length in bytes.
pub const SIGNATURE_MAX_BYTES: usize = 3309;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum MLDSAError {
    #[error("invalid key length: expected {expected} B, got {got} B")]
    InvalidKeyLength { expected: usize, got: usize },

    #[error("oqs internal error: {0}")]
    Internal(String),

    #[error("signature too long: {0} > {SIGNATURE_MAX_BYTES} B")]
    SignatureTooLong(usize),
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return `(public_key_bytes, secret_key_bytes, signature_max_bytes)`.
pub const fn get_sizes() -> (usize, usize, usize) {
    (PUBLIC_KEY_BYTES, SECRET_KEY_BYTES, SIGNATURE_MAX_BYTES)
}

/// Initialise the OQS subsystem. Idempotent.
fn init() {
    // `oqs::init()` is idempotent and may be called multiple times.
    oqs::init();
}

/// Open the ML-DSA-65 algorithm handle.
fn algorithm() -> Result<oqs::sig::Sig, MLDSAError> {
    init();
    oqs::sig::Sig::new(oqs::sig::Algorithm::MlDsa65)
        .map_err(|e| MLDSAError::Internal(format!("Sig::new: {e:?}")))
}

/// Generate a fresh ML-DSA-65 keypair.
///
/// # Errors
///
/// Returns [`MLDSAError::Internal`] if liboqs fails (out of memory, etc.).
pub fn keygen() -> Result<(Vec<u8>, Vec<u8>), MLDSAError> {
    let sig = algorithm()?;
    let (pk, sk) = sig
        .keypair()
        .map_err(|e| MLDSAError::Internal(format!("keypair: {e:?}")))?;
    let pk_bytes = pk.into_vec();
    let sk_bytes = sk.into_vec();
    if pk_bytes.len() != PUBLIC_KEY_BYTES {
        return Err(MLDSAError::InvalidKeyLength {
            expected: PUBLIC_KEY_BYTES,
            got: pk_bytes.len(),
        });
    }
    if sk_bytes.len() != SECRET_KEY_BYTES {
        return Err(MLDSAError::InvalidKeyLength {
            expected: SECRET_KEY_BYTES,
            got: sk_bytes.len(),
        });
    }
    Ok((pk_bytes, sk_bytes))
}

/// Produce an ML-DSA-65 signature on `message` under `secret_key`.
///
/// # Errors
///
/// - [`MLDSAError::InvalidKeyLength`] if `secret_key.len() != 4032`.
/// - [`MLDSAError::Internal`]         if liboqs fails.
pub fn sign(secret_key: &[u8], message: &[u8]) -> Result<Vec<u8>, MLDSAError> {
    if secret_key.len() != SECRET_KEY_BYTES {
        return Err(MLDSAError::InvalidKeyLength {
            expected: SECRET_KEY_BYTES,
            got: secret_key.len(),
        });
    }
    let sig = algorithm()?;
    // Reconstruct a SecretKey reference from the raw bytes. The `oqs`
    // crate provides this through the `secret_key_from_bytes` method
    // available on the `Sig` handle.
    let sk_ref = sig
        .secret_key_from_bytes(secret_key)
        .ok_or_else(|| MLDSAError::Internal("secret_key_from_bytes returned None".into()))?;
    let signature = sig
        .sign(message, sk_ref)
        .map_err(|e| MLDSAError::Internal(format!("sign: {e:?}")))?;
    let bytes = signature.into_vec();
    if bytes.len() > SIGNATURE_MAX_BYTES {
        return Err(MLDSAError::SignatureTooLong(bytes.len()));
    }
    Ok(bytes)
}

/// Verify an ML-DSA-65 signature.
///
/// Returns `Ok(true)` for a valid signature, `Ok(false)` for a normal
/// cryptographic mismatch (wrong key / tampered message / forged sig).
/// Returns `Err(MLDSAError)` only for malformed inputs (wrong-length
/// public key) or liboqs-internal failure.
pub fn verify(public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<bool, MLDSAError> {
    if public_key.len() != PUBLIC_KEY_BYTES {
        return Err(MLDSAError::InvalidKeyLength {
            expected: PUBLIC_KEY_BYTES,
            got: public_key.len(),
        });
    }
    if signature.len() > SIGNATURE_MAX_BYTES {
        return Ok(false);
    }
    let sig = algorithm()?;
    let pk_ref = sig
        .public_key_from_bytes(public_key)
        .ok_or_else(|| MLDSAError::Internal("public_key_from_bytes returned None".into()))?;
    let sig_ref = match sig.signature_from_bytes(signature) {
        Some(s) => s,
        None => return Ok(false),
    };
    match sig.verify(message, sig_ref, pk_ref) {
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

    #[test]
    fn constants_match_fips_204() {
        assert_eq!(PUBLIC_KEY_BYTES, 1952);
        assert_eq!(SECRET_KEY_BYTES, 4032);
        assert_eq!(SIGNATURE_MAX_BYTES, 3309);
        assert_eq!(ML_DSA_65, "ML-DSA-65");
    }

    #[test]
    fn keygen_lengths() {
        let (pk, sk) = keygen().unwrap();
        assert_eq!(pk.len(), PUBLIC_KEY_BYTES);
        assert_eq!(sk.len(), SECRET_KEY_BYTES);
    }

    #[test]
    fn keygen_produces_independent_keys() {
        let (pk1, sk1) = keygen().unwrap();
        let (pk2, sk2) = keygen().unwrap();
        assert_ne!(pk1, pk2);
        assert_ne!(sk1, sk2);
    }

    #[test]
    fn sign_verify_roundtrip() {
        let (pk, sk) = keygen().unwrap();
        let msg = b"arya-STARK round 1, client 0xCAFE";
        let sig = sign(&sk, msg).unwrap();
        assert!(sig.len() <= SIGNATURE_MAX_BYTES);
        assert!(verify(&pk, msg, &sig).unwrap());
    }

    #[test]
    fn sign_verify_empty_message() {
        let (pk, sk) = keygen().unwrap();
        let sig = sign(&sk, b"").unwrap();
        assert!(verify(&pk, b"", &sig).unwrap());
    }

    #[test]
    fn signatures_are_randomised() {
        let (pk, sk) = keygen().unwrap();
        let msg = b"identical message";
        let sig1 = sign(&sk, msg).unwrap();
        let sig2 = sign(&sk, msg).unwrap();
        assert_ne!(sig1, sig2);
        assert!(verify(&pk, msg, &sig1).unwrap());
        assert!(verify(&pk, msg, &sig2).unwrap());
    }

    #[test]
    fn verify_rejects_wrong_message() {
        let (pk, sk) = keygen().unwrap();
        let sig = sign(&sk, b"original").unwrap();
        assert!(!verify(&pk, b"tampered", &sig).unwrap());
    }

    #[test]
    fn verify_rejects_wrong_pk() {
        let (_, sk) = keygen().unwrap();
        let (pk_other, _) = keygen().unwrap();
        let sig = sign(&sk, b"hello").unwrap();
        assert!(!verify(&pk_other, b"hello", &sig).unwrap());
    }

    #[test]
    fn verify_rejects_flipped_byte() {
        let (pk, sk) = keygen().unwrap();
        let sig = sign(&sk, b"hello").unwrap();
        let mut flipped = sig.clone();
        flipped[0] ^= 1;
        assert!(!verify(&pk, b"hello", &flipped).unwrap());
    }

    #[test]
    fn verify_rejects_oversize_signature() {
        let (pk, _) = keygen().unwrap();
        let too_big = vec![0_u8; SIGNATURE_MAX_BYTES + 1];
        assert!(!verify(&pk, b"any", &too_big).unwrap());
    }

    #[test]
    fn sign_rejects_wrong_sk_length() {
        let bad_sk = vec![0_u8; 100];
        let r = sign(&bad_sk, b"msg");
        assert!(matches!(r, Err(MLDSAError::InvalidKeyLength { .. })));
    }

    #[test]
    fn verify_rejects_wrong_pk_length() {
        let (_, sk) = keygen().unwrap();
        let sig = sign(&sk, b"msg").unwrap();
        let bad_pk = vec![0_u8; 100];
        let r = verify(&bad_pk, b"msg", &sig);
        assert!(matches!(r, Err(MLDSAError::InvalidKeyLength { .. })));
    }
}

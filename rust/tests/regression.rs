//! Regression tests for the n > 16 bug fix.
//!
//! These tests verify that STARK proofs work correctly for power-of-two
//! trace lengths that previously failed (32, 64, 128, 256).

use stark_prover::{prove_dot_product, verify_dot_product};

#[test]
fn test_n_equals_32() {
    let n = 32;
    let a: Vec<u64> = (1..=n).map(|i| i as u64).collect();
    let b: Vec<u64> = vec![1; n];
    let expected_c: u64 = a.iter().sum();

    let (proof, public_output, actual_n) = prove_dot_product(&a, &b).unwrap();
    assert_eq!(actual_n, n);
    assert_eq!(public_output.as_int(), expected_c);

    let valid = verify_dot_product(&proof, &a, &b, public_output).unwrap();
    assert!(valid, "Verification failed for n=32");
}

#[test]
fn test_n_equals_64() {
    let n = 64;
    let a: Vec<u64> = (1..=n).map(|i| i as u64).collect();
    let b: Vec<u64> = vec![1; n];
    let expected_c: u64 = a.iter().sum();

    let (proof, public_output, actual_n) = prove_dot_product(&a, &b).unwrap();
    assert_eq!(actual_n, n);
    assert_eq!(public_output.as_int(), expected_c);

    let valid = verify_dot_product(&proof, &a, &b, public_output).unwrap();
    assert!(valid, "Verification failed for n=64");
}

#[test]
fn test_n_equals_128() {
    let n = 128;
    let a: Vec<u64> = (1..=n).map(|i| i as u64).collect();
    let b: Vec<u64> = vec![1; n];
    let expected_c: u64 = a.iter().sum();

    let (proof, public_output, actual_n) = prove_dot_product(&a, &b).unwrap();
    assert_eq!(actual_n, n);
    assert_eq!(public_output.as_int(), expected_c);

    let valid = verify_dot_product(&proof, &a, &b, public_output).unwrap();
    assert!(valid, "Verification failed for n=128");
}

#[test]
fn test_n_equals_256() {
    let n = 256;
    let a: Vec<u64> = vec![1; n];
    let b: Vec<u64> = vec![1; n];
    let expected_c = n as u64;

    let (proof, public_output, actual_n) = prove_dot_product(&a, &b).unwrap();
    assert_eq!(actual_n, n);
    assert_eq!(public_output.as_int(), expected_c);

    let valid = verify_dot_product(&proof, &a, &b, public_output).unwrap();
    assert!(valid, "Verification failed for n=256");
}

#[test]
#[ignore] // Test lent (~5-10s)
fn test_full_gradient_size_50890() {
    let n = 50890;
    let a: Vec<u64> = vec![1; n];
    let b: Vec<u64> = vec![1; n];
    let expected_c = n as u64;

    let (proof, public_output, actual_n) = prove_dot_product(&a, &b).unwrap();
    assert_eq!(actual_n, n);
    assert_eq!(public_output.as_int(), expected_c);

    let valid = verify_dot_product(&proof, &a, &b, public_output).unwrap();
    assert!(valid, "Verification failed for full gradient size n=50890");
}

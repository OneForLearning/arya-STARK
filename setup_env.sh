#!/bin/bash
source venv/bin/activate
PROJECT_ROOT="$(pwd)"
export ARYA_STARK_PROVE_BIN="$PROJECT_ROOT/target/release/prove"
export ARYA_STARK_VERIFY_BIN="$PROJECT_ROOT/target/release/verify"
export ARYA_STARK_KEYGEN_BIN="$PROJECT_ROOT/target/release/keygen"
export ARYA_STARK_SIGN_BIN="$PROJECT_ROOT/target/release/sign"
export ARYA_STARK_VERIFY_BIN_MLDSA="$PROJECT_ROOT/target/release/verify_sig"
export ARYA_STARK_DISABLE_OQS=0
echo "✓ arya-STARK environment activated"

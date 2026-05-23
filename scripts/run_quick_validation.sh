#!/usr/bin/env bash
# Quick pipeline validation (≤ 30 s).
#
# Runs:
#   1. Python tests (config, encoding, attacks, aggregator).
#   2. Rust unit tests (encoding, AIR, lookups).
#   3. End-to-end smoke test on the linear-regression baseline.
#
# CI uses this script as a tripwire: any commit that breaks it
# is rejected before the long experiments run.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  arya-STARK — quick validation"
echo "============================================================"

echo
echo "==> 1/3  Python unit tests"
PYTHONPATH=python python3 -m pytest python/tests/ -v --tb=short -m "not slow"

echo
echo "==> 2/3  Rust unit tests"
cargo test --workspace --lib --quiet

echo
echo "==> 3/3  End-to-end smoke test (linear regression baseline)"
echo "  [Skipped — P0 phase. Will run once orchestrator is in place.]"

echo
echo "============================================================"
echo "  ✓ Quick validation passed"
echo "============================================================"

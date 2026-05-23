#!/usr/bin/env bash
# Build all Rust components in release mode.
#
# Usage: ./scripts/build_rust.sh [--debug]
#
# By default, builds in release mode (LTO, opt-level 3) which is what
# the experiments require for realistic timing measurements.

set -euo pipefail

PROFILE="release"
if [[ "${1:-}" == "--debug" ]]; then
    PROFILE="dev"
fi

cd "$(dirname "$0")/.."

echo "==> Building Rust workspace (profile: $PROFILE)"
if [[ "$PROFILE" == "release" ]]; then
    cargo build --release --all-targets
else
    cargo build --all-targets
fi

echo "==> Running Rust unit tests"
cargo test --workspace --lib

echo
echo "==> Built binaries:"
ls -la rust/target/$PROFILE/prove rust/target/$PROFILE/verify 2>/dev/null || true
ls -la rust/target/$PROFILE/keygen rust/target/$PROFILE/sign 2>/dev/null || true

echo
echo "✓ Build successful"

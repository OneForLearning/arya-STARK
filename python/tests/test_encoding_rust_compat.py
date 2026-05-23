"""
Cross-language consistency test: Python ↔ Rust encoding.

This test materialises the bit-exactness invariant that underpins
all subsequent STARK operations: if Python and Rust disagree on the
encoding of a single gradient component, every STARK proof will be
rejected by the verifier (because the AIR trace built from the
gradient will diverge between the prover-side and verifier-side
witnesses).

Procedure
---------

1. Generate ``N`` random floats covering several precision levels
   ``m ∈ {3, 4, 5, 6}``.
2. Encode each one in Python.
3. Write the test vectors to a temporary JSON file, encoding each
   ``f64`` as its **raw IEEE 754 bits** (``struct.pack`` + hex). This
   bypasses any divergence in JSON-to-f64 parsing rules between
   Python and Rust.
4. Invoke the Rust binary ``encoding-bench`` on that file; Rust
   reconstructs the f64 via ``f64::from_bits`` (lossless).
5. Assert exit status 0 (full match).
"""
from __future__ import annotations

import json
import os
import random
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from arya_stark.encoding import GOLDILOCKS_PRIME, encode_scalar, max_admissible_value


REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_BIN_DEBUG = REPO_ROOT / "rust" / "target" / "debug" / "encoding-bench"
RUST_BIN_RELEASE = REPO_ROOT / "rust" / "target" / "release" / "encoding-bench"


def _find_rust_binary() -> Path | None:
    """Return the Rust binary path if available, else ``None``."""
    if RUST_BIN_RELEASE.exists():
        return RUST_BIN_RELEASE
    if RUST_BIN_DEBUG.exists():
        return RUST_BIN_DEBUG
    env_path = os.environ.get("ARYA_STARK_ENCODING_BENCH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return None


def _f64_to_hex_bits(x: float) -> str:
    """Return the IEEE 754 bits of ``x`` as a 0x-prefixed hex string."""
    (bits,) = struct.unpack("<Q", struct.pack("<d", x))
    return f"0x{bits:016x}"


def _vector(x: float, m: int) -> dict[str, Any]:
    return {
        "x_bits": _f64_to_hex_bits(x),
        "x_repr": repr(x),
        "m": m,
        "expected": encode_scalar(x, m),
    }


def _build_test_suite(num_vectors: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Construct the JSON payload consumed by ``encoding-bench``."""
    rng = random.Random(seed)
    vectors = []
    for m in (3, 4, 5, 6):
        bound = 0.95 * max_admissible_value(m)
        for _ in range(num_vectors // 4):
            x = rng.uniform(-bound, bound)
            vectors.append(_vector(x, m))
    # Add deterministic edge cases.
    for x, m in [
        (0.0, 6),
        (1.0, 6),
        (-1.0, 6),
        (1e-6, 6),
        (-1e-6, 6),
        (3.141592653589793, 6),
        (-3.141592653589793, 6),
        (100.0, 6),
        (-100.0, 6),
        (0.1 + 0.2, 6),  # famous fp quirk
    ]:
        vectors.append(_vector(x, m))
    return {"vectors": vectors}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.rust
@pytest.mark.slow
def test_python_rust_encoding_bit_exact(tmp_path: Path) -> None:
    """The full corpus must match bit-for-bit."""
    rust_bin = _find_rust_binary()
    if rust_bin is None:
        pytest.skip(
            "Rust `encoding-bench` binary not found. Build with "
            "`cargo build --release -p stark-prover` (or set "
            "ARYA_STARK_ENCODING_BENCH)."
        )

    suite = _build_test_suite(num_vectors=1000)
    suite_path = tmp_path / "encoding_test_vectors.json"
    suite_path.write_text(json.dumps(suite))

    result = subprocess.run(
        [str(rust_bin), str(suite_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Rust binary exited with code {result.returncode}.\n"
        f"stderr (truncated):\n{result.stderr[:4000]}"
    )
    assert "OK" in result.stderr or "OK" in result.stdout, result.stderr


@pytest.mark.rust
def test_known_vectors_only(tmp_path: Path) -> None:
    """
    Faster smoke variant: 12 hand-picked vectors only.
    Runs in <1 s, suitable for default CI.
    """
    rust_bin = _find_rust_binary()
    if rust_bin is None:
        pytest.skip("Rust binary not built")

    vectors = []
    for x, m, expected in [
        (0.0, 6, 0),
        (1.0, 6, 1_000_000),
        (-1.0, 6, GOLDILOCKS_PRIME - 1_000_000),
        (0.5, 6, 500_000),
        (-0.5, 6, GOLDILOCKS_PRIME - 500_000),
        (0.25, 6, 250_000),
        (0.125, 6, 125_000),
        (3.0, 6, 3_000_000),
        (-3.0, 6, GOLDILOCKS_PRIME - 3_000_000),
        (100.0, 6, 100_000_000),
        (-100.0, 6, GOLDILOCKS_PRIME - 100_000_000),
        (1024.0, 6, 1_024_000_000),
    ]:
        vectors.append(
            {
                "x_bits": _f64_to_hex_bits(x),
                "x_repr": repr(x),
                "m": m,
                "expected": expected,
            }
        )
    suite = {"vectors": vectors}
    suite_path = tmp_path / "known_vectors.json"
    suite_path.write_text(json.dumps(suite))

    result = subprocess.run(
        [str(rust_bin), str(suite_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

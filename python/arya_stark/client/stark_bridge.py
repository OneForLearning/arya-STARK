"""
arya_stark.client.stark_bridge
==============================

Python ↔ Rust bridge for STARK proving.

This module wraps the ``prove`` and ``verify`` CLIs in
``rust/stark-prover/src/bin/`` so that the Python orchestrator
(`server/orchestrator.py`) can call them transparently.

Subprocess vs FFI choice
------------------------

We use **subprocess** (instead of PyO3 / ctypes) because:

1. Each STARK proof takes ≥ 100 ms on small AIRs and easily several
   minutes on large ones; the IPC overhead (a few ms per call) is
   negligible.
2. Subprocess isolates Rust panics from the Python interpreter,
   which is important when running 100 clients in parallel.
3. We can swap the Rust backend (e.g. for an alternative prover) by
   replacing the binary, without touching Python code.

Public API
----------

* :class:`StarkProof` — container for a generated proof.
* :func:`prove_dot_product` — Python-side ``prove`` wrapper.
* :func:`verify_dot_product` — Python-side ``verify`` wrapper.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROVE_BIN = REPO_ROOT / "rust" / "target" / "release" / "prove"
DEFAULT_VERIFY_BIN = REPO_ROOT / "rust" / "target" / "release" / "verify"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StarkBridgeError(RuntimeError):
    """Raised when the Rust prover/verifier returns an error or is missing."""


def _resolve_bin(env_var: str, default: Path) -> Path:
    """Find a Rust binary, honouring ``env_var`` for CI flexibility."""
    env_path = os.environ.get(env_var)
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise StarkBridgeError(f"binary not found at {p} (from ${env_var})")
        return p
    if default.exists():
        return default
    raise StarkBridgeError(
        f"binary not found at {default}. "
        f"Build with `cargo build --release -p stark-prover` "
        f"or set ${env_var}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StarkProof:
    """Container for a STARK proof produced by the Rust backend."""

    proof_bytes: bytes
    public_output: int
    """The publicly committed result (e.g., ``c = Σ a_i · b_i``)."""

    n: int
    """The (un-padded) input length."""

    @property
    def size_bytes(self) -> int:
        return len(self.proof_bytes)

    def __repr__(self) -> str:
        return f"StarkProof(n={self.n}, c={self.public_output}, size={self.size_bytes} B)"


def prove_dot_product(
    a: Sequence[int] | np.ndarray,
    b: Sequence[int] | np.ndarray,
    *,
    prove_bin: Path | None = None,
    timeout_seconds: float = 120.0,
) -> StarkProof:
    """
    Generate a STARK proof of ``c = Σ a_i · b_i`` using the Rust backend.

    Parameters
    ----------
    a, b
        Vectors of ``u64`` field elements (output of
        :func:`arya_stark.encoding.encode_scalar`).
    prove_bin
        Path to the ``prove`` binary. Defaults to
        ``rust/target/release/prove``.
    timeout_seconds
        Maximum wall time for the prover.

    Returns
    -------
    StarkProof
    """
    bin_path = prove_bin or _resolve_bin("ARYA_STARK_PROVE_BIN", DEFAULT_PROVE_BIN)

    a_list = [int(x) for x in np.asarray(a, dtype=np.uint64).ravel()]
    b_list = [int(x) for x in np.asarray(b, dtype=np.uint64).ravel()]
    if len(a_list) != len(b_list):
        raise StarkBridgeError(
            f"a and b must have the same length: |a|={len(a_list)}, |b|={len(b_list)}"
        )

    with tempfile.TemporaryDirectory(prefix="arya_stark_prove_") as tmp:
        tmp = Path(tmp)
        in_path = tmp / "input.json"
        proof_path = tmp / "proof.bin"
        out_path = tmp / "output.json"

        in_path.write_text(json.dumps({"a": a_list, "b": b_list}))

        try:
            subprocess.run(
                [str(bin_path), str(in_path), str(proof_path), str(out_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as e:
            raise StarkBridgeError(
                f"Rust prover failed (rc={e.returncode}): {e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise StarkBridgeError(
                f"Rust prover timed out after {timeout_seconds}s"
            ) from e

        proof_bytes = proof_path.read_bytes()
        out_data = json.loads(out_path.read_text())

        return StarkProof(
            proof_bytes=proof_bytes,
            public_output=int(out_data["c"]),
            n=int(out_data["n"]),
        )


def verify_dot_product(
    proof: bytes | StarkProof,
    a: Sequence[int] | np.ndarray,
    b: Sequence[int] | np.ndarray,
    claimed_c: int,
    *,
    verify_bin: Path | None = None,
    timeout_seconds: float = 30.0,
) -> bool:
    """
    Verify a previously-generated STARK proof using the Rust backend.

    Returns ``True`` iff the proof is valid for the given ``(a, b, c)``.

    Notes
    -----
    Returns ``False`` on cryptographic rejection (wrong c, tampered
    proof). Raises :class:`StarkBridgeError` on malformed inputs (e.g.,
    proof bytes corrupted, length mismatch).
    """
    bin_path = verify_bin or _resolve_bin("ARYA_STARK_VERIFY_BIN", DEFAULT_VERIFY_BIN)

    a_list = [int(x) for x in np.asarray(a, dtype=np.uint64).ravel()]
    b_list = [int(x) for x in np.asarray(b, dtype=np.uint64).ravel()]

    # Extract proof bytes if proof is a StarkProof object.
    if isinstance(proof, StarkProof):
        proof_bytes = proof.proof_bytes
    else:
        proof_bytes = proof

    with tempfile.TemporaryDirectory(prefix="arya_stark_verify_") as tmp:
        tmp = Path(tmp)
        in_path = tmp / "input.json"
        proof_path = tmp / "proof.bin"

        in_path.write_text(
            json.dumps({"a": a_list, "b": b_list, "c": int(claimed_c)})
        )
        proof_path.write_bytes(proof_bytes)

        result = subprocess.run(
            [str(bin_path), str(in_path), str(proof_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise StarkBridgeError(
            f"verify error (rc={result.returncode}): {result.stderr}"
        )


__all__ = [
    "StarkBridgeError",
    "StarkProof",
    "prove_dot_product",
    "verify_dot_product",
]

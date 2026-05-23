"""
Cross-language consistency test: Python ↔ Rust ML-DSA-65.

Both implementations follow FIPS 204; signatures are interoperable.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from arya_stark.crypto import mldsa as py_mldsa


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = REPO_ROOT / "rust" / "target" / "release"
DEBUG_DIR = REPO_ROOT / "rust" / "target" / "debug"


def _find_bin(name: str, env_var: str) -> Path | None:
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    for d in (RELEASE_DIR, DEBUG_DIR):
        candidate = d / name
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def rust_bins() -> dict[str, Path]:
    bins = {
        "keygen": _find_bin("keygen", "ARYA_STARK_KEYGEN_BIN"),
        "sign": _find_bin("sign", "ARYA_STARK_SIGN_BIN"),
        "verify_sig": _find_bin("verify_sig", "ARYA_STARK_VERIFY_BIN_MLDSA"),
    }
    missing = [name for name, p in bins.items() if p is None]
    if missing:
        pytest.skip(f"Rust ML-DSA binaries missing: {missing}")
    return {k: v for k, v in bins.items() if v is not None}


@pytest.mark.rust
def test_rust_signs_python_verifies(
    rust_bins: dict[str, Path], tmp_path: Path
) -> None:
    pk_path = tmp_path / "pk.bin"
    sk_path = tmp_path / "sk.bin"
    msg_path = tmp_path / "msg.bin"
    sig_path = tmp_path / "sig.bin"

    msg = b"arya-STARK round 0: gradient hash 0xabcd"
    msg_path.write_bytes(msg)

    subprocess.run(
        [str(rust_bins["keygen"]), str(pk_path), str(sk_path)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [str(rust_bins["sign"]), str(sk_path), str(msg_path), str(sig_path)],
        check=True, capture_output=True,
    )

    pk = pk_path.read_bytes()
    sig = sig_path.read_bytes()
    assert py_mldsa.verify(pk, msg, sig)


@pytest.mark.rust
def test_python_signs_rust_verifies(
    rust_bins: dict[str, Path], tmp_path: Path
) -> None:
    pk_path = tmp_path / "pk.bin"
    msg_path = tmp_path / "msg.bin"
    sig_path = tmp_path / "sig.bin"

    pk, sk = py_mldsa.keygen()
    msg = b"arya-STARK reverse direction"
    sig = py_mldsa.sign(sk, msg)

    pk_path.write_bytes(pk)
    msg_path.write_bytes(msg)
    sig_path.write_bytes(sig)

    result = subprocess.run(
        [str(rust_bins["verify_sig"]), str(pk_path), str(msg_path), str(sig_path)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


@pytest.mark.rust
def test_rust_signs_python_rejects_tampered(
    rust_bins: dict[str, Path], tmp_path: Path
) -> None:
    pk_path = tmp_path / "pk.bin"
    sk_path = tmp_path / "sk.bin"
    msg_path = tmp_path / "msg.bin"
    sig_path = tmp_path / "sig.bin"
    msg = b"original"
    msg_path.write_bytes(msg)
    subprocess.run(
        [str(rust_bins["keygen"]), str(pk_path), str(sk_path)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [str(rust_bins["sign"]), str(sk_path), str(msg_path), str(sig_path)],
        check=True, capture_output=True,
    )
    pk = pk_path.read_bytes()
    sig = sig_path.read_bytes()
    bad_sig = bytearray(sig)
    bad_sig[42] ^= 0xFF
    assert not py_mldsa.verify(pk, msg, bytes(bad_sig))
    assert not py_mldsa.verify(pk, b"forgery", sig)

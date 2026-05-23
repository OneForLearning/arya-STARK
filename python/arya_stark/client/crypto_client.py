"""
arya_stark.client.crypto_client
===============================

Cryptographic wrapper around HonestClient that adds STARK proving
and ML-DSA signing to each local update.

This is the **P7 integration layer** that bridges the FL pipeline
(P4-P6) with the cryptographic primitives (P1-P3).

Public API
----------
* :class:`CryptoClient`     — wraps HonestClient with crypto.
* :class:`SignedUpdate`     — client update + proof + signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arya_stark.client.honest_client import HonestClient, LocalUpdate
from arya_stark.client.stark_bridge import (
    StarkProof,
    prove_dot_product,
    verify_dot_product,
)
from arya_stark.crypto.mldsa import sign, verify
from arya_stark.encoding import decode_vector, encode_vector


@dataclass
class SignedUpdate:
    """
    Client update with cryptographic attestation.

    Attributes
    ----------
    client_id : int
    delta : np.ndarray
        Gradient in float32 (shape (d,)). This is what the server
        aggregates.
    delta_fp : np.ndarray
        Gradient encoded in 𝔽_p (dtype=uint64, shape (d,)).
    proof : StarkProof
        zk-STARK proof object (contains proof_bytes and public_output).
    signature : bytes
        ML-DSA-65 signature over (proof.proof_bytes || delta_fp).
    n_samples : int
        Number of local training samples (for weighted aggregation).
    final_loss : float
        Final local loss after training.
    final_acc : float
        Final local accuracy after training.
    """

    client_id: int
    delta: np.ndarray  # float32
    delta_fp: np.ndarray  # encoded in 𝔽_p
    proof: StarkProof  # CHANGED: store full object, not just bytes
    signature: bytes
    n_samples: int
    final_loss: float
    final_acc: float


class CryptoClient:
    """
    Wrapper that adds STARK proving + ML-DSA signing to HonestClient.

    For P7, we prove a **single representative dot-product** as a
    proof-of-concept. The full gradient proof (all dot-products in
    the gradient computation) will be implemented in P8 with the MLP
    AIR.

    Parameters
    ----------
    honest_client : HonestClient
        The underlying honest client (handles local training).
    secret_key : bytes
        ML-DSA-65 secret key for signing.
    public_key : bytes
        ML-DSA-65 public key (stored for verification).
    modulus : int
        Field modulus 𝔽_p (default 2^61 - 1).
    precision : int
        Fixed-point precision (default 6 decimal places).
    """

    def __init__(
        self,
        honest_client: HonestClient,
        secret_key: bytes,
        public_key: bytes,
        modulus: int = 2**61 - 1,
        precision: int = 6,
    ) -> None:
        self.honest_client = honest_client
        self.secret_key = secret_key
        self.public_key = public_key
        self.modulus = modulus
        self.precision = precision

    @property
    def client_id(self) -> int:
        return self.honest_client.client_id

    @property
    def n_samples(self) -> int:
        return self.honest_client.n_samples

    def compute_update(
        self, global_params: np.ndarray, round_number: int
    ) -> SignedUpdate:
        """
        Compute local update with cryptographic attestation.

        Steps:
          1. Train locally → delta (float32)
          2. Encode delta → 𝔽_p
          3. Prove gradient correctness (representative dot-product)
          4. Sign (proof || delta_fp) with ML-DSA-65
          5. Return SignedUpdate

        Returns
        -------
        SignedUpdate
            Contains delta_fp, proof, signature, metadata.
        """
        # Step 1: Local training (honest gradient).
        update: LocalUpdate = self.honest_client.compute_update(
            global_params, round_number
        )

        # Step 2: Encode delta → 𝔽_p.
        delta_fp = encode_vector(update.delta, m=self.precision)

        # Step 3: Generate STARK proof.
        stark_proof = self._prove_representative_computation(delta_fp)

        # Step 4: Sign (proof_bytes || delta_fp).
        # We sign the proof bytes (not the full StarkProof object).
        message = stark_proof.proof_bytes + delta_fp.tobytes()
        signature = sign(self.secret_key, message)

        # Step 5: Return signed update (store full StarkProof).
        return SignedUpdate(
            client_id=self.client_id,
            delta=update.delta,  # float32 for aggregation
            delta_fp=delta_fp,    # 𝔽_p for proof/signature
            proof=stark_proof,    # Store full StarkProof (not just bytes)
            signature=signature,
            n_samples=update.n_samples,
            final_loss=update.final_loss,
            final_acc=update.final_acc,
        )

    def _prove_representative_computation(self, delta_fp: np.ndarray) -> StarkProof:
        """
        Generate a STARK proof for a representative computation.

        For P7, we prove: dot_product(delta[:10], [1, 1, ..., 1]) = sum(delta[:10]).
        This demonstrates the crypto pipeline; the full gradient proof is P8.

        Parameters
        ----------
        delta_fp : np.ndarray
            Full gradient in 𝔽_p (shape (d,), dtype=uint64).

        Returns
        -------
        StarkProof
            STARK proof object.
        """
        # Take first 10 components (or pad if d < 10).
        n = min(10, len(delta_fp))
        a = delta_fp[:n].copy()
        if n < 10:
            a = np.concatenate([a, np.zeros(10 - n, dtype=np.uint64)])
        b = np.ones(10, dtype=np.uint64)

        # Generate proof via Rust bridge.
        proof = prove_dot_product(a, b)
        return proof


def verify_signed_update(
    update: SignedUpdate, public_key: bytes, modulus: int = 2**61 - 1
) -> tuple[bool, bool]:
    """
    Verify a signed update's cryptographic attestations.

    Returns
    -------
    sig_valid : bool
        True if ML-DSA signature is valid.
    proof_valid : bool
        True if STARK proof is valid.
    """
    # Verify ML-DSA signature (over proof_bytes || delta_fp).
    message = update.proof.proof_bytes + update.delta_fp.tobytes()
    sig_valid = verify(public_key, message, update.signature)

    # Verify STARK proof.
    # Reconstruct inputs: first 10 components of delta_fp.
    n = min(10, len(update.delta_fp))
    a = update.delta_fp[:n].copy()
    if n < 10:
        a = np.concatenate([a, np.zeros(10 - n, dtype=np.uint64)])
    b = np.ones(10, dtype=np.uint64)
    
    # Use the claimed c from the proof object (not recomputed).
    # This avoids modular arithmetic errors in Python.
    c = update.proof.public_output
    
    proof_valid = verify_dot_product(update.proof, a, b, c)

    return sig_valid, proof_valid


__all__ = [
    "CryptoClient",
    "SignedUpdate",
    "verify_signed_update",
]

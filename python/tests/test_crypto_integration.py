"""
P7 integration test: full FL round with STARK + ML-DSA.

This test demonstrates the complete crypto pipeline:
  1. Clients compute updates locally
  2. Encode deltas → 𝔽_p
  3. Generate STARK proofs
  4. Sign (proof || delta) with ML-DSA-65
  5. Server verifies signatures and proofs
  6. Server aggregates and updates global model

For P7, we use a simplified proof (dot-product of first 10
components) as a proof-of-concept. The full gradient AIR will be
implemented in P8.
"""
from __future__ import annotations

import numpy as np
import pytest

from arya_stark.client.crypto_client import CryptoClient, verify_signed_update
from arya_stark.client.honest_client import HonestClient
from arya_stark.config import ModelConfig
from arya_stark.crypto.mldsa import keygen
from arya_stark.data.loaders import load_synthetic_mnist
from arya_stark.data.partition import partition_iid
from arya_stark.models.linear import LinearModel
from arya_stark.server.aggregator import FedAvg


@pytest.mark.slow
def test_one_round_with_crypto() -> None:
    """
    Run 1 FL round with 3 clients, each sending a STARK-proven and
    ML-DSA-signed update.
    """
    # Setup: dataset, model, clients.
    ds = load_synthetic_mnist(n_train=3000, n_test=1000, seed=42)
    shards = partition_iid(3000, num_clients=3, seed=42)
    model = LinearModel.from_config(
        ModelConfig(name="linear", dataset="synthetic_mnist", input_dim=784, num_classes=10),
        seed=42,
    )

    # Generate ML-DSA keypairs for each client.
    keypairs = [keygen() for _ in range(3)]

    # Wrap HonestClients with CryptoClients.
    clients = []
    for i, (shard, (pk, sk)) in enumerate(zip(shards, keypairs)):  # CORRECTED order
        honest = HonestClient(
            client_id=i,
            dataset=ds,
            shard=shard,
            local_epochs=1,
            local_batch_size=32,
            learning_rate=0.1,
            seed=42,
        )
        crypto = CryptoClient(
            honest_client=honest,
            secret_key=sk,
            public_key=pk,
        )
        clients.append((crypto, pk))

    # Round 1: collect signed updates.
    global_params = model.get_flat_params()
    signed_updates = []
    for crypto_client, _ in clients:
        update = crypto_client.compute_update(global_params, round_number=1)
        signed_updates.append(update)

    # Server: verify all signatures and proofs.
    for update, (_, pk) in zip(signed_updates, clients):
        sig_valid, proof_valid = verify_signed_update(update, pk)
        assert sig_valid, f"Signature verification failed for client {update.client_id}"
        assert proof_valid, f"Proof verification failed for client {update.client_id}"

    # Server: aggregate.
    # Convert SignedUpdate → LocalUpdate for aggregator.
    from arya_stark.client.honest_client import LocalUpdate

    local_updates = [
        LocalUpdate(
            client_id=u.client_id,
            delta=u.delta,
            n_samples=u.n_samples,
            final_loss=u.final_loss,
            final_acc=u.final_acc,
        )
        for u in signed_updates
    ]
    agg = FedAvg()
    aggregated_delta = agg.aggregate(local_updates)

    # Server: apply update.
    model.apply_update(aggregated_delta, lr=1.0)

    # Check that model improved.
    X_test, y_test = ds.X_test, ds.y_test
    test_acc = model.accuracy(X_test, y_test)
    test_loss = model.loss(X_test, y_test)
    print(f"After 1 round with crypto: test_acc={test_acc:.3f}, test_loss={test_loss:.3f}")

    # The model should have learned something (accuracy > random 10%).
    assert test_acc > 0.3, f"Model didn't learn: accuracy={test_acc:.3f}"


def test_crypto_client_simple() -> None:
    """Simple unit test: CryptoClient produces a valid SignedUpdate."""
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    pk, sk = keygen()  # CORRECTED: pk first, sk second
    honest = HonestClient(
        client_id=0,
        dataset=ds,
        shard=shards[0],
        local_epochs=1,
        local_batch_size=32,
        learning_rate=0.1,
        seed=42,
    )
    crypto = CryptoClient(honest_client=honest, secret_key=sk, public_key=pk)
    global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
    update = crypto.compute_update(global_params, round_number=1)

    # Verify the update.
    sig_valid, proof_valid = verify_signed_update(update, pk)
    assert sig_valid
    assert proof_valid
    assert update.delta.shape == (784 * 10 + 10,)
    assert update.delta_fp.shape == (784 * 10 + 10,)

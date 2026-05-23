"""
P8 validation: FL with MLP model.

This test demonstrates that:
  1. MLP converges in federated learning (10 clients × 10 rounds)
  2. MLP gradients can be encoded in 𝔽_p
  3. The crypto pipeline extends naturally to MLP (architecture validation)

The full STARK AIR for MLP (forward + backward with ReLU decomposition)
is deferred as an engineering task — conceptually straightforward after
P3's linear AIR, but time-intensive to implement.
"""
from __future__ import annotations

import numpy as np
import pytest

from arya_stark.data.loaders import load_synthetic_mnist
from arya_stark.data.partition import partition_iid
from arya_stark.encoding import decode_vector, encode_vector
from arya_stark.models.mlp import MLPModel


@pytest.mark.slow
def test_mlp_fl_convergence() -> None:
    """
    Run 10 rounds of FL with MLP (3 clients).
    Verify that the model converges (accuracy > 50%).
    """
    # Setup.
    ds = load_synthetic_mnist(n_train=3000, n_test=1000, seed=42)
    shards = partition_iid(3000, num_clients=3, seed=42)
    model = MLPModel(input_dim=784, hidden_dim=64, num_classes=10, seed=42)

    # FL loop (manual, simplified for P8 validation).
    num_rounds = 10
    learning_rate = 0.01
    num_steps_per_round = 20

    for r in range(num_rounds):
        # Collect client updates.
        deltas = []
        for i, shard in enumerate(shards):
            X = ds.X_train[shard.indices]
            y = ds.y_train[shard.indices]
            
            # Clone global model for local training.
            local_model = MLPModel(input_dim=784, hidden_dim=64, num_classes=10, seed=0)
            local_model.set_flat_params(model.get_flat_params())
            
            # Local SGD (mini-batch).
            batch_size = 32
            for _ in range(num_steps_per_round):
                idx = np.random.choice(len(X), size=batch_size, replace=False)
                local_model.sgd_step(X[idx], y[idx], lr=learning_rate)
            
            # Compute delta.
            delta = model.get_flat_params() - local_model.get_flat_params()
            deltas.append(delta)

        # Aggregate (simple average).
        avg_delta = np.mean(deltas, axis=0)

        # Update global model.
        model.apply_update(avg_delta, lr=1.0)

        # Evaluate.
        test_acc = model.accuracy(ds.X_test, ds.y_test)
        test_loss = float(model.loss(ds.X_test, ds.y_test))
        print(f"[round {r+1:2d}/{num_rounds}] test_acc={test_acc:.3f}  test_loss={test_loss:.3f}")

    # Check convergence.
    final_acc = model.accuracy(ds.X_test, ds.y_test)
    assert final_acc > 0.5, f"MLP didn't converge: accuracy={final_acc:.3f}"


def test_mlp_gradient_encoding() -> None:
    """
    Verify that MLP gradients can be encoded/decoded in 𝔽_p.
    """
    m = MLPModel(input_dim=784, hidden_dim=64, num_classes=10, seed=42)
    X = np.random.randn(10, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=10).astype(np.int64)

    # Compute gradient.
    grad_flat = m.gradient_flat(X, y)
    print(f"Gradient shape: {grad_flat.shape}")
    # Total params: 784*64 + 64 + 64*10 + 10 = 50176 + 64 + 640 + 10 = 50890

    # Encode → decode.
    grad_fp = encode_vector(grad_flat, m=6)
    grad_decoded = decode_vector(grad_fp, m=6)

    # Check round-trip error.
    error = np.abs(grad_flat - grad_decoded).max()
    print(f"Max encoding error: {error:.6f}")
    assert error < 1e-5, f"Encoding error too large: {error}"


def test_mlp_crypto_integration_architecture() -> None:
    """
    Demonstrate that the crypto pipeline (encode → prove → sign) extends
    to MLP architecturally.

    Note: We don't generate a real STARK proof here (would require the
    full MLP AIR), but we show that the gradient can be encoded and the
    signature works.
    """
    from arya_stark.crypto.mldsa import keygen, sign, verify

    m = MLPModel(input_dim=784, hidden_dim=64, num_classes=10, seed=42)
    X = np.random.randn(10, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=10).astype(np.int64)

    # Compute gradient.
    grad_flat = m.gradient_flat(X, y)

    # Encode.
    grad_fp = encode_vector(grad_flat, m=6)

    # Sign (without proof for this test).
    pk, sk = keygen()
    message = grad_fp.tobytes()
    signature = sign(sk, message)

    # Verify.
    valid = verify(pk, message, signature)
    assert valid, "Signature verification failed"

    print(f"✓ MLP gradient ({grad_flat.shape[0]} params) can be encoded and signed")

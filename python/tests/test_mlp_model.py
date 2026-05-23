"""Tests for MLPModel."""
from __future__ import annotations

import numpy as np

from arya_stark.config import ModelConfig
from arya_stark.models.mlp import MLPModel, cross_entropy, relu, relu_derivative, softmax


def test_relu() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y = relu(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    assert np.allclose(y, expected)


def test_relu_derivative() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    dy = relu_derivative(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    assert np.allclose(dy, expected)


def test_softmax() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]], dtype=np.float32)
    probs = softmax(logits)
    # Each row should sum to 1.
    assert np.allclose(probs.sum(axis=-1), [1.0, 1.0])
    # All probs should be in [0, 1].
    assert (probs >= 0).all() and (probs <= 1).all()


def test_cross_entropy() -> None:
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    loss = cross_entropy(probs, labels)
    # Should be small (good predictions).
    assert loss < 0.2


def test_mlp_from_config() -> None:
    cfg = ModelConfig(name="mlp", dataset="mnist", input_dim=784, num_classes=10)
    m = MLPModel.from_config(cfg, seed=42)
    assert m.W1.shape == (784, 128)  # default hidden_dim=128
    assert m.b1.shape == (128,)
    assert m.W2.shape == (128, 10)
    assert m.b2.shape == (10,)


def test_mlp_forward() -> None:
    m = MLPModel(input_dim=10, hidden_dim=8, num_classes=3, seed=42)
    X = np.random.randn(5, 10).astype(np.float32)
    logits = m.forward(X)
    assert logits.shape == (5, 3)


def test_mlp_loss() -> None:
    m = MLPModel(input_dim=10, hidden_dim=8, num_classes=3, seed=42)
    X = np.random.randn(5, 10).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    loss = m.loss(X, y)
    assert isinstance(loss, (float, np.floating))
    assert loss > 0


def test_mlp_accuracy() -> None:
    m = MLPModel(input_dim=10, hidden_dim=8, num_classes=3, seed=42)
    X = np.random.randn(5, 10).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    acc = m.accuracy(X, y)
    assert 0.0 <= acc <= 1.0


def test_mlp_gradient_shape() -> None:
    m = MLPModel(input_dim=10, hidden_dim=8, num_classes=3, seed=42)
    X = np.random.randn(5, 10).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    dW1, db1, dW2, db2 = m.gradient(X, y)
    assert dW1.shape == (10, 8)
    assert db1.shape == (8,)
    assert dW2.shape == (8, 3)
    assert db2.shape == (3,)


def test_mlp_flat_params_round_trip() -> None:
    m1 = MLPModel(input_dim=10, hidden_dim=8, num_classes=3, seed=42)
    flat = m1.get_flat_params()
    # Total params: 10*8 + 8 + 8*3 + 3 = 80 + 8 + 24 + 3 = 115
    assert flat.shape == (115,)
    m2 = MLPModel(input_dim=10, hidden_dim=8, num_classes=3, seed=0)
    m2.set_flat_params(flat)
    assert np.allclose(m2.W1, m1.W1)
    assert np.allclose(m2.b1, m1.b1)
    assert np.allclose(m2.W2, m1.W2)
    assert np.allclose(m2.b2, m1.b2)


def test_mlp_sgd_step_decreases_loss() -> None:
    m = MLPModel(input_dim=10, hidden_dim=8, num_classes=3, seed=42)
    X = np.random.randn(10, 10).astype(np.float32)
    y = np.random.randint(0, 3, size=10).astype(np.int64)
    loss_before = m.loss(X, y)
    m.sgd_step(X, y, lr=0.1)
    loss_after = m.loss(X, y)
    # Loss should decrease (at least not increase significantly).
    assert loss_after <= loss_before + 0.01


def test_mlp_gradient_correctness_via_finite_differences() -> None:
    """Check gradient via finite differences on a tiny MLP."""
    m = MLPModel(input_dim=5, hidden_dim=3, num_classes=2, seed=42)
    X = np.random.randn(2, 5).astype(np.float32)
    y = np.array([0, 1], dtype=np.int64)

    # Analytical gradient.
    grad_flat = m.gradient_flat(X, y)

    # Numerical gradient via finite differences.
    eps = 1e-4
    flat_params = m.get_flat_params()
    num_grad = np.zeros_like(flat_params)
    for i in range(flat_params.shape[0]):
        p = flat_params.copy()
        p[i] += eps
        m.set_flat_params(p)
        loss_plus = m.loss(X, y)
        p[i] -= 2 * eps
        m.set_flat_params(p)
        loss_minus = m.loss(X, y)
        num_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    # Restore original.
    m.set_flat_params(flat_params)
    # ReLU introduces non-smoothness, so we need a slightly larger tolerance.
    assert np.allclose(grad_flat, num_grad, atol=3e-3)

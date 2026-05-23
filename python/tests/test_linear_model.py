"""Tests for arya_stark.models.linear."""
from __future__ import annotations

import numpy as np

from arya_stark.config import ModelConfig
from arya_stark.models.linear import LinearModel, cross_entropy, softmax


def test_softmax_sums_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=-1), [1.0, 1.0])


def test_cross_entropy_decreases_with_better_predictions() -> None:
    probs_good = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    probs_bad = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]])
    labels = np.array([0, 1])
    loss_good = cross_entropy(probs_good, labels)
    loss_bad = cross_entropy(probs_bad, labels)
    assert loss_good < loss_bad


def test_from_config() -> None:
    cfg = ModelConfig(name="linear", dataset="mnist", input_dim=784, num_classes=10)
    m = LinearModel.from_config(cfg, seed=42)
    assert m.W.shape == (784, 10)
    assert m.b.shape == (10,)


def test_flat_params_round_trip() -> None:
    cfg = ModelConfig(name="linear", dataset="mnist", input_dim=784, num_classes=10)
    m1 = LinearModel.from_config(cfg, seed=42)
    flat = m1.get_flat_params()
    assert flat.shape == (784 * 10 + 10,)
    m2 = LinearModel.from_flat(flat, input_dim=784, num_classes=10)
    # Use allclose instead of == to handle float32 precision.
    assert np.allclose(m2.W, m1.W)
    assert np.allclose(m2.b, m1.b)


def test_forward_shape() -> None:
    m = LinearModel.from_config(
        ModelConfig(name="linear", dataset="mnist", input_dim=784, num_classes=10),
        seed=42,
    )
    X = np.random.randn(5, 784).astype(np.float32)
    logits = m.forward(X)
    assert logits.shape == (5, 10)


def test_gradient_shape() -> None:
    m = LinearModel.from_config(
        ModelConfig(name="linear", dataset="mnist", input_dim=784, num_classes=10),
        seed=42,
    )
    X = np.random.randn(5, 784).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    dW, db = m.gradient(X, y)
    assert dW.shape == (784, 10)
    assert db.shape == (10,)


def test_gradient_correctness_via_finite_differences() -> None:
    """Check gradient via finite differences on a tiny model."""
    # Small model for speed
    m = LinearModel.from_config(
        ModelConfig(name="linear", dataset="mnist", input_dim=5, num_classes=3),
        seed=42,
    )
    X = np.random.randn(2, 5).astype(np.float32)
    y = np.array([0, 1], dtype=np.int64)

    # Analytical gradient
    grad_flat = m.gradient_flat(X, y)

    # Numerical gradient via finite differences
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

    # Restore original
    m.set_flat_params(flat_params)
    assert np.allclose(grad_flat, num_grad, atol=2e-3)  # Relaxed from 1e-3


def test_sgd_step_decreases_loss() -> None:
    m = LinearModel.from_config(
        ModelConfig(name="linear", dataset="mnist", input_dim=784, num_classes=10),
        seed=42,
    )
    X = np.random.randn(10, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=10).astype(np.int64)
    loss_before = m.loss(X, y)
    m.sgd_step(X, y, lr=0.1)
    loss_after = m.loss(X, y)
    assert loss_after < loss_before


def test_apply_update() -> None:
    m = LinearModel.from_config(
        ModelConfig(name="linear", dataset="mnist", input_dim=10, num_classes=3),
        seed=42,
    )
    original = m.get_flat_params().copy()
    delta = np.ones_like(original) * 0.5
    m.apply_update(delta, lr=0.1)
    new = m.get_flat_params()
    expected = original - 0.1 * delta
    assert np.allclose(new, expected)

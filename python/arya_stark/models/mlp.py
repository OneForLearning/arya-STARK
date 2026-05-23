"""
arya_stark.models.mlp
=====================

Multi-Layer Perceptron (MLP) with ReLU activations.

This module implements a simple 2-layer MLP for federated learning:
  - Input layer → hidden layer (W₁, b₁) + ReLU
  - Hidden layer → output layer (W₂, b₂) + softmax
  - Cross-entropy loss

Used in P8 to demonstrate STARK proving for non-linear models.

Architecture
------------
::

    x ∈ ℝ^d_in
      ↓ W₁ (d_in × d_hidden) + b₁
    h_pre ∈ ℝ^d_hidden
      ↓ ReLU
    h ∈ ℝ^d_hidden
      ↓ W₂ (d_hidden × d_out) + b₂
    logits ∈ ℝ^d_out
      ↓ softmax
    probs ∈ ℝ^d_out

Public API
----------
* :class:`MLPModel` — 2-layer MLP with ReLU.
"""
from __future__ import annotations

import numpy as np

from arya_stark.config import ModelConfig


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    return (x > 0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over the last axis.

    Parameters
    ----------
    logits : np.ndarray
        Shape (..., num_classes).

    Returns
    -------
    np.ndarray
        Probabilities, same shape as logits.
    """
    # Subtract max for numerical stability.
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Cross-entropy loss: -mean(log(probs[labels])).

    Parameters
    ----------
    probs : np.ndarray
        Shape (batch_size, num_classes).
    labels : np.ndarray
        Shape (batch_size,), dtype int.

    Returns
    -------
    float
        Scalar loss.
    """
    batch_size = probs.shape[0]
    # Extract probs[i, labels[i]] for each sample.
    log_probs = np.log(probs[np.arange(batch_size), labels] + 1e-12)
    return -np.mean(log_probs)


class MLPModel:
    """
    2-layer MLP: input → hidden (ReLU) → output (softmax).

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dim : int
        Number of hidden units.
    num_classes : int
        Number of output classes.
    seed : int
        Random seed for weight initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Initialize weights with small random values.
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (hidden_dim, num_classes)).astype(np.float32)
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    @classmethod
    def from_config(cls, config: ModelConfig, seed: int = 0) -> MLPModel:
        """Create MLP from ModelConfig."""
        if config.name != "mlp":
            raise ValueError(f"Expected model='mlp', got '{config.name}'")
        # Default hidden_dim = 128 if not specified.
        hidden_dim = getattr(config, "hidden_dim", 128)
        return cls(
            input_dim=config.input_dim,
            hidden_dim=hidden_dim,
            num_classes=config.num_classes,
            seed=seed,
        )

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: X → hidden (ReLU) → logits.

        Parameters
        ----------
        X : np.ndarray
            Shape (batch_size, input_dim).

        Returns
        -------
        np.ndarray
            Logits, shape (batch_size, num_classes).
        """
        # Hidden layer.
        h_pre = X @ self.W1 + self.b1  # (batch, hidden)
        h = relu(h_pre)
        # Output layer.
        logits = h @ self.W2 + self.b2  # (batch, num_classes)
        return logits

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Parameters
        ----------
        X : np.ndarray
            Shape (batch_size, input_dim).
        y : np.ndarray
            Shape (batch_size,), dtype int.

        Returns
        -------
        float
            Scalar loss.
        """
        logits = self.forward(X)
        probs = softmax(logits)
        return cross_entropy(probs, y)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray
            Shape (batch_size, input_dim).
        y : np.ndarray
            Shape (batch_size,), dtype int.

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        logits = self.forward(X)
        preds = np.argmax(logits, axis=-1)
        return float(np.mean(preds == y))

    def gradient(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients via backpropagation.

        Parameters
        ----------
        X : np.ndarray
            Shape (batch_size, input_dim).
        y : np.ndarray
            Shape (batch_size,), dtype int.

        Returns
        -------
        dW1 : np.ndarray
            Gradient w.r.t. W1, shape (input_dim, hidden_dim).
        db1 : np.ndarray
            Gradient w.r.t. b1, shape (hidden_dim,).
        dW2 : np.ndarray
            Gradient w.r.t. W2, shape (hidden_dim, num_classes).
        db2 : np.ndarray
            Gradient w.r.t. b2, shape (num_classes,).
        """
        batch_size = X.shape[0]

        # Forward pass (store intermediates).
        h_pre = X @ self.W1 + self.b1  # (batch, hidden)
        h = relu(h_pre)
        logits = h @ self.W2 + self.b2  # (batch, num_classes)
        probs = softmax(logits)

        # Backward pass.
        # Gradient of cross-entropy + softmax w.r.t. logits.
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), y] -= 1.0
        d_logits /= batch_size

        # Gradient w.r.t. W2, b2.
        dW2 = h.T @ d_logits  # (hidden, num_classes)
        db2 = d_logits.sum(axis=0)  # (num_classes,)

        # Gradient w.r.t. h.
        d_h = d_logits @ self.W2.T  # (batch, hidden)

        # Gradient through ReLU.
        d_h_pre = d_h * relu_derivative(h_pre)  # (batch, hidden)

        # Gradient w.r.t. W1, b1.
        dW1 = X.T @ d_h_pre  # (input_dim, hidden)
        db1 = d_h_pre.sum(axis=0)  # (hidden,)

        return dW1, db1, dW2, db2

    def get_flat_params(self) -> np.ndarray:
        """
        Flatten all parameters into a single vector.

        Returns
        -------
        np.ndarray
            Shape (total_params,), dtype float32.
        """
        return np.concatenate(
            [
                self.W1.ravel(),
                self.b1.ravel(),
                self.W2.ravel(),
                self.b2.ravel(),
            ]
        )

    def set_flat_params(self, flat: np.ndarray) -> None:
        """
        Set parameters from a flat vector.

        Parameters
        ----------
        flat : np.ndarray
            Shape (total_params,), dtype float32.
        """
        idx = 0
        # W1
        n1 = self.input_dim * self.hidden_dim
        self.W1 = flat[idx : idx + n1].reshape(self.input_dim, self.hidden_dim)
        idx += n1
        # b1
        n2 = self.hidden_dim
        self.b1 = flat[idx : idx + n2]
        idx += n2
        # W2
        n3 = self.hidden_dim * self.num_classes
        self.W2 = flat[idx : idx + n3].reshape(self.hidden_dim, self.num_classes)
        idx += n3
        # b2
        n4 = self.num_classes
        self.b2 = flat[idx : idx + n4]

    def gradient_flat(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient and return as a flat vector.

        Returns
        -------
        np.ndarray
            Shape (total_params,), dtype float32.
        """
        dW1, db1, dW2, db2 = self.gradient(X, y)
        return np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])

    def sgd_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        """
        Perform one SGD step: params -= lr * gradient.

        Parameters
        ----------
        X : np.ndarray
            Shape (batch_size, input_dim).
        y : np.ndarray
            Shape (batch_size,), dtype int.
        lr : float
            Learning rate.
        """
        dW1, db1, dW2, db2 = self.gradient(X, y)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def apply_update(self, delta: np.ndarray, lr: float) -> None:
        """
        Apply a delta (gradient) update: params -= lr * delta.

        Parameters
        ----------
        delta : np.ndarray
            Flat gradient vector, shape (total_params,).
        lr : float
            Learning rate.
        """
        current = self.get_flat_params()
        self.set_flat_params(current - lr * delta)


__all__ = [
    "MLPModel",
    "relu",
    "relu_derivative",
    "softmax",
    "cross_entropy",
]

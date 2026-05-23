"""
arya_stark.models.linear
========================

Multinomial (softmax) logistic regression in NumPy.

The model is ``f(x; W, b) = softmax(W x + b)`` with cross-entropy
loss. We expose a minimal API focused on what the FL orchestrator
needs:

* :class:`LinearModel.from_config` — initialise weights from a
  ``ModelConfig`` (e.g., ``input_dim=784``, ``num_classes=10``).
* :meth:`forward`                 — predict logits.
* :meth:`gradient`                — compute ∂Loss/∂(W, b) on a batch.
* :meth:`apply_update`            — apply a delta to the parameters.
* :meth:`get_flat_params`         — pack ``(W, b)`` into a flat vector.
* :meth:`set_flat_params`         — unpack from a flat vector.
* :meth:`accuracy`                — top-1 accuracy on a batch.

The flat-vector view is the canonical representation for federated
aggregation (every client → server message is a flat vector of
length ``d * num_classes + num_classes``).

NumPy was chosen over PyTorch deliberately:
* arya-STARK's gradient witness is a numpy computation; using torch
  would require shadowing each step.
* No autograd needed: the gradient of cross-entropy wrt softmax
  inputs has a closed form ``(softmax - one_hot)``.
* Zero CUDA dependency, deterministic, fast for small models.

For MLP / CNN models (P8+), we will switch to torch (autograd
needed for ReLU bit-decomposition tricks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arya_stark.config import ModelConfig


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy ``-mean log p[label]``."""
    eps = 1e-12
    n = probs.shape[0]
    return float(-np.log(probs[np.arange(n), labels] + eps).mean())


# ---------------------------------------------------------------------------
# LinearModel
# ---------------------------------------------------------------------------


@dataclass
class LinearModel:
    """
    Multinomial logistic regression: ``y = softmax(W x + b)``.

    Attributes
    ----------
    W : np.ndarray of shape ``(input_dim, num_classes)``
    b : np.ndarray of shape ``(num_classes,)``
    """

    W: np.ndarray
    b: np.ndarray

    # ----- Construction -----

    @classmethod
    def from_config(cls, model: ModelConfig, *, seed: int = 42) -> "LinearModel":
        """Initialise with He initialisation."""
        rng = np.random.default_rng(seed)
        d = model.input_dim
        c = model.num_classes
        W = rng.standard_normal((d, c)).astype(np.float32) * np.sqrt(2.0 / d)
        b = np.zeros(c, dtype=np.float32)
        return cls(W=W, b=b)

    @classmethod
    def from_flat(
        cls, flat: np.ndarray, *, input_dim: int, num_classes: int
    ) -> "LinearModel":
        d, c = input_dim, num_classes
        if flat.shape != (d * c + c,):
            raise ValueError(
                f"flat vector has length {flat.shape[0]}, expected {d*c + c}"
            )
        W = flat[: d * c].reshape(d, c).astype(np.float32)
        b = flat[d * c :].astype(np.float32)
        return cls(W=W, b=b)

    # ----- Properties -----

    @property
    def input_dim(self) -> int:
        return self.W.shape[0]

    @property
    def num_classes(self) -> int:
        return self.W.shape[1]

    @property
    def num_params(self) -> int:
        return self.W.size + self.b.size

    # ----- Flat parameter view -----

    def get_flat_params(self) -> np.ndarray:
        """Concatenate ``W`` (row-major) and ``b`` into a single 1-D array."""
        return np.concatenate([self.W.ravel(), self.b]).astype(np.float32)

    def set_flat_params(self, flat: np.ndarray) -> None:
        """Inverse of :meth:`get_flat_params`."""
        d, c = self.input_dim, self.num_classes
        if flat.shape != (d * c + c,):
            raise ValueError(
                f"flat shape {flat.shape} does not match (d*c + c) = ({d*c + c},)"
            )
        self.W = flat[: d * c].reshape(d, c).astype(np.float32)
        self.b = flat[d * c :].astype(np.float32)

    # ----- Forward / loss / accuracy -----

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute logits ``X W + b``."""
        return X @ self.W + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=-1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return cross_entropy(self.predict_proba(X), y)

    # ----- Gradient -----

    def gradient(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute ``(∂L/∂W, ∂L/∂b)`` on a batch ``(X, y)``.

        Cross-entropy gradient through softmax:

            dL/dz = (softmax(z) - one_hot(y)) / B
            dL/dW = X^T @ dL/dz
            dL/db = sum(dL/dz, axis=0)

        Returns gradients shaped like ``W`` and ``b`` respectively.
        """
        n = X.shape[0]
        probs = softmax(self.forward(X))
        # Subtract one-hot
        probs[np.arange(n), y] -= 1.0
        dz = probs / n
        dW = X.T @ dz
        db = dz.sum(axis=0)
        return dW.astype(np.float32), db.astype(np.float32)

    def gradient_flat(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Same as :meth:`gradient` but returns a flat vector."""
        dW, db = self.gradient(X, y)
        return np.concatenate([dW.ravel(), db]).astype(np.float32)

    # ----- Apply update -----

    def apply_update(self, delta: np.ndarray, *, lr: float = 1.0) -> None:
        """
        Apply ``new_params = old_params - lr * delta`` in flat space.

        Used by the orchestrator to apply the aggregated gradient.
        """
        flat = self.get_flat_params() - lr * delta.astype(np.float32)
        self.set_flat_params(flat)

    # ----- Mini-batch SGD step (used by clients) -----

    def sgd_step(
        self, X: np.ndarray, y: np.ndarray, *, lr: float
    ) -> tuple[float, float]:
        """
        Perform a single SGD step on ``(X, y)``.

        Returns ``(loss_before, accuracy_before)`` for logging.
        """
        loss = self.loss(X, y)
        acc = self.accuracy(X, y)
        grad = self.gradient_flat(X, y)
        self.apply_update(grad, lr=lr)
        return loss, acc


__all__ = [
    "LinearModel",
    "softmax",
    "cross_entropy",
]

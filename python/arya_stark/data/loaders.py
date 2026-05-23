"""
arya_stark.data.loaders
=======================

Dataset loaders for arya-STARK federated learning.

Supports two data sources:

1. **MNIST (real)** via ``sklearn.datasets.fetch_openml`` —
   downloads ~60 MB on first use; cached afterward.
2. **Synthetic MNIST-like** — deterministic per-seed Gaussian-mixture
   dataset matching MNIST shape (60 000 train + 10 000 test, 784
   features, 10 classes). Used as the default in CI / dev where
   network access may be limited.

The two sources expose the **same numpy interface**::

    X_train: np.ndarray of shape (n_train, 784), dtype float32, range [0, 1]
    y_train: np.ndarray of shape (n_train,), dtype int64, values in 0..9
    X_test:  np.ndarray of shape (n_test, 784),  dtype float32, range [0, 1]
    y_test:  np.ndarray of shape (n_test,),  dtype int64

The orchestrator (P4+) is therefore agnostic to the source.

Public API
----------
* :func:`load_dataset`           — high-level dispatch on dataset name.
* :func:`load_synthetic_mnist`   — deterministic synthetic data.
* :func:`load_real_mnist`        — actual MNIST via fetch_openml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Numpy-backed dataset, identical shape across loaders."""

    X_train: np.ndarray  # (n_train, d)  float32, range [0, 1]
    y_train: np.ndarray  # (n_train,)    int64,   range [0, num_classes)
    X_test: np.ndarray   # (n_test, d)   float32
    y_test: np.ndarray   # (n_test,)     int64
    name: str
    """Human-readable identifier (e.g. 'synthetic_mnist', 'mnist')."""

    @property
    def n_train(self) -> int:
        return self.X_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.X_test.shape[0]

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y_train.max()) + 1


# ---------------------------------------------------------------------------
# Synthetic MNIST-like dataset (default for CI / dev)
# ---------------------------------------------------------------------------


def load_synthetic_mnist(
    n_train: int = 60_000,
    n_test: int = 10_000,
    input_dim: int = 784,
    num_classes: int = 10,
    seed: int = 42,
    class_separation: float = 1.5,
) -> Dataset:
    """
    Generate a deterministic MNIST-shape synthetic dataset.

    Each class is sampled from a multivariate Gaussian whose mean is a
    sparse direction in ``ℝ^input_dim``. ``class_separation`` controls
    how far apart the means are (higher = easier classification).

    With the default parameters, a linear classifier reaches ~85 %
    accuracy after ~10 epochs of SGD — comparable to a vanilla
    softmax regression on real MNIST.

    The function is fully deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)

    # Sparse class-mean directions (each class has random support
    # of ~30 % of the features, with ±class_separation magnitude).
    support = rng.random((num_classes, input_dim)) < 0.3
    signs = rng.choice([-1.0, 1.0], size=(num_classes, input_dim))
    means = (support.astype(np.float32) * signs.astype(np.float32)) * class_separation

    def _sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        # Balanced class assignment.
        y = rng.integers(0, num_classes, size=n)
        # Noise + per-class mean.
        noise = rng.standard_normal(size=(n, input_dim)).astype(np.float32)
        X = noise + means[y]
        # Squash to [0, 1] via sigmoid (mimics MNIST normalisation).
        X = 1.0 / (1.0 + np.exp(-X.astype(np.float64))).astype(np.float32)
        return X, y.astype(np.int64)

    X_train, y_train = _sample(n_train)
    X_test, y_test = _sample(n_test)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        name="synthetic_mnist",
    )


# ---------------------------------------------------------------------------
# Real MNIST loader (production)
# ---------------------------------------------------------------------------


def load_real_mnist(cache_dir: Path | None = None) -> Dataset:
    """
    Load real MNIST via ``sklearn.datasets.fetch_openml``.

    Caches to ``cache_dir`` (default: ``~/.arya_stark_cache/mnist``).
    First call downloads ~50 MB; subsequent calls are instant.

    Raises
    ------
    ImportError
        If ``sklearn`` is not installed.
    RuntimeError
        On network failure.
    """
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as e:
        raise ImportError(
            "real MNIST requires scikit-learn; "
            "use load_synthetic_mnist() instead, or install sklearn."
        ) from e

    cache_dir = cache_dir or Path.home() / ".arya_stark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = cache_dir / "mnist_cached.npz"
    if cached.exists():
        npz = np.load(cached)
        return Dataset(
            X_train=npz["X_train"],
            y_train=npz["y_train"],
            X_test=npz["X_test"],
            y_test=npz["y_test"],
            name="mnist",
        )

    # Network fetch.
    bundle = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
    X = bundle.data.astype(np.float32) / 255.0
    y = bundle.target.astype(np.int64)

    # Standard 60K/10K split (sklearn returns 70K rows).
    X_train, X_test = X[:60_000], X[60_000:]
    y_train, y_test = y[:60_000], y[60_000:]

    np.savez_compressed(
        cached, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        name="mnist",
    )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

DatasetName = Literal["synthetic_mnist", "mnist", "fashion_mnist", "cifar10"]


def load_dataset(
    name: DatasetName | str,
    *,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> Dataset:
    """Load a dataset by name. See module docstring for the canonical names."""
    if name == "synthetic_mnist":
        return load_synthetic_mnist(seed=seed)
    if name == "mnist":
        # Fall back to synthetic if env var requests it (for offline CI).
        if os.environ.get("ARYA_STARK_FORCE_SYNTHETIC", "0") == "1":
            return load_synthetic_mnist(seed=seed)
        return load_real_mnist(cache_dir=cache_dir)
    raise NotImplementedError(
        f"Dataset {name!r} not implemented yet. "
        f"Available: synthetic_mnist, mnist."
    )


__all__ = [
    "Dataset",
    "DatasetName",
    "load_dataset",
    "load_synthetic_mnist",
    "load_real_mnist",
]

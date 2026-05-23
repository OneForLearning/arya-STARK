"""Tests for arya_stark.data.loaders."""
from __future__ import annotations

import pytest

from arya_stark.data.loaders import Dataset, load_dataset, load_synthetic_mnist


def test_synthetic_mnist_shape() -> None:
    ds = load_synthetic_mnist()
    assert ds.X_train.shape == (60_000, 784)
    assert ds.y_train.shape == (60_000,)
    assert ds.X_test.shape == (10_000, 784)
    assert ds.y_test.shape == (10_000,)
    assert ds.num_classes == 10
    assert ds.input_dim == 784


def test_synthetic_mnist_range() -> None:
    ds = load_synthetic_mnist()
    assert ds.X_train.min() >= 0.0
    assert ds.X_train.max() <= 1.0
    assert ds.y_train.min() == 0
    assert ds.y_train.max() == 9


def test_synthetic_mnist_deterministic() -> None:
    ds1 = load_synthetic_mnist(seed=123)
    ds2 = load_synthetic_mnist(seed=123)
    assert (ds1.X_train == ds2.X_train).all()
    assert (ds1.y_train == ds2.y_train).all()


def test_synthetic_mnist_different_seeds() -> None:
    ds1 = load_synthetic_mnist(seed=1)
    ds2 = load_synthetic_mnist(seed=2)
    assert not (ds1.X_train == ds2.X_train).all()


def test_load_dataset_synthetic() -> None:
    ds = load_dataset("synthetic_mnist", seed=42)
    assert ds.name == "synthetic_mnist"
    assert ds.n_train == 60_000


def test_load_dataset_unknown_raises() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        load_dataset("unknown_dataset")

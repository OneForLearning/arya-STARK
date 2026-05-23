"""Tests for arya_stark.data.partition."""
from __future__ import annotations

import numpy as np
import pytest

from arya_stark.data.partition import (
    partition,
    partition_iid,
    partition_non_iid_dirichlet,
)


def test_partition_iid_coverage() -> None:
    """All samples appear exactly once."""
    shards = partition_iid(1000, num_clients=10, seed=42)
    all_indices = np.concatenate([s.indices for s in shards])
    assert sorted(all_indices) == list(range(1000))


def test_partition_iid_balanced() -> None:
    """Shards are roughly balanced."""
    shards = partition_iid(1000, num_clients=10, seed=42)
    sizes = [len(s) for s in shards]
    assert min(sizes) >= 90
    assert max(sizes) <= 110


def test_partition_non_iid_coverage() -> None:
    """All samples appear exactly once in non-IID split."""
    labels = np.repeat(np.arange(10), 100)  # 1000 samples, 10 classes
    shards = partition_non_iid_dirichlet(labels, num_clients=5, alpha=0.5, seed=42)
    all_indices = np.concatenate([s.indices for s in shards])
    assert sorted(all_indices) == list(range(1000))


def test_partition_non_iid_skew() -> None:
    """Lower alpha produces skewed shards (not all classes per client)."""
    labels = np.repeat(np.arange(10), 100)
    shards = partition_non_iid_dirichlet(labels, num_clients=10, alpha=0.1, seed=42)
    # With alpha=0.1, *most* clients see ≤ 3 classes. One outlier seeing
    # all 10 is statistically possible but rare.
    n_classes_per_client = [len(np.unique(labels[s.indices])) for s in shards]
    avg_classes = np.mean(n_classes_per_client)
    # Average should be << 10 (e.g., ≤ 5). Perfect IID would be 10 for all.
    assert avg_classes < 6.0, f"Average {avg_classes:.1f} classes per client is too high (no skew detected)"


def test_partition_dispatcher_iid() -> None:
    labels = np.arange(100)
    shards = partition(labels, num_clients=5, distribution="iid", seed=42)
    assert len(shards) == 5


def test_partition_dispatcher_non_iid() -> None:
    labels = np.repeat(np.arange(10), 100)  # 1000 samples, 10 classes
    shards = partition(
        labels, num_clients=5, distribution="non_iid_dirichlet", alpha=0.5, seed=42
    )
    assert len(shards) == 5


def test_partition_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown distribution"):
        partition(np.arange(100), num_clients=5, distribution="unknown")

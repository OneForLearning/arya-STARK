"""
arya_stark.data.partition
=========================

Partition a dataset across ``N`` clients for federated learning.

Two partitioning schemes (matching ``FLConfig.data_distribution``):

1. **IID** — each client gets a uniform random sub-sample of the
   training set. Sizes are roughly balanced.
2. **Non-IID Dirichlet** — Hsu et al. (2019). Each class's samples
   are distributed across clients via a Dirichlet(α) draw. Lower α
   → more skewed (each client sees fewer classes); α → ∞ recovers
   IID.

Public API
----------
* :class:`ClientShard` — a per-client view of `(X, y)` indices.
* :func:`partition_iid`
* :func:`partition_non_iid_dirichlet`
* :func:`partition`            — high-level dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ClientShard:
    """Indices owned by a single client."""

    client_id: int
    indices: np.ndarray  # 1-D int64

    def __len__(self) -> int:
        return self.indices.shape[0]


def partition_iid(
    n_samples: int,
    num_clients: int,
    *,
    seed: int = 42,
) -> list[ClientShard]:
    """
    IID partition: shuffle indices and split into roughly equal chunks.

    Returns ``num_clients`` shards. The last shard may be slightly
    larger (or smaller) due to integer division.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    chunks = np.array_split(perm, num_clients)
    return [
        ClientShard(client_id=i, indices=chunk.astype(np.int64))
        for i, chunk in enumerate(chunks)
    ]


def partition_non_iid_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    *,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 1,
) -> list[ClientShard]:
    """
    Non-IID partition à la Hsu et al. 2019.

    For each class ``c``, samples are split across clients according
    to a Dirichlet(α, ..., α) draw of length ``num_clients``. Lower
    α produces more skewed partitions (e.g. α=0.1 → each client
    typically sees only 2-3 classes).

    Parameters
    ----------
    labels
        ``(n,)`` array of class labels.
    num_clients
    alpha
        Dirichlet concentration parameter. Default 0.5 = moderate skew.
    seed
    min_samples_per_client
        Floor on shard size; if a draw produces an empty shard, we
        re-sample (up to 10 retries) to satisfy this constraint.

    Raises
    ------
    RuntimeError
        If a valid partition cannot be found in 10 retries (try
        higher ``alpha`` or fewer clients).
    """
    rng = np.random.default_rng(seed)
    n_samples = labels.shape[0]
    classes = np.unique(labels)
    num_classes = classes.size

    for attempt in range(10):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        # For each class, sample a Dirichlet split.
        for c in classes:
            class_idxs = np.where(labels == c)[0]
            rng.shuffle(class_idxs)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            # Convert proportions to integer slice points.
            split_points = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
            chunks = np.split(class_idxs, split_points)
            for i, chunk in enumerate(chunks):
                client_indices[i].extend(chunk.tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_samples_per_client:
            return [
                ClientShard(
                    client_id=i,
                    indices=np.array(ci, dtype=np.int64),
                )
                for i, ci in enumerate(client_indices)
            ]
        # Retry with a different seed.
        rng = np.random.default_rng(seed + attempt + 1)

    raise RuntimeError(
        f"Could not produce a valid Dirichlet({alpha}) partition with "
        f"min ≥ {min_samples_per_client} samples per client after 10 retries. "
        f"Try a higher alpha or fewer clients."
    )


def partition(
    labels: np.ndarray,
    num_clients: int,
    *,
    distribution: Literal["iid", "non_iid_dirichlet"] = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> list[ClientShard]:
    """High-level dispatcher matching ``FLConfig.data_distribution``."""
    if distribution == "iid":
        return partition_iid(labels.shape[0], num_clients, seed=seed)
    if distribution == "non_iid_dirichlet":
        return partition_non_iid_dirichlet(
            labels, num_clients, alpha=alpha, seed=seed
        )
    raise ValueError(f"unknown distribution: {distribution}")


__all__ = [
    "ClientShard",
    "partition",
    "partition_iid",
    "partition_non_iid_dirichlet",
]

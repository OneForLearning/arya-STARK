"""
arya_stark.server.aggregator
============================

Server-side aggregation of client updates.

This module provides a unified interface for several aggregators:

* :class:`FedAvg`        — vanilla federated averaging (McMahan et al. 2017).
* :class:`FedAvgWeighted` — weighted by ``n_samples`` per client.
* :class:`GBREA`         — Gradient-Based Robust Euclidean Aggregation
                            (arya-STARK's core defense). Combines ℓ₂ clipping
                            + coordinate-wise trimmed mean + Shamir secret-
                            shared masked distances (P7).
* :class:`Krum`          — Blanchard et al. 2017. Selects the gradient
                            closest to the consensus.
* :class:`Median`        — coordinate-wise median (naïve baseline).
* :class:`TrimmedMean`   — Yin et al. 2018. Coordinate-wise trimmed mean.

Every aggregator implements the :class:`Aggregator` protocol:

    aggregate(updates: Sequence[LocalUpdate]) → np.ndarray  # flat delta

The returned vector is the aggregated **delta** that the server
will subtract from the global parameters (with the global learning
rate from `FLConfig`).
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from arya_stark.client.honest_client import LocalUpdate


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Aggregator(Protocol):
    """Common protocol for all aggregators."""

    name: str

    def aggregate(self, updates: Sequence[LocalUpdate]) -> np.ndarray:
        """Aggregate per-client deltas into a single delta vector."""
        ...


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------


class FedAvg:
    """
    Vanilla FedAvg: ``aggregate = mean over clients``.

    All clients are weighted equally, regardless of shard size.
    Equivalent to FedAvgWeighted iff all shards have the same size.
    """

    name = "fedavg"

    def aggregate(self, updates: Sequence[LocalUpdate]) -> np.ndarray:
        if not updates:
            raise ValueError("no updates to aggregate")
        deltas = np.stack([u.delta for u in updates], axis=0)
        return deltas.mean(axis=0).astype(np.float32)


class FedAvgWeighted:
    """
    Weighted FedAvg: ``aggregate = Σ (n_i / N) * delta_i``.

    ``n_i`` is the number of samples used by client ``i``, ``N = Σ n_i``.
    This is the original McMahan et al. 2017 formulation.
    """

    name = "fedavg_weighted"

    def aggregate(self, updates: Sequence[LocalUpdate]) -> np.ndarray:
        if not updates:
            raise ValueError("no updates to aggregate")
        weights = np.array([u.n_samples for u in updates], dtype=np.float32)
        weights = weights / weights.sum()
        deltas = np.stack([u.delta for u in updates], axis=0)
        return (weights[:, None] * deltas).sum(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# GBREA (Gradient-Based Robust Euclidean Aggregation)
# ---------------------------------------------------------------------------


class GBREA:
    """
    GBREA: arya-STARK's Byzantine-robust aggregator.

    Combines three defense layers:

    1. **ℓ₂ clipping** — clip each gradient to max norm ``clip_norm``.
       Mitigates magnitude-based attacks (e.g. random_gaussian with
       large σ).
    2. **Coordinate-wise trimmed mean** — for each coordinate, discard
       the ``beta`` largest and ``beta`` smallest values, then average
       the rest. ``beta = floor(N * trim_ratio)``. Mitigates targeted
       poisoning (e.g. sign_flip, label_flip).
    3. **Shamir secret-shared masked distances** (P7) — detect
       malicious clients via pairwise Euclidean distances. Requires
       multi-round MPC; deferred to the crypto integration phase.

    For P5, layers 1 and 2 are active; layer 3 is a no-op placeholder.

    Parameters
    ----------
    clip_norm : float
        Maximum ℓ₂ norm per gradient. Default 5.0.
    trim_ratio : float
        Fraction of outliers to trim per coordinate. Default 0.1
        (10% from each tail → 20% total trimmed if N ≥ 10).
    use_shamir : bool
        Enable Shamir layer (P7+). Default False.
    """

    name = "gbrea"

    def __init__(
        self,
        clip_norm: float = 5.0,
        trim_ratio: float = 0.1,
        use_shamir: bool = False,
    ) -> None:
        self.clip_norm = clip_norm
        self.trim_ratio = trim_ratio
        self.use_shamir = use_shamir

    def aggregate(self, updates: Sequence[LocalUpdate]) -> np.ndarray:
        if not updates:
            raise ValueError("no updates to aggregate")

        # Stack deltas into (N, d) array.
        deltas = np.stack([u.delta for u in updates], axis=0)
        N, d = deltas.shape

        # Layer 1: ℓ₂ clipping.
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.clip_norm / (norms + 1e-12))
        deltas_clipped = deltas * scale

        # Layer 2: coordinate-wise trimmed mean.
        beta = int(np.floor(N * self.trim_ratio))
        if beta > 0 and N >= 2 * beta + 1:
            # Sort each coordinate independently.
            sorted_deltas = np.sort(deltas_clipped, axis=0)
            # Trim beta from each tail.
            trimmed = sorted_deltas[beta : N - beta, :]
            result = trimmed.mean(axis=0).astype(np.float32)
        else:
            # Not enough clients to trim → fallback to mean.
            result = deltas_clipped.mean(axis=0).astype(np.float32)

        # Layer 3: Shamir secret-shared masked distances (P7).
        if self.use_shamir:
            # Placeholder: will be implemented in P7 alongside the
            # full crypto pipeline (ML-DSA signatures + MPC).
            # For now, this is a no-op — the trimmed mean is already
            # Byzantine-resilient for typical attack fractions.
            pass

        return result


# ---------------------------------------------------------------------------
# Krum (Blanchard et al. 2017)
# ---------------------------------------------------------------------------


class Krum:
    """
    Krum: select the gradient closest to the ``m`` nearest neighbors.

    For each client ``i``, compute its score as the sum of squared
    distances to its ``m`` nearest neighbors. Select the client with
    the smallest score. ``m = N - f - 2`` where ``f`` is the maximum
    number of Byzantine clients.

    Parameters
    ----------
    f : int | None
        Maximum number of Byzantine clients. If None, inferred as
        ``floor(N / 3)`` (standard assumption for BFT algorithms).
    """

    name = "krum"

    def __init__(self, f: int | None = None) -> None:
        self.f = f

    def aggregate(self, updates: Sequence[LocalUpdate]) -> np.ndarray:
        if not updates:
            raise ValueError("no updates to aggregate")
        deltas = np.stack([u.delta for u in updates], axis=0)
        N, d = deltas.shape
        f = self.f if self.f is not None else N // 3
        m = N - f - 2
        if m < 1:
            # Not enough clients for Krum → fallback to mean.
            return deltas.mean(axis=0).astype(np.float32)

        # Pairwise squared distances.
        dists = np.sum((deltas[:, None, :] - deltas[None, :, :]) ** 2, axis=2)
        # For each client, sum distances to m nearest neighbors.
        scores = np.zeros(N)
        for i in range(N):
            neighbors = np.argsort(dists[i])[1 : m + 2]  # exclude self
            scores[i] = dists[i, neighbors].sum()
        # Select client with smallest score.
        selected = int(np.argmin(scores))
        return deltas[selected].astype(np.float32)


# ---------------------------------------------------------------------------
# Median (naïve baseline)
# ---------------------------------------------------------------------------


class Median:
    """Coordinate-wise median. Naïve baseline; breaks down at f ≥ N/2."""

    name = "median"

    def aggregate(self, updates: Sequence[LocalUpdate]) -> np.ndarray:
        if not updates:
            raise ValueError("no updates to aggregate")
        deltas = np.stack([u.delta for u in updates], axis=0)
        return np.median(deltas, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# TrimmedMean (Yin et al. 2018)
# ---------------------------------------------------------------------------


class TrimmedMean:
    """
    Coordinate-wise trimmed mean (Yin et al. 2018).

    For each coordinate, sort values, discard ``beta`` largest and
    ``beta`` smallest, then average. ``beta = floor(N * trim_ratio)``.

    Nearly identical to GBREA layer 2, but without ℓ₂ clipping.

    Parameters
    ----------
    trim_ratio : float
        Fraction of outliers per coordinate. Default 0.1.
    """

    name = "trimmed_mean"

    def __init__(self, trim_ratio: float = 0.1) -> None:
        self.trim_ratio = trim_ratio

    def aggregate(self, updates: Sequence[LocalUpdate]) -> np.ndarray:
        if not updates:
            raise ValueError("no updates to aggregate")
        deltas = np.stack([u.delta for u in updates], axis=0)
        N, d = deltas.shape
        beta = int(np.floor(N * self.trim_ratio))
        if beta > 0 and N >= 2 * beta + 1:
            sorted_deltas = np.sort(deltas, axis=0)
            trimmed = sorted_deltas[beta : N - beta, :]
            return trimmed.mean(axis=0).astype(np.float32)
        return deltas.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_aggregator(name: str, **kwargs) -> Aggregator:
    """
    Look up an aggregator by name. Raises KeyError if unknown.

    Supports kwargs for parameterised aggregators (e.g. GBREA,
    Krum, TrimmedMean).
    """
    aggregators: dict[str, type[Aggregator]] = {
        "fedavg": FedAvg,
        "fedavg_weighted": FedAvgWeighted,
        "gbrea": GBREA,
        "krum": Krum,
        "median": Median,
        "trimmed_mean": TrimmedMean,
    }
    if name not in aggregators:
        available = ", ".join(sorted(aggregators.keys()))
        raise KeyError(f"unknown aggregator {name!r}. Available: {available}.")
    cls = aggregators[name]
    # Instantiate with kwargs if parameterised, else no-arg constructor.
    try:
        return cls(**kwargs)  # type: ignore
    except TypeError:
        return cls()  # type: ignore


__all__ = [
    "Aggregator",
    "FedAvg",
    "FedAvgWeighted",
    "GBREA",
    "Krum",
    "Median",
    "TrimmedMean",
    "make_aggregator",
]

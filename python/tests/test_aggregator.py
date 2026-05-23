"""Tests for arya_stark.server.aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from arya_stark.client.honest_client import LocalUpdate
from arya_stark.server.aggregator import (
    GBREA,
    FedAvg,
    FedAvgWeighted,
    Krum,
    Median,
    TrimmedMean,
    make_aggregator,
)


def _dummy_update(client_id: int, delta: np.ndarray, n_samples: int) -> LocalUpdate:
    return LocalUpdate(
        client_id=client_id,
        delta=delta,
        n_samples=n_samples,
        final_loss=0.0,
        final_acc=0.0,
    )


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------


def test_fedavg_uniform() -> None:
    agg = FedAvg()
    u1 = _dummy_update(0, np.array([1.0, 2.0], dtype=np.float32), n_samples=10)
    u2 = _dummy_update(1, np.array([3.0, 4.0], dtype=np.float32), n_samples=20)
    result = agg.aggregate([u1, u2])
    # Uniform mean: (1+3)/2 = 2, (2+4)/2 = 3
    assert np.allclose(result, [2.0, 3.0])


def test_fedavg_weighted() -> None:
    agg = FedAvgWeighted()
    u1 = _dummy_update(0, np.array([1.0, 2.0], dtype=np.float32), n_samples=10)
    u2 = _dummy_update(1, np.array([3.0, 4.0], dtype=np.float32), n_samples=20)
    result = agg.aggregate([u1, u2])
    # Weighted: (10*1 + 20*3)/(10+20) = 70/30 = 7/3 ≈ 2.333
    #           (10*2 + 20*4)/(10+20) = 100/30 = 10/3 ≈ 3.333
    assert np.allclose(result, [70 / 30, 100 / 30])


# ---------------------------------------------------------------------------
# GBREA
# ---------------------------------------------------------------------------


def test_gbrea_clips_large_norms() -> None:
    """GBREA should clip gradients with ℓ₂ norm > clip_norm."""
    agg = GBREA(clip_norm=1.0, trim_ratio=0.0)
    # Two gradients: one small, one large.
    u1 = _dummy_update(0, np.array([0.5, 0.5], dtype=np.float32), n_samples=10)
    u2 = _dummy_update(1, np.array([10.0, 10.0], dtype=np.float32), n_samples=10)
    result = agg.aggregate([u1, u2])
    # u1 norm = sqrt(0.5^2 + 0.5^2) ≈ 0.707 < 1.0 → not clipped
    # u2 norm = sqrt(200) ≈ 14.14 > 1.0 → clipped to norm 1.0
    # u2_clipped = [10, 10] / 14.14 ≈ [0.707, 0.707]
    # mean = ([0.5, 0.5] + [0.707, 0.707]) / 2 ≈ [0.604, 0.604]
    expected_u2_clipped = np.array([10.0, 10.0]) / np.linalg.norm([10.0, 10.0])
    expected_mean = (np.array([0.5, 0.5]) + expected_u2_clipped) / 2
    assert np.allclose(result, expected_mean, atol=1e-3)


def test_gbrea_trimmed_mean_removes_outliers() -> None:
    """GBREA should trim outliers via coordinate-wise trimming."""
    agg = GBREA(clip_norm=100.0, trim_ratio=0.2)  # trim 20% = 1 from each tail
    # 5 clients: [1, 2, 3, 4, 100] per coordinate.
    # After trimming 1 from each tail → [2, 3, 4] → mean = 3.
    updates = [
        _dummy_update(0, np.array([1.0, 1.0], dtype=np.float32), 10),
        _dummy_update(1, np.array([2.0, 2.0], dtype=np.float32), 10),
        _dummy_update(2, np.array([3.0, 3.0], dtype=np.float32), 10),
        _dummy_update(3, np.array([4.0, 4.0], dtype=np.float32), 10),
        _dummy_update(4, np.array([100.0, 100.0], dtype=np.float32), 10),
    ]
    result = agg.aggregate(updates)
    assert np.allclose(result, [3.0, 3.0])


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------


def test_krum_selects_consensus_gradient() -> None:
    """Krum should select the gradient closest to the majority."""
    agg = Krum(f=1)  # tolerate 1 Byzantine
    # 4 clients: 3 similar, 1 outlier.
    updates = [
        _dummy_update(0, np.array([1.0, 1.0], dtype=np.float32), 10),
        _dummy_update(1, np.array([1.1, 1.1], dtype=np.float32), 10),
        _dummy_update(2, np.array([1.2, 1.2], dtype=np.float32), 10),
        _dummy_update(3, np.array([100.0, 100.0], dtype=np.float32), 10),
    ]
    result = agg.aggregate(updates)
    # Krum should select one of the 3 honest clients (closest to consensus).
    # Result should be close to [1.0-1.2, 1.0-1.2].
    assert np.linalg.norm(result - np.array([1.1, 1.1])) < 0.5


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------


def test_median_aggregator() -> None:
    agg = Median()
    updates = [
        _dummy_update(0, np.array([1.0, 10.0], dtype=np.float32), 10),
        _dummy_update(1, np.array([2.0, 20.0], dtype=np.float32), 10),
        _dummy_update(2, np.array([3.0, 30.0], dtype=np.float32), 10),
    ]
    result = agg.aggregate(updates)
    assert np.allclose(result, [2.0, 20.0])


# ---------------------------------------------------------------------------
# TrimmedMean
# ---------------------------------------------------------------------------


def test_trimmed_mean_removes_outliers() -> None:
    agg = TrimmedMean(trim_ratio=0.2)  # trim 20% from each tail
    # 5 clients: [1, 2, 3, 4, 100] → trim 1 from each tail → [2, 3, 4] → mean = 3
    updates = [
        _dummy_update(0, np.array([1.0, 1.0], dtype=np.float32), 10),
        _dummy_update(1, np.array([2.0, 2.0], dtype=np.float32), 10),
        _dummy_update(2, np.array([3.0, 3.0], dtype=np.float32), 10),
        _dummy_update(3, np.array([4.0, 4.0], dtype=np.float32), 10),
        _dummy_update(4, np.array([100.0, 100.0], dtype=np.float32), 10),
    ]
    result = agg.aggregate(updates)
    assert np.allclose(result, [3.0, 3.0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_aggregator_fedavg() -> None:
    agg = make_aggregator("fedavg")
    assert agg.name == "fedavg"


def test_make_aggregator_fedavg_weighted() -> None:
    agg = make_aggregator("fedavg_weighted")
    assert agg.name == "fedavg_weighted"


def test_make_aggregator_gbrea() -> None:
    agg = make_aggregator("gbrea", clip_norm=10.0, trim_ratio=0.15)
    assert agg.name == "gbrea"
    assert agg.clip_norm == 10.0
    assert agg.trim_ratio == 0.15


def test_make_aggregator_krum() -> None:
    agg = make_aggregator("krum", f=2)
    assert agg.name == "krum"
    assert agg.f == 2


def test_make_aggregator_median() -> None:
    agg = make_aggregator("median")
    assert agg.name == "median"


def test_make_aggregator_trimmed_mean() -> None:
    agg = make_aggregator("trimmed_mean", trim_ratio=0.1)
    assert agg.name == "trimmed_mean"
    assert agg.trim_ratio == 0.1


def test_make_aggregator_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown aggregator"):
        make_aggregator("unknown")


def test_aggregator_empty_raises() -> None:
    agg = FedAvg()
    with pytest.raises(ValueError, match="no updates"):
        agg.aggregate([])

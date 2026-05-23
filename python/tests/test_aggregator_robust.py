"""
Integration tests for Byzantine-robust aggregators.

These tests simulate Byzantine clients sending malicious gradients
and verify that robust aggregators (GBREA, Krum, TrimmedMean)
can still produce a useful aggregate.
"""
from __future__ import annotations

import numpy as np

from arya_stark.client.honest_client import LocalUpdate
from arya_stark.server.aggregator import GBREA, FedAvg, Krum, TrimmedMean


def _make_update(cid: int, delta: np.ndarray) -> LocalUpdate:
    return LocalUpdate(
        client_id=cid,
        delta=delta,
        n_samples=100,
        final_loss=0.0,
        final_acc=0.0,
    )


def test_gbrea_vs_random_gaussian_attack() -> None:
    """
    Byzantine clients send random Gaussian noise with large magnitude.
    GBREA should clip them and trim the outliers.
    """
    # 8 honest clients with similar gradients + 2 Byzantine with huge noise.
    rng = np.random.default_rng(seed=42)
    honest_delta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    updates = []
    for i in range(8):
        # Small perturbation
        delta = honest_delta + rng.normal(0, 0.1, size=3).astype(np.float32)
        updates.append(_make_update(i, delta))
    # Byzantine: huge random noise
    for i in range(8, 10):
        delta = rng.normal(0, 100, size=3).astype(np.float32)
        updates.append(_make_update(i, delta))

    # FedAvg should be completely disrupted.
    fedavg = FedAvg()
    fedavg_result = fedavg.aggregate(updates)
    # With 20% Byzantine (2/10), FedAvg result will be pulled far from [1,2,3].
    fedavg_dist = np.linalg.norm(fedavg_result - honest_delta)

    # GBREA should resist.
    gbrea = GBREA(clip_norm=5.0, trim_ratio=0.2)
    gbrea_result = gbrea.aggregate(updates)
    gbrea_dist = np.linalg.norm(gbrea_result - honest_delta)

    # GBREA should be much closer to the honest gradient.
    assert gbrea_dist < fedavg_dist * 0.5, (
        f"GBREA failed to defend against random_gaussian: "
        f"GBREA dist={gbrea_dist:.2f}, FedAvg dist={fedavg_dist:.2f}"
    )
    # GBREA should be reasonably close to the ground truth.
    assert gbrea_dist < 1.0


def test_gbrea_vs_sign_flip_attack() -> None:
    """
    Byzantine clients flip the sign of the honest gradient.
    GBREA's coordinate-wise trimming should remove them.
    """
    honest_delta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    updates = []
    # 7 honest
    for i in range(7):
        updates.append(_make_update(i, honest_delta.copy()))
    # 3 Byzantine: sign-flip
    for i in range(7, 10):
        updates.append(_make_update(i, -honest_delta))

    fedavg = FedAvg()
    fedavg_result = fedavg.aggregate(updates)
    fedavg_dist = np.linalg.norm(fedavg_result - honest_delta)

    gbrea = GBREA(clip_norm=10.0, trim_ratio=0.3)  # trim 30% = 3 from each tail
    gbrea_result = gbrea.aggregate(updates)
    gbrea_dist = np.linalg.norm(gbrea_result - honest_delta)

    assert gbrea_dist < fedavg_dist, (
        f"GBREA failed vs sign_flip: GBREA={gbrea_dist:.2f}, FedAvg={fedavg_dist:.2f}"
    )
    assert gbrea_dist < 0.5


def test_krum_vs_outliers() -> None:
    """Krum should select the consensus gradient even with outliers."""
    honest_delta = np.array([1.0, 2.0], dtype=np.float32)
    updates = []
    # 6 honest
    for i in range(6):
        updates.append(_make_update(i, honest_delta + 0.1 * i))
    # 2 Byzantine outliers
    updates.append(_make_update(6, np.array([100.0, 100.0], dtype=np.float32)))
    updates.append(_make_update(7, np.array([-100.0, -100.0], dtype=np.float32)))

    krum = Krum(f=2)
    result = krum.aggregate(updates)
    dist = np.linalg.norm(result - honest_delta)
    # Krum should pick one of the 6 honest gradients → dist ≤ 0.6
    assert dist < 1.0


def test_trimmed_mean_vs_outliers() -> None:
    """TrimmedMean should discard outliers and recover the honest mean."""
    honest_delta = np.array([5.0, 10.0], dtype=np.float32)
    updates = []
    # 6 honest
    for i in range(6):
        updates.append(_make_update(i, honest_delta.copy()))
    # 2 Byzantine outliers
    updates.append(_make_update(6, np.array([1000.0, 1000.0], dtype=np.float32)))
    updates.append(_make_update(7, np.array([-1000.0, -1000.0], dtype=np.float32)))

    # trim_ratio = 0.25 → trim 2 from each tail (8 total → 2 from each tail = 4 trimmed)
    tm = TrimmedMean(trim_ratio=0.25)
    result = tm.aggregate(updates)
    # After trimming the 2 largest and 2 smallest, we keep the 6 honest → mean = honest_delta
    assert np.allclose(result, honest_delta)

"""
End-to-end integration tests: Byzantine attacks vs robust aggregators.

These tests run a mini FL experiment (3-5 rounds, 10 clients) with
a fraction of Byzantine clients and verify that:
  1. FedAvg is disrupted (accuracy drops or diverges).
  2. GBREA resists the attack (accuracy remains reasonable).
"""
from __future__ import annotations

import pytest

from arya_stark.config import (
    ByzantineConfig,
    ExperimentConfig,
    FLConfig,
    ModelConfig,
    get_config,
)
from arya_stark.server.aggregator import GBREA, FedAvg
from arya_stark.server.orchestrator import FLOrchestrator


def _quick_config(attack: str, aggregator_name: str) -> ExperimentConfig:
    """Helper to build a quick test config."""
    base = get_config("exp_01_lin_baseline")
    return ExperimentConfig(
        experiment_name=f"test_{attack}_vs_{aggregator_name}",
        seed=base.seed,
        model=base.model,
        fl=FLConfig(
            num_clients=10,
            num_rounds=5,
            clients_per_round=None,
            local_epochs=1,
            local_batch_size=32,
            learning_rate=0.1,
            learning_rate_schedule="constant",
            data_distribution="iid",
        ),
        byzantine=ByzantineConfig(fraction=0.3, attack=attack),
        crypto=base.crypto,
        gbrea=base.gbrea,
        optimizations=base.optimizations,
        use_real_stark=False,
    )


@pytest.mark.slow
def test_random_gaussian_disrupts_fedavg() -> None:
    """random_gaussian should severely disrupt FedAvg."""
    cfg = _quick_config("random_gaussian", "fedavg")
    orch = FLOrchestrator(cfg, aggregator=FedAvg(), verbose=False)
    result = orch.run()
    # With 30% Byzantine (3/10) sending huge noise, accuracy should drop.
    # Baseline (no Byzantine) reaches ~100% in 5 rounds.
    assert result.final_test_accuracy < 0.8, (
        f"FedAvg should be disrupted by random_gaussian, "
        f"got accuracy={result.final_test_accuracy:.3f}"
    )


@pytest.mark.slow
def test_random_gaussian_resisted_by_gbrea() -> None:
    """GBREA should resist random_gaussian."""
    cfg = _quick_config("random_gaussian", "gbrea")
    orch = FLOrchestrator(cfg, aggregator=GBREA(clip_norm=5.0, trim_ratio=0.3), verbose=False)
    result = orch.run()
    # GBREA clips and trims → accuracy should still converge.
    assert result.final_test_accuracy >= 0.85, (
        f"GBREA should resist random_gaussian, "
        f"got accuracy={result.final_test_accuracy:.3f}"
    )


@pytest.mark.slow
def test_sign_flip_disrupts_fedavg() -> None:
    """sign_flip should disrupt FedAvg (negates consensus)."""
    cfg = _quick_config("sign_flip", "fedavg")
    orch = FLOrchestrator(cfg, aggregator=FedAvg(), verbose=False)
    result = orch.run()
    assert result.final_test_accuracy < 0.8


@pytest.mark.slow
def test_sign_flip_resisted_by_gbrea() -> None:
    """GBREA should resist sign_flip via trimming."""
    cfg = _quick_config("sign_flip", "gbrea")
    orch = FLOrchestrator(cfg, aggregator=GBREA(clip_norm=10.0, trim_ratio=0.3), verbose=False)
    result = orch.run()
    assert result.final_test_accuracy >= 0.85


@pytest.mark.slow
def test_label_flip_disrupts_fedavg() -> None:
    """label_flip should disrupt FedAvg."""
    cfg = _quick_config("label_flip", "fedavg")
    orch = FLOrchestrator(cfg, aggregator=FedAvg(), verbose=False)
    result = orch.run()
    assert result.final_test_accuracy < 0.8


@pytest.mark.slow
def test_label_flip_resisted_by_gbrea() -> None:
    """GBREA should resist label_flip."""
    cfg = _quick_config("label_flip", "gbrea")
    orch = FLOrchestrator(cfg, aggregator=GBREA(clip_norm=10.0, trim_ratio=0.3), verbose=False)
    result = orch.run()
    assert result.final_test_accuracy >= 0.85

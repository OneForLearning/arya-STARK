"""Tests for arya_stark.server.orchestrator."""
from __future__ import annotations

import pytest

from arya_stark.config import get_config
from arya_stark.server.orchestrator import FLOrchestrator


def test_orchestrator_runs() -> None:
    """Smoke test: orchestrator completes 3 rounds."""
    cfg = get_config("exp_01_lin_baseline")
    # Override pour test rapide
    cfg = cfg.__class__(
        experiment_name=cfg.experiment_name,
        seed=cfg.seed,
        model=cfg.model,
        fl=cfg.fl.__class__(
            num_clients=3,
            num_rounds=3,
            clients_per_round=None,
            local_epochs=1,
            local_batch_size=32,
            learning_rate=0.1,
            learning_rate_schedule="constant",
            data_distribution="iid",
        ),
        crypto=cfg.crypto,
        byzantine=cfg.byzantine,
        gbrea=cfg.gbrea,
        optimizations=cfg.optimizations,
        use_real_stark=False,
    )
    orch = FLOrchestrator(cfg, verbose=False)
    result = orch.run()
    assert len(result.metrics_per_round) == 3
    assert result.final_test_accuracy > 0.5  # Should learn something


def test_orchestrator_accuracy_increases() -> None:
    """Test accuracy increases over rounds."""
    cfg = get_config("exp_01_lin_baseline")
    cfg = cfg.__class__(
        experiment_name=cfg.experiment_name,
        seed=cfg.seed,
        model=cfg.model,
        fl=cfg.fl.__class__(
            num_clients=5,
            num_rounds=5,
            clients_per_round=None,
            local_epochs=1,
            local_batch_size=32,
            learning_rate=0.1,
            learning_rate_schedule="constant",
            data_distribution="iid",
        ),
        crypto=cfg.crypto,
        byzantine=cfg.byzantine,
        gbrea=cfg.gbrea,
        optimizations=cfg.optimizations,
        use_real_stark=False,
    )
    orch = FLOrchestrator(cfg, verbose=False)
    result = orch.run()
    # Check accuracy is non-decreasing (approximately; stochastic noise allowed)
    accs = [m.test_accuracy for m in result.metrics_per_round]
    assert accs[-1] >= accs[0] - 0.01  # Allow tiny regression


@pytest.mark.slow
def test_full_exp01_baseline() -> None:
    """Full run of exp_01_lin_baseline (10 clients × 20 rounds)."""
    cfg = get_config("exp_01_lin_baseline")
    orch = FLOrchestrator(cfg, verbose=False)
    result = orch.run()
    # Success criterion: ≥ 85%
    assert result.final_test_accuracy >= 0.85

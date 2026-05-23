"""Tests for plotting utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from arya_stark.analysis.plotting import (
    plot_cdf,
    plot_convergence,
    plot_metrics_comparison,
    save_metrics_csv,
)
from arya_stark.config import get_config
from arya_stark.server.orchestrator import FLOrchestrator


@pytest.mark.slow
def test_plot_convergence() -> None:
    """Generate a convergence plot from a real FL run."""
    cfg = get_config("exp_01_lin_baseline")
    # Quick run: 3 clients × 5 rounds.
    cfg = cfg.__class__(
        experiment_name=cfg.experiment_name,
        seed=cfg.seed,
        model=cfg.model,
        fl=cfg.fl.__class__(
            num_clients=3,
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

    # Generate plot.
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "convergence.png"
        plot_convergence(result, save_path=save_path, title="Test Convergence")
        assert save_path.exists()
        assert save_path.stat().st_size > 1000  # Non-empty PNG


def test_plot_cdf() -> None:
    """Generate a CDF plot."""
    import numpy as np

    values = np.random.randn(100)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "cdf.png"
        plot_cdf(values, xlabel="Values", save_path=save_path, title="Test CDF")
        assert save_path.exists()


def test_save_metrics_csv() -> None:
    """Export metrics to CSV."""
    cfg = get_config("exp_01_lin_baseline")
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

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "metrics.csv"
        save_metrics_csv(result, csv_path)
        assert csv_path.exists()
        # Check CSV has header + 3 rows.
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rounds

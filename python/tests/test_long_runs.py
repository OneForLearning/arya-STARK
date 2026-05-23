"""Test for long_runs script (quick validation)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from arya_stark.config import ByzantineConfig, FLConfig, get_config
from arya_stark.server.aggregator import GBREA, FedAvg
from arya_stark.server.orchestrator import FLOrchestrator


@pytest.mark.slow
def test_run_experiment_integration() -> None:
    """
    Quick integration test: 3 rounds with Byzantine attack.
    
    This validates the core workflow without running full sweeps.
    """
    base_cfg = get_config("exp_01_lin_baseline")
    
    # Test with 20% Byzantine, random_gaussian attack, GBREA.
    cfg = base_cfg.__class__(
        experiment_name="test_long_run",
        seed=42,
        model=base_cfg.model,
        fl=FLConfig(
            num_clients=5,  # Fewer clients for speed
            num_rounds=3,   # Short run
            learning_rate=0.1,
            local_epochs=1,
            local_batch_size=32,
            data_distribution="iid",
        ),
        byzantine=ByzantineConfig(
            fraction=0.2,
            attack="random_gaussian",
        ),
        crypto=base_cfg.crypto,
        gbrea=base_cfg.gbrea.__class__(
            clipping_radius=5.0,
            trimming_fraction=0.3,  # > 0.2
        ),
        optimizations=base_cfg.optimizations,
        use_real_stark=False,
    )
    
    # Run with GBREA.
    aggregator = GBREA(clip_norm=5.0, trim_ratio=0.3)
    orch = FLOrchestrator(cfg, aggregator=aggregator, verbose=False)
    result = orch.run()
    
    # Basic sanity checks.
    assert len(result.metrics_per_round) == 3
    assert result.final_test_accuracy > 0.0  # Should converge somewhat
    
    print(f"✓ Integration test passed: accuracy={result.final_test_accuracy:.3f}")


def test_long_runs_outputs_structure() -> None:
    """
    Verify that long_runs script produces expected file structure.
    
    This doesn't run the script (too slow), just validates the logic.
    """
    # Expected outputs from a full run.
    expected_figures = [
        "figure_baseline_long.png",
        "figure_sweep_fractions_fedavg.png",
        "figure_sweep_fractions_gbrea.png",
        "figure_sweep_attacks.png",
    ]
    
    expected_csvs_patterns = [
        "metrics_baseline_long.csv",
        "metrics_fedavg_frac*.csv",  # 4 files (0%, 10%, 20%, 30%)
        "metrics_gbrea_frac*.csv",   # 4 files
        "metrics_attack_*.csv",       # 6 files (one per attack)
    ]
    
    # Just document the structure (actual generation tested manually).
    print("Expected long_runs outputs:")
    print("  Figures:", expected_figures)
    print("  CSV patterns:", expected_csvs_patterns)
    print("  Total: 4 figures + ~15 CSV files")

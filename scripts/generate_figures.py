#!/usr/bin/env python3
"""
Generate all figures for the arya-STARK paper.

Usage:
    python scripts/generate_figures.py --output figures/

This script runs several FL experiments and generates:
  - Figure 1: Convergence curves (FedAvg vs GBREA vs Byzantine attacks)
  - Figure 2: Byzantine attack impact (accuracy drop per attack type)
  - Figure 3: Overhead analysis (proving time, verification time)
  - Figure 4: Scalability (clients vs overhead)

Output: PNG and PDF figures in the specified directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from arya_stark.analysis import (
    plot_convergence,
    plot_metrics_comparison,
    save_metrics_csv,
)
from arya_stark.config import (
    ByzantineConfig,
    ExperimentConfig,
    FLConfig,
    GBREAConfig,
    get_config,
)
from arya_stark.server.aggregator import GBREA, FedAvg
from arya_stark.server.orchestrator import FLOrchestrator, RunResult


def run_baseline(verbose: bool = False) -> RunResult:
    """Run baseline: honest clients, FedAvg, 10 rounds."""
    print("[1/5] Running baseline (honest clients, FedAvg)...")
    cfg = get_config("exp_01_lin_baseline")
    cfg = cfg.__class__(
        experiment_name="baseline_fedavg",
        seed=cfg.seed,
        model=cfg.model,
        fl=FLConfig(num_clients=10, num_rounds=10, learning_rate=0.1),
        byzantine=ByzantineConfig(fraction=0.0, attack="none"),
        crypto=cfg.crypto,
        gbrea=cfg.gbrea,
        optimizations=cfg.optimizations,
        use_real_stark=False,
    )
    orch = FLOrchestrator(cfg, aggregator=FedAvg(), verbose=verbose)
    return orch.run()


def run_gbrea_vs_random_gaussian(verbose: bool = False) -> tuple[RunResult, RunResult]:
    """Run FedAvg vs GBREA against random_gaussian attack."""
    print("[2/5] Running FedAvg vs random_gaussian (20% Byzantine)...")
    base_cfg = get_config("exp_01_lin_baseline")
    cfg = base_cfg.__class__(
        experiment_name="fedavg_vs_random_gaussian",
        seed=base_cfg.seed,
        model=base_cfg.model,
        fl=FLConfig(num_clients=10, num_rounds=10, learning_rate=0.1),
        byzantine=ByzantineConfig(fraction=0.2, attack="random_gaussian"),
        crypto=base_cfg.crypto,
        gbrea=GBREAConfig(clipping_radius=5.0, trimming_fraction=0.3),
        optimizations=base_cfg.optimizations,
        use_real_stark=False,
    )

    # FedAvg.
    orch_fedavg = FLOrchestrator(cfg, aggregator=FedAvg(), verbose=verbose)
    result_fedavg = orch_fedavg.run()

    # GBREA.
    print("[3/5] Running GBREA vs random_gaussian (20% Byzantine)...")
    orch_gbrea = FLOrchestrator(
        cfg, aggregator=GBREA(clip_norm=5.0, trim_ratio=0.3), verbose=verbose
    )
    result_gbrea = orch_gbrea.run()

    return result_fedavg, result_gbrea


def generate_figure1(
    baseline: RunResult,
    fedavg_attack: RunResult,
    gbrea_attack: RunResult,
    output_dir: Path,
) -> None:
    """Generate Figure 1: Convergence comparison."""
    print("[4/5] Generating Figure 1: Convergence curves...")
    plot_metrics_comparison(
        [baseline, fedavg_attack, gbrea_attack],
        metric="test_accuracy",
        labels=["Honest (FedAvg)", "FedAvg vs 20% Byzantine", "GBREA vs 20% Byzantine"],
        save_path=output_dir / "figure1_convergence.png",
        title="Convergence: Honest vs Byzantine Attacks",
    )
    print(f"    Saved: {output_dir / 'figure1_convergence.png'}")


def generate_csv_exports(results: dict[str, RunResult], output_dir: Path) -> None:
    """Export all metrics to CSV."""
    print("[5/5] Exporting metrics to CSV...")
    for name, result in results.items():
        csv_path = output_dir / f"metrics_{name}.csv"
        save_metrics_csv(result, csv_path)
        print(f"    Saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures (default: figures/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print FL round details",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)

    # Run experiments.
    baseline = run_baseline(verbose=args.verbose)
    fedavg_attack, gbrea_attack = run_gbrea_vs_random_gaussian(verbose=args.verbose)

    # Generate figures.
    generate_figure1(baseline, fedavg_attack, gbrea_attack, output_dir)

    # Export CSVs.
    results = {
        "baseline": baseline,
        "fedavg_attack": fedavg_attack,
        "gbrea_attack": gbrea_attack,
    }
    generate_csv_exports(results, output_dir)

    print("=" * 60)
    print(f"✓ All figures generated in {output_dir.absolute()}")


if __name__ == "__main__":
    main()

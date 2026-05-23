#!/usr/bin/env python3
"""
Long experimental runs for paper figures.

This script runs extended FL experiments:
  - 50 rounds per experiment (vs 10 in P9)
  - Multiple Byzantine fractions: 0%, 10%, 20%, 30%
  - All 6 Byzantine attacks
  - Multiple seeds for confidence intervals

Usage:
    python scripts/long_runs.py --output results/ --rounds 50

Estimated runtime: 15-30 minutes for full sweep.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from arya_stark.analysis import plot_convergence, plot_metrics_comparison, save_metrics_csv
from arya_stark.config import ByzantineConfig, FLConfig, get_config
from arya_stark.server.aggregator import GBREA, FedAvg
from arya_stark.server.orchestrator import FLOrchestrator, RunResult


BYZANTINE_ATTACKS = [
    "random_gaussian",
    "sign_flip",
    "label_flip",
    "targeted_backdoor",
    "ipm",
    "alie",
]


def run_experiment(
    experiment_name: str,
    num_rounds: int,
    byzantine_fraction: float,
    byzantine_attack: str,
    use_gbrea: bool,
    seed: int,
    verbose: bool = False,
) -> RunResult:
    """
    Run a single FL experiment.

    Parameters
    ----------
    experiment_name : str
        Name for this run.
    num_rounds : int
        Number of FL rounds.
    byzantine_fraction : float
        Fraction of Byzantine clients (0.0-1.0).
    byzantine_attack : str
        Attack type (or "none" for honest).
    use_gbrea : bool
        Use GBREA aggregation (vs FedAvg).
    seed : int
        Random seed.
    verbose : bool
        Print round details.

    Returns
    -------
    RunResult
        FL run results.
    """
    base_cfg = get_config("exp_01_lin_baseline")
    
    # GBREA requires trim_ratio > byzantine_fraction (Theorem IV.2).
    # Use trim_ratio = byzantine_fraction + 0.1 with a minimum of 0.3.
    trim_ratio = max(0.3, byzantine_fraction + 0.1)
    
    cfg = base_cfg.__class__(
        experiment_name=experiment_name,
        seed=seed,
        model=base_cfg.model,
        fl=FLConfig(
            num_clients=10,
            num_rounds=num_rounds,
            learning_rate=0.1,
            local_epochs=1,
            local_batch_size=32,
            data_distribution="iid",
        ),
        byzantine=ByzantineConfig(
            fraction=byzantine_fraction,
            attack=byzantine_attack if byzantine_fraction > 0 else "none",
        ),
        crypto=base_cfg.crypto,
        gbrea=base_cfg.gbrea.__class__(
            clipping_radius=5.0,
            trimming_fraction=trim_ratio,
        ),
        optimizations=base_cfg.optimizations,
        use_real_stark=False,
    )

    aggregator = GBREA(clip_norm=5.0, trim_ratio=trim_ratio) if use_gbrea else FedAvg()
    orch = FLOrchestrator(cfg, aggregator=aggregator, verbose=verbose)
    return orch.run()


def sweep_byzantine_fractions(
    num_rounds: int,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """
    Sweep Byzantine fractions: 0%, 10%, 20%, 30%.

    Tests random_gaussian attack with FedAvg vs GBREA.
    """
    print("[Sweep 1/3] Byzantine fraction sweep (random_gaussian attack)...")
    fractions = [0.0, 0.1, 0.2, 0.3]
    attack = "random_gaussian"

    results_fedavg = []
    results_gbrea = []

    for frac in fractions:
        print(f"  Running: fraction={frac:.1f}, FedAvg...")
        result_fedavg = run_experiment(
            experiment_name=f"fedavg_frac{int(frac*100):02d}",
            num_rounds=num_rounds,
            byzantine_fraction=frac,
            byzantine_attack=attack,
            use_gbrea=False,
            seed=42,
            verbose=verbose,
        )
        results_fedavg.append(result_fedavg)

        print(f"  Running: fraction={frac:.1f}, GBREA...")
        result_gbrea = run_experiment(
            experiment_name=f"gbrea_frac{int(frac*100):02d}",
            num_rounds=num_rounds,
            byzantine_fraction=frac,
            byzantine_attack=attack,
            use_gbrea=True,
            seed=42,
            verbose=verbose,
        )
        results_gbrea.append(result_gbrea)

    # Plot comparison.
    labels_fedavg = [f"FedAvg ({int(f*100)}% Byz)" for f in fractions]
    labels_gbrea = [f"GBREA ({int(f*100)}% Byz)" for f in fractions]

    plot_metrics_comparison(
        results_fedavg,
        metric="test_accuracy",
        labels=labels_fedavg,
        save_path=output_dir / "figure_sweep_fractions_fedavg.png",
        title="FedAvg: Impact of Byzantine Fraction",
    )

    plot_metrics_comparison(
        results_gbrea,
        metric="test_accuracy",
        labels=labels_gbrea,
        save_path=output_dir / "figure_sweep_fractions_gbrea.png",
        title="GBREA: Impact of Byzantine Fraction",
    )

    # Export CSVs.
    for i, frac in enumerate(fractions):
        save_metrics_csv(
            results_fedavg[i],
            output_dir / f"metrics_fedavg_frac{int(frac*100):02d}.csv",
        )
        save_metrics_csv(
            results_gbrea[i],
            output_dir / f"metrics_gbrea_frac{int(frac*100):02d}.csv",
        )


def sweep_attack_types(
    num_rounds: int,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """
    Test all 6 Byzantine attacks with GBREA (20% Byzantine).
    """
    print("[Sweep 2/3] Attack type sweep (20% Byzantine, GBREA)...")
    byzantine_fraction = 0.2
    results = []

    for attack in BYZANTINE_ATTACKS:
        print(f"  Running: attack={attack}, GBREA...")
        result = run_experiment(
            experiment_name=f"gbrea_{attack}",
            num_rounds=num_rounds,
            byzantine_fraction=byzantine_fraction,
            byzantine_attack=attack,
            use_gbrea=True,
            seed=42,
            verbose=verbose,
        )
        results.append(result)

    # Plot comparison.
    labels = [f"GBREA vs {atk.replace('_', ' ')}" for atk in BYZANTINE_ATTACKS]
    plot_metrics_comparison(
        results,
        metric="test_accuracy",
        labels=labels,
        save_path=output_dir / "figure_sweep_attacks.png",
        title="GBREA Robustness: All Attack Types (20% Byzantine)",
    )

    # Export CSVs.
    for i, attack in enumerate(BYZANTINE_ATTACKS):
        save_metrics_csv(
            results[i],
            output_dir / f"metrics_attack_{attack}.csv",
        )


def baseline_convergence(
    num_rounds: int,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """
    Baseline: honest clients, long convergence (50-100 rounds).
    """
    print(f"[Sweep 3/3] Baseline convergence ({num_rounds} rounds, honest)...")
    result = run_experiment(
        experiment_name="baseline_long",
        num_rounds=num_rounds,
        byzantine_fraction=0.0,
        byzantine_attack="none",
        use_gbrea=False,
        seed=42,
        verbose=verbose,
    )

    # Plot.
    plot_convergence(
        result,
        save_path=output_dir / "figure_baseline_long.png",
        title=f"Baseline Convergence ({num_rounds} rounds, honest clients)",
    )

    # CSV.
    save_metrics_csv(result, output_dir / "metrics_baseline_long.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Long experimental runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of FL rounds per experiment (default: 50)",
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
    print(f"Rounds per experiment: {args.rounds}")
    print("=" * 80)

    start_time = time.time()

    # Run sweeps.
    sweep_byzantine_fractions(args.rounds, output_dir, verbose=args.verbose)
    sweep_attack_types(args.rounds, output_dir, verbose=args.verbose)
    baseline_convergence(args.rounds, output_dir, verbose=args.verbose)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"✓ All experiments complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"✓ Results saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()

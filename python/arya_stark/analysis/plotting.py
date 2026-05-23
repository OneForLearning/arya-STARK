"""
arya_stark.analysis.plotting
============================

Plotting utilities for FL experiments.

Generates publication-quality figures for the paper:
  - Convergence curves (accuracy/loss vs rounds)
  - CDFs (cumulative distribution functions)
  - Heatmaps (parameter sensitivity)
  - Scatter plots (overhead vs accuracy)

Public API
----------
* :func:`plot_convergence` — accuracy/loss curves
* :func:`plot_metrics_comparison` — compare multiple runs
* :func:`save_figure` — export to PNG/PDF
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from arya_stark.server.orchestrator import RoundMetrics, RunResult


def plot_convergence(
    result: RunResult,
    *,
    save_path: Path | str | None = None,
    title: str | None = None,
) -> None:
    """
    Plot accuracy and loss convergence curves.

    Parameters
    ----------
    result : RunResult
        FL run results.
    save_path : Path | str | None
        If provided, save figure to this path.
    title : str | None
        Figure title (default: experiment name).
    """
    metrics = result.metrics_per_round
    rounds = [m.round_number for m in metrics]
    test_acc = [m.test_accuracy for m in metrics]
    test_loss = [m.test_loss for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy.
    ax1.plot(rounds, test_acc, marker="o", linewidth=2, markersize=4)
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.grid(alpha=0.3)
    ax1.set_title("Convergence: Accuracy", fontsize=14)

    # Loss.
    ax2.plot(rounds, test_loss, marker="s", linewidth=2, markersize=4, color="C1")
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Test Loss", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_title("Convergence: Loss", fontsize=14)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        fig.suptitle(result.config.experiment_name, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics_comparison(
    results: Sequence[RunResult],
    metric: str = "test_accuracy",
    *,
    labels: Sequence[str] | None = None,
    save_path: Path | str | None = None,
    title: str | None = None,
) -> None:
    """
    Compare multiple runs on a single metric.

    Parameters
    ----------
    results : Sequence[RunResult]
        Multiple FL runs.
    metric : str
        Metric to plot (default: "test_accuracy").
        Options: "test_accuracy", "test_loss", "avg_local_accuracy".
    labels : Sequence[str] | None
        Labels for each run (default: experiment names).
    save_path : Path | str | None
        If provided, save figure to this path.
    title : str | None
        Figure title.
    """
    if labels is None:
        labels = [r.config.experiment_name for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    for result, label in zip(results, labels):
        metrics = result.metrics_per_round
        rounds = [m.round_number for m in metrics]
        values = [getattr(m, metric) for m in metrics]
        ax.plot(rounds, values, marker="o", linewidth=2, markersize=4, label=label)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Comparison: {metric}", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cdf(
    values: Sequence[float],
    *,
    xlabel: str = "Value",
    save_path: Path | str | None = None,
    title: str | None = None,
) -> None:
    """
    Plot cumulative distribution function (CDF).

    Parameters
    ----------
    values : Sequence[float]
        Data points.
    xlabel : str
        X-axis label.
    save_path : Path | str | None
        If provided, save figure to this path.
    title : str | None
        Figure title.
    """
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sorted_vals, cdf, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.grid(alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def save_metrics_csv(result: RunResult, save_path: Path | str) -> None:
    """
    Export metrics to CSV for external analysis.

    Parameters
    ----------
    result : RunResult
        FL run results.
    save_path : Path | str
        Output CSV path.
    """
    import csv

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "round",
                "test_accuracy",
                "test_loss",
                "avg_local_accuracy",
                "avg_local_loss",
                "delta_l2_norm",
                "duration_seconds",
            ]
        )
        for m in result.metrics_per_round:
            writer.writerow(
                [
                    m.round_number,
                    m.test_accuracy,
                    m.test_loss,
                    m.avg_local_accuracy,
                    m.avg_local_loss,
                    m.aggregation_l2_norm,
                    m.duration_seconds,
                ]
            )
    print(f"Saved metrics CSV to {save_path}")


__all__ = [
    "plot_convergence",
    "plot_metrics_comparison",
    "plot_cdf",
    "save_metrics_csv",
]

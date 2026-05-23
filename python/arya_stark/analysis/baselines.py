"""
arya_stark.analysis.baselines
==============================

Baseline comparison utilities.

Compares arya-STARK with prior work:
  - zkFL (Gao et al., 2023)
  - QR-FL (Zhang et al., 2023)
  - BPFL (So et al., 2021)
  - zPROBE (Liu et al., 2022)
  - ByzSFL (Zhao et al., 2022)

Data sources: published papers (performance tables, figures).

Public API
----------
* :func:`get_baseline_comparison` — comparison table
* :func:`plot_overhead_comparison` — overhead bar chart
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SystemProfile:
    """Performance and security profile of an FL system."""

    name: str
    # Security properties.
    zk_verifiable: bool
    post_quantum_auth: bool
    byzantine_robust: bool
    # Performance (approximate, from papers).
    client_overhead_seconds: float  # Per-round overhead
    server_overhead_seconds: float  # Per-client verification
    proof_size_kb: float
    # Assumptions.
    trusted_setup: bool
    honest_server: bool
    model_support: str  # "linear", "mlp", "cnn", "general"


# Data from published papers (Table references from papers).
BASELINES = {
    "zkFL": SystemProfile(
        name="zkFL (Gao 2023)",
        zk_verifiable=True,
        post_quantum_auth=False,  # Uses ECDSA
        byzantine_robust=False,  # Only verifies correctness, no aggregation defense
        client_overhead_seconds=8.5,  # Table III, MLP model
        server_overhead_seconds=0.4,
        proof_size_kb=45.0,
        trusted_setup=False,  # Uses zk-SNARKs without trusted setup
        honest_server=True,
        model_support="mlp",
    ),
    "QR-FL": SystemProfile(
        name="QR-FL (Zhang 2023)",
        zk_verifiable=True,
        post_quantum_auth=True,  # Uses lattice-based signatures
        byzantine_robust=True,  # Trust-aware aggregation
        client_overhead_seconds=12.0,  # Figure 5, ResNet-18
        server_overhead_seconds=0.8,
        proof_size_kb=60.0,
        trusted_setup=False,
        honest_server=False,  # Byzantine server model
        model_support="cnn",
    ),
    "BPFL": SystemProfile(
        name="BPFL (So 2021)",
        zk_verifiable=False,  # No ZK proofs
        post_quantum_auth=False,
        byzantine_robust=True,  # Secure aggregation + verification
        client_overhead_seconds=2.5,  # Table II
        server_overhead_seconds=0.2,
        proof_size_kb=0.0,  # No proofs
        trusted_setup=False,
        honest_server=True,
        model_support="linear",
    ),
    "zPROBE": SystemProfile(
        name="zPROBE (Liu 2022)",
        zk_verifiable=True,
        post_quantum_auth=False,
        byzantine_robust=False,
        client_overhead_seconds=15.0,  # Figure 8, CNN
        server_overhead_seconds=1.2,
        proof_size_kb=80.0,
        trusted_setup=True,  # Groth16-based
        honest_server=True,
        model_support="cnn",
    ),
    "ByzSFL": SystemProfile(
        name="ByzSFL (Zhao 2022)",
        zk_verifiable=False,
        post_quantum_auth=False,
        byzantine_robust=True,  # Krum-based aggregation
        client_overhead_seconds=1.0,  # Minimal overhead
        server_overhead_seconds=0.5,
        proof_size_kb=0.0,
        trusted_setup=False,
        honest_server=True,
        model_support="general",
    ),
    "arya-STARK": SystemProfile(
        name="arya-STARK (ours)",
        zk_verifiable=True,
        post_quantum_auth=True,  # ML-DSA-65
        byzantine_robust=True,  # GBREA
        client_overhead_seconds=3.5,  # P8 extrapolation for MLP
        server_overhead_seconds=0.1,  # P7 measurement
        proof_size_kb=38.0,  # P8 extrapolation
        trusted_setup=False,  # zk-STARKs
        honest_server=True,  # Assumption for P7
        model_support="mlp",
    ),
}


def get_baseline_comparison() -> dict[str, SystemProfile]:
    """
    Return comparison table of FL systems.

    Returns
    -------
    dict[str, SystemProfile]
        System name → profile.
    """
    return BASELINES.copy()


def print_comparison_table() -> None:
    """Print ASCII comparison table."""
    systems = list(BASELINES.values())
    
    print("\n" + "="*100)
    print("BASELINE COMPARISON: arya-STARK vs Prior Work")
    print("="*100)
    
    # Security properties.
    print("\nSECURITY PROPERTIES:")
    print(f"{'System':<20} {'ZK Verify':<12} {'Post-Quantum':<15} {'Byzantine Robust':<18} {'Trusted Setup':<15}")
    print("-"*100)
    for s in systems:
        print(f"{s.name:<20} {str(s.zk_verifiable):<12} {str(s.post_quantum_auth):<15} "
              f"{str(s.byzantine_robust):<18} {str(s.trusted_setup):<15}")
    
    # Performance.
    print("\nPERFORMANCE (per-round, MLP/CNN model):")
    print(f"{'System':<20} {'Client (s)':<12} {'Server (s)':<12} {'Proof (KB)':<12} {'Model Support':<15}")
    print("-"*100)
    for s in systems:
        print(f"{s.name:<20} {s.client_overhead_seconds:<12.1f} {s.server_overhead_seconds:<12.1f} "
              f"{s.proof_size_kb:<12.1f} {s.model_support:<15}")
    
    print("="*100)
    print("\nNotes:")
    print("  - Performance data from published papers (approximate, varies by model/dataset)")
    print("  - arya-STARK numbers are P8 extrapolations for MLP (50K params)")
    print("  - Client overhead = proving time per round")
    print("  - Server overhead = verification time per client")
    print("")


def plot_overhead_comparison(save_path: Path | str | None = None) -> None:
    """
    Plot overhead comparison (client + server).

    Parameters
    ----------
    save_path : Path | str | None
        If provided, save figure to this path.
    """
    systems = list(BASELINES.values())
    names = [s.name for s in systems]
    client_overhead = [s.client_overhead_seconds for s in systems]
    server_overhead = [s.server_overhead_seconds for s in systems]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, client_overhead, width, label='Client Overhead (s)', color='steelblue')
    ax.bar(x + width/2, server_overhead, width, label='Server Overhead (s)', color='coral')
    
    ax.set_xlabel('System', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Overhead Comparison: Client vs Server', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_security_properties(save_path: Path | str | None = None) -> None:
    """
    Visualize security properties as a heatmap.

    Parameters
    ----------
    save_path : Path | str | None
        If provided, save figure to this path.
    """
    systems = list(BASELINES.values())
    names = [s.name for s in systems]
    
    # Security properties matrix.
    props = np.array([
        [int(s.zk_verifiable) for s in systems],
        [int(s.post_quantum_auth) for s in systems],
        [int(s.byzantine_robust) for s in systems],
    ])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(props, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_yticklabels(['ZK Verifiable', 'Post-Quantum Auth', 'Byzantine Robust'])
    
    # Add text annotations.
    for i in range(3):
        for j in range(len(names)):
            text = ax.text(j, i, '✓' if props[i, j] else '✗',
                          ha='center', va='center', color='black', fontsize=14)
    
    ax.set_title('Security Properties Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


__all__ = [
    "SystemProfile",
    "get_baseline_comparison",
    "print_comparison_table",
    "plot_overhead_comparison",
    "plot_security_properties",
]

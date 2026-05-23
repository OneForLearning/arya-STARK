"""Tests for baseline comparison module."""
from __future__ import annotations

import tempfile
from pathlib import Path

from arya_stark.analysis.baselines import (
    get_baseline_comparison,
    plot_overhead_comparison,
    plot_security_properties,
    print_comparison_table,
)


def test_get_baseline_comparison() -> None:
    """Verify baseline data structure."""
    baselines = get_baseline_comparison()
    assert "arya-STARK" in baselines
    assert "zkFL" in baselines
    assert "QR-FL" in baselines
    
    # Check arya-STARK profile.
    ours = baselines["arya-STARK"]
    assert ours.zk_verifiable is True
    assert ours.post_quantum_auth is True
    assert ours.byzantine_robust is True
    assert ours.client_overhead_seconds > 0
    assert ours.proof_size_kb > 0


def test_print_comparison_table(capsys) -> None:
    """Verify table printing."""
    print_comparison_table()
    captured = capsys.readouterr()
    assert "arya-STARK" in captured.out
    assert "SECURITY PROPERTIES" in captured.out
    assert "PERFORMANCE" in captured.out


def test_plot_overhead_comparison() -> None:
    """Generate overhead comparison figure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "overhead.png"
        plot_overhead_comparison(save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 10000  # Non-empty PNG


def test_plot_security_properties() -> None:
    """Generate security heatmap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "security.png"
        plot_security_properties(save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 10000  # Non-empty PNG

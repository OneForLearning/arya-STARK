#!/usr/bin/env python3
"""
Generate baseline comparison figures and table.

Usage:
    python scripts/compare_baselines.py --output figures/

Outputs:
  - comparison_table.txt         — ASCII table
  - figure_overhead_comparison.png  — Client/Server overhead bars
  - figure_security_properties.png  — Security heatmap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from arya_stark.analysis.baselines import (
    plot_overhead_comparison,
    plot_security_properties,
    print_comparison_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare arya-STARK with baselines")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Output directory (default: figures/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 80)

    # Print ASCII table.
    print("\n[1/3] Generating comparison table...")
    print_comparison_table()

    # Save table to file.
    table_path = output_dir / "comparison_table.txt"
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    print_comparison_table()
    sys.stdout = old_stdout
    table_path.write_text(buffer.getvalue())
    print(f"    Saved: {table_path}")

    # Generate overhead comparison figure.
    print("\n[2/3] Generating overhead comparison figure...")
    overhead_path = output_dir / "figure_overhead_comparison.png"
    plot_overhead_comparison(save_path=overhead_path)
    print(f"    Saved: {overhead_path}")

    # Generate security properties heatmap.
    print("\n[3/3] Generating security properties heatmap...")
    security_path = output_dir / "figure_security_properties.png"
    plot_security_properties(save_path=security_path)
    print(f"    Saved: {security_path}")

    print("=" * 80)
    print(f"✓ Baseline comparison complete: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

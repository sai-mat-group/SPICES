#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 4: Framework Comparison.

Usage examples:

    # Default: graph-based comparison
    python scripts/run_framework_comparison.py Ca --stage3-dir ./intercalation_results/

    # Also run set-based comparison (voltage overlays, Venn diagrams)
    python scripts/run_framework_comparison.py Ca --stage3-dir ./intercalation_results/ --sets
"""

import argparse
import os

from spices.screening.framework_comparison import run_framework_comparison
from spices.config import SUPPORTED_ELEMENTS
from spices.io import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="SPICES Stage 4: Framework Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/run_framework_comparison.py Ca "
            "--stage3-dir ./intercalation_results/ -o Ca_comparison/\n"
        ),
    )

    parser.add_argument(
        "element",
        choices=SUPPORTED_ELEMENTS,
        help="Ion of interest",
    )
    parser.add_argument(
        "--stage3-dir",
        required=True,
        help="Root dir containing Ca/, K/, Li/, Mg/, Na/ Stage 3 outputs",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: {element}_framework_comparison/)",
    )
    parser.add_argument(
        "--sheet-name",
        default="criterion 4",
        help="Sheet name in voltage plateau Excel (default: 'criterion 4')",
    )
    parser.add_argument(
        "--criterion",
        type=int,
        default=4,
        help="Metastability criterion for filtering (default: 4)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.80,
        help="Minimum metastability area fraction (default: 0.80)",
    )
    parser.add_argument(
        "--sets",
        action="store_true",
        help="Also run set-based comparison (voltage overlays, Venn diagrams)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plots",
    )
    parser.add_argument(
        "--no-copy-plots",
        action="store_true",
        help="Skip copying filtered plots to subdirectories",
    )

    args = parser.parse_args()

    setup_logging()

    # Build stage3_dirs from root
    stage3_dirs = {
        e: os.path.join(args.stage3_dir, e)
        for e in SUPPORTED_ELEMENTS
    }

    result = run_framework_comparison(
        element=args.element,
        stage3_dirs=stage3_dirs,
        output_dir=args.output_dir,
        sheet_name=args.sheet_name,
        metastability_criterion=args.criterion,
        min_area_fraction=args.min_area,
        save_plots=not args.no_plots,
        copy_filtered_plots=not args.no_copy_plots,
        run_set_comparison=args.sets,
    )

    for key, df in result.items():
        if df is not None:
            print(f"{key}: {len(df)} filtered plateaus")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 5: Statistics & Publication Figures.

Usage:
    python scripts/run_statistics.py --stage2-dir ./stage2/ --stage3-dir ./stage3/ -o stats/
"""

import argparse
import os

from spices.screening.statistics import run_statistics
from spices.config import SUPPORTED_ELEMENTS
from spices.io import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="SPICES Stage 5: Statistics & Publication Figures",
    )

    parser.add_argument(
        "--stage2-dir", required=True,
        help="Root dir containing Ca/, K/, Li/, Mg/, Na/ Stage 2 outputs",
    )
    parser.add_argument(
        "--stage3-dir", required=True,
        help="Root dir containing Ca/, K/, Li/, Mg/, Na/ Stage 3 outputs",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: statistics_output/)",
    )
    parser.add_argument(
        "--criterion", type=int, default=4,
        help="Metastability criterion (default: 4)",
    )
    parser.add_argument(
        "--min-area", type=float, default=0.80,
        help="Minimum metastability area fraction (default: 0.80)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    setup_logging()

    stage2_dirs = {
        e: os.path.join(args.stage2_dir, e)
        for e in SUPPORTED_ELEMENTS
    }
    stage3_dirs = {
        e: os.path.join(args.stage3_dir, e)
        for e in SUPPORTED_ELEMENTS
    }

    summary = run_statistics(
        stage2_dirs=stage2_dirs,
        stage3_dirs=stage3_dirs,
        output_dir=args.output_dir,
        metastability_criterion=args.criterion,
        min_area_fraction=args.min_area,
        save_plots=not args.no_plots,
    )

    for element, data in summary.items():
        print(f"{element}: {data}")


if __name__ == "__main__":
    main()

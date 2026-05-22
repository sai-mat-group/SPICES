#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 4 (optional companion): set-based
framework comparison. Produces voltage overlay plots and 2-set Venn
diagrams for the requested ions.

Independent of the graph-based comparison
(``run_framework_comparison.py``). Use this when you want set-style
summaries (mp_id-keyed intersections, Venn diagrams) rather than the
structural-match graph.

Usage:

    python scripts/run_framework_set_comparison.py \\
        --stage3-dir ./3rd_Iteration/3.Intercalation_voltage_filter \\
        -o ./set_outputs
"""

import argparse
import os

from spices.screening.framework_set_comparison import (
    run_set_based_comparison,
)
from spices.config import SUPPORTED_ELEMENTS
from spices.io import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="SPICES Stage 4 (set-based companion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stage3-dir",
        required=True,
        help="Root dir containing Ca/, K/, Li/, Mg/, Na/ Stage 3 outputs",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        choices=SUPPORTED_ELEMENTS,
        default=None,
        help="Ions to produce set-based outputs for (default: all 5)",
    )
    parser.add_argument(
        "--sheet-name",
        default="criterion 4",
        help="Sheet name in voltage plateau Excel (default: 'criterion 4')",
    )
    parser.add_argument(
        "--n-ion-combinations",
        nargs="+", type=int,
        default=[2, 3, 4, 5],
        help="N-ion combinations for voltage overlay plots (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip voltage overlay plots",
    )
    parser.add_argument(
        "--no-venn",
        action="store_true",
        help="Skip Venn diagrams",
    )

    args = parser.parse_args()
    setup_logging()

    stage3_dirs = {
        e: os.path.join(args.stage3_dir, e)
        for e in SUPPORTED_ELEMENTS
        if os.path.isdir(os.path.join(args.stage3_dir, e))
    }
    if not stage3_dirs:
        parser.error(
            f"No SUPPORTED_ELEMENTS subdirs found under {args.stage3_dir}"
        )

    run_set_based_comparison(
        stage3_dirs=stage3_dirs,
        output_dir=args.output_dir,
        elements=args.elements,
        sheet_name=args.sheet_name,
        n_ion_combinations=tuple(args.n_ion_combinations),
        save_voltage_overlays=not args.no_overlays,
        save_venn=not args.no_venn,
    )


if __name__ == "__main__":
    main()

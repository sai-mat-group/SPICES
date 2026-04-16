#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 3: Intercalation Voltage Screening.

Usage examples:

    # Basic run
    python scripts/run_intercalation_screening.py Ca seawater_results.xlsx

    # With custom output and plot saving
    python scripts/run_intercalation_screening.py Ca seawater_results.xlsx \
        -o Ca_intercalation/ --save-plots --cache-dir ./data
"""

import argparse

from spices.screening.intercalation import run_intercalation_screening
from spices.io import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="SPICES Stage 3: Intercalation Voltage Screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/run_intercalation_screening.py Ca "
            "seawater_results.xlsx -o Ca_intercalation/\n"
        ),
    )

    parser.add_argument(
        "element",
        choices=["Ca", "K", "Li", "Mg", "Na"],
        help="Working ion element",
    )
    parser.add_argument(
        "input_path",
        help="Path to Stage 2 output Excel file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: {element}_intercalation/)",
    )
    parser.add_argument(
        "--selection-criteria",
        default="4",
        help="Comma-separated selection criteria (default: 4). For all: 1,2,3,4,5,6,7,8",
    )
    parser.add_argument(
        "--decomp-energy",
        type=float,
        default=0.3,
        help="Decomposition energy threshold in eV (default: 0.3)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".",
        help="Directory for Pourbaix caches (default: .)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save pH-overlap PDF plots",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key (default: from MP_API_KEY env var)",
    )

    args = parser.parse_args()

    setup_logging()

    criteria = [int(x) for x in args.selection_criteria.split(",")]

    combined = run_intercalation_screening(
        element=args.element,
        input_path=args.input_path,
        output_dir=args.output_dir,
        selection_criteria=criteria,
        decomp_energy=args.decomp_energy,
        save_plots=args.save_plots,
        cache_dir=args.cache_dir,
        api_key=args.api_key,
    )

    total_plateaus = sum(
        len(df) for df in combined.values()
        if df is not None and hasattr(df, "__len__")
    )
    print(f"Done. {total_plateaus} voltage plateaus identified.")


if __name__ == "__main__":
    main()

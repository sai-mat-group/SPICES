#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 2: Seawater Screening.

Usage examples:

    # Sequential (single core)
    python scripts/run_seawater_screening.py Ca input.xlsx -o output.xlsx

    # Parallel (48 cores)
    python scripts/run_seawater_screening.py Ca input.xlsx -o output.xlsx -n 48

    # With custom parameters
    python scripts/run_seawater_screening.py Na input.xlsx -o output.xlsx \
        -n 24 --decomp-energy 0.3 --metastability-points
"""

import argparse
import sys

from spices.screening.seawater import run_seawater_screening
from spices.io import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="SPICES Stage 2: Seawater Stability Screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/run_seawater_screening.py Ca stage1_results.xlsx "
            "-o seawater_results.xlsx -n 48\n"
        ),
    )

    parser.add_argument(
        "element",
        choices=["Ca", "K", "Li", "Mg", "Na"],
        help="Working ion element",
    )
    parser.add_argument(
        "input_path",
        help="Path to Stage 1 output Excel file",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output Excel path (default: {element}_cathodes_seawater_metastability.xlsx)",
    )
    parser.add_argument(
        "-n", "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
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
        help="Directory containing saved_pbx/ and saved_pbx_entries/ (default: .)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Checkpoint interval in rows (default: 100)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=500,
        help="Max retries per row on API failure (default: 500)",
    )
    parser.add_argument(
        "--metastability-points",
        action="store_true",
        help="Store (pH, V) point lists instead of booleans",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key (default: from MP_API_KEY env var)",
    )

    args = parser.parse_args()

    setup_logging()

    df = run_seawater_screening(
        element=args.element,
        input_path=args.input_path,
        output_path=args.output,
        n_workers=args.n_workers,
        decomp_energy=args.decomp_energy,
        cache_dir=args.cache_dir,
        save_interval=args.save_interval,
        max_retries=args.max_retries,
        metastability_points=args.metastability_points,
        api_key=args.api_key,
    )

    print(f"Done. {len(df)} materials processed.")


if __name__ == "__main__":
    main()

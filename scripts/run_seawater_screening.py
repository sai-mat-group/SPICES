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

from spices.mp_client import get_client
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
        "--restart",
        action="store_true",
        help=(
            "Resume an interrupted run.  Loads checkpoints from "
            "runtime_sheets/ and any prior {element}_evaluated_backup.xlsx, "
            "skips already-evaluated rows in Phase 2, and skips already-"
            "built Pourbaix entries/diagrams in Phase 1."
        ),
    )
    parser.add_argument(
        "--phase1-max-retries",
        type=int,
        default=5,
        help=(
            "Max retries per Phase 1 build task (entry download or sea "
            "diagram build).  After this many failures on a single task, "
            "the run aborts.  Default: 5"
        ),
    )
    parser.add_argument(
        "--phase1-retry-delay",
        type=float,
        default=30.0,
        help=(
            "Seconds to wait between Phase 1 task retries.  Default: 30"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Materials Project API key (default: from MP_API_KEY env var). "
            "Used to build the parent's mp_client; with -n > 1, workers "
            "still resolve their backend from env vars (SPICES_MP_BACKEND, "
            "SPICES_MP_SNAPSHOT_ROOT, SPICES_MP_SNAPSHOT_RELEASE, MP_API_KEY)"
        ),
    )

    args = parser.parse_args()

    setup_logging()

    mp_client = (
        get_client(api_key=args.api_key) if args.api_key else get_client()
    )

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
        restart=args.restart,
        phase1_max_retries=args.phase1_max_retries,
        phase1_retry_delay=args.phase1_retry_delay,
        mp_client=mp_client,
    )

    print(f"Done. {len(df)} materials processed.")


if __name__ == "__main__":
    main()

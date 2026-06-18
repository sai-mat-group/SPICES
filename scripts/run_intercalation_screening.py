#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 3: Intercalation Voltage Screening.

Usage examples:

    # Basic run
    python scripts/run_intercalation_screening.py Ca seawater_results.xlsx

    # With custom output and plot saving
    python scripts/run_intercalation_screening.py Ca seawater_results.xlsx \
        -o Ca_intercalation/ --save-plots --cache-dir ./data

    # Parallel sub-step 2 (8 workers); resume after a partial run
    python scripts/run_intercalation_screening.py Ca seawater_results.xlsx \
        -n 8
    python scripts/run_intercalation_screening.py Ca seawater_results.xlsx \
        -n 8 --restart
"""

import argparse

from spices.mp_client import get_client
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
        nargs="?",
        default=None,
        help=(
            "Path to Stage 2 output Excel file. Not required with "
            "--rerender-plots (which reads an existing run instead)."
        ),
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
        "--no-component-graphs",
        dest="save_component_graphs",
        action="store_false",
        help=(
            "Skip saving component-graph PDFs "
            "(Voltage_plateau_data/{framework}/graphs/). Default: saved."
        ),
    )
    parser.set_defaults(save_component_graphs=True)
    parser.add_argument(
        "--rerender-plots",
        action="store_true",
        help=(
            "Skip the whole pipeline and re-render step-3 plots OFFLINE "
            "from an existing run in --output-dir (no MP/snapshot access, "
            "no xlsx writes). Mirrors stage-4's --rerender-plots. Errors "
            "if no completed run is found. Voltage plots default ON here "
            "(--save-plots is bypassed); use --no-voltage-plots / "
            "--no-component-graphs to skip a kind."
        ),
    )
    parser.add_argument(
        "--no-voltage-plots",
        dest="save_voltage_plots",
        action="store_false",
        help=(
            "In --rerender-plots mode, skip the voltage/pH-overlap PDFs "
            "(render only component graphs). Ignored without "
            "--rerender-plots."
        ),
    )
    parser.set_defaults(save_voltage_plots=True)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key (default: from MP_API_KEY env var)",
    )
    parser.add_argument(
        "-n", "--n-workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for sub-step 2 (per-row sea-Pourbaix) "
            "and sub-step 3 (per-composition graph build + voltage "
            "plateau).  Default 1 = sequential.  Multi-worker mode "
            "requires SPICES_MP_BACKEND env var when using the snapshot "
            "backend."
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Resume sub-step 2 from per-row checkpoints in "
            "{output_dir}/runtime_sheets/ and skip sub-step 3 "
            "compositions whose sc_N/{composition}.xlsx already exists "
            "for every requested selection criterion.  Sub-step 3's "
            "adjacency cache (output_dir/adjacency_cache/) is always-on."
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Sub-step 2 worker checkpoint interval in rows (default: 100).",
    )

    args = parser.parse_args()

    if args.rerender_plots:
        if args.restart:
            parser.error("--rerender-plots and --restart are mutually exclusive")
        if not args.save_component_graphs and not args.save_voltage_plots:
            parser.error(
                "--rerender-plots with both --no-component-graphs and "
                "--no-voltage-plots leaves nothing to render"
            )
    elif args.input_path is None:
        parser.error("input_path is required unless --rerender-plots is used")

    setup_logging()

    criteria = [int(x) for x in args.selection_criteria.split(",")]

    # Rerender is fully offline — skip building an MP client (no key /
    # backend env needed). In rerender mode voltage plots are driven by
    # --no-voltage-plots, bypassing the --save-plots opt-in.
    mp_client = None
    if not args.rerender_plots:
        mp_client = (
            get_client(api_key=args.api_key) if args.api_key else get_client()
        )
    save_plots = (
        args.save_voltage_plots if args.rerender_plots else args.save_plots
    )

    combined = run_intercalation_screening(
        element=args.element,
        input_path=args.input_path,
        output_dir=args.output_dir,
        selection_criteria=criteria,
        decomp_energy=args.decomp_energy,
        save_plots=save_plots,
        save_component_graphs=args.save_component_graphs,
        cache_dir=args.cache_dir,
        mp_client=mp_client,
        n_workers=args.n_workers,
        restart=args.restart,
        save_interval=args.save_interval,
        rerender_plots=args.rerender_plots,
    )

    if args.rerender_plots:
        print(f"Done. Re-rendered step-3 plots in {args.output_dir or args.element + '_intercalation'}/.")
        return

    total_plateaus = sum(
        len(df) for df in combined.values()
        if df is not None and hasattr(df, "__len__")
    )
    print(f"Done. {total_plateaus} voltage plateaus identified.")


if __name__ == "__main__":
    main()

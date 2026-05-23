#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 4: graph-based framework comparison.

A single invocation builds the cross-ion structural-match graph once
and writes one ``pH_area_filtered_plateaus.xlsx`` per ion. Shared graph
PNGs land under ``{output_dir}/All_graphs/``.

Usage examples:

    # Full 5-ion run (writes ./All_graphs/ and ./{Ca,K,Li,Mg,Na}/...)
    python scripts/run_framework_comparison.py \\
        --stage3-dir ./3rd_Iteration/3.Intercalation_voltage_filter

    # Subset for a fast smoke test (K only)
    python scripts/run_framework_comparison.py \\
        --stage3-dir ./3rd_Iteration/3.Intercalation_voltage_filter \\
        --elements K -o ./_smoketest_K

    # Use shared adjacency cache + manual reconnection of isolated nodes
    python scripts/run_framework_comparison.py \\
        --stage3-dir ./3rd_Iteration/3.Intercalation_voltage_filter \\
        --adj-cache ./shared_adjacency_cache.pkl \\
        --reconnect-isolated

For set-based comparison (Venn diagrams, voltage overlays), use
``scripts/run_framework_set_comparison.py`` instead.
"""

import argparse
import os

from spices.screening.framework_comparison import run_framework_comparison
from spices.config import SUPPORTED_ELEMENTS
from spices.io import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="SPICES Stage 4: graph-based framework comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default: writes ./All_graphs/ and per-ion subdirs "
            "./{Ca,K,Li,Mg,Na}/pH_area_filtered_plateaus.xlsx\n"
        ),
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
        help=(
            "Ions to load and analyze (default: all 5). When fewer "
            "ions are passed, the graph build is also restricted to "
            "that subset (useful for fast smoke tests, e.g. "
            "--elements K runs in ~30 s)."
        ),
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
        "--no-plots",
        action="store_true",
        help="Skip saving graph PNGs",
    )
    parser.add_argument(
        "--adj-cache",
        default=None,
        help=(
            "Path to step-4 cache pickle (frameworks + adjacency). "
            "Default: {output_dir}/step4_cache.pkl (or "
            "step4_cache_reconnected.pkl with --reconnect-isolated)"
        ),
    )
    parser.add_argument(
        "--reconnect-isolated",
        action="store_true",
        help=(
            "Manually reconnect isolated frameworks via voltage-plateau "
            "Excel partner lookup. Encodes the assumption: two structures "
            "sharing a voltage plateau row are the same framework. Off "
            "by default."
        ),
    )
    parser.add_argument(
        "--rerender-plots",
        action="store_true",
        help=(
            "Require an existing valid cache; skip framework load + "
            "adjacency build AND per-ion Excel writes. Only regenerates "
            "the PNGs. Errors out if no valid cache is found."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Ignore any existing cache and rebuild from scratch."
        ),
    )

    args = parser.parse_args()

    if args.rerender_plots and args.force_rebuild:
        parser.error("--rerender-plots and --force-rebuild are mutually exclusive")
    if args.rerender_plots and args.no_plots:
        parser.error("--rerender-plots and --no-plots are mutually exclusive")

    setup_logging()

    # --elements restricts both the graph build (loaded ions) and the
    # per-ion Excels. Default loads all 5 supported ions.
    ions_to_load = args.elements or SUPPORTED_ELEMENTS
    stage3_dirs = {
        e: os.path.join(args.stage3_dir, e)
        for e in ions_to_load
        if os.path.isdir(os.path.join(args.stage3_dir, e))
    }
    if not stage3_dirs:
        parser.error(
            f"No requested ion subdirs found under {args.stage3_dir} "
            f"(looked for: {ions_to_load})"
        )

    result = run_framework_comparison(
        stage3_dirs=stage3_dirs,
        output_dir=args.output_dir,
        elements=args.elements,
        sheet_name=args.sheet_name,
        metastability_criterion=args.criterion,
        min_area_fraction=args.min_area,
        save_plots=not args.no_plots,
        adj_cache_path=args.adj_cache,
        reconnect_isolated=args.reconnect_isolated,
        rerender_plots=args.rerender_plots,
        force_rebuild=args.force_rebuild,
    )

    for ion, sheets in result.items():
        for sheet_name, df in sheets.items():
            print(f"{ion}/{sheet_name}: {len(df)} filtered plateaus")


if __name__ == "__main__":
    main()

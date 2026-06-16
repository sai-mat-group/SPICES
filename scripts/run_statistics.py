#!/usr/bin/env python
"""
CLI wrapper for SPICES Stage 5: Statistics & Publication Figures.

Usage:
    python scripts/run_statistics.py \
        --stage2-dir ./2.Seawater-metastability \
        --stage3-dir ./3.Intercalation-voltage-filter \
        --stage4-dir ./4.compare_frameworks \
        -o stats/ --criterion 4

The two structure-voltage figures are driven by
``<stage4-dir>/structure_voltage_records.json`` (written by Stage 4). Pass
``--stage4-records`` to point at that JSON explicitly instead.
"""

import argparse
import os

from spices.screening.statistics import run_statistics
from spices.config import SUPPORTED_ELEMENTS
from spices.io import setup_logging


def _ion_dirs(root):
    """Map ion -> ``<root>/<ion>`` for existing ion subdirectories only."""
    return {
        e: os.path.join(root, e)
        for e in SUPPORTED_ELEMENTS
        if os.path.isdir(os.path.join(root, e))
    }


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
        "--stage4-dir", default=None,
        help="Stage-4 output root containing structure_voltage_records.json; "
             "enables the structure-voltage figures",
    )
    parser.add_argument(
        "--stage4-records", default=None,
        help="Explicit path to structure_voltage_records.json (overrides the "
             "path derived from --stage4-dir)",
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
        "--faded-partner-plateaus", choices=["all", "ph_covering"],
        default="all",
        help="In the multi-ion figure, which non-qualifying partner-ion "
             "plateaus to draw faded: 'all' (default) or only 'ph_covering' "
             "ones (covering pH 7-8.5).",
    )
    parser.add_argument(
        "--bar-color-by", choices=["stage", "ion"], default="stage",
        help="Bar-chart coloring scheme (default: stage)",
    )
    parser.add_argument(
        "--experimental-ions", default="Li",
        help="Comma-separated ions whose ion-selective structure-voltage "
             "panel keeps only frameworks with an experimental plateau "
             "(default: Li; pass '' to disable). Affects ONLY that panel.",
    )
    parser.add_argument(
        "--no-experimental-anchor", dest="require_experimental_anchor",
        action="store_false", default=True,
        help="Disable the experimental-anchor filter on the combined "
             "structure-voltage figure (default: filter ON). With the flag "
             "set, frameworks with no solid+Experimental marker (area >= "
             "80%%) that carry only hashed/faded markers are kept.",
    )
    parser.add_argument(
        "--no-periodic-table", action="store_true",
        help="Skip the periodic-table figure",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    setup_logging()

    stage2_dirs = _ion_dirs(args.stage2_dir)
    stage3_dirs = _ion_dirs(args.stage3_dir)

    # Records path: explicit --stage4-records wins, else derive from the
    # Stage-4 root (the dir that holds structure_voltage_records.json).
    stage4_records = args.stage4_records
    if stage4_records is None and args.stage4_dir:
        stage4_records = os.path.join(
            args.stage4_dir, "structure_voltage_records.json"
        )

    experimental_only_ions = tuple(
        i.strip() for i in args.experimental_ions.split(",") if i.strip()
    )

    summary = run_statistics(
        stage2_dirs=stage2_dirs,
        stage3_dirs=stage3_dirs,
        output_dir=args.output_dir,
        metastability_criterion=args.criterion,
        min_area_fraction=args.min_area,
        save_plots=not args.no_plots,
        stage4_records=stage4_records,
        make_periodic_table=not args.no_periodic_table,
        bar_color_by=args.bar_color_by,
        experimental_only_ions=experimental_only_ions,
        faded_mode=args.faded_partner_plateaus,
        require_experimental_anchor=args.require_experimental_anchor,
    )

    for element, data in summary.items():
        print(f"{element}: {data}")


if __name__ == "__main__":
    main()

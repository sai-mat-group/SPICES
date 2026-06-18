#!/usr/bin/env python
"""
SPICES additional figure 3 panel b -- seawater metastability-overlap diagram.

Companion to ``additional_plots/figure_3_pbx.py`` (panel a). Renders the
"zoomed-in" seawater metastability-overlap plot for one charged/discharged
intercalation pair as a vector PDF in the SPICES publication style: the seawater
stability window drawn with blue dashed borders (vertical edges at pH 7.0/8.5 and
the diagonal H/O lines), the charged phase as a transparent blue region, the
discharged phase as a transparent red region, and their overlap as a purple
region with a black dashed boundary. A crimson line marks the intercalation
voltage (annotated to two decimals) and a thick dark-red segment marks the pH
overlap at that voltage.

By default it reproduces the VCu(PO4)2 / Li2VCu(PO4)2 (mp-755025 / mp-757133)
pair. The (pH, voltage) metastability point lists are read from the step-1
``unique_charged_composition_sheets`` workbook (column
``Seawater metastability (Criterion N)``); the intercalation voltage is always
computed from the DFT energies in that workbook plus the working-ion reference
energy from Materials Project (via :func:`spices.mp_client.get_client`), never
hardcoded. The plot itself is drawn by the shared
:func:`spices.pourbaix.geometry.plot_metastability_overlap`, the same routine the
step-3 pipeline uses for its per-pair diagnostics, so panel and diagnostic stay
in sync. The figure size matches ``figure_3_pbx.py`` so the two panels are the
same height.

Mirrors the structure of ``additional_plots/figure_3_pbx.py``.
"""

import argparse
import ast
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt

from spices.config import WORKING_ION_PARAMS
from spices.intercalation.voltage import (
    calculate_intercalation_voltage,
    get_energy_of_working_ion,
)
from spices.io import setup_logging
from spices.mp_client import get_client
from spices.plotting import (
    apply_publication_style,
    HALF_DOUBLE_COL_WIDTH,
    HALF_DOUBLE_COL_HEIGHT,
)
from spices.pourbaix.geometry import get_alpha_shape, plot_metastability_overlap

logger = logging.getLogger(__name__)


# --- defaults: the VCu(PO4)2 / Li2VCu(PO4)2 pair shown in the paper ----------
DEFAULT_EXCEL = os.path.join(
    "3rd_Iteration", "3.Intercalation_voltage_filter", "Li",
    "unique_charged_composition_sheets", "VCu(PO4)2.xlsx",
)
DEFAULT_SHEET = "merged subgroup 1"
DEFAULT_CHARGED_MPID = "mp-755025"      # VCu(PO4)2    (de-intercalated / charged)
DEFAULT_DISCHARGED_MPID = "mp-757133"   # Li2VCu(PO4)2 (intercalated / discharged)
DEFAULT_CRITERION = 4
DEFAULT_WORKING_ION = "Li"

FIG_WIDTH = HALF_DOUBLE_COL_WIDTH    # ~3.37 in (== figure_3_pbx panel a)
FIG_HEIGHT = HALF_DOUBLE_COL_HEIGHT  # ~2.52 in (== figure_3_pbx panel a)


def _row_by_mpid(df, mpid):
    """Return the single workbook row whose ``MP id`` equals ``mpid``."""
    hits = df[df["MP id"] == mpid]
    if hits.empty:
        raise ValueError(f"MP id {mpid!r} not found in the sheet.")
    return hits.iloc[0]


def _metastability_points(row, criterion):
    """Parse the ``Seawater metastability (Criterion N)`` cell into a point list."""
    coords = row[f"Seawater metastability (Criterion {criterion})"]
    if isinstance(coords, str):
        coords = ast.literal_eval(coords)
    return coords


def compute_voltage(charged_row, discharged_row, working_ion, mp_client):
    """Intercalation voltage vs SHE from the workbook energies (never hardcoded).

    Replicates ``spices.screening.intercalation._get_voltage_plateau_df``: the
    voltage-vs-ion from :func:`calculate_intercalation_voltage` plus the working
    ion's standard reduction potential.
    """
    params = WORKING_ION_PARAMS[working_ion]
    ion_energy = get_energy_of_working_ion(working_ion, mp_client)
    voltage_vs_ion = calculate_intercalation_voltage(
        discharged_formula=discharged_row["Formula"],
        discharged_energy_per_formula=discharged_row["E (eV/formula)"],
        charged_formula=charged_row["Formula"],
        charged_energy_per_formula=charged_row["E (eV/formula)"],
        working_ion=working_ion,
        working_ion_valency=params["valency"],
        working_ion_energy=ion_energy,
    )
    return voltage_vs_ion + params["voltage_vs_SHE"]


def plot_figure(charged_row, discharged_row, voltage, criterion, out_path,
                fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT):
    """Render the metastability-overlap panel as a vector PDF at ``out_path``."""
    poly_charged = get_alpha_shape(_metastability_points(charged_row, criterion))
    poly_discharged = get_alpha_shape(
        _metastability_points(discharged_row, criterion))

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")
    plot_metastability_overlap(
        ax, poly_charged, poly_discharged, voltage,
        A_phase_name=f"{charged_row['Formula']}\n{charged_row['MP id']}",
        B_phase_name=f"{discharged_row['Formula']}\n{discharged_row['MP id']}",
    )
    # Save at the exact figure size (not bbox="tight"): constrained layout already
    # fits the outside-right legend and all labels inside the canvas, so the panel
    # is exactly fig_width x fig_height -- giving the requested 0.5 x double-column
    # width and the same height as figure_3_pbx panel a.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out_path, format="pdf")
    logger.info("Wrote %s (V = %.3f V vs SHE)", out_path, voltage)
    plt.close(fig)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPICES Figure 3 panel b: seawater metastability-overlap diagram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--excel", default=DEFAULT_EXCEL,
                        help="step-1 composition workbook holding the metastability points")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="sheet name to read")
    parser.add_argument("--charged-mpid", default=DEFAULT_CHARGED_MPID,
                        help="MP id of the charged (de-intercalated) phase")
    parser.add_argument("--discharged-mpid", default=DEFAULT_DISCHARGED_MPID,
                        help="MP id of the discharged (intercalated) phase")
    parser.add_argument("--criterion", type=int, default=DEFAULT_CRITERION,
                        help="seawater metastability criterion number (default 4)")
    parser.add_argument("--working-ion", default=DEFAULT_WORKING_ION,
                        help="working-ion element symbol (default Li)")
    parser.add_argument("-o", "--output-dir", default="additional_plots/figures",
                        help="Directory for the output PDF")
    parser.add_argument("--api-key", default=None,
                        help="MP API key for the live backend (default: MP_API_KEY env)")
    parser.add_argument("--snapshot-root", default=None,
                        help="MP snapshot root (or SPICES_MP_SNAPSHOT_ROOT env)")
    parser.add_argument("--snapshot-release", default=None,
                        help="MP snapshot release (or SPICES_MP_SNAPSHOT_RELEASE env)")
    parser.add_argument("--fig-width", type=float, default=FIG_WIDTH,
                        help="Figure width in inches (default 0.5 x DOUBLE_COL_WIDTH)")
    parser.add_argument("--fig-height", type=float, default=FIG_HEIGHT,
                        help="Figure height in inches (matches figure_3_pbx panel a)")
    args = parser.parse_args(argv)

    setup_logging()

    df = pd.read_excel(args.excel, sheet_name=args.sheet)
    charged_row = _row_by_mpid(df, args.charged_mpid)
    discharged_row = _row_by_mpid(df, args.discharged_mpid)

    # Snapshot backend when a root is given (flag or env); else live API.
    if args.snapshot_root or os.environ.get("SPICES_MP_SNAPSHOT_ROOT"):
        mp_client = get_client(
            backend="snapshot",
            snapshot_root=args.snapshot_root,
            release=args.snapshot_release,
        )
    else:
        mp_client = get_client(api_key=args.api_key)

    voltage = compute_voltage(
        charged_row, discharged_row, args.working_ion, mp_client)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_figure(
        charged_row, discharged_row, voltage, args.criterion,
        os.path.join(args.output_dir, "figure_3_metastability.pdf"),
        fig_width=args.fig_width, fig_height=args.fig_height,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
SPICES additional figure 3 -- Ni seawater Pourbaix diagram.

A single-element Ni Pourbaix diagram drawn in the SPICES publication style and
written as a vector PDF. It reproduces the seawater diagram
``3rd_Iteration/5.Calculating_statistics/pourbaix-diagram-plots/sea-pbx-transparent.png``
(notebook ``pourbaix-diagram-plot.ipynb`` cell 12): teal Ni solids, cream Ni ions,
dark-red water-stability lines, blue seawater-stability lines and the hatched
seawater stability window (pH 7-8.5). The colour scheme is taken verbatim from the
source; the base diagram is drawn by the shared :class:`CustomPourbaixPlotter`, which
already supports every feature the source used (the -0.3 / +0.5 V seawater shifts
included).

On top of the source diagram this figure adds two vertical "dimension" double-arrows:

* a blue one at low pH spanning the two seawater lines, labelled "2 V" (the extended
  seawater stability window, 1.23 + 0.3 + 0.5 = 2.03 V); and
* a dark-red one spanning the two water lines, labelled "1.23 V" (the thermodynamic
  water stability window).

Each label sits between its pair of lines, with the two arrowheads pointing away from
the text toward each line; text, arrows and the stability lines share one colour. The
two arrows are placed at different pH (defaults 2.5 and 5.0) because the 2 V window
vertically contains the 1.23 V window -- at the same pH the labels would collide.

Pourbaix entries are sourced from the cached MP 2025 snapshot via
:func:`spices.mp_client.get_client` (pass ``--snapshot-root`` or set
``SPICES_MP_SNAPSHOT_ROOT``), reusing the on-disk :class:`PourbaixCache`. Mirrors the
structure of ``additional_plots/figure_1_pbx_diag.py``.
"""

import argparse
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PREFAC

from spices.io import setup_logging
from spices.mp_client import get_client
from spices.plotting import (
    apply_publication_style,
    HALF_DOUBLE_COL_WIDTH,
    HALF_DOUBLE_COL_HEIGHT,
)
from spices.pourbaix.cache import PourbaixCache
from spices.pourbaix.plotting import CustomPourbaixPlotter

logger = logging.getLogger(__name__)


# --- figure constants -------------------------------------------------------
# Colours verbatim from the source notebook (cell 12): teal solids, cream ions,
# dark-red water lines, blue seawater lines.
SOLID_COLOR = "#326771"
ION_COLOR = "#dce2c8"
DOMAIN_BOUNDARY = "black"
WATER_LINE_COLOR = "#931621"  # dark red -- water stability lines (1.23 V window)
SEAWATER_COLOR = "blue"       # blue -- seawater stability lines (2 V window)

CHEMSYS = "Ni"
LIMITS = ((0, 14), (-2, 3))

# Requested width: half the double-column (double-panel) width; keep the source
# 6:4.5 aspect ratio for the height. Shared with figure_3_metastability.py (panel b)
# via spices.plotting so both figure-3 panels are exactly the same height.
FIG_WIDTH = HALF_DOUBLE_COL_WIDTH    # ~3.37 in
FIG_HEIGHT = HALF_DOUBLE_COL_HEIGHT  # ~2.52 in

# Default pH at which to draw each dimension double-arrow (overridable via CLI).
BLUE_ARROW_PH = 2.5   # "2 V"  seawater window
RED_ARROW_PH = 5.0    # "1.23 V" water window

# Seawater line shifts (must match CustomPourbaixPlotter): H down 0.3, O up 0.5.
SEAWATER_H_SHIFT = -0.3
SEAWATER_O_SHIFT = 0.5

# pH ticks: SPICES step-5 spacing of 2, minus the integer 8 (it sits only 0.5 pH
# from the seawater edge 8.5 and would overlap), plus the seawater edges 7 and 8.5.
PH_TICKS = [0, 2, 4, 6, 7, 8.5, 10, 12, 14]
SEAWATER_TICKS = {"7", "8.5"}


def _label_fontprops(size):
    """Helvetica (regular) :class:`FontProperties` at ``size`` pt (annotation labels)."""
    return FontProperties(family="Helvetica", weight="normal", size=size)


def load_ni_entries(mp_client, cache_dir="."):
    """Read-or-download the Ni PourbaixEntry list from the on-disk cache."""
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )
    return cache.read_or_download_pourbaix_entries(CHEMSYS, mp_client)


def build_diagram(entries):
    """Build the single-element Ni PourbaixDiagram (no comp/conc dict needed)."""
    return PourbaixDiagram(entries)


def _render_base(diagram, ax, limits):
    """Draw the source seawater Ni diagram with the shared plotter, then re-style.

    Uses the exact colour/flag set from notebook cell 12 (teal solids, cream ions,
    dark-red water lines, blue seawater lines + hatched pH 7-8.5 window).
    """
    plotter = CustomPourbaixPlotter(diagram)
    plotter.get_pourbaix_plot(
        limits=limits,
        ax=ax,
        label_domains=False,
        show_water_lines=True,
        show_neutral_axes=False,
        highlight_entry=None,
        domain_color=SOLID_COLOR,
        domain_boundary_color=DOMAIN_BOUNDARY,
        water_line_color=WATER_LINE_COLOR,
        domain_boundary_width=1,
        spine_width=0.8,
        separate_colours_for_ions_and_solids=True,
        solid_colour=SOLID_COLOR,
        ion_colour=ION_COLOR,
        show_seawater_lines=True,
        seawater_lines_colour=SEAWATER_COLOR,
        show_seawater_stability_window=True,
    )
    _restyle_axes(ax, limits)


def _restyle_axes(ax, limits):
    """Override the plotter's 20 pt labels / thick ticks with the SPICES step-5 style.

    pH ticks follow :data:`PH_TICKS` (spacing 2, minus 8, plus the seawater edges
    7 and 8.5); the 7/8.5 labels are coloured like the seawater lines. Helvetica and
    the step-5 font sizes are inherited from ``apply_publication_style`` -- only the
    colour of the two seawater ticks changes.
    """
    ax.set_title("")
    ax.set_xlabel("pH", fontsize=plt.rcParams["axes.labelsize"])
    ax.set_ylabel("Applied potential (V vs SHE)",
                  fontsize=plt.rcParams["axes.labelsize"])
    ax.tick_params(axis="both", which="both", direction="in", length=3,
                   width=0.8, top=False, right=False,
                   labelsize=plt.rcParams["xtick.labelsize"])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])

    # Explicit ticks: spacing 2 minus 8, plus the seawater edges 7 and 8.5
    # (a FixedLocator, so no MultipleLocator needed).
    ax.set_xticks(PH_TICKS)
    ax.set_xticklabels(
        [("8.5" if abs(t - 8.5) < 1e-6 else str(int(round(t)))) for t in PH_TICKS])
    for label in ax.get_xticklabels():
        if label.get_text() in SEAWATER_TICKS:
            label.set_color(SEAWATER_COLOR)
            label.set_fontweight("bold")


def _draw_window_arrow(ax, ph, y_lo, y_hi, label, color, gap_frac=0.18, lw=0.8):
    """Vertical text-broken dimension arrow at ``ph`` spanning ``y_lo``..``y_hi``.

    The label sits at the midpoint; two arrows fan out from it -- one head to the
    top line, one head to the bottom line (both pointing away from the text). Text,
    arrows and (caller's) lines share ``color``.
    """
    y_mid = 0.5 * (y_lo + y_hi)
    gap = gap_frac * (y_hi - y_lo)
    arrow = dict(arrowstyle="->", color=color, lw=lw,
                 shrinkA=0, shrinkB=0, mutation_scale=5)
    ax.annotate("", xy=(ph, y_hi), xytext=(ph, y_mid + gap),   # up to top line
                arrowprops=arrow, zorder=6)
    ax.annotate("", xy=(ph, y_lo), xytext=(ph, y_mid - gap),   # down to bottom line
                arrowprops=arrow, zorder=6)
    ax.text(ph, y_mid, label, color=color, ha="center", va="center", zorder=7,
            fontproperties=_label_fontprops(plt.rcParams["axes.labelsize"]))


def _add_window_arrows(ax, blue_ph=BLUE_ARROW_PH, red_ph=RED_ARROW_PH):
    """Add the "2 V" (seawater, blue) and "1.23 V" (water, red) dimension arrows.

    Line values are recomputed from PREFAC exactly as the plotter draws the lines.
    """
    # Seawater window (blue): H line shifted down 0.3, O line up 0.5 -> ~2.03 V span.
    blue_lo = -blue_ph * PREFAC + SEAWATER_H_SHIFT
    blue_hi = -blue_ph * PREFAC + 1.23 + SEAWATER_O_SHIFT
    _draw_window_arrow(ax, blue_ph, blue_lo, blue_hi, "2 V", SEAWATER_COLOR)

    # Water window (dark red): H line to O line -> 1.23 V span.
    red_lo = -red_ph * PREFAC
    red_hi = -red_ph * PREFAC + 1.23
    _draw_window_arrow(ax, red_ph, red_lo, red_hi, "1.23 V", WATER_LINE_COLOR)


def plot_figure(diagram, out_path, fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT,
                limits=LIMITS, blue_ph=BLUE_ARROW_PH, red_ph=RED_ARROW_PH):
    """Render the Ni seawater Pourbaix diagram as a vector PDF at ``out_path``."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")

    _render_base(diagram, ax, limits)
    _add_window_arrows(ax, blue_ph=blue_ph, red_ph=red_ph)

    # Save at the exact figure size (not bbox="tight"): constrained layout already
    # fits all labels inside the canvas, so the panel is exactly fig_width x
    # fig_height and shares its height with figure_3_metastability.py panel b.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out_path, format="pdf")
    logger.info("Wrote %s", out_path)
    plt.close(fig)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPICES Figure 3: Ni seawater Pourbaix diagram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", default="additional_plots/figures",
                        help="Directory for the output PDF")
    parser.add_argument("--cache-dir", default=".",
                        help="Directory holding saved_pbx_entries/ (default: .)")
    parser.add_argument("--api-key", default=None,
                        help="MP API key for the live backend (default: MP_API_KEY env)")
    parser.add_argument("--snapshot-root", default=None,
                        help="MP snapshot root (or SPICES_MP_SNAPSHOT_ROOT env)")
    parser.add_argument("--snapshot-release", default=None,
                        help="MP snapshot release (or SPICES_MP_SNAPSHOT_RELEASE env)")
    parser.add_argument("--fig-width", type=float, default=FIG_WIDTH,
                        help="Figure width in inches (default 0.5 x DOUBLE_COL_WIDTH)")
    parser.add_argument("--fig-height", type=float, default=FIG_HEIGHT,
                        help="Figure height in inches")
    parser.add_argument("--blue-arrow-ph", type=float, default=BLUE_ARROW_PH,
                        help="pH of the blue '2 V' seawater-window arrow")
    parser.add_argument("--red-arrow-ph", type=float, default=RED_ARROW_PH,
                        help="pH of the red '1.23 V' water-window arrow")
    args = parser.parse_args(argv)

    setup_logging()

    # Snapshot backend when a root is given (flag or env); else live API.
    if args.snapshot_root or os.environ.get("SPICES_MP_SNAPSHOT_ROOT"):
        mp_client = get_client(
            backend="snapshot",
            snapshot_root=args.snapshot_root,
            release=args.snapshot_release,
        )
    else:
        mp_client = get_client(api_key=args.api_key)

    entries = load_ni_entries(mp_client, cache_dir=args.cache_dir)
    diagram = build_diagram(entries)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_figure(
        diagram, os.path.join(args.output_dir, "figure_3_pbx.pdf"),
        fig_width=args.fig_width, fig_height=args.fig_height,
        blue_ph=args.blue_arrow_ph, red_ph=args.red_arrow_ph,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
SPICES additional figure: Figure 1 panels c & d -- Li-Fe-P Pourbaix diagrams.

These two panels convey the scientific reasoning of the workflow (they are not
derived from the step-5 statistics): they show how Li-Fe-P phase stability shifts
with the aqueous ion concentration.

Panel c: composition Li:Fe:P = 1:1:1, aqueous concentrations Li = 0.75,
         Fe = 0.25, P = 0.33. The region where LiFePO4 is stable is filled violet
         and labelled "LiFePO4(s)"; every other region is white and unlabelled
         (to avoid clutter).
Panel d: same composition at the dilute limit (all concentrations = 1e-6).
         LiFePO4 is no longer stable; the analogous region around the operating
         point (pH 8, -0.4 V vs SHE) is filled the same violet and labelled with
         whatever phase(s) are stable there.

Both panels use the SPICES publication style and are written as vector PDFs for
assembly in Inkscape. Pourbaix entries are sourced from the cached MP 2025
snapshot via :func:`spices.mp_client.get_client` (set ``SPICES_MP_BACKEND=snapshot``
and ``SPICES_MP_SNAPSHOT_ROOT``), reusing the on-disk :class:`PourbaixCache`.

The base diagram (white domains, black boundaries, dark-red water lines) is drawn
by the shared :class:`CustomPourbaixPlotter`; the exact-violet highlight and the
single-region label are painted on top here, so the shared plotter -- which other
figures rely on -- stays untouched.

Reproduces 3rd_Iteration/5.Calculating_statistics/pourbaix-diagram-plots/
LFPO-pbx-2.png (panel c) and LFPO-pbx-1.png (panel d), re-styled for publication.
"""

import argparse
import logging
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, MultiEntry

from spices.io import setup_logging
from spices.mp_client import get_client
from spices.plotting import apply_publication_style, SINGLE_COL_WIDTH
from spices.pourbaix.cache import PourbaixCache
from spices.pourbaix.plotting import CustomPourbaixPlotter

logger = logging.getLogger(__name__)


# --- figure constants -------------------------------------------------------
# Violet identical to PIPELINE_BAR_COLORS[1] in spices/screening/statistics.py
# (the "metastable in seawater" bar in pipeline_bar_chart.pdf). Hardcoded so we
# do not import the heavy statistics module just for one colour string.
HIGHLIGHT_COLOR = "#6b6ecf"
WATER_LINE_COLOR = "#931621"  # dark red, as in the original LFPO-pbx notebooks
DOMAIN_COLOR = "white"

CHEMSYS = "Li-Fe-P"
COMP_DICT = {"Li": 1, "Fe": 1, "P": 1}
CONC_DICT_C = {"Li": 0.75, "Fe": 0.25, "P": 0.33}  # panel c
CONC_DICT_D = {"Li": 1e-6, "Fe": 1e-6, "P": 1e-6}  # panel d (dilute limit)

LIMITS = ((0, 14), (-1.5, 1.5))
REF_PH, REF_V = 8.0, -0.4  # operating point that selects the highlighted region
PANEL_C_LABEL = "LiFePO4(s)"


def _subscript_formula(text):
    """Subscript stoichiometric numerals for matplotlib mathtext.

    e.g. ``"LiFePO4(s)"`` -> ``"LiFePO$_{4}$(s)"``. Copied (not imported) from
    ``spices.screening.statistics._subscript_formula`` to avoid the heavy
    statistics import chain for one regex.
    """
    return re.sub(r"(\d+(\.\d+)?)", r"$_{\1}$", text)


def _format_ion_charge(match):
    """Convert pymatgen's ``[+1]`` / ``[-2]`` charge bracket to a mathtext superscript.

    e.g. ``"[+1]"`` -> ``"$^{+}$"`` and ``"[-2]"`` -> ``"$^{2−}$"`` (magnitude
    then sign, magnitude dropped when 1, using a true minus sign).
    """
    sign = "+" if match.group(1) == "+" else "−"  # U+2212 minus
    magnitude = "" if match.group(2) == "1" else match.group(2)
    return f"$^{{{magnitude}{sign}}}$"


def _entry_latex(entry):
    """Publication label for one Pourbaix sub-entry.

    ``to_latex_string`` already subscripts the formula and keeps the "(s)" suffix;
    we additionally turn ion charge brackets into proper superscripts.
    """
    return re.sub(r"\[([+-])(\d+)\]", _format_ion_charge, entry.to_latex_string())


def _entry_phase_label(entry):
    """Label for a stable (Multi)Entry, solids first (e.g. the electrode phase leads).

    For a MultiEntry this lists the full coexisting assemblage, e.g.
    ``"Fe3(PO4)2(s) + Li+ + HPO4 2-"`` (with proper sub/superscripts), conveying
    that Li and P dissolve into solution at the dilute limit.
    """
    subentries = entry.entry_list if isinstance(entry, MultiEntry) else [entry]
    # Stable-sort solids (phase_type "Solid") ahead of dissolved ions.
    subentries = sorted(subentries, key=lambda e: e.phase_type != "Solid")
    return " + ".join(_entry_latex(e) for e in subentries)


def load_lfp_entries(mp_client, cache_dir="."):
    """Read-or-download the Li-Fe-P PourbaixEntry list from the on-disk cache."""
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )
    return cache.read_or_download_pourbaix_entries(CHEMSYS, mp_client)


def build_diagram(entries, conc_dict):
    """Build a Li-Fe-P PourbaixDiagram (1:1:1) at the given ion concentrations."""
    return PourbaixDiagram(entries, comp_dict=COMP_DICT, conc_dict=conc_dict)


def plot_panel(diagram, out_path, label_text=None, panel_width=None,
               ref_ph=REF_PH, ref_v=REF_V, limits=LIMITS):
    """Render one Pourbaix panel and save it as a vector PDF.

    Parameters
    ----------
    diagram : PourbaixDiagram
    out_path : str
        Destination PDF path.
    label_text : str or None
        Literal label for the highlighted region (panel c). If ``None``, the
        label is derived from the phase(s) stable at ``(ref_ph, ref_v)`` (panel d).
    panel_width : float or None
        Figure width in inches (default ``SINGLE_COL_WIDTH``); height = width * 2/3.
    ref_ph, ref_v : float
        Operating point selecting the region to highlight.
    limits : ((xlo, xhi), (ylo, yhi))

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style()

    width = panel_width or SINGLE_COL_WIDTH
    fig, ax = plt.subplots(figsize=(width, width * 2 / 3))

    # Base diagram: all domains white, no labels, dark-red water lines, no
    # neutral axes. highlight_entry stays None -- we overpaint the highlight
    # ourselves at full opacity so the colour matches #6b6ecf exactly (the
    # plotter fills domains at alpha=0.5, which would wash the hex out).
    plotter = CustomPourbaixPlotter(diagram)
    plotter.get_pourbaix_plot(
        limits=limits,
        ax=ax,
        label_domains=False,
        show_water_lines=True,
        show_neutral_axes=False,
        highlight_entry=None,
        domain_color=DOMAIN_COLOR,
        domain_boundary_color="black",
        water_line_color=WATER_LINE_COLOR,
        domain_boundary_width=0.4,
        spine_width=0.8,
        separate_colours_for_ions_and_solids=False,
    )

    # Highlight the region containing the operating point, in exact violet.
    # get_stable_entry returns an object that is a key of _stable_domain_vertices
    # (both are built together in PourbaixDiagram.__init__), so the lookup is
    # identity-safe.
    highlight = diagram.get_stable_entry(ref_ph, ref_v)
    verts = np.asarray(diagram._stable_domain_vertices[highlight])
    closed = np.vstack([verts, verts[0]])
    ax.fill(closed[:, 0], closed[:, 1], color=HIGHLIGHT_COLOR, alpha=1.0,
            zorder=2)
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=0.4, zorder=3)

    # Label ONLY the highlighted region.
    text = _subscript_formula(label_text) if label_text else \
        _entry_phase_label(highlight)
    center = verts.mean(axis=0)
    ax.annotate(text, center, ha="center", va="center",
                fontsize=plt.rcParams["axes.labelsize"], zorder=4)

    # Re-apply the publication style: get_pourbaix_plot hardcodes 20 pt axis
    # labels / bold title / thick inward ticks, which override apply_publication_style.
    ax.set_title("")
    ax.set_xlabel("pH", fontsize=plt.rcParams["axes.labelsize"])
    ax.set_ylabel("Applied potential (V vs SHE)",
                  fontsize=plt.rcParams["axes.labelsize"])
    ax.tick_params(axis="both", which="both", direction="in", length=3,
                   width=0.8, top=True, right=True,
                   labelsize=plt.rcParams["xtick.labelsize"])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    logger.info("Wrote %s", out_path)
    plt.close(fig)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPICES Figure 1 (panels c, d): Li-Fe-P Pourbaix diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", default="additional_plots/figures",
                        help="Directory for the two output PDFs")
    parser.add_argument("--cache-dir", default=".",
                        help="Directory holding saved_pbx_entries/ (default: .)")
    parser.add_argument("--api-key", default=None,
                        help="MP API key for the live backend (default: MP_API_KEY env)")
    parser.add_argument("--snapshot-root", default=None,
                        help="MP snapshot root (or SPICES_MP_SNAPSHOT_ROOT env)")
    parser.add_argument("--snapshot-release", default=None,
                        help="MP snapshot release (or SPICES_MP_SNAPSHOT_RELEASE env)")
    parser.add_argument("--ref-ph", type=float, default=REF_PH,
                        help="pH of the operating point selecting the highlighted region")
    parser.add_argument("--ref-v", type=float, default=REF_V,
                        help="Voltage (V vs SHE) of that operating point")
    parser.add_argument("--panel-width", type=float, default=None,
                        help="Panel width in inches (default SINGLE_COL_WIDTH = 3.5)")
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

    entries = load_lfp_entries(mp_client, cache_dir=args.cache_dir)

    pbx_c = build_diagram(entries, CONC_DICT_C)
    pbx_d = build_diagram(entries, CONC_DICT_D)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_panel(
        pbx_c, os.path.join(args.output_dir, "figure_1_panel_c.pdf"),
        label_text=PANEL_C_LABEL, panel_width=args.panel_width,
        ref_ph=args.ref_ph, ref_v=args.ref_v,
    )
    plot_panel(
        pbx_d, os.path.join(args.output_dir, "figure_1_panel_d.pdf"),
        label_text=None, panel_width=args.panel_width,
        ref_ph=args.ref_ph, ref_v=args.ref_v,
    )


if __name__ == "__main__":
    main()

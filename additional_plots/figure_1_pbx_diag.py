#!/usr/bin/env python
"""
SPICES additional figure: Figure 1 panels c & d -- Li-Fe-P Pourbaix diagrams.

These two panels convey the scientific reasoning of the workflow (they are not
derived from the step-5 statistics): they show how Li-Fe-P phase stability shifts
with the aqueous ion concentration. They are drawn together as one double panel
(c left, d right) whose total width is 1.4x the single-column width.

Panel c: composition Li:Fe:P = 1:1:1, aqueous concentrations Li = 0.75,
         Fe = 0.25, P = 0.33. The region where LiFePO4 is stable is filled violet
         and labelled "LiFePO4(s)". Only this region is bordered; every other
         region is white, unlabelled and borderless (to avoid clutter).
Panel d: same composition at the dilute limit (all concentrations = 1e-6).
         LiFePO4 is no longer stable: the footprint it occupied in panel c is now
         tiled by several coexisting assemblages. That footprint is subdivided,
         each sub-region filled a distinct muted colour and bordered, and every
         sub-region is labelled -- entirely within the axes box -- with the
         phase(s) stable there, using non-crossing leader arrows that fan out to
         the top, left and bottom margins of the plot.

The figure uses the SPICES publication style and is written as a vector PDF for
assembly in Inkscape. Pourbaix entries are sourced from the cached MP 2025
snapshot via :func:`spices.mp_client.get_client` (set ``SPICES_MP_BACKEND=snapshot``
and ``SPICES_MP_SNAPSHOT_ROOT``), reusing the on-disk :class:`PourbaixCache`.

The base diagram (white domains, dark-red water lines) is drawn by the shared
:class:`CustomPourbaixPlotter` with white domain borders, so the shared plotter --
which other figures rely on -- stays untouched; the violet highlight (panel c),
the footprint subdivision (panel d) and all labels are painted on top here.

Reproduces 3rd_Iteration/5.Calculating_statistics/pourbaix-diagram-plots/
LFPO-pbx-2.png (panel c) and LFPO-pbx-1.png (panel d), re-styled for publication.
"""

import argparse
import logging
import math
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, MultiEntry
from shapely import make_valid
from shapely.geometry import Polygon, box

from spices.io import setup_logging
from spices.mp_client import get_client
from spices.plotting import apply_publication_style, SINGLE_COL_WIDTH, DOUBLE_COL_WIDTH
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

# Panel d: one muted, distinguishable colour per sub-region (largest first).
# Curated Tableau-muted hues, deliberately excluding any blue-violet so they do
# not read as panel c's "LiFePO4 stable" violet. Edit freely; cycles if there
# are ever more sub-regions than colours.
PALETTE_D = [
    "#4e79a7",  # blue
    "#f28e2b",  # orange
    "#59a14f",  # green
    "#e15759",  # red
    "#76b7b2",  # teal
    "#edc948",  # yellow
    "#ff9da7",  # salmon
    "#9c755f",  # brown
    "#bab0ac",  # warm grey
    "#8cd17d",  # light green
    "#86bcb6",  # light teal
    "#d37295",  # rose
]

CHEMSYS = "Li-Fe-P"
COMP_DICT = {"Li": 1, "Fe": 1, "P": 1}
CONC_DICT_C = {"Li": 0.75, "Fe": 0.25, "P": 0.33}  # panel c
CONC_DICT_D = {"Li": 1e-6, "Fe": 1e-6, "P": 1e-6}  # panel d (dilute limit)

LIMITS = ((0, 14), (-1.5, 1.5))
REF_PH, REF_V = 8.0, -0.4  # operating point that selects the LiFePO4 region
PANEL_C_LABEL = "LiFePO4(s)"

FIG_WIDTH = 0.65 * DOUBLE_COL_WIDTH  # total width of the c+d double panel (in.)
FIG_HEIGHT = 2.6                     # in.; square-ish boxes leave room for labels
LABEL_FONTSIZE_D = 3.8          # pt; small so all 12 in-box labels stay legible
SUBREGION_BORDER_WIDTH = 0.3    # pt; per sub-region outline (panel d)
FOOTPRINT_BORDER_WIDTH = 0.6    # pt; outer LiFePO4 footprint outline (panel d)
AREA_TOL = 1e-4                 # ignore numerically-empty clip remnants


def _label_fontprops(size):
    """Helvetica Light :class:`FontProperties` at ``size`` pt (stable-phase labels)."""
    return FontProperties(family="Helvetica", weight="light", size=size)


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


def _polygon_parts(geom):
    """Yield the exterior coordinate array (Nx2) of each polygonal part of ``geom``.

    A shapely clip can return a Polygon, MultiPolygon, or GeometryCollection;
    non-areal parts (lines/points from edge-touching clips) are skipped. The
    Pourbaix sub-regions are convex with no holes, so interior rings are ignored.
    """
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        geoms = [geom]
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        geoms = list(geom.geoms)
    else:
        geoms = []
    for g in geoms:
        if g.geom_type == "Polygon" and not g.is_empty:
            yield np.asarray(g.exterior.coords)


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


def _render_base(diagram, ax, limits):
    """Draw the shared white-domain / dark-red-water-line base diagram and re-style.

    Domain borders are drawn in the fill colour (white) so the base contributes
    no visible boundaries -- the panels add their own borders only where wanted.
    Water lines are a separate code path and stay dark red.
    """
    plotter = CustomPourbaixPlotter(diagram)
    plotter.get_pourbaix_plot(
        limits=limits,
        ax=ax,
        label_domains=False,
        show_water_lines=True,
        show_neutral_axes=False,
        highlight_entry=None,
        domain_color=DOMAIN_COLOR,
        domain_boundary_color=DOMAIN_COLOR,   # white -> base borders invisible
        water_line_color=WATER_LINE_COLOR,
        domain_boundary_width=0.4,
        spine_width=0.8,
        separate_colours_for_ions_and_solids=False,
    )
    _restyle_axes(ax, limits)


def _restyle_axes(ax, limits):
    """Override the plotter's hardcoded 20 pt labels / thick ticks with SPICES style.

    Top/right tick marks are removed (the box spines stay).
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
    ax.xaxis.set_major_locator(MultipleLocator(2))  # pH ticks 0,2,4,...,14
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])


def render_panel_c(ax, diagram, label_text=PANEL_C_LABEL,
                   ref_ph=REF_PH, ref_v=REF_V, limits=LIMITS):
    """Draw panel c (LiFePO4 stable region highlighted violet) into ``ax``."""
    _render_base(diagram, ax, limits)

    # Highlight the LiFePO4 region in exact violet at full opacity (the base
    # fills domains at alpha 0.5, which would wash the hex out). get_stable_entry
    # returns an object that is a key of _stable_domain_vertices, so the lookup is
    # identity-safe.
    highlight = diagram.get_stable_entry(ref_ph, ref_v)
    verts = np.asarray(diagram._stable_domain_vertices[highlight])
    closed = np.vstack([verts, verts[0]])
    ax.fill(closed[:, 0], closed[:, 1], color=HIGHLIGHT_COLOR, alpha=1.0, zorder=2)
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=0.4, zorder=3)

    # Label ONLY the highlighted region, in Helvetica Light.
    text = _subscript_formula(label_text) if label_text else \
        _entry_phase_label(highlight)
    ax.annotate(text, verts.mean(axis=0), ha="center", va="center",
                fontproperties=_label_fontprops(plt.rcParams["axes.labelsize"]),
                zorder=4)


def _footprint(pbx_c, ref_ph, ref_v, limits):
    """The LiFePO4 stability polygon from panel c, clipped to the plot box."""
    lfp = pbx_c.get_stable_entry(ref_ph, ref_v)
    (xlo, xhi), (ylo, yhi) = limits
    poly = make_valid(Polygon(np.asarray(pbx_c._stable_domain_vertices[lfp])))
    return poly.intersection(box(xlo, ylo, xhi, yhi))


def _subdivide(pbx_d, foot):
    """Sub-regions tiling ``foot`` at the dilute limit, as ``(entry, geom)`` largest-first."""
    regions = []
    for entry, verts in pbx_d._stable_domain_vertices.items():
        sub = foot.intersection(make_valid(Polygon(np.asarray(verts))))
        if sub.area > AREA_TOL:
            regions.append((entry, sub))
    regions.sort(key=lambda es: es[1].area, reverse=True)
    return regions


def _strip_math(text):
    """Approximate visible character count of a mathtext label (drop $ _ ^ { } \\)."""
    return re.sub(r"[\${}^_\\]", "", text)


def _place_inside_labels(ax, regions, foot, limits, fontsize):
    """Label every sub-region with non-crossing leaders, all inside the axes box.

    The footprint is a thin wedge: a dense cluster of small regions on the left
    and a few regions at the right tip. Regions left of the footprint centroid
    are labelled on the left (a vertical stack that the left white strip holds
    without overlap, arrows pointing right). The remaining right-tip regions --
    too close in pH for side-by-side labels -- are sent to the top or bottom
    white area (by whether they sit above or below the centroid) and greedily
    stacked into rows near their own pH so the wide labels never collide; their
    arrows are near-vertical. Left labels live in the pH < footprint zone and
    top/bottom labels in the pH >= centroid zone, so the two fans never cross.
    """
    (xlo, xhi), (ylo, yhi) = limits
    mx = 0.03 * (xhi - xlo)
    my = 0.05 * (yhi - ylo)
    x0, y0, x1, y1 = xlo + mx, ylo + my, xhi - mx, yhi - my
    fc = foot.centroid
    fontprops = _label_fontprops(fontsize)
    char_ph = 0.20 * (fontsize / 4.0)   # approx label width per character, in pH
    row_gap = 0.05 * (yhi - ylo)
    gap_ph = 0.02 * (xhi - xlo)

    def emit(entry, target, anchor, ha, va):
        ax.annotate(_entry_phase_label(entry), xy=target, xycoords="data",
                    xytext=anchor, textcoords="data", ha=ha, va=va,
                    fontproperties=fontprops, annotation_clip=True, zorder=5,
                    arrowprops=dict(arrowstyle="->", lw=0.4, color="0.35",
                                    shrinkA=2, shrinkB=2, mutation_scale=4))

    # Only the leftmost tight pH column goes to the left strip; the rest (central
    # and right-tip regions) go to the top/bottom white areas, split by their own
    # median V so both are used. Keeping the left set to a narrow pH column makes
    # its near-horizontal leaders genuinely non-crossing.
    fxmin, _, fxmax, _ = foot.bounds
    left_cut = fxmin + 0.22 * (fxmax - fxmin)
    left = [(entry, sub.centroid) for entry, sub in regions if sub.centroid.x < left_cut]
    rest = [(entry, sub.centroid) for entry, sub in regions if sub.centroid.x >= left_cut]
    top, bottom = [], []
    if rest:
        v_sorted = sorted(c.y for _, c in rest)
        v_split = v_sorted[len(v_sorted) // 2]   # upper half -> top, lower -> bottom
        top = [(e, c) for e, c in rest if c.y >= v_split]
        bottom = [(e, c) for e, c in rest if c.y < v_split]

    # Left: stack just left of the cluster (small Δx => sorting by V cannot cross),
    # text reading leftward into the white strip.
    x_left = max(x0, fxmin - 0.3)
    left.sort(key=lambda ec: ec[1].y)
    for i, (entry, c) in enumerate(left):
        emit(entry, (c.x, c.y), (x_left, y0 + (i + 0.5) / len(left) * (y1 - y0)),
             "right", "center")

    # Top / bottom: greedy row-stacking near each region's pH, near-vertical arrows.
    def place_rows(group, y_base, y_dir, va):
        group.sort(key=lambda ec: ec[1].x)
        row_right = []   # right extent (pH) of the last label placed in each row
        for entry, c in group:
            half = 0.5 * len(_strip_math(_entry_phase_label(entry))) * char_ph
            lx = min(max(c.x, x0 + half), x1 - half)   # keep the label inside box
            row = next((r for r, rx in enumerate(row_right)
                        if lx - half >= rx + gap_ph), len(row_right))
            if row == len(row_right):
                row_right.append(lx + half)
            else:
                row_right[row] = lx + half
            emit(entry, (c.x, c.y), (lx, y_base + y_dir * row * row_gap), "center", va)

    place_rows(top, y1, -1, "top")
    place_rows(bottom, y0, +1, "bottom")


def render_panel_d(ax, pbx_d, pbx_c, ref_ph=REF_PH, ref_v=REF_V, limits=LIMITS,
                   label_fontsize=LABEL_FONTSIZE_D):
    """Draw panel d (LiFePO4 footprint subdivided at the dilute limit) into ``ax``.

    The polygon LiFePO4 occupies in panel c is overlaid on the dilute-limit
    diagram and split into the assemblages stable within it; each sub-region is
    filled a distinct muted colour, bordered, and labelled inside the box.
    """
    _render_base(pbx_d, ax, limits)

    foot = _footprint(pbx_c, ref_ph, ref_v, limits)
    regions = _subdivide(pbx_d, foot)
    logger.info("Panel d: %d sub-regions tile the LiFePO4 footprint", len(regions))

    # Fill + border each sub-region in its own muted colour.
    for i, (entry, sub) in enumerate(regions):
        color = PALETTE_D[i % len(PALETTE_D)]
        for ring in _polygon_parts(sub):
            ax.fill(ring[:, 0], ring[:, 1], color=color, alpha=1.0, zorder=2)
            ax.plot(ring[:, 0], ring[:, 1], color="black",
                    linewidth=SUBREGION_BORDER_WIDTH, zorder=3)

    # Outline the whole footprint to echo panel c's LiFePO4 boundary.
    for ring in _polygon_parts(foot):
        ax.plot(ring[:, 0], ring[:, 1], color="black",
                linewidth=FOOTPRINT_BORDER_WIDTH, zorder=4)

    _place_inside_labels(ax, regions, foot, limits, label_fontsize)


def plot_figure(pbx_c, pbx_d, out_path, fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT,
                ref_ph=REF_PH, ref_v=REF_V, limits=LIMITS,
                label_fontsize=LABEL_FONTSIZE_D):
    """Render the c+d double panel as one vector PDF (total width ``fig_width``)."""
    apply_publication_style()
    fig, (ax_c, ax_d) = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), layout="constrained")

    render_panel_c(ax_c, pbx_c, label_text=PANEL_C_LABEL,
                   ref_ph=ref_ph, ref_v=ref_v, limits=limits)

    render_panel_d(ax_d, pbx_d, pbx_c, ref_ph=ref_ph, ref_v=ref_v, limits=limits,
                   label_fontsize=label_fontsize)

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    logger.info("Wrote %s", out_path)
    plt.close(fig)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SPICES Figure 1 (panels c, d): Li-Fe-P Pourbaix double panel",
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
    parser.add_argument("--ref-ph", type=float, default=REF_PH,
                        help="pH of the operating point selecting the LiFePO4 region")
    parser.add_argument("--ref-v", type=float, default=REF_V,
                        help="Voltage (V vs SHE) of that operating point")
    parser.add_argument("--fig-width", type=float, default=FIG_WIDTH,
                        help="Total figure width in inches (default 0.65 x DOUBLE_COL_WIDTH)")
    parser.add_argument("--fig-height", type=float, default=FIG_HEIGHT,
                        help="Figure height in inches")
    parser.add_argument("--label-fontsize-d", type=float, default=LABEL_FONTSIZE_D,
                        help="Font size (pt) of the panel-d in-box labels")
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
    plot_figure(
        pbx_c, pbx_d, os.path.join(args.output_dir, "figure_1_panel_cd.pdf"),
        fig_width=args.fig_width, fig_height=args.fig_height,
        ref_ph=args.ref_ph, ref_v=args.ref_v, label_fontsize=args.label_fontsize_d,
    )


if __name__ == "__main__":
    main()

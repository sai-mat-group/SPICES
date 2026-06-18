"""
Geometric overlap calculations for Pourbaix phase regions and water
stability lines, including alpha-shape analysis for stability area
computations.
"""

import re

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

from spices.config import (
    PREFAC, H_LINE, O_LINE, DEFAULT_OER_SHIFT, DEFAULT_HER_SHIFT,
)
from spices.plotting import (
    apply_publication_style,
    HALF_DOUBLE_COL_WIDTH,
    HALF_DOUBLE_COL_HEIGHT,
)

apply_publication_style()


# --- metastability-overlap plot colours (figure 3 panel b) ------------------
# Verbatim from the seawater-metastability reference figure: transparent blue
# charged region, transparent red discharged region, purple overlap.
CHARGED_COLOR = "#648FFF"          # transparent blue: charged (de-intercalated) phase
DISCHARGED_COLOR = "#DC267F"       # transparent red: discharged (intercalated) phase
INTERSECTION_COLOR = "#7161ef"     # purple: metastability overlap
SEAWATER_LINE_COLOR = "blue"       # blue dashed seawater-stability-window borders
VOLTAGE_LINE_COLOR = "crimson"     # intercalation-voltage line + label
VOLTAGE_OVERLAP_COLOR = "#720026"  # dark red: voltage-pH overlap segment
PLOT_BACKGROUND_COLOR = "#EDF2F3"  # light-grey panel background


def _subscript_formula(text):
    """Subscript a formula's stoichiometric digits for matplotlib mathtext.

    ``text`` may be ``"Formula"`` or ``"Formula\\nmp-id"``; only the first line
    is subscripted so an mp-id's digits are left intact. E.g.
    ``"VCu(PO4)2\\nmp-755025"`` -> ``"VCu(PO$_{4}$)$_{2}$\\nmp-755025"``.
    Ported from ``spices.screening.statistics._subscript_formula``.
    """
    parts = text.split("\n")
    parts[0] = re.sub(r"(\d+(\.\d+)?)", r"$_{\1}$", parts[0])
    return "\n".join(parts)


def check_overlap(polygon_coords, line_equation1, line_equation2,
                  range_of_pH=(0, 14), show_plot=False):
    """
    Check if a Pourbaix phase polygon overlaps with the water stability
    region bounded by two lines.

    Parameters
    ----------
    polygon_coords : list of tuple
        (pH, voltage) vertices of the phase polygon.
    line_equation1 : tuple of float
        (a, b, c) for line ax + by + c = 0 (typically the H line).
    line_equation2 : tuple of float
        (a, b, c) for line ax + by + c = 0 (typically the O line).
    range_of_pH : tuple of float
        (min_pH, max_pH) range over which to check overlap.
    show_plot : bool
        If True, display a matplotlib plot (default False).

    Returns
    -------
    bool
        True if the phase polygon overlaps with the water stability
        region, False otherwise.
    """
    polygon = Polygon(polygon_coords)

    a1, b1, c1 = line_equation1
    a2, b2, c2 = line_equation2

    x_min, x_max = range_of_pH

    line1 = LineString([
        (x_min, -(a1 * x_min + c1) / b1),
        (x_max, -(a1 * x_max + c1) / b1),
    ])
    line2 = LineString([
        (x_min, -(a2 * x_min + c2) / b2),
        (x_max, -(a2 * x_max + c2) / b2),
    ])

    area_between_lines = unary_union([line1, line2]).convex_hull

    if show_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x, y = polygon.exterior.coords.xy
        ax.plot(x, y, label="Phase polygon", color="blue")
        ax.plot(*line1.xy, label="H line", linestyle="--", color="red")
        ax.plot(*line2.xy, label="O line", linestyle="--", color="green")
        ax.scatter(x, y, color="black", marker="o")
        ax.set_xlabel("pH")
        ax.set_ylabel("V (SHE)")
        ax.legend()
        plt.show()

    return polygon.intersects(area_between_lines)


def check_overlap_with_water_stability_lines(pH, voltage,
                                             OER_shift=0.0,
                                             HER_shift=0.0):
    """
    Check if a (pH, voltage) point lies within the water stability
    window, optionally extended by OER/HER overpotential shifts.

    Parameters
    ----------
    pH : float
        pH value.
    voltage : float
        Electrode potential (V vs SHE).
    OER_shift : float
        Positive shift applied to the O line (OER overpotential).
        Default 0.0 (standard water window).
    HER_shift : float
        Negative shift applied to the H line (HER overpotential).
        Default 0.0 (standard water window).

    Returns
    -------
    bool
        True if the point is within the (extended) water stability
        window.
    """
    y_H = -(H_LINE[0] * pH + H_LINE[2]) / H_LINE[1] + HER_shift
    y_O = -(O_LINE[0] * pH + O_LINE[2]) / O_LINE[1] + OER_shift

    return min(y_H, y_O) <= voltage <= max(y_H, y_O)


def check_overlap_with_extended_water_stability_lines(pH, voltage,
                                                      OER_shift=0.5,
                                                      HER_shift=-0.3):
    """
    Check if a (pH, voltage) point lies *only* in the extended region
    (between the standard and shifted water stability lines) and NOT
    within the standard water stability window.

    This identifies points that are stable only due to kinetic
    overpotentials.

    Parameters
    ----------
    pH : float
    voltage : float
    OER_shift : float
        Positive shift for OER overpotential (default 0.5 V).
    HER_shift : float
        Negative shift for HER overpotential (default -0.3 V).

    Returns
    -------
    bool
        True if the point is in the extended-only region.
    """
    # HER extended region
    y_H = -(H_LINE[0] * pH + H_LINE[2]) / H_LINE[1]
    y_H_ext = y_H + HER_shift
    if min(y_H, y_H_ext) <= voltage <= max(y_H, y_H_ext):
        return True

    # OER extended region
    y_O = -(O_LINE[0] * pH + O_LINE[2]) / O_LINE[1]
    y_O_ext = y_O + OER_shift
    if min(y_O, y_O_ext) <= voltage <= max(y_O, y_O_ext):
        return True

    return False


# ====================================================================
# Alpha-shape and stability-area geometry (Stage 3)
# ====================================================================


def get_alpha_shape(points, alpha=0):
    """
    Compute the alpha shape (concave hull) for a set of 2D points.

    Parameters
    ----------
    points : list of tuple or np.ndarray
        (pH, voltage) coordinate pairs.
    alpha : float
        Between 0 (convex hull) and 1 (concave hull).

    Returns
    -------
    shapely.geometry.Polygon or LineString or Point
        The alpha shape geometry.

    Raises
    ------
    ImportError
        If the ``alphashape`` package is not installed.
    """
    try:
        import alphashape
    except ImportError:
        raise ImportError(
            "The 'alphashape' library is required. "
            "Install it with `pip install alphashape`."
        )

    pts = np.array(points)
    return alphashape.alphashape(pts, alpha)


def find_ph_range_at_voltage(
    points_A,
    points_B,
    voltage,
    OER_shift=DEFAULT_OER_SHIFT,
    HER_shift=DEFAULT_HER_SHIFT,
    save_plot=False,
    save_plot_path=None,
    A_phase_name="Phase A",
    B_phase_name="Phase B",
):
    """
    Find the pH range where both phase A and phase B are stable
    at a given intercalation voltage.

    Parameters
    ----------
    points_A : list of tuple
        (pH, voltage) points for phase A stability region.
    points_B : list of tuple
        (pH, voltage) points for phase B stability region.
    voltage : float
        The intercalation voltage to test.
    OER_shift : float
    HER_shift : float
    save_plot : bool
        If True, save a plot of the intersection.
    save_plot_path : str or None
        File path for the plot.
    A_phase_name, B_phase_name : str

    Returns
    -------
    tuple of float or empty tuple
        ``(min_pH, max_pH)`` if overlap exists, ``()`` otherwise.
    """
    poly_A = get_alpha_shape(points_A)
    poly_B = get_alpha_shape(points_B)
    intersection_region = poly_A.intersection(poly_B)

    all_pH = [p for p, _ in list(points_A) + list(points_B)]
    min_pH, max_pH = min(all_pH) - 1, max(all_pH) + 1
    test_line = LineString([(min_pH, voltage), (max_pH, voltage)])
    overlap_line = (
        intersection_region.intersection(test_line)
        if not intersection_region.is_empty
        else None
    )

    if save_plot:
        _plot_ph_range_intersection(
            poly_A, poly_B, voltage, OER_shift, HER_shift,
            save_plot_path, A_phase_name, B_phase_name,
        )

    if intersection_region.is_empty:
        return ()

    # Extract pH coordinates from the overlap geometry
    segs = []
    if isinstance(overlap_line, LineString):
        segs = [overlap_line]
    elif isinstance(overlap_line, Point) and not overlap_line.is_empty:
        return (overlap_line.x, overlap_line.x)
    elif overlap_line.geom_type in ("MultiLineString", "GeometryCollection"):
        segs = [g for g in overlap_line.geoms if isinstance(g, LineString)]

    if not segs:
        return ()

    xs = []
    for seg in segs:
        xs.extend([x for x, _ in seg.coords])

    if not xs:
        return ()

    return (min(xs), max(xs))


def find_stability_area_around_voltage(
    points_A,
    points_B,
    voltage,
    OER_shift=0.5,
    HER_shift=-0.3,
):
    """
    Compute the fraction of the water-stability region (pH 7-8.5)
    occupied by the overlap of two phase stability regions.

    Parameters
    ----------
    points_A : list of tuple
        (pH, voltage) points for phase A.
    points_B : list of tuple
        (pH, voltage) points for phase B.
    voltage : float
        The intercalation voltage (used to split above/below).
    OER_shift : float
    HER_shift : float

    Returns
    -------
    tuple of float
        ``(total_area_fraction, below_V_fraction, above_V_fraction)``
        where fractions are relative to the total water-bounded
        parallelogram area.
    """
    poly_A = get_alpha_shape(points_A)
    poly_B = get_alpha_shape(points_B)
    intersection_region = poly_A.intersection(poly_B)

    if intersection_region.is_empty or intersection_region.area == 0:
        return (0.0, 0.0, 0.0)

    # Water-stability parallelogram (pH 7–8.5)
    E_HER_7 = HER_shift + (-PREFAC) * 7
    E_OER_7 = 1.23 + OER_shift + (-PREFAC) * 7
    E_HER_8_5 = HER_shift + (-PREFAC) * 8.5
    E_OER_8_5 = 1.23 + OER_shift + (-PREFAC) * 8.5

    water_region = Polygon([
        (7.0, E_HER_7),
        (8.5, E_HER_8_5),
        (8.5, E_OER_8_5),
        (7.0, E_OER_7),
    ])
    total_water_area = water_region.area

    # Clip intersection to water region
    overlap_in_region = intersection_region.intersection(water_region)
    if overlap_in_region.is_empty or overlap_in_region.area == 0:
        return (0.0, 0.0, 0.0)

    area_overlap = overlap_in_region.area

    # Split into below-voltage and above-voltage regions
    min_y = min(E_HER_7, E_HER_8_5) - 10.0
    max_y = max(E_OER_7, E_OER_8_5) + 10.0

    lower_half = Polygon([
        (7.0, min_y), (8.5, min_y),
        (8.5, voltage), (7.0, voltage),
    ])
    upper_half = Polygon([
        (7.0, voltage), (8.5, voltage),
        (8.5, max_y), (7.0, max_y),
    ])

    under_region = overlap_in_region.intersection(lower_half)
    above_region = overlap_in_region.intersection(upper_half)

    area_under = under_region.area if not under_region.is_empty else 0.0
    area_above = above_region.area if not above_region.is_empty else 0.0

    return (
        area_overlap / total_water_area,
        area_under / total_water_area,
        area_above / total_water_area,
    )


def plot_metastability_overlap(
    ax,
    poly_A,
    poly_B,
    voltage,
    A_phase_name,
    B_phase_name,
    OER_shift=DEFAULT_OER_SHIFT,
    HER_shift=DEFAULT_HER_SHIFT,
    x_range=(6.0, 9.5),
    ph_window=(7.0, 8.5),
    ylim=(-1.0, 1.5),
):
    """Draw a seawater metastability-overlap diagram onto ``ax``.

    Reproduces the publication look of figure 3 panel b: the seawater stability
    window drawn with blue dashed borders (vertical lines at the ``ph_window``
    edges plus the diagonal H/O lines), the charged phase as a transparent blue
    region, the discharged phase as a transparent red region, and their overlap
    as a purple region with a black dashed boundary. A crimson line marks the
    intercalation ``voltage`` (annotated to two decimals) and a thick dark-red
    segment marks the pH overlap at that voltage.

    ``A_phase_name``/``B_phase_name`` may be ``"Formula"`` or ``"Formula\\nmp-id"``;
    stoichiometric digits in the formula are subscripted for the legend.

    The legend is placed outside the axes on the right via a figure legend, so
    ``ax`` must belong to a figure using ``layout="constrained"`` (keeps the
    total figure size fixed while shrinking the axes to make room).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from shapely.geometry import MultiLineString, GeometryCollection

    x_min, x_max = x_range
    lw = plt.rcParams["axes.linewidth"]

    ax.set_facecolor(PLOT_BACKGROUND_COLOR)

    # --- seawater stability window: blue dashed borders ---
    a_H, b_H, c_H = PREFAC, 1, -HER_shift
    a_O, b_O, c_O = PREFAC, 1, -1.23 - OER_shift
    line_H = LineString([
        (x_min, -(a_H * x_min + c_H) / b_H),
        (x_max, -(a_H * x_max + c_H) / b_H),
    ])
    line_O = LineString([
        (x_min, -(a_O * x_min + c_O) / b_O),
        (x_max, -(a_O * x_max + c_O) / b_O),
    ])
    ax.plot(*line_H.xy, linestyle="--", color=SEAWATER_LINE_COLOR, linewidth=lw)
    ax.plot(*line_O.xy, linestyle="--", color=SEAWATER_LINE_COLOR, linewidth=lw)
    for ph in ph_window:
        ax.axvline(ph, linestyle="--", color=SEAWATER_LINE_COLOR, linewidth=lw)

    # --- phase metastability regions (transparent blue / red) ---
    for poly, name, color, alpha in [
        (poly_A, _subscript_formula(A_phase_name), CHARGED_COLOR, 0.3),
        (poly_B, _subscript_formula(B_phase_name), DISCHARGED_COLOR, 0.2),
    ]:
        if poly.is_empty:
            continue
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha, label=name)
        elif isinstance(poly, LineString):
            ax.plot(*poly.xy, color=color, label=name)
        elif isinstance(poly, Point):
            ax.scatter([poly.x], [poly.y], color=color, label=name)

    # --- metastability overlap (purple) + black dashed boundary ---
    # The dashed black border is part of the fill patch (rgba facecolor carries the
    # alpha; the edge stays opaque) so it also shows in the legend swatch.
    intersection_region = poly_A.intersection(poly_B)
    if not intersection_region.is_empty and isinstance(intersection_region, Polygon):
        xI, yI = intersection_region.exterior.xy
        ax.fill(xI, yI, facecolor=to_rgba(INTERSECTION_COLOR, 0.6),
                edgecolor="k", linestyle="--", linewidth=lw,
                label="Metastability\noverlap")

    # --- intercalation voltage line + two-decimal annotation ---
    ax.plot([x_min, x_max], [voltage, voltage],
            color=VOLTAGE_LINE_COLOR, linewidth=lw)
    ax.text(x_max - 0.1, voltage + 0.04, f"{voltage:.2f} V",
            color=VOLTAGE_LINE_COLOR, fontweight="bold", ha="right", va="bottom")

    # --- voltage-pH overlap: thick dark-red segment at the voltage ---
    overlap_line = (
        intersection_region.intersection(
            LineString([(x_min, voltage), (x_max, voltage)]))
        if not intersection_region.is_empty else None
    )
    if overlap_line is not None and not overlap_line.is_empty:
        if isinstance(overlap_line, LineString):
            segs = [overlap_line]
        elif isinstance(overlap_line, (MultiLineString, GeometryCollection)):
            segs = [g for g in overlap_line.geoms if isinstance(g, LineString)]
        else:
            segs = []
        labeled = False
        for seg in segs:
            ax.plot(*seg.xy, color=VOLTAGE_OVERLAP_COLOR, linewidth=1.5 * lw,
                    label=None if labeled else "Voltage")
            labeled = True

    # --- axes cosmetics ---
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(ylim)
    ax.set_xlabel("pH")
    ax.set_ylabel("Applied potential (V vs SHE)")

    # x-ticks every 0.5; colour the seawater-window edges blue + bold.
    ticks = np.arange(x_min, x_max + 1e-9, 0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    window_labels = {f"{ph:.1f}" for ph in ph_window}
    for label in ax.get_xticklabels():
        if label.get_text() in window_labels:
            label.set_color(SEAWATER_LINE_COLOR)
            label.set_fontweight("bold")

    # legend outside, right (figure legend → constrained layout keeps size fixed)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.figure.legend(handles, labels, loc="outside right upper",
                           borderaxespad=0.0)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(lw)

    # Thin the dashed-black border on the metastability-overlap swatch only (the
    # legend Patch inherits the on-plot fill's linewidth, which looks too heavy at
    # swatch scale); the on-plot boundary keeps its full lw.
    legend_handles = getattr(leg, "legend_handles", None) or leg.legendHandles
    for handle, label in zip(legend_handles, labels):
        if label == "Metastability\noverlap":
            handle.set_linewidth(0.5 * lw)

    # Cut the axes<->legend gap to 40% (the gap of an "outside" legend is set by the
    # constrained-layout w_pad, not borderaxespad), giving the plot more width.
    eng = ax.figure.get_layout_engine()
    if eng is not None:
        eng.set(w_pad=0.4 * plt.rcParams["figure.constrained_layout.w_pad"])


def _plot_ph_range_intersection(
    poly_A, poly_B, voltage, OER_shift, HER_shift,
    save_plot_path, A_phase_name, B_phase_name,
):
    """Render the per-pair metastability-overlap diagram to a vector PDF.

    Thin wrapper around :func:`plot_metastability_overlap` that creates the
    figure at the figure-3 panel-b size and saves it.
    """
    import matplotlib.pyplot as plt

    apply_publication_style()
    fig, ax = plt.subplots(
        figsize=(HALF_DOUBLE_COL_WIDTH, HALF_DOUBLE_COL_HEIGHT),
        layout="constrained",
    )
    plot_metastability_overlap(
        ax, poly_A, poly_B, voltage, A_phase_name, B_phase_name,
        OER_shift=OER_shift, HER_shift=HER_shift,
    )
    if save_plot_path:
        # Save at the exact figure size (constrained layout already fits the
        # outside-right legend inside the canvas), matching figure-3 panel b.
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(save_plot_path, format="pdf")
    plt.close(fig)

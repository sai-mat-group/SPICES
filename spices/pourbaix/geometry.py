"""
Geometric overlap calculations for Pourbaix phase regions and water
stability lines, including alpha-shape analysis for stability area
computations.
"""

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

from spices.config import PREFAC, H_LINE, O_LINE


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
    OER_shift=0.5,
    HER_shift=-0.3,
    save_plot=False,
    save_plot_path=None,
    title=None,
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
    title : str or None
    A_phase_name, B_phase_name : str

    Returns
    -------
    tuple of float or empty tuple
        ``(min_pH, max_pH)`` if overlap exists, ``()`` otherwise.
    """
    poly_A = get_alpha_shape(points_A)
    poly_B = get_alpha_shape(points_B)
    intersection_region = poly_A.intersection(poly_B)

    if save_plot:
        _plot_ph_range_intersection(
            poly_A, poly_B, intersection_region, voltage,
            OER_shift, HER_shift, save_plot_path, title,
            A_phase_name, B_phase_name,
        )

    if intersection_region.is_empty:
        return ()

    # Horizontal test line at the given voltage
    all_pH = [p for p, _ in list(points_A) + list(points_B)]
    min_pH, max_pH = min(all_pH) - 1, max(all_pH) + 1
    test_line = LineString([(min_pH, voltage), (max_pH, voltage)])
    overlap_line = intersection_region.intersection(test_line)

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


def _plot_ph_range_intersection(
    poly_A, poly_B, intersection_region, voltage,
    OER_shift, HER_shift, save_plot_path, title,
    A_phase_name, B_phase_name,
):
    """Internal helper: plot the pH-range intersection diagram."""
    import matplotlib.pyplot as plt

    plt.figure(dpi=150, figsize=(6.4, 3.8))
    if title:
        plt.title(title)
    plt.xlabel("pH")
    plt.ylabel("Voltage (V)")
    plt.grid(True)

    # Water-stability lines
    x_min, x_max = 6, 9.5
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
    plt.plot(*line_H.xy, linestyle="-.", label="H (meta)stability")
    plt.plot(*line_O.xy, linestyle="-.", label="O (meta)stability")

    # Phase polygons
    for poly, name, fill_color, edge_color in [
        (poly_A, A_phase_name, "indianred", "chocolate"),
        (poly_B, B_phase_name, "dodgerblue", "slateblue"),
    ]:
        if poly.is_empty:
            continue
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.3, label=name, color=fill_color)
            plt.plot(x, y, color=edge_color)
        elif isinstance(poly, LineString):
            plt.plot(*poly.xy, label=name, color=fill_color)
        elif isinstance(poly, Point):
            plt.scatter([poly.x], [poly.y], label=name, color=edge_color)

    # Intersection boundary
    if not intersection_region.is_empty and isinstance(intersection_region, Polygon):
        xI, yI = intersection_region.exterior.xy
        plt.plot(xI, yI, "--", label="Intersection", color="k")

    plt.legend(
        loc="upper left", bbox_to_anchor=(1.05, 1),
        borderaxespad=0., fontsize=8,
    )
    plt.tight_layout()
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=200)
    plt.close()

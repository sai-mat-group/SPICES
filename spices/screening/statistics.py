"""
Stage 5: Statistics and publication-ready figures.

Aggregates statistics across all 5 working ions, applies the final
pH + area filters, and generates the pipeline figures: a per-ion bar
chart, per-ion filter funnels, the screening-space periodic table, and
the structure-voltage figures (ion-selective + multi-ion).

Replaces ``stat_calculation_utils.py`` and the ``stats-codes.ipynb``,
``Bar-plots.ipynb``, ``Venn-diagram-plots.ipynb``, ``periodic-table.ipynb``,
and ``structure-voltage-*.ipynb`` notebooks.
"""

import ast
import json
import logging
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

from spices.config import (
    DEFAULT_ANIONS,
    DEFAULT_HER_SHIFT,
    DEFAULT_OER_SHIFT,
    DEFAULT_TRANSITION_METALS,
    ION_COLORS,
    PREFAC,
    SEAWATER_PH_RANGE,
    SUPPORTED_ELEMENTS,
)
from spices.plotting import (
    DOUBLE_COL_WIDTH,
    apply_publication_style,
)

# NOTE: ``apply_publication_style()`` is intentionally NOT called at import
# time (Fix #8). Each plotting function calls it (idempotent) instead, so
# merely importing this module no longer mutates global matplotlib state.

logger = logging.getLogger(__name__)


# ====================================================================
# Statistics functions
# ====================================================================


def get_stability_bool(x):
    """
    Convert a metastability cell to boolean.

    Handles string representations of lists (from Excel), actual
    lists/tuples, NaN/None, and numeric values (any non-zero, non-NaN
    number -> True, so ``int 2`` and ``float 2.0`` agree; Fix #11).

    Parameters
    ----------
    x : str or list or float or int or bool or None

    Returns
    -------
    bool
    """
    if isinstance(x, str):
        return len(ast.literal_eval(x)) > 0
    if isinstance(x, (list, tuple)):
        return len(x) > 0
    if x is None:
        return False
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return False
    if isinstance(x, (bool, int, float, np.integer, np.floating, np.bool_)):
        return bool(x)
    return False


def _ph_range_covers(tuple_str, lo, hi, tol=0.02):
    """
    Test whether a ``"(pH_min, pH_max)"`` string COVERS the window
    ``[lo, hi]`` (placement, not just width; Fix #5).

    ``tol`` (~0.02) absorbs interpolation jitter at the pH grid step.
    Returns False on non-str / parse error.
    """
    if not isinstance(tuple_str, str):
        return False
    try:
        a, b = ast.literal_eval(tuple_str)
    except (ValueError, SyntaxError, TypeError):
        return False
    return (a <= lo + tol) and (b >= hi - tol)


def _get_pH_range_bool(tuple_str, min_pH_range=1.5):
    """
    Width-only pH test (legacy).

    Kept for backward compatibility; prefer :func:`_ph_range_covers`,
    which also checks placement against the seawater window.
    """
    if not isinstance(tuple_str, str):
        return False
    try:
        pH_min, pH_max = ast.literal_eval(tuple_str)
    except (ValueError, SyntaxError, TypeError):
        return False
    return round(pH_max - pH_min, 1) >= min_pH_range


def count_phase_stabilities(stability_excel, criteria=None):
    """
    Count (meta)stable phases per criterion from Stage 2 output.

    Parameters
    ----------
    stability_excel : str or pd.DataFrame
        Path to the Stage 2 seawater stability Excel file, or a DataFrame.
    criteria : list of int or None
        Selection criteria to count (default: ``[4]``). A missing
        criterion column is warned about and counted as 0 (Fix #3).

    Returns
    -------
    pd.Series
        Counts: ``"Seawater stability"`` plus one
        ``"Seawater metastability (Criterion N) bool"`` per criterion.
    """
    if criteria is None:
        criteria = [4]

    if isinstance(stability_excel, pd.DataFrame):
        df = stability_excel
    else:
        df = pd.read_excel(stability_excel)

    counts = {}
    if "Seawater stability" in df.columns:
        counts["Seawater stability"] = int(
            df["Seawater stability"].apply(get_stability_bool).sum()
        )

    for sc in criteria:
        col = f"Seawater metastability (Criterion {sc})"
        bool_col = f"{col} bool"
        if col in df.columns:
            counts[bool_col] = int(df[col].apply(get_stability_bool).sum())
        else:
            logger.warning(
                "Stage-2 column %r absent; counting 0 for criterion %d.",
                col, sc,
            )
            counts[bool_col] = 0

    series = pd.Series(counts)
    logger.info(
        "Total phases: %d. Counts:\n%s", len(df), series.to_string(),
    )
    return series


def count_phases_with_voltage_stability(
    voltage_excel_path,
    stability_excel_path,
    metastability_criterion=4,
    pH_window=None,
    min_area_fraction=None,
    experimental=None,
):
    """
    Count unique MP IDs with both seawater (meta)stability AND
    intercalation voltage data, optionally filtered.

    Parameters
    ----------
    voltage_excel_path : str or pd.DataFrame
        Path to the Stage 3 voltage plateau Excel, or a DataFrame.
        When a path is given the ``criterion N`` sheet is read.
    stability_excel_path : str or pd.DataFrame
        Path to the Stage 2 stability Excel, or a DataFrame.
    metastability_criterion : int
    pH_window : tuple of float or None
        E.g. ``(7.0, 8.5)``. Keeps only plateaus whose pH range COVERS
        the window (Fix #5). Must lie within the seawater window.
    min_area_fraction : float or None
        Minimum metastability area fraction (0-1).
    experimental : bool or None
        If set, keep only plateaus with a Charged/Discharged
        experimental flag equal to ``experimental`` (Fix #10).

    Returns
    -------
    dict
        ``{'total_phases': int, 'total_plots': int, 'filtered_df': DataFrame}``
    """
    if isinstance(stability_excel_path, pd.DataFrame):
        stability_df = stability_excel_path
    else:
        stability_df = pd.read_excel(stability_excel_path)

    if isinstance(voltage_excel_path, pd.DataFrame):
        voltage_df = voltage_excel_path
    else:
        voltage_df = pd.read_excel(
            voltage_excel_path,
            sheet_name=f"criterion {metastability_criterion}",
        )

    pH_col = f"pH range metastability criterion {metastability_criterion}"

    if pH_window is not None:
        lo, hi = pH_window
        sw_lo, sw_hi = SEAWATER_PH_RANGE
        if lo < sw_lo or hi > sw_hi:
            raise ValueError(
                f"pH_window {pH_window} must lie within the seawater "
                f"window {SEAWATER_PH_RANGE}."
            )
        voltage_df = voltage_df[
            voltage_df[pH_col].apply(
                lambda x: _ph_range_covers(str(x), lo, hi)
            )
        ]

    if min_area_fraction is not None:
        voltage_df = voltage_df[
            voltage_df["Total metastability area fraction"]
            >= min_area_fraction
        ]

    if experimental is not None:
        for col in ("Charged Is experimental", "Discharged Is experimental"):
            if col not in voltage_df.columns:
                raise ValueError(
                    f"Column {col!r} is required for the experimental "
                    f"filter but is absent from the voltage data."
                )
        voltage_df = voltage_df[
            (voltage_df["Charged Is experimental"] == experimental)
            | (voltage_df["Discharged Is experimental"] == experimental)
        ]

    # Count unique MP IDs present in both stability and voltage data.
    all_valid_mpids = set(stability_df["MP id"].unique())
    charged_mpids = {
        mpid for mpid in voltage_df["Charged MP id"].unique()
        if mpid in all_valid_mpids
    }
    discharged_mpids = {
        mpid for mpid in voltage_df["Discharged MP id"].unique()
        if mpid in all_valid_mpids
    }
    all_mpids = discharged_mpids.union(charged_mpids)

    result = {
        "total_phases": len(all_mpids),
        "total_plots": voltage_df["Plot number"].nunique(),
        "filtered_df": voltage_df,
    }

    logger.info(
        "Criterion %d: %d phases, %d plots (pH=%s, area>=%s, expt=%s)",
        metastability_criterion, result["total_phases"],
        result["total_plots"], pH_window, min_area_fraction, experimental,
    )
    return result


def _resolve_stage2_excel(ion_dir, ion):
    """
    Resolve the Stage-2 seawater-metastability Excel for *ion* (Fix #1).

    Rule (per user spec):

    1. Prefer the canonical ``"{ion}_cathodes_seawater_metastability.xlsx"``,
       searched recursively (handles nested ``run2/`` subdirs, e.g. K).
       If more than one canonical file exists, warn and pick the first.
    2. Otherwise fall back to the ``.xlsx`` files directly in *ion_dir*:
       exactly one -> use it with a warning; zero -> ``FileNotFoundError``;
       two or more (none canonical) -> ``ValueError`` (ambiguous).

    The canonical-name match means stale ``*_old.xlsx`` and decoy
    ``*_mp_data.xlsx`` files are ignored cleanly when the real file exists.
    """
    canonical = f"{ion}_cathodes_seawater_metastability.xlsx"
    matches = sorted(glob(os.path.join(ion_dir, "**", canonical), recursive=True))
    if matches:
        if len(matches) > 1:
            logger.warning(
                "Multiple %r files under %s; using %s",
                canonical, ion_dir, matches[0],
            )
        return matches[0]

    xlsx = sorted(glob(os.path.join(ion_dir, "*.xlsx")))
    if len(xlsx) == 1:
        logger.warning(
            "No %r in %s; reading the single .xlsx present: %s",
            canonical, ion_dir, os.path.basename(xlsx[0]),
        )
        return xlsx[0]
    if not xlsx:
        raise FileNotFoundError(
            f"No .xlsx found in {ion_dir} (expected {canonical})"
        )
    raise ValueError(
        f"Multiple .xlsx files in {ion_dir} and none named {canonical}: "
        f"{[os.path.basename(p) for p in xlsx]}. Rename the correct file "
        f"to {canonical} to disambiguate."
    )


# ====================================================================
# Visualization helpers
# ====================================================================


def _subscript_formula(formula_mpid):
    """
    Subscript the digits of a chemical formula for matplotlib mathtext.

    Ported verbatim from the structure-voltage notebooks. The input may
    be ``"Formula"`` or ``"Formula\\nmp-id"``; the mp-id line is preserved.

    Example: ``"FePO4\\nmp-25303"`` -> ``"FePO$_{4}$\\nmp-25303"``.
    """
    parts = formula_mpid.split("\n")
    formatted = re.sub(r"(\d+(\.\d+)?)", r"$_{\1}$", parts[0])
    if len(parts) > 1:
        return f"{formatted}\n{parts[1]}"
    return formatted


def _set_formula_xticklabels(ax, labels, tick_fontsize, tick_pad=15):
    """
    Set rotated x-tick labels to the formula only, returning the mp-id of
    each tick for separate styling (E6, change 1a).

    ``labels`` are the ``"{formula-mathtext}\\n{mp-id}"`` strings produced
    by :func:`_subscript_formula`. The formula becomes the tick label; the
    mp-id is returned (empty string when a label has no mp-id line) so the
    caller can draw it smaller and in a different color via
    :func:`_draw_mpid_labels` after layout.

    ``tick_pad`` (points) pushes the formula away from the x-spine so the
    label text does not clip the spine line; matplotlib's default
    (~3.5 pt) is too small for the 45-degree anchored labels here.
    """
    formulas, mpids = [], []
    for lab in labels:
        parts = lab.split("\n")
        formulas.append(parts[0])
        mpids.append(parts[1] if len(parts) > 1 else "")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        formulas, rotation=45, ha="center", va="top",
        rotation_mode="anchor", fontsize=tick_fontsize,
    )
    ax.tick_params(axis="x", pad=tick_pad)
    return mpids


def _draw_mpid_labels(ax, mpids, mpid_fontsize, mpid_color):
    """
    Draw the mp-id of each x-tick one line below its 45-degree formula label,
    smaller and in ``mpid_color`` (change 1a / change 4), so the
    formula+mp-id reads as one stacked, rotated block.

    Uses a fixed offset (in points) perpendicular to the 45-degree baseline,
    so placement is length-independent (no canvas bbox measurement). Should be
    called AFTER ``tight_layout``; the figure is then saved with
    ``bbox_inches="tight"`` to include the added text.
    """
    # One line below the formula, following the 45-degree block: shifting
    # (0, -lh) by +45 deg CCW gives (+0.71*lh, -0.71*lh) in offset points.
    lh = 4.0 * mpid_fontsize
    dx, dy = 0 * lh, -1.5 * lh
    for i, mpid in enumerate(mpids):
        if not mpid:
            continue
        ax.annotate(
            mpid, xy=(i, 0), xycoords=("data", "axes fraction"),
            xytext=(dx, dy), textcoords="offset points",
            rotation=45, rotation_mode="anchor", ha="center", va="top",
            color=mpid_color, fontsize=mpid_fontsize, zorder=5,
        )


def _water_stability_ref_lines(ref_pH=None, apply_shifts=True):
    """
    O/H water-stability reference voltages derived from config (E5).

    Replaces the notebooks' literal ``1.23`` / ``-0.71`` axhlines.
    Evaluates the O and H lines at *ref_pH* (default: midpoint of
    ``SEAWATER_PH_RANGE`` = 7.75), optionally adding the seawater
    OER/HER overpotential shifts.

    Returns ``(O, H)``. At ref_pH=7.75 with shifts -> ~ (1.272, -0.758).
    """
    if ref_pH is None:
        ref_pH = sum(SEAWATER_PH_RANGE) / 2.0
    o_line = 1.23 - PREFAC * ref_pH   # from O_LINE: V = 1.23 - PREFAC*pH
    h_line = -PREFAC * ref_pH          # from H_LINE: V = -PREFAC*pH
    if apply_shifts:
        o_line += DEFAULT_OER_SHIFT
        h_line += DEFAULT_HER_SHIFT
    return o_line, h_line


# Periodic-table layout: (atomic_number, symbol, name, mass, row, col).
# Ported verbatim from peridoic_table/periodic-table.ipynb (cell 2);
# f-block rows omitted (placeholder labels drawn instead).
_PERIODIC_TABLE_ELEMENTS = [
    (1, "H", "Hydrogen", 1.008, 1, 1),
    (2, "He", "Helium", 4.0026, 1, 18),
    (3, "Li", "Lithium", 6.94, 2, 1),
    (4, "Be", "Beryllium", 9.0122, 2, 2),
    (5, "B", "Boron", 10.81, 2, 13),
    (6, "C", "Carbon", 12.011, 2, 14),
    (7, "N", "Nitrogen", 14.007, 2, 15),
    (8, "O", "Oxygen", 15.999, 2, 16),
    (9, "F", "Fluorine", 18.998, 2, 17),
    (10, "Ne", "Neon", 20.180, 2, 18),
    (11, "Na", "Sodium", 22.990, 3, 1),
    (12, "Mg", "Magnesium", 24.305, 3, 2),
    (13, "Al", "Aluminum", 26.982, 3, 13),
    (14, "Si", "Silicon", 28.085, 3, 14),
    (15, "P", "Phosphorus", 30.974, 3, 15),
    (16, "S", "Sulfur", 32.06, 3, 16),
    (17, "Cl", "Chlorine", 35.45, 3, 17),
    (18, "Ar", "Argon", 39.948, 3, 18),
    (19, "K", "Potassium", 39.098, 4, 1),
    (20, "Ca", "Calcium", 40.078, 4, 2),
    (21, "Sc", "Scandium", 44.956, 4, 3),
    (22, "Ti", "Titanium", 47.867, 4, 4),
    (23, "V", "Vanadium", 50.942, 4, 5),
    (24, "Cr", "Chromium", 51.996, 4, 6),
    (25, "Mn", "Manganese", 54.938, 4, 7),
    (26, "Fe", "Iron", 55.845, 4, 8),
    (27, "Co", "Cobalt", 58.933, 4, 9),
    (28, "Ni", "Nickel", 58.693, 4, 10),
    (29, "Cu", "Copper", 63.546, 4, 11),
    (30, "Zn", "Zinc", 65.38, 4, 12),
    (31, "Ga", "Gallium", 69.723, 4, 13),
    (32, "Ge", "Germanium", 72.63, 4, 14),
    (33, "As", "Arsenic", 74.922, 4, 15),
    (34, "Se", "Selenium", 78.971, 4, 16),
    (35, "Br", "Bromine", 79.904, 4, 17),
    (36, "Kr", "Krypton", 83.798, 4, 18),
    (37, "Rb", "Rubidium", 85.468, 5, 1),
    (38, "Sr", "Strontium", 87.62, 5, 2),
    (39, "Y", "Yttrium", 88.906, 5, 3),
    (40, "Zr", "Zirconium", 91.224, 5, 4),
    (41, "Nb", "Niobium", 92.906, 5, 5),
    (42, "Mo", "Molybdenum", 95.95, 5, 6),
    (43, "Tc", "Technetium", 98, 5, 7),
    (44, "Ru", "Ruthenium", 101.07, 5, 8),
    (45, "Rh", "Rhodium", 102.91, 5, 9),
    (46, "Pd", "Palladium", 106.42, 5, 10),
    (47, "Ag", "Silver", 107.87, 5, 11),
    (48, "Cd", "Cadmium", 112.41, 5, 12),
    (49, "In", "Indium", 114.82, 5, 13),
    (50, "Sn", "Tin", 118.71, 5, 14),
    (51, "Sb", "Antimony", 121.76, 5, 15),
    (52, "Te", "Tellurium", 127.60, 5, 16),
    (53, "I", "Iodine", 126.90, 5, 17),
    (54, "Xe", "Xenon", 131.29, 5, 18),
    (55, "Cs", "Cesium", 132.91, 6, 1),
    (56, "Ba", "Barium", 137.33, 6, 2),
    (72, "Hf", "Hafnium", 178.49, 6, 4),
    (73, "Ta", "Tantalum", 180.95, 6, 5),
    (74, "W", "Tungsten", 183.84, 6, 6),
    (75, "Re", "Rhenium", 186.21, 6, 7),
    (76, "Os", "Osmium", 190.23, 6, 8),
    (77, "Ir", "Iridium", 192.22, 6, 9),
    (78, "Pt", "Platinum", 195.08, 6, 10),
    (79, "Au", "Gold", 196.97, 6, 11),
    (80, "Hg", "Mercury", 200.59, 6, 12),
    (81, "Tl", "Thallium", 204.38, 6, 13),
    (82, "Pb", "Lead", 207.2, 6, 14),
    (83, "Bi", "Bismuth", 208.98, 6, 15),
    (84, "Po", "Polonium", 209, 6, 16),
    (85, "At", "Astatine", 210, 6, 17),
    (86, "Rn", "Radon", 222, 6, 18),
    (87, "Fr", "Francium", 223, 7, 1),
    (88, "Ra", "Radium", 226, 7, 2),
    (104, "Rf", "Rutherfordium", 267, 7, 4),
    (105, "Db", "Dubnium", 268, 7, 5),
    (106, "Sg", "Seaborgium", 271, 7, 6),
    (107, "Bh", "Bohrium", 270, 7, 7),
    (108, "Hs", "Hassium", 277, 7, 8),
    (109, "Mt", "Meitnerium", 278, 7, 9),
    (110, "Ds", "Darmstadtium", 281, 7, 10),
    (111, "Rg", "Roentgenium", 282, 7, 11),
    (112, "Cn", "Copernicium", 285, 7, 12),
    (113, "Nh", "Nihonium", 286, 7, 13),
    (114, "Fl", "Flerovium", 289, 7, 14),
    (115, "Mc", "Moscovium", 290, 7, 15),
    (116, "Lv", "Livermorium", 293, 7, 16),
    (117, "Ts", "Tennessine", 294, 7, 17),
    (118, "Og", "Oganesson", 294, 7, 18),
]


# ====================================================================
# Visualization
# ====================================================================

# Pipeline bar-chart palette: cyan (Total), violet (Metastable), yellow
# (Stable), as used in the 3rd_Iteration bar plots (skyblue + tab20b 2/20 +
# tab20b 10/20).
PIPELINE_BAR_COLORS = ["#87ceeb", "#6b6ecf", "#e7ba52"]

# Left-to-right ion order for the bar chart (publication panel b).
PIPELINE_BAR_ION_ORDER = ["Li", "Mg", "Ca", "Na", "K"]


def plot_pipeline_bar_chart(
    stage_data,
    save_path=None,
    figsize=(0.35 * DOUBLE_COL_WIDTH, 110 / 25.4),
    color_by="stage",
    yscale="log",
):
    """
    Grouped bar chart of phase counts per ion per pipeline stage.

    Parameters
    ----------
    stage_data : dict
        Maps ion -> dict of ``{stage_label: count}``.
    save_path : str or None
        Output path; the format follows the extension (``.pdf`` -> vector).
    figsize : tuple
        Default = panel-b size (0.35 * double-col width wide, 110 mm tall)
        = (2.36, 4.33) in: a tall narrow panel.
    color_by : {"stage", "ion"}
        ``"stage"`` (default): one consistent color per stage from
        ``PIPELINE_BAR_COLORS`` (cyan, violet, yellow); legend maps
        stage -> color; ion identity is the x-axis. ``"ion"``: bars use
        ``ION_COLORS[ion]`` with stage encoded by alpha; legend maps
        stage -> opacity via neutral-grey proxy patches. Both legends are
        honest (Fix #2).
    yscale : {"log", "linear"}
        Counts span ~1344 -> 3, so "log" (default) is usually needed;
        a positive floor is set so bars start sensibly (Fix #6).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    elements = list(stage_data.keys())
    stage_labels = list(stage_data[elements[0]].keys())
    n_stages = len(stage_labels)
    n_elements = len(elements)

    x = np.arange(n_elements)
    width = 0.8 / n_stages

    fig, ax = plt.subplots(figsize=figsize)

    stage_palette = [
        PIPELINE_BAR_COLORS[i % len(PIPELINE_BAR_COLORS)] for i in range(n_stages)
    ]
    stage_alphas = [0.3 + 0.7 * (i / max(n_stages - 1, 1)) for i in range(n_stages)]

    for i, stage in enumerate(stage_labels):
        values = [stage_data[e].get(stage, 0) for e in elements]
        offset = (i - n_stages / 2 + 0.5) * width
        if color_by == "ion":
            colors = [ION_COLORS.get(e, "gray") for e in elements]
            alpha = stage_alphas[i]
            label = None
        else:  # "stage"
            colors = stage_palette[i]
            alpha = 0.9
            label = stage
        bars = ax.bar(
            x + offset, values, width, label=label,
            color=colors, alpha=alpha, edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:  # skip zero-height bars (undefined on log)
                # Left-align labels on interior bars (the middle "Metastable"
                # bar) so digits start at the bar's left edge and clear the
                # neighbouring label; center the outer (first/last) bars.
                if 0 < i < n_stages - 1:
                    tx, halign = bar.get_x(), "left"
                else:
                    tx, halign = bar.get_x() + bar.get_width() / 2, "center"
                ax.text(
                    tx, bar.get_height(), str(val),
                    ha=halign, va="bottom", fontsize=5, fontweight="bold",
                )

    ax.set_ylabel("No. of structures")
    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontweight="bold")

    # Headroom above the tallest bar so the top legend clears it.
    all_vals = [stage_data[e].get(s, 0) for e in elements for s in stage_labels]
    max_val = max(all_vals) if all_vals else 1
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8, top=max_val * 4)
    else:
        ax.set_ylim(top=max_val * 1.3)
    ax.grid(axis="y", alpha=0.3)

    if color_by == "ion":
        proxies = [
            Patch(facecolor="grey", alpha=stage_alphas[i], edgecolor="black",
                  label=stage)
            for i, stage in enumerate(stage_labels)
        ]
        ax.legend(handles=proxies, loc="upper right", ncol=1)
    else:
        ax.legend(loc="upper right", ncol=1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Bar chart saved to %s", save_path)

    return fig


def plot_nested_venn(
    element,
    counts,
    labels,
    save_path=None,
):
    """
    Filter funnel: concentric, area-proportional circles showing the
    progressive narrowing of the screening at each pipeline stage.

    (This is a nested-circle funnel, not a true Venn diagram and not
    "vertically" stacked; Fix #7.) Each circle's radius is proportional
    to ``sqrt(count / counts[0])``, so areas are proportional to counts.
    Labels are placed to the side with leader lines to avoid overlap when
    counts are close.

    Parameters
    ----------
    element : str
        Ion name (for title and color).
    counts : list of int
        Decreasing counts, e.g. [metastable, with_voltage, filtered].
    labels : list of str
        Label for each level.
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.set_aspect("equal")

    max_radius = 2.0
    n = len(counts)
    color = ION_COLORS.get(element, "gray")

    radii = []
    for i, count in enumerate(counts):
        radius = max_radius * np.sqrt(count / counts[0]) if counts[0] else 0
        radii.append(radius)
        circle = Circle(
            (0, 0), radius,
            fill=True, facecolor=color,
            alpha=min(0.15 + 0.2 * i, 1.0),
            edgecolor="black", linewidth=1.0,
        )
        ax.add_patch(circle)

    # Stack labels on the right with leader lines to each circle's top.
    label_x = max_radius + 0.8
    label_ys = np.linspace(max_radius, -max_radius, n) if n > 1 else [0]
    for label, count, radius, ly in zip(labels, counts, radii, label_ys):
        ax.annotate(
            f"{label}\n({count})",
            xy=(0, radius), xytext=(label_x, ly),
            ha="left", va="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="grey"),
        )

    ax.set_xlim(-max_radius - 0.5, max_radius + 3.0)
    ax.set_ylim(-max_radius - 0.5, max_radius + 0.5)
    ax.set_title(element, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Filter funnel saved to %s", save_path)

    return fig


def plot_periodic_table(
    transition_metals=None,
    anions=None,
    save_path=None,
    figsize=(0.7 * DOUBLE_COL_WIDTH, 0.7 * DOUBLE_COL_WIDTH * 8 / 18),
):
    """
    Periodic table highlighting the screened transition-metal and anion
    sets (E1). Highlight membership is by element SYMBOL and derived from
    config, so updating ``DEFAULT_TRANSITION_METALS`` / ``DEFAULT_ANIONS``
    (e.g. adding C, Si) updates both the highlighting and the count labels.

    Parameters
    ----------
    transition_metals : list of str or None
        Defaults to ``DEFAULT_TRANSITION_METALS``.
    anions : list of str or None
        Defaults to ``DEFAULT_ANIONS``.
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style()
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if transition_metals is None:
        transition_metals = DEFAULT_TRANSITION_METALS
    if anions is None:
        anions = DEFAULT_ANIONS
    tm_set = set(transition_metals)
    anion_set = set(anions)

    tm_color = "#87CEEB"
    anion_color = "#F0E442"
    base_color = "#D7D7E4"

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(0.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Rounded, near-flush cells: bold atomic number (top-left) + larger,
    # non-bold symbol only (element name & atomic mass dropped for a clean
    # panel look). ``gap`` insets each box inside its unit cell so the rounded
    # corners read and neighbours sit nearly flush with a thin white edge;
    # a small ``gap`` keeps that inter-box white hairline-thin. The number is
    # tucked into the far top-left corner (small, non-bold) and the symbol is
    # nudged just below centre so the two never crowd each other.
    gap, radius, margin = 0.005, 0.12, 0.13  # cell inset / corner radius / number inset
    side = 1 - gap
    sym_dy = 0.08  # drop symbol slightly below cell centre
    for num, sym, name, mass, row, col in _PERIODIC_TABLE_ELEMENTS:
        x, y = col, 9 - row  # invert y for top-down view
        color = base_color
        if sym in tm_set:
            color = tm_color
        elif sym in anion_set:
            color = anion_color
        ax.add_patch(patches.FancyBboxPatch(
            (x - side / 2, y - side / 2), side, side,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0.4, edgecolor="white", facecolor=color,
        ))
        ax.text(x - side / 2 + margin, y + side / 2 - margin, str(num),
                ha="left", va="top", fontsize=3.0, fontweight="normal")
        ax.text(x, y - sym_dy, sym, ha="center", va="center",
                fontsize=7, fontweight="light")

    # Lanthanide / actinide placeholders.
    ax.text(3, 9 - 6, "57–71", ha="center", va="center",
            fontsize=3, fontstyle="italic")
    ax.text(3, 9 - 7, "89–103", ha="center", va="center",
            fontsize=3, fontstyle="italic")

    # Group legend: small rounded colour-swatch + name chips, placed side by
    # side on one row in the empty upper-middle. Swatch is sized to roughly the
    # 6 pt label height (not much larger); names only (counts dropped).
    chips = []

    def _legend_chip(cx, cy, swatch_color, label):
        sw = 0.42  # ~ the 6 pt label height in data units
        ax.add_patch(patches.FancyBboxPatch(
            (cx, cy - sw / 2), sw, sw,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0, facecolor=swatch_color, zorder=3,
        ))
        txt = ax.text(cx + sw + 0.18, cy, label, ha="left", va="center",
                      fontsize=6, color="#222222", zorder=3)
        chips.append((cx, cy, sw, txt))

    _legend_chip(3.6, 7.3, tm_color, "Transition metals")
    _legend_chip(8.0, 7.3, anion_color, "Anions")

    # Optional soft grey round-box framing both chips (matches the pipeline
    # bar-chart legend). Disabled for now — flip ``draw_legend_frame`` to enable.
    # The extent is measured from the rendered label text so the box always
    # wraps the labels regardless of wording.
    draw_legend_frame = False
    if draw_legend_frame:
        fig.canvas.draw()
        inv = ax.transData.inverted()
        xs, ys = [], []
        for cx, cy, sw, txt in chips:
            xs.append(cx)
            ys.extend([cy - sw / 2, cy + sw / 2])
            bb = txt.get_window_extent().transformed(inv)
            xs.append(bb.x1)
            ys.extend([bb.y0, bb.y1])
        pad = 0.5
        ax.add_patch(patches.FancyBboxPatch(
            (min(xs) - pad, min(ys) - pad),
            (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad,
            boxstyle="round,pad=0,rounding_size=0.3",
            linewidth=0.8, edgecolor="0.7", facecolor="0.96", zorder=2,
        ))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Periodic table saved to %s", save_path)

    return fig


# ====================================================================
# Structure-voltage figures
# ====================================================================

def build_structure_voltage_data(
    records,
    mode,
    top_n=None,
    experimental_only_ions=None,
    faded_mode="all",
    require_experimental_anchor=True,
):
    """
    Build structure-voltage plot data from Stage-4 records (E2).

    Parameters
    ----------
    records : dict or str
        The Stage-4 ``structure_voltage_records.json`` content (an
        already-loaded dict) or a path to it. Produced by
        :func:`spices.screening.framework_comparison.build_structure_voltage_records`.
    mode : {"ion_selective", "multi_ion"}
        Keeps only components whose ``route`` equals ``mode``.
    top_n : int or None
        (ion-selective only) keep only the top-N components per ion by their
        maximum qualifying area fraction.
    experimental_only_ions : iterable of str or None
        (ion-selective only) for each listed ion, drop a component that has NO
        ``Experimental`` qualifying marker. ``None`` (default) keeps all.
    faded_mode : {"all", "ph_covering"}
        (multi-ion only) ``"all"`` shows every faded partner-ion marker;
        ``"ph_covering"`` hides faded markers whose plateau does not cover the
        pH window. Routing / ``n_ions`` come from the records and are unchanged.
    require_experimental_anchor : bool
        When ``True`` (default), drop any component that carries a hashed or
        faded marker but has no ``status="solid"`` + ``type="Experimental"``
        anchor: the weaker secondary markers are only shown for hosts with at
        least one strong experimental plateau. Set ``False`` to disable the
        filter and recover the pre-filter behaviour.

    Returns
    -------
    For ``mode="ion_selective"``:
        ``{ion: {label_key: [(V, type, style), ...]}}``
    For ``mode="multi_ion"``:
        ``(data, n_ions)`` where
        ``data = {label_key: {ion: [(V, type, style), ...]}}`` and
        ``n_ions = {label_key: int}``.
    ``style`` is ``None`` (solid), ``"hashed"``, or ``"faded"``.
    """
    if mode not in ("ion_selective", "multi_ion"):
        raise ValueError(f"Unknown mode {mode!r}")

    if isinstance(records, str):
        if not os.path.exists(records):
            raise FileNotFoundError(
                f"Structure-voltage records not found: {records}"
            )
        with open(records) as fh:
            records = json.load(fh)
    if records.get("schema_version") != 1:
        raise ValueError(
            "Unsupported structure_voltage_records schema_version "
            f"{records.get('schema_version')!r} (expected 1)"
        )

    comps = [c for c in records.get("components", []) if c["route"] == mode]

    if require_experimental_anchor:
        def _drop(c):
            has_weak = any(
                m["status"] in ("hashed", "faded") for m in c["markers"]
            )
            has_anchor = any(
                m["status"] == "solid" and m["type"] == "Experimental"
                for m in c["markers"]
            )
            return has_weak and not has_anchor
        comps = [c for c in comps if not _drop(c)]

    def _style(marker):
        # "solid" -> None (plain marker); "hashed"/"faded" pass through.
        return None if marker["status"] == "solid" else marker["status"]

    if mode == "ion_selective":
        exp_ions = set(experimental_only_ions or ())
        # ion-selective components have exactly one (qualifying) ion; all their
        # markers are "solid"/"hashed".
        per_ion = defaultdict(list)  # ion -> [(label_key, entries, max_area)]
        for c in comps:
            markers = c["markers"]
            if not markers:
                continue
            ion = markers[0]["ion"]
            if ion in exp_ions and not any(
                m["type"] == "Experimental" for m in markers
            ):
                continue  # drop a component with no experimental plateau
            entries = [(m["V"], m["type"], _style(m)) for m in markers]
            max_area = max(m["area"] for m in markers)
            per_ion[ion].append((c["label_key"], entries, max_area))

        data = {}
        for ion in ION_COLORS:
            comp_list = per_ion.get(ion)
            if not comp_list:
                continue
            if top_n is not None:
                comp_list = sorted(
                    comp_list, key=lambda t: t[2], reverse=True
                )[:top_n]
            data[ion] = {
                label_key: entries
                for label_key, entries, _ in sorted(
                    comp_list, key=lambda t: t[0]
                )
            }
        return data

    # mode == "multi_ion"
    data = {}
    n_ions = {}
    for c in comps:
        label_key = c["label_key"]
        ion_entries = defaultdict(list)
        for m in c["markers"]:
            if (faded_mode == "ph_covering"
                    and m["status"] == "faded" and not m["covers_pH"]):
                continue
            ion_entries[m["ion"]].append((m["V"], m["type"], _style(m)))
        data[label_key] = dict(ion_entries)
        n_ions[label_key] = c["n_ions"]
        if len(ion_entries) < 2:
            logger.warning(
                "Multi-ion column %r shows %d ion(s) with a visible marker "
                "(n_ions=%d); a partner is undrawable or hidden by "
                "faded_mode=%r.",
                label_key, len(ion_entries), c["n_ions"], faded_mode,
            )
    return data, n_ions


def _render_ion_selective_panel(
    ax, data, *, ion_colors, ref_pH, apply_shifts,
    tick_fontsize, label_fontsize, y_limits,
    hatch_rect_height, marker_size,
):
    """Render the single-ion panel of the combined structure-voltage figure.

    Columns are grouped along the x-axis in ``ION_COLORS`` order; within an
    ion the builder already sorts columns by ``label_key``. Returns the set
    of ions actually drawn and the mp-id list (one per x-tick, empty string
    when absent) so the caller can place mp-id labels after layout.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x_labels = []
    x_counter = 0
    ions_present = set()
    for ion in ION_COLORS:
        structures = data.get(ion)
        if not structures:
            continue
        ions_present.add(ion)
        for structure, entries in structures.items():
            x_labels.append(_subscript_formula(structure))
            volts = [e[0] for e in entries]
            if len(volts) > 1:
                ax.plot([x_counter, x_counter], [min(volts), max(volts)],
                        linestyle="--", color="grey", linewidth=0.8, zorder=1)
            for entry in entries:
                v, typ = entry[0], entry[1]
                style = entry[2] if len(entry) > 2 else None
                marker = "h" if str(typ).lower() == "experimental" else "o"
                ax.scatter(x_counter, v, marker=marker, s=marker_size,
                           color=ion_colors.get(ion, "gray"),
                           edgecolor="black", linewidths=0.6, zorder=3)
                if style == "hashed":
                    ax.add_patch(Rectangle(
                        (x_counter - 0.2, v - hatch_rect_height / 2.0),
                        0.4, hatch_rect_height,
                        facecolor="none",
                        edgecolor=ion_colors.get(ion, "gray"),
                        hatch="////", linewidth=0, zorder=4,
                    ))
            x_counter += 1

    o_line, h_line = _water_stability_ref_lines(ref_pH, apply_shifts)
    ax.axhline(o_line, color="blue", linestyle="--", linewidth=0.8)
    ax.axhline(h_line, color="blue", linestyle="--", linewidth=0.8)

    mpids = _set_formula_xticklabels(ax, x_labels, tick_fontsize)
    ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)
    ax.set_ylabel("Intercalation Voltage (V vs SHE)", fontsize=label_fontsize)
    ax.set_ylim(y_limits)
    if x_counter:
        ax.set_xlim(-0.5, x_counter - 0.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.tick_params(direction="in", length=3, width=0.8, which="major")
    ax.tick_params(direction="in", length=1.5, width=0.6, which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return ions_present, mpids


def _render_multi_ion_panel(
    ax, data, n_ions, *, ion_colors, background_colors,
    ref_pH, apply_shifts,
    tick_fontsize, label_fontsize, y_limits,
    hatch_rect_height, marker_size,
):
    """Render the multi-ion panel of the combined structure-voltage figure.

    Frameworks are ordered by ``n_ions`` (2-ion first, then 3, then 4);
    contiguous groups are shaded with ``background_colors[ni]`` and labelled
    with a ``"{n} ions"`` header in-band. Faded partner-ion markers are drawn
    translucent (``alpha=0.3``). Returns ``(ions_present, mpids)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    structures = sorted(data.keys(), key=lambda k: (n_ions.get(k, 0), k))
    n_structures = len(structures)
    x_labels = []

    grp_start = 0
    while grp_start < n_structures:
        ni = n_ions.get(structures[grp_start])
        grp_end = grp_start
        while (grp_end + 1 < n_structures
               and n_ions.get(structures[grp_end + 1]) == ni):
            grp_end += 1
        if ni in background_colors:
            ax.axvspan(grp_start - 0.5, grp_end + 0.5,
                       facecolor=background_colors[ni], alpha=0.7, zorder=0)
        if ni:
            ax.text(
                (grp_start + grp_end) / 2.0, 0.97, f"{ni} ions",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=tick_fontsize, fontweight="bold", zorder=5,
            )
        grp_start = grp_end + 1

    ions_present = set()
    for idx, structure in enumerate(structures):
        x_labels.append(_subscript_formula(structure))
        ions = data[structure]
        all_volts = [e[0] for entries in ions.values() for e in entries]
        if len(all_volts) > 1:
            ax.plot([idx, idx], [min(all_volts), max(all_volts)],
                    color="grey", linestyle="--", linewidth=0.8, zorder=1)
        for ion, entries in ions.items():
            ions_present.add(ion)
            for entry in entries:
                v, typ = entry[0], str(entry[1]).lower()
                style = entry[2] if len(entry) > 2 else None
                faded = style == "faded"
                marker = "h" if "experimental" in typ else "o"
                ax.scatter(idx, v, marker=marker, s=marker_size,
                           facecolor=ion_colors.get(ion, "gray"),
                           edgecolor="black", linewidths=0.6,
                           alpha=0.3 if faded else 1.0, zorder=3)
                if style == "hashed":
                    ax.add_patch(Rectangle(
                        (idx - 0.2, v - hatch_rect_height / 2.0),
                        0.4, hatch_rect_height, facecolor="none",
                        edgecolor=ion_colors.get(ion, "gray"),
                        hatch="////", linewidth=0, zorder=4,
                    ))

    o_line, h_line = _water_stability_ref_lines(ref_pH, apply_shifts)
    ax.axhline(o_line, color="blue", linestyle="--", linewidth=0.8)
    ax.axhline(h_line, color="blue", linestyle="--", linewidth=0.8)

    mpids = _set_formula_xticklabels(ax, x_labels, tick_fontsize)
    ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)
    ax.set_ylabel("Intercalation Voltage (V vs SHE)", fontsize=label_fontsize)
    ax.set_ylim(y_limits)
    if n_structures:
        ax.set_xlim(-0.5, n_structures - 0.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.tick_params(direction="in", length=3, width=0.8, which="major")
    ax.tick_params(direction="in", length=1.5, width=0.6, which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return ions_present, mpids


def plot_structure_voltages_combined(
    records,
    *,
    ion_colors=None,
    background_colors=None,
    y_limits=(-1.0, 1.5),
    ref_pH=None,
    apply_shifts=True,
    save_path=None,
    figsize=(7.2, 6),
    tick_fontsize=7,
    label_fontsize=8,
    mpid_fontsize=None,
    mpid_color="blue",
    qualify_area=0.70,
    hatch_area=0.80,
    experimental_only_ions=None,
    faded_mode="all",
    require_experimental_anchor=True,
):
    """
    Combined ion-selective + multi-ion structure-voltage figure (PDF).

    Top panel: single-ion intercalating frameworks (columns grouped in
    ``ION_COLORS`` order). Bottom panel: multi-ion frameworks (columns
    ordered by ion count, shaded by group, with an in-band ``"{n} ions"``
    header). A single legend on the right combines the working-ion patches
    with the marker / hatch / faded swatches.

    Width matches the package's double-column convention (7.2 in, see
    :mod:`spices.plotting`); height accommodates both panels comfortably.
    Hexagon markers = experimental, circles = theoretical; ``hashed`` =
    metastability area between ``qualify_area`` and ``hatch_area``; ``faded``
    = partner ion that intercalates the host but is not seawater-stable
    (drawn translucent). Water-stability reference lines are pH-shifted via
    :func:`_water_stability_ref_lines`; marker voltages are the raw
    Stage-3 ``"V vs SHE (V)"`` (unchanged).

    Parameters
    ----------
    records : dict or str
        Stage-4 ``structure_voltage_records.json`` (path or loaded dict).
    require_experimental_anchor : bool
        Pass-through to :func:`build_structure_voltage_data`. When ``True``
        (default), frameworks that carry only weaker (hashed/faded) markers
        without an experimental solid anchor are dropped from both panels.
    experimental_only_ions, faded_mode
        Pass-through to :func:`build_structure_voltage_data` for the
        respective panels.
    ion_colors, background_colors, y_limits, ref_pH, apply_shifts, figsize,
    tick_fontsize, label_fontsize, mpid_fontsize, mpid_color,
    qualify_area, hatch_area
        Cosmetic / reference-line controls. The default fontsizes are tuned
        for the 7.2-in double-column width; marker / hatch sizes scale with
        ``tick_fontsize``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_publication_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if ion_colors is None:
        ion_colors = ION_COLORS
    if background_colors is None:
        background_colors = {2: "#D6EAF8", 3: "#E8DAEF",
                             4: "#E9F7EF", 5: "#FDEBD0"}
    if mpid_fontsize is None:
        mpid_fontsize = tick_fontsize * 0.6

    # Marker / hatch cosmetics scale with the tick fontsize so the tuning
    # above flows through to the panel helpers.
    marker_size = max(12, int((tick_fontsize ** 2) * 0.6))
    hatch_rect_height = 0.12

    sv_ion = build_structure_voltage_data(
        records, "ion_selective",
        experimental_only_ions=experimental_only_ions,
        require_experimental_anchor=require_experimental_anchor,
    )
    sv_multi, n_ions = build_structure_voltage_data(
        records, "multi_ion",
        faded_mode=faded_mode,
        require_experimental_anchor=require_experimental_anchor,
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[6, 1], height_ratios=[1, 1],
        hspace=0.6, wspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[:, 1])
    ax_legend.axis("off")

    ions_top, mpids_top = _render_ion_selective_panel(
        ax_top, sv_ion,
        ion_colors=ion_colors,
        ref_pH=ref_pH, apply_shifts=apply_shifts,
        tick_fontsize=tick_fontsize, label_fontsize=label_fontsize,
        y_limits=y_limits, hatch_rect_height=hatch_rect_height,
        marker_size=marker_size,
    )
    ions_bot, mpids_bot = _render_multi_ion_panel(
        ax_bot, sv_multi, n_ions,
        ion_colors=ion_colors, background_colors=background_colors,
        ref_pH=ref_pH, apply_shifts=apply_shifts,
        tick_fontsize=tick_fontsize, label_fontsize=label_fontsize,
        y_limits=y_limits, hatch_rect_height=hatch_rect_height,
        marker_size=marker_size,
    )

    ions_present = ions_top | ions_bot
    ion_handles = [
        Patch(facecolor=ion_colors.get(i, "gray"), edgecolor="black", label=i)
        for i in ION_COLORS if i in ions_present
    ]
    marker_handles = [
        Line2D([0], [0], marker="h", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=7, label="Experimental"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=6, label="Theoretical"),
        Patch(facecolor="none", edgecolor="black", hatch="////",
              label=f"{int(qualify_area * 100)}% < area < "
                    f"{int(hatch_area * 100)}%"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=6, alpha=0.4,
               label="Other ion\n(not seawater stable)"),
    ]
    ax_legend.legend(
        handles=ion_handles + marker_handles,
        loc="center left", frameon=True, fontsize=tick_fontsize,
        handlelength=1.5, borderaxespad=0.0,
    )

    # mp-id labels drawn after axes positions settle so the rotated offsets
    # are honoured; bbox_inches="tight" then absorbs the overflow.
    _draw_mpid_labels(ax_top, mpids_top, mpid_fontsize, mpid_color)
    _draw_mpid_labels(ax_bot, mpids_bot, mpid_fontsize, mpid_color)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info("Combined structure-voltage plot saved to %s", save_path)
    return fig


# ====================================================================
# Runner
# ====================================================================


def run_statistics(
    stage2_dirs,
    stage3_dirs,
    output_dir=None,
    metastability_criterion=4,
    pH_window=(7.0, 8.5),
    min_area_fraction=0.80,
    save_plots=True,
    stage4_records=None,
    make_periodic_table=True,
    bar_color_by="stage",
    yscale="log",
    experimental_only_ions=("Li",),
    faded_mode="all",
    require_experimental_anchor=True,
):
    """
    Run the full Stage-5 statistics pipeline across all ions.

    Parameters
    ----------
    stage2_dirs : dict
        Maps ion -> Stage 2 output directory (seawater metastability Excel).
    stage3_dirs : dict
        Maps ion -> Stage 3 output directory (``All_voltage_plateaus.xlsx``).
    output_dir : str or None
    metastability_criterion : int
        Honored end-to-end (Fix #3): used for the Stage-2 count and the
        Stage-3 ``criterion N`` sheet.
    pH_window : tuple of float
    min_area_fraction : float
    save_plots : bool
    stage4_records : str or None
        Path to the Stage-4 ``structure_voltage_records.json``. When given and
        present on disk, the combined structure-voltage figure is produced
        (single PDF, two panels); when given but missing, a warning is logged
        and the figure is skipped.
    make_periodic_table : bool
    bar_color_by : {"stage", "ion"}
    yscale : {"log", "linear"}
    experimental_only_ions : iterable of str or None
        Ions whose ion-selective panel keeps only frameworks with at least
        one experimental plateau (change 3). Defaults to ``("Li",)`` since
        Li has the most ion-selective hosts.
    faded_mode : {"all", "ph_covering"}
        Controls which non-qualifying partner-ion markers are drawn faded in
        the multi-ion panel. ``"all"`` (default) draws every partner plateau;
        ``"ph_covering"`` draws only those covering the pH window.
    require_experimental_anchor : bool
        When ``True`` (default), frameworks that carry hashed/faded markers
        but lack a ``status="solid"`` + ``type="Experimental"`` anchor are
        dropped from both panels. Set ``False`` to recover the pre-filter
        behaviour and see every framework.

    Returns
    -------
    dict
        Maps ion -> summary dict with counts at each stage.
    """
    if output_dir is None:
        output_dir = "statistics_output"
    os.makedirs(output_dir, exist_ok=True)

    elements = [e for e in SUPPORTED_ELEMENTS if e in stage2_dirs]
    summary = {}

    # --- Per-ion statistics ---
    for element in elements:
        logger.info("=== %s ===", element)
        s2_dir = stage2_dirs[element]
        s3_dir = stage3_dirs.get(element)

        if not os.path.isdir(s2_dir):  # Fix #4
            logger.warning(
                "Stage-2 dir for %s missing: %s; skipping.", element, s2_dir,
            )
            continue
        try:
            s2_path = _resolve_stage2_excel(s2_dir, element)  # Fix #1
        except FileNotFoundError as exc:
            logger.warning("%s; skipping %s.", exc, element)
            continue
        # (ValueError from ambiguous candidates propagates intentionally.)

        s2_df = pd.read_excel(s2_path)  # read ONCE (Fix #9)

        stability_counts = count_phase_stabilities(
            s2_df, criteria=[metastability_criterion],  # Fix #3
        )

        elem_summary = {
            "total_phases": len(s2_df),
            "seawater_stable": int(stability_counts.get("Seawater stability", 0)),
        }
        key = f"Seawater metastability (Criterion {metastability_criterion}) bool"
        elem_summary[f"metastable_c{metastability_criterion}"] = int(
            stability_counts.get(key, 0)
        )

        # --- Voltage-stable phases ---
        if s3_dir:
            v_path = os.path.join(s3_dir, "All_voltage_plateaus.xlsx")
            if os.path.exists(v_path):
                try:
                    v_df = pd.read_excel(  # read ONCE (Fix #9)
                        v_path,
                        sheet_name=f"criterion {metastability_criterion}",
                    )
                except (ValueError, KeyError) as exc:  # Fix #3
                    logger.warning(
                        "No 'criterion %d' sheet in %s (%s); skipping voltage "
                        "counts for %s.",
                        metastability_criterion, v_path, exc, element,
                    )
                    v_df = None

                if v_df is not None:
                    v_result = count_phases_with_voltage_stability(
                        v_df, s2_df, metastability_criterion,
                    )
                    elem_summary["voltage_phases"] = v_result["total_phases"]
                    elem_summary["voltage_plots"] = v_result["total_plots"]

                    v_filtered = count_phases_with_voltage_stability(
                        v_df, s2_df, metastability_criterion,
                        pH_window=pH_window,
                        min_area_fraction=min_area_fraction,
                    )
                    elem_summary["filtered_phases"] = v_filtered["total_phases"]
                    elem_summary["filtered_plots"] = v_filtered["total_plots"]

        summary[element] = elem_summary

    # --- Export summary ---
    summary_df = pd.DataFrame(summary).T
    summary_path = os.path.join(output_dir, "pipeline_summary.xlsx")
    summary_df.to_excel(summary_path)
    logger.info("Summary saved to %s", summary_path)

    # --- Plots ---
    if save_plots:
        plotted = list(summary.keys())

        # Bar chart — fixed left-to-right ion order for the publication panel.
        bar_order = [e for e in PIPELINE_BAR_ION_ORDER if e in plotted]
        bar_order += [e for e in plotted if e not in bar_order]
        bar_data = {}
        for e in bar_order:
            s = summary[e]
            bar_data[e] = {
                "Total electrodes": s.get("total_phases", 0),
                "Metastable in seawater": s.get(
                    f"metastable_c{metastability_criterion}", 0
                ),
                "Stable in seawater": s.get("seawater_stable", 0),
            }
        if bar_data:
            plot_pipeline_bar_chart(
                bar_data,
                save_path=os.path.join(output_dir, "pipeline_bar_chart.pdf"),
                color_by=bar_color_by, yscale=yscale,
            )

        # Per-ion filter funnel
        venn_dir = os.path.join(output_dir, "nested_venn")
        os.makedirs(venn_dir, exist_ok=True)
        for e in plotted:
            s = summary[e]
            counts = [
                s.get(f"metastable_c{metastability_criterion}", 0),
                s.get("voltage_phases", 0),
                s.get("filtered_phases", 0),
            ]
            labels = [
                f"Metastable (C{metastability_criterion})",
                "With voltage",
                "pH + area filtered",
            ]
            plot_nested_venn(
                e, counts, labels,
                save_path=os.path.join(venn_dir, f"{e}_nested_venn.png"),
            )

        # Periodic table (no pipeline data needed)
        if make_periodic_table:
            plot_periodic_table(
                save_path=os.path.join(output_dir, "periodic_table.pdf"),
            )

        # Combined structure-voltage figure (driven by the Stage-4 records
        # JSON). Both panels (single-ion top, multi-ion bottom) share one
        # legend and ship as a single PDF.
        if stage4_records and os.path.exists(stage4_records):
            with open(stage4_records) as fh:
                sv_records = json.load(fh)
            qa = sv_records.get("qualify_area", 0.70)
            ha = sv_records.get("hatch_area", 0.80)
            plot_structure_voltages_combined(
                sv_records,
                qualify_area=qa, hatch_area=ha,
                experimental_only_ions=experimental_only_ions,
                faded_mode=faded_mode,
                require_experimental_anchor=require_experimental_anchor,
                save_path=os.path.join(output_dir, "structure_voltages.pdf"),
            )
        elif stage4_records:
            logger.warning(
                "No structure_voltage_records.json at %s; skipping the "
                "combined structure-voltage figure. Re-run Stage-4 to "
                "generate it.",
                stage4_records,
            )

    return summary

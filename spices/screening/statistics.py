"""
Stage 5: Statistics and publication-ready figures.

Aggregates statistics across all 5 elements, applies final filters,
and generates bar charts and nested Venn diagrams for the pipeline
results.

Replaces ``stat_calculation_utils.py`` and the ``stats-codes.ipynb``,
``Bar-plots.ipynb``, and ``Venn-diagram-plots.ipynb`` notebooks.
"""

import ast
import logging
import os

import numpy as np
import pandas as pd

from spices.config import SUPPORTED_ELEMENTS
from spices.screening.framework_comparison import ION_COLORS

logger = logging.getLogger(__name__)


# ====================================================================
# Statistics functions
# ====================================================================


def get_stability_bool(x):
    """
    Convert metastability data to boolean.

    Handles string representations of lists (from Excel), actual
    lists/tuples, NaN, and numeric values.

    Parameters
    ----------
    x : str or list or float or None

    Returns
    -------
    bool
    """
    if isinstance(x, str):
        return len(ast.literal_eval(x)) > 0
    if isinstance(x, (list, tuple)):
        return len(x) > 0
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    if isinstance(x, (bool, int, np.bool_)):
        return bool(x)
    return False


def _get_pH_range_bool(tuple_str, min_pH_range=1.5):
    """Check if a pH range string spans at least *min_pH_range*."""
    if not isinstance(tuple_str, str):
        return False

    try:
        pH_min, pH_max = ast.literal_eval(tuple_str)
    except (ValueError, SyntaxError):
        return False

    return round(pH_max - pH_min, 1) >= min_pH_range


def count_phase_stabilities(
    stability_excel_path,
    criteria=None,
):
    """
    Count (meta)stable phases per criterion from Stage 2 output.

    Parameters
    ----------
    stability_excel_path : str
        Path to Stage 2 seawater stability Excel file.
    criteria : list of int or None
        Selection criteria to count (default: [4]).

    Returns
    -------
    pd.Series
        Counts per criterion plus seawater stability.
    """
    if criteria is None:
        criteria = [4]

    df = pd.read_excel(stability_excel_path)

    for sc in criteria:
        col = f"Seawater metastability (Criterion {sc})"
        if col in df.columns:
            df[f"{col} bool"] = df[col].apply(get_stability_bool)

    count_cols = []
    if "Seawater stability" in df.columns:
        count_cols.append("Seawater stability")
    count_cols += [
        f"Seawater metastability (Criterion {sc}) bool"
        for sc in criteria
        if f"Seawater metastability (Criterion {sc}) bool" in df.columns
    ]

    counts = df[count_cols].sum()

    logger.info(
        "Total phases: %d. Counts:\n%s", len(df), counts.to_string(),
    )
    return counts


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
        Path to Stage 3 voltage plateau Excel, or a DataFrame.
    stability_excel_path : str or pd.DataFrame
        Path to Stage 2 stability Excel, or a DataFrame.
    metastability_criterion : int
    pH_window : tuple of float or None
        E.g. ``(7.0, 8.5)``. Filters to plateaus with full pH coverage.
    min_area_fraction : float or None
        Minimum metastability area fraction (0-1).
    experimental : bool or None
        If True, keep only plateaus with at least one experimental phase.

    Returns
    -------
    dict
        ``{'total_phases': int, 'total_plots': int, 'filtered_df': DataFrame}``
    """
    # Load DataFrames
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

    # Apply filters
    pH_col = f"pH range metastability criterion {metastability_criterion}"

    if pH_window is not None:
        pH_range = pH_window[1] - pH_window[0]
        voltage_df = voltage_df[
            voltage_df[pH_col].apply(
                lambda x: _get_pH_range_bool(str(x), pH_range)
            )
        ]

    if min_area_fraction is not None:
        voltage_df = voltage_df[
            voltage_df["Total metastability area fraction"]
            >= min_area_fraction
        ]

    if experimental is not None:
        voltage_df = voltage_df[
            (voltage_df.get("Charged Is experimental") == experimental)
            | (voltage_df.get("Discharged Is experimental") == experimental)
        ]

    # Count unique MP IDs present in both stability and voltage data
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


# ====================================================================
# Visualization
# ====================================================================


def plot_pipeline_bar_chart(
    stage_data,
    save_path=None,
    figsize=(14, 8),
):
    """
    Generate a grouped bar chart of phase counts per element per stage.

    Parameters
    ----------
    stage_data : dict
        Maps element -> dict of ``{stage_label: count}``.
        Example: ``{"Li": {"Seawater stable": 5, "Metastable (C4)": 1042}, ...}``
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    elements = list(stage_data.keys())
    stage_labels = list(stage_data[elements[0]].keys())
    n_stages = len(stage_labels)
    n_elements = len(elements)

    x = np.arange(n_elements)
    width = 0.8 / n_stages

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for i, stage in enumerate(stage_labels):
        values = [stage_data[e].get(stage, 0) for e in elements]
        offset = (i - n_stages / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width, label=stage,
            color=[ION_COLORS.get(e, "gray") for e in elements],
            alpha=0.3 + 0.7 * (i / max(n_stages - 1, 1)),
            edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(val), ha="center", va="bottom", fontsize=7,
                )

    ax.set_xlabel("Working Ion", fontsize=14)
    ax.set_ylabel("Number of Phases", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(elements)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Bar chart saved to %s", save_path)

    return fig


def plot_nested_venn(
    element,
    counts,
    labels,
    save_path=None,
):
    """
    Plot vertically nested circles showing progressive pipeline filtering.

    Parameters
    ----------
    element : str
        Ion name (for title and color).
    counts : list of int
        Decreasing counts, e.g. [total, criterion_pass, pH_pass, area_pass].
    labels : list of str
        Label for each level.
    save_path : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(4, 6), dpi=150)
    ax.set_aspect("equal")

    max_radius = 2.0
    n = len(counts)
    color = ION_COLORS.get(element, "gray")

    for i, (count, label) in enumerate(zip(counts, labels)):
        if counts[0] == 0:
            radius = 0
        else:
            radius = max_radius * np.sqrt(count / counts[0])
        y_center = 0

        circle = Circle(
            (0, y_center), radius,
            fill=True, facecolor=color,
            alpha=0.15 + 0.2 * i,
            edgecolor="black", linewidth=1.5,
        )
        ax.add_patch(circle)

        ax.text(
            0, y_center + radius + 0.15,
            f"{label}\n({count})",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xlim(-max_radius - 1, max_radius + 1)
    ax.set_ylim(-max_radius - 1, max_radius + 1.5)
    ax.set_title(element, fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Nested Venn saved to %s", save_path)

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
):
    """
    Run the full statistics pipeline across all elements.

    Parameters
    ----------
    stage2_dirs : dict
        Maps element -> path to Stage 2 output directory.
        Each dir must contain the seawater metastability Excel.
    stage3_dirs : dict
        Maps element -> path to Stage 3 output directory.
        Each dir must contain ``All_voltage_plateaus.xlsx``.
    output_dir : str or None
    metastability_criterion : int
    pH_window : tuple of float
    min_area_fraction : float
    save_plots : bool

    Returns
    -------
    dict
        Maps element -> summary dict with counts at each stage.
    """
    if output_dir is None:
        output_dir = "statistics_output"
    os.makedirs(output_dir, exist_ok=True)

    elements = [e for e in SUPPORTED_ELEMENTS if e in stage2_dirs]
    summary = {}

    # --- Per-element statistics ---
    for element in elements:
        logger.info("=== %s ===", element)
        s2_dir = stage2_dirs[element]
        s3_dir = stage3_dirs.get(element)

        # Find Stage 2 Excel
        s2_files = [
            f for f in os.listdir(s2_dir) if f.endswith(".xlsx")
        ]
        if not s2_files:
            logger.warning("No Excel files in %s", s2_dir)
            continue
        s2_path = os.path.join(s2_dir, s2_files[0])

        # Count stabilities
        stability_counts = count_phase_stabilities(s2_path)

        elem_summary = {
            "total_phases": len(pd.read_excel(s2_path)),
            "seawater_stable": int(stability_counts.get("Seawater stability", 0)),
        }
        key = f"Seawater metastability (Criterion {metastability_criterion}) bool"
        elem_summary[f"metastable_c{metastability_criterion}"] = int(
            stability_counts.get(key, 0)
        )

        # Count voltage-stable phases
        if s3_dir:
            v_path = os.path.join(s3_dir, "All_voltage_plateaus.xlsx")
            if os.path.exists(v_path):
                # Unfiltered
                v_result = count_phases_with_voltage_stability(
                    v_path, s2_path, metastability_criterion,
                )
                elem_summary["voltage_phases"] = v_result["total_phases"]
                elem_summary["voltage_plots"] = v_result["total_plots"]

                # With pH + area filters
                v_filtered = count_phases_with_voltage_stability(
                    v_path, s2_path, metastability_criterion,
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
        import matplotlib
        matplotlib.use("Agg")

        # Bar chart
        bar_data = {}
        for e in elements:
            s = summary.get(e, {})
            bar_data[e] = {
                "Total MP phases": s.get("total_phases", 0),
                f"Metastable (C{metastability_criterion})": s.get(
                    f"metastable_c{metastability_criterion}", 0
                ),
                "With voltage": s.get("voltage_phases", 0),
                "pH + area filtered": s.get("filtered_phases", 0),
            }
        plot_pipeline_bar_chart(
            bar_data,
            save_path=os.path.join(output_dir, "pipeline_bar_chart.png"),
        )

        # Nested Venn per element
        venn_dir = os.path.join(output_dir, "nested_venn")
        os.makedirs(venn_dir, exist_ok=True)
        for e in elements:
            s = summary.get(e, {})
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

    return summary

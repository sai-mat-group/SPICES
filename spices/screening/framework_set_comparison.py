"""
Stage 4 (optional companion): set-based framework comparison.

Provides pairwise/N-wise intersections, ion-selective subsets, voltage
overlay plots, and 2-set Venn diagrams across the 5 working ions. Use
this when you want set-style summaries that ignore the structural-match
graph (e.g. quick "shared mp-ids per ion pair" checks).

This module is independent of the graph-based comparison in
``framework_comparison``. Both share ``FrameworkStructure`` loading.

Note on the underlying identity bug fix: ``FrameworkStructure`` has no
``__eq__`` override and hashes by ``mp_id`` only, so set-membership
tests on ``FrameworkStructure`` instances fall back to Python identity
and never match across different load instances. All set operations
here compare via ``mp_id`` keys instead.
"""

import logging
import os
import warnings
from itertools import combinations

import pandas as pd
from pymatgen.core import Composition

from spices.config import SUPPORTED_ELEMENTS
from spices.plotting import apply_publication_style
from spices.screening.framework_comparison import (
    FrameworkStructure,
    ION_COLORS,
    formula_to_mathtext,
    load_all_frameworks_for_element,
)

apply_publication_style()

logger = logging.getLogger(__name__)


# ====================================================================
# Set operations (mp_id-keyed; identity-safe)
# ====================================================================


def _dedup_by_mpid(frameworks_iterable):
    """Return a list of FrameworkStructure objects, one per unique mp_id."""
    seen = {}
    for fw in frameworks_iterable:
        if fw.mp_id not in seen:
            seen[fw.mp_id] = fw
    return list(seen.values())


def framework_set_intersection(
    req_charged, req_discharged, other_charged, other_discharged,
):
    """Frameworks from req that share an mp_id with any framework in other.

    Returns a ``set`` of ``FrameworkStructure`` objects from the req side.
    Matching is by ``mp_id``, not by ``StructureMatcher``.
    """
    other_mpids = (
        {fw.mp_id for fw in other_charged}
        | {fw.mp_id for fw in other_discharged}
    )
    result = {}
    for fw in list(req_charged) + list(req_discharged):
        if fw.mp_id in other_mpids and fw.mp_id not in result:
            result[fw.mp_id] = fw
    return set(result.values())


def multi_ion_common_frameworks(
    req_charged, req_discharged,
    other_charged_list, other_discharged_list,
):
    """Frameworks in req whose mp_id is shared with ALL other ions."""
    if len(other_charged_list) != len(other_discharged_list):
        raise ValueError("Charged and discharged lists must be same length")

    req_mpids = (
        {fw.mp_id for fw in req_charged}
        | {fw.mp_id for fw in req_discharged}
    )
    for chg, dis in zip(other_charged_list, other_discharged_list):
        other_mpids = {fw.mp_id for fw in chg} | {fw.mp_id for fw in dis}
        req_mpids &= other_mpids

    result = {}
    for fw in list(req_charged) + list(req_discharged):
        if fw.mp_id in req_mpids and fw.mp_id not in result:
            result[fw.mp_id] = fw
    return set(result.values())


def ion_selective_frameworks(
    req_charged, req_discharged,
    other_charged_list, other_discharged_list,
):
    """Frameworks in req whose mp_id is NOT in any other ion."""
    if len(other_charged_list) != len(other_discharged_list):
        raise ValueError("Charged and discharged lists must be same length")

    other_mpids = set()
    for chg, dis in zip(other_charged_list, other_discharged_list):
        other_mpids |= {fw.mp_id for fw in chg}
        other_mpids |= {fw.mp_id for fw in dis}

    result = {}
    for fw in list(req_charged) + list(req_discharged):
        if fw.mp_id not in other_mpids and fw.mp_id not in result:
            result[fw.mp_id] = fw
    return set(result.values())


# ====================================================================
# Voltage overlay plots
# ====================================================================


def _get_working_ion_fraction(phase_formula, framework_formula, working_ion):
    """Compute working ion fraction x in M_x[Framework]."""
    framework_weight = Composition(
        framework_formula
    ).get_reduced_formula_and_factor()[1]

    if working_ion in phase_formula:
        phase_comp = Composition(phase_formula)
        num_ion = phase_comp.get_el_amt_dict().get(working_ion, 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fw_comp = phase_comp.replace({working_ion: {working_ion: 0}})

        num_fw = (
            fw_comp.get_reduced_formula_and_factor()[1] / framework_weight
        )
        return num_ion / num_fw

    return 0


def voltage_step_plot(
    plot_num, df, ion_name, fig=None,
    save=True, save_dir="./voltage_plots",
):
    """Generate a voltage vs. ion-fraction step plot."""
    import matplotlib.pyplot as plt

    plot_df = df[df["Plot number"] == plot_num]
    if plot_df.empty:
        raise ValueError(f"No data for plot number: {plot_num}")

    framework_formula = plot_df["Framework formula"].iloc[0]
    most_charged = plot_df["Charged Formula"].iloc[0]
    charged_frac = _get_working_ion_fraction(
        most_charged, framework_formula, ion_name,
    )
    discharged_fracs = [
        _get_working_ion_fraction(f, framework_formula, ion_name)
        for f in plot_df["Discharged Formula"].tolist()
    ]
    voltages = plot_df["V vs SHE (V)"].tolist()

    x_vals = [charged_frac] + discharged_fracs
    y_vals = [voltages[0]] + voltages

    if fig is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
    else:
        ax = fig.axes[0] if fig.axes else fig.add_subplot(1, 1, 1)

    ax.axhline(1.23, color="blue", linestyle="--")
    ax.axhline(-0.71, color="blue", linestyle="--")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.minorticks_on()
    ax.grid(which="major", color="#CCCCCC", linestyle="-", linewidth=0.6)
    ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=0.4)

    ax.step(
        x_vals, y_vals,
        label=f"M = {ion_name}; plot {plot_num}",
        color=ION_COLORS.get(ion_name, "black"),
    )
    ax.set_ylim(-0.8, 1.3)
    ax.set_xlabel(
        r"$x$ in $M_x$" + formula_to_mathtext(framework_formula),
    )
    ax.set_ylabel("Applied potential (V vs SHE)")
    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        borderaxespad=0.5, frameon=True,
    )
    legend.get_frame().set_edgecolor("black")

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"voltage_plot_{framework_formula}_plot{plot_num}.png"
        fig.savefig(os.path.join(save_dir, fname))
        plt.close(fig)

    return fig


def overlay_multi_ion_voltage_plots(
    req_framework_set, req_df,
    other_framework_sets, other_dfs,
    req_ion, other_ions,
    save_dir, save=True,
):
    """Overlay voltage plots for shared frameworks across multiple ions."""
    for framework in req_framework_set:
        plot_filter = (
            (req_df["Discharged MP id"] == framework.mp_id)
            | (req_df["Charged MP id"] == framework.mp_id)
        )
        plot_nums = req_df[plot_filter]["Plot number"].unique()

        for plot_num in plot_nums:
            figure = None
            for i, other_set in enumerate(other_framework_sets):
                matching = [
                    f for f in other_set if f.mp_id == framework.mp_id
                ]
                for match_fw in matching:
                    pf = (
                        (other_dfs[i]["Discharged MP id"] == match_fw.mp_id)
                        | (other_dfs[i]["Charged MP id"] == match_fw.mp_id)
                    )
                    other_plot_nums = (
                        other_dfs[i][pf]["Plot number"].unique()
                    )
                    for pn in other_plot_nums:
                        figure = voltage_step_plot(
                            pn, other_dfs[i], other_ions[i],
                            fig=figure, save=False,
                        )

            voltage_step_plot(
                plot_num, req_df, req_ion,
                fig=figure, save=save, save_dir=save_dir,
            )


def save_all_combination_voltage_plots(
    frameworks_dict, df_dict, ion_of_interest,
    n_ions, save_dir="./Voltage_plateaus",
):
    """Generate overlaid plots for all N-ion combinations."""
    other_ions = [
        ion for ion in frameworks_dict if ion != ion_of_interest
    ]
    combs = combinations(other_ions, n_ions - 1)

    for comb in combs:
        req_set = multi_ion_common_frameworks(
            req_charged=frameworks_dict[ion_of_interest][1],
            req_discharged=frameworks_dict[ion_of_interest][0],
            other_charged_list=[frameworks_dict[i][1] for i in comb],
            other_discharged_list=[frameworks_dict[i][0] for i in comb],
        )
        if not req_set:
            continue

        other_sets = []
        other_dfs = []
        for other in comb:
            oset = multi_ion_common_frameworks(
                req_charged=frameworks_dict[other][1],
                req_discharged=frameworks_dict[other][0],
                other_charged_list=(
                    [frameworks_dict[i][1] for i in comb if i != other]
                    + [frameworks_dict[ion_of_interest][1]]
                ),
                other_discharged_list=(
                    [frameworks_dict[i][0] for i in comb if i != other]
                    + [frameworks_dict[ion_of_interest][0]]
                ),
            )
            other_sets.append(oset)
            other_dfs.append(df_dict[other])

        leaf = f"{ion_of_interest}-" + "-".join(comb)
        path = os.path.join(save_dir, f"{n_ions}-ions", leaf)

        overlay_multi_ion_voltage_plots(
            req_set, df_dict[ion_of_interest],
            other_sets, other_dfs,
            ion_of_interest, list(comb),
            save_dir=path, save=True,
        )


# ====================================================================
# Venn diagrams
# ====================================================================


def venn_2set(
    frameworks_dict, ion_a, ion_b,
    intersection_set=None,
    save_dir="./Venn-diags", save=True,
):
    """Generate a 2-set Venn diagram comparing two ions."""
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2 as _venn2

    set_a_mpids = (
        {fw.mp_id for fw in frameworks_dict[ion_a][0]}
        | {fw.mp_id for fw in frameworks_dict[ion_a][1]}
    )
    set_b_mpids = (
        {fw.mp_id for fw in frameworks_dict[ion_b][0]}
        | {fw.mp_id for fw in frameworks_dict[ion_b][1]}
    )

    if intersection_set is None:
        intersection_set = multi_ion_common_frameworks(
            frameworks_dict[ion_a][1], frameworks_dict[ion_a][0],
            [frameworks_dict[ion_b][1]], [frameworks_dict[ion_b][0]],
        )
    n_common = len({fw.mp_id for fw in intersection_set})

    colors = {
        "Li": "crimson", "Na": "gold",
        "K": "#988ED5", "Mg": "#348ABD", "Ca": "#8EBA42",
    }

    plt.figure(figsize=(3.5, 3.5))
    _venn2(
        subsets=(
            len(set_a_mpids) - n_common,
            len(set_b_mpids) - n_common,
            n_common,
        ),
        set_labels=(ion_a, ion_b),
        set_colors=(
            colors.get(ion_a, "gray"), colors.get(ion_b, "gray"),
        ),
        alpha=0.3,
    )

    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{ion_a}-{ion_b}.pdf"))
    plt.close()


# ====================================================================
# Top-level runner
# ====================================================================


def run_set_based_comparison(
    stage3_dirs,
    output_dir=".",
    elements=None,
    sheet_name="criterion 4",
    n_ion_combinations=(2, 3, 4, 5),
    save_voltage_overlays=True,
    save_venn=True,
):
    """
    Run set-based framework comparison: voltage overlays and Venn diagrams.

    Independent of the graph-based comparison. Loads frameworks per ion,
    then for each requested element produces overlay plots for all
    N-ion combinations and 2-set Venn diagrams.

    Parameters
    ----------
    stage3_dirs : dict
        Maps element symbol to Stage 3 output directory.
    output_dir : str
        Root output directory (default: ``.``). Per-element subdirs are
        created underneath.
    elements : list of str or None
        Ions to produce outputs for. Defaults to ``SUPPORTED_ELEMENTS``.
    sheet_name : str
    n_ion_combinations : iterable of int
    save_voltage_overlays : bool
    save_venn : bool
    """
    if elements is None:
        elements = list(SUPPORTED_ELEMENTS)

    os.makedirs(output_dir, exist_ok=True)

    # Load frameworks + voltage dfs for all ions
    frameworks_dict = {}
    df_dict = {}
    for ion in stage3_dirs:
        s3_dir = stage3_dirs[ion]
        excel_path = os.path.join(s3_dir, "All_voltage_plateaus.xlsx")
        discharged_dir = os.path.join(s3_dir, "structs")
        charged_dir = os.path.join(s3_dir, "All_charged_structures")
        if not os.path.exists(charged_dir):
            charged_dir = os.path.join(s3_dir, "All_charged_strutures")

        logger.info("Loading frameworks for %s from %s", ion, s3_dir)
        dis_set, chg_set, vdf = load_all_frameworks_for_element(
            excel_path, sheet_name, discharged_dir, charged_dir, ion,
        )
        frameworks_dict[ion] = (dis_set, chg_set)
        df_dict[ion] = vdf

    for element in elements:
        element_out = os.path.join(output_dir, element)
        os.makedirs(element_out, exist_ok=True)
        other_ions = [
            ion for ion in frameworks_dict if ion != element
        ]

        # Voltage overlays
        if save_voltage_overlays:
            for n in n_ion_combinations:
                save_all_combination_voltage_plots(
                    frameworks_dict, df_dict, element, n,
                    save_dir=os.path.join(element_out, "Voltage_plateaus"),
                )

        # Venn diagrams (pairwise with each other ion)
        if save_venn:
            for other in other_ions:
                inter = framework_set_intersection(
                    frameworks_dict[element][1],
                    frameworks_dict[element][0],
                    frameworks_dict[other][1],
                    frameworks_dict[other][0],
                )
                venn_2set(
                    frameworks_dict, element, other,
                    intersection_set=inter,
                    save_dir=os.path.join(element_out, "Venn-diags"),
                )
        logger.info("Completed set-based outputs for %s", element)

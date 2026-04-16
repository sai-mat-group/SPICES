"""
Stage 3: Intercalation voltage filter screening.

This module implements the 4-step pipeline for computing intercalation
voltages and identifying voltage plateaus for cathode materials:

1. Filter Stage 2 output by metastability, group by charged composition
2. Group structures by similarity (StructureMatcher) into subgroups
3. Compute voltage plateaus on pseudo-binary convex hull
4. Consolidate all results into ``All_voltage_plateaus.xlsx``

Replaces the per-element ``step-1/2/3/4`` notebooks with a single
parameterized implementation.
"""

import ast
import logging
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from spices.config import (
    WORKING_ION_PARAMS,
    DEFAULT_SELECTION_CRITERIA,
    INTERCALATION_DECOMP_ENERGY_THRESHOLD,
    DEFAULT_OER_SHIFT,
    DEFAULT_HER_SHIFT,
)
from spices.mp_client import get_client
from spices.pourbaix.cache import PourbaixCache
from spices.pourbaix.geometry import (
    find_ph_range_at_voltage,
    find_stability_area_around_voltage,
)
from spices.intercalation.voltage import (
    calculate_intercalation_voltage,
    get_energy_of_working_ion,
)
from spices.intercalation.hull import filter_df_to_get_on_hull_phases
from spices.intercalation.structure import (
    create_subgroups,
    subgroups_to_excel_sheets,
    add_charged_struct_info,
)
from spices.dataframe_ops import update_seawater_metastability_columns_by_mp_id

logger = logging.getLogger(__name__)

# ====================================================================
# Column name mappings
# ====================================================================

TO_CHARGED = {
    "MP id": "Charged MP id",
    "Formula": "Charged Formula",
    "Pbx elements per formula": "Pbx elements per charged formula",
    "Base XC Functional": "Charged base XC Functional",
    "Correction XC functional": "Charged correction XC Functional",
    "GGA(+U)/R2SCAN mixing adjustment": "Charged GGA(+U)/R2SCAN mixing adjustment",
    "E (eV/formula)": "Charged Structure E (eV/formula)",
    "Energy Above Hull (eV)": "Charged E above hull (eV)",
    "Thermo-type": "Charged Thermo-type",
    "Crystal Structure": "Charged Crystal Structure",
    "Space Group": "Charged Space Group",
    "Structure stable in water": "Charged structure stable in water",
    "Structure metastable in water": "Charged structure metastable in water",
    "No of working ions per framework": "No of working ions per charged framework",
    "Is experimental": "Charged Is experimental",
}

TO_DISCHARGED = {
    "MP id": "Discharged MP id",
    "Formula": "Discharged Formula",
    "Pbx elements per formula": "Pbx elements per discharged formula",
    "Base XC Functional": "Discharged base XC Functional",
    "Correction XC functional": "Discharged correction XC Functional",
    "GGA(+U)/R2SCAN mixing adjustment": "Discharged GGA(+U)/R2SCAN mixing adjustment",
    "E (eV/formula)": "Discharged Structure E (eV/formula)",
    "Energy Above Hull (eV)": "Discharged E above hull (eV)",
    "Thermo-type": "Discharged Thermo-type",
    "Crystal Structure": "Discharged Crystal Structure",
    "Space Group": "Discharged Space Group",
    "Structure stable in water": "Discharged structure stable in water",
    "Structure metastable in water": "Discharged structure metastable in water",
    "No of working ions per framework": "No of working ions per discharged framework",
    "Is experimental": "Discharged Is experimental",
}

CHARGED_COLUMNS = list(TO_CHARGED.values())
DISCHARGED_COLUMNS = list(TO_DISCHARGED.values())


def get_columns_order(selection_criterion):
    """Return the standard column order for voltage plateau output."""
    return [
        "Charged MP id",
        "Charged Formula",
        "Discharged MP id",
        "Discharged Formula",
        "V vs SHE (V)",
        f"pH range metastability criterion {selection_criterion}",
        "Charged Structure E (eV/formula)",
        "Charged base XC Functional",
        "Charged correction XC Functional",
        "Pbx elements per charged formula",
        "Charged E above hull (eV)",
        "Charged Thermo-type",
        "Charged Crystal Structure",
        "Charged Space Group",
        "Charged structure stable in water",
        "Charged structure metastable in water",
        "Charged GGA(+U)/R2SCAN mixing adjustment",
        "No of working ions per charged framework",
        "No of working ions per discharged framework",
        "Discharged Structure E (eV/formula)",
        "Discharged base XC Functional",
        "Discharged correction XC Functional",
        "Pbx elements per discharged formula",
        "Discharged E above hull (eV)",
        "Discharged Thermo-type",
        "Discharged Crystal Structure",
        "Discharged Space Group",
        "Discharged structure stable in water",
        "Discharged structure metastable in water",
        "Discharged GGA(+U)/R2SCAN mixing adjustment",
        "Charged Is experimental",
        "Discharged Is experimental",
        "Total metastability area fraction",
        "Below V metastability area fraction",
        "Above V metastability area fraction",
    ]


# ====================================================================
# Helper functions
# ====================================================================


def _convert_metastability_list_to_bool(x):
    """Convert string representation of metastability list to bool."""
    if isinstance(x, str):
        return len(ast.literal_eval(x)) > 0
    elif isinstance(x, (list, tuple)):
        return len(x) > 0
    elif isinstance(x, float) and np.isnan(x):
        return False
    return False


def _intelligent_outer_merge(df1, df2):
    """
    Outer-merge two DataFrames, handling dtype mismatches by casting
    conflicting columns to ``object``.
    """
    common_cols = set(df1.columns).intersection(set(df2.columns))
    mismatched = [col for col in common_cols if df1[col].dtype != df2[col].dtype]
    for col in mismatched:
        df1[col] = df1[col].astype("object")
        df2[col] = df2[col].astype("object")

    if mismatched:
        warnings.warn(f"Dtype mismatches fixed by converting to 'object': {mismatched}")

    return df1.merge(df2, how="outer")


def merge_subgroups(all_sheets):
    """
    Merge subgroup sheets that share common MP IDs via outer merge.

    Parameters
    ----------
    all_sheets : dict
        Mapping from sheet name to DataFrame.

    Returns
    -------
    dict
        Updated dict with ``'merged subgroup N'`` keys added.
    """
    subgroup_names = [name for name in all_sheets if name.startswith("subgroup")]

    if not subgroup_names:
        raise ValueError("all_sheets contains no 'subgroup' sheets")

    already_merged = set()
    merged_count = 1

    for i in range(len(subgroup_names)):
        if subgroup_names[i] in already_merged:
            continue

        sheet1 = all_sheets[subgroup_names[i]]

        for j in range(i + 1, len(subgroup_names)):
            if subgroup_names[j] in already_merged:
                continue

            sheet2 = all_sheets[subgroup_names[j]]

            ids1 = set(sheet1["MP id"].tolist())
            ids2 = set(sheet2["MP id"].tolist())

            if not ids1.intersection(ids2):
                continue

            logger.debug("Merging subgroups %d and %d", i + 1, j + 1)
            sheet1 = _intelligent_outer_merge(sheet1, sheet2)
            already_merged.add(subgroup_names[j])

        all_sheets[f"merged subgroup {merged_count}"] = sheet1
        merged_count += 1

    return all_sheets


def _append_if_new(df, new_df):
    """
    Append voltage plateau rows only if they are not a subset of any
    existing ``'Plot number'`` group.
    """
    if len(new_df) == 0:
        return df

    relevant_cols = [
        "Charged MP id",
        "Charged Formula",
        "Discharged MP id",
        "Discharged Formula",
        "V vs SHE (V)",
    ]

    new_set = set(map(tuple, new_df[relevant_cols].to_records(index=False)))

    for _, group in df.groupby("Plot number"):
        existing_set = set(map(tuple, group[relevant_cols].to_records(index=False)))
        if new_set.issubset(existing_set):
            return df

    next_idx = int(df["Plot number"].max()) + 1
    new_df = new_df.copy()
    new_df.insert(loc=0, column="Plot number", value=[next_idx] * len(new_df))

    return pd.concat([df, new_df], ignore_index=True)


def populate_seawater_metastability_points_for_seawater_stable_phases(
    df,
    selection_criteria,
    cache,
    mp_client,
    element_str,
    decomp_energy_for_metastability=0.3,
):
    """
    Fill in metastability point data for phases that are seawater-stable
    but lack metastability coordinate data.

    Parameters
    ----------
    df : pd.DataFrame
    selection_criteria : list of int
    cache : PourbaixCache
    mp_client : MPRester
    element_str : str
    decomp_energy_for_metastability : float

    Returns
    -------
    pd.DataFrame
    """
    if len(df) <= 1:
        return df

    df_stable = df[df.get("Seawater stability") == 1]
    if len(df_stable) == 0:
        return df

    metastability_cols = [
        f"Seawater metastability (Criterion {i})" for i in selection_criteria
    ]

    # Ensure columns exist with object dtype
    for col in metastability_cols:
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype=object)
        else:
            df[col] = df[col].astype(object)

    for index, row in df_stable.iterrows():
        if any(pd.isna(row.get(col)) for col in metastability_cols):
            updated_row = update_seawater_metastability_columns_by_mp_id(
                row,
                cache=cache,
                mp_client=mp_client,
                element_str=element_str,
                metastability_points=True,
                check_stability=True,
                all_selection_criteria=len(selection_criteria) > 1,
                decomp_energy_for_metastability=decomp_energy_for_metastability,
            )

            for col, val in updated_row.items():
                try:
                    df.at[index, col] = val
                except ValueError:
                    df[col] = df[col].astype(object)
                    df.at[index, col] = val

            # Update boolean columns
            for sc in selection_criteria:
                col = f"Seawater metastability (Criterion {sc})"
                val = df.loc[index, col]
                if isinstance(val, (list, tuple)):
                    df.loc[index, f"{col}_bool"] = int(len(val) > 0)
                elif isinstance(val, str):
                    df.loc[index, f"{col}_bool"] = int(len(ast.literal_eval(val)) > 0)
                else:
                    df.loc[index, f"{col}_bool"] = 0

    return df


# ====================================================================
# Core voltage plateau computation
# ====================================================================


def _get_voltage_plateau_df(
    df,
    df_on_hull,
    working_ion,
    working_ion_valency,
    working_ion_voltage_vs_SHE,
    working_ion_energies,
    stability_criterion=3,
    save_plot=False,
    save_plot_path="./",
    voltage_plateau_df=None,
):
    """
    Recursive function to compute voltage plateau data for a fixed
    charged terminal composition.

    Returns a DataFrame with charged/discharged phase info, intercalation
    voltage (vs SHE), pH range, and metastability area fractions.
    """
    # Initialize output DataFrame
    if not isinstance(voltage_plateau_df, pd.DataFrame):
        voltage_plateau_df = pd.DataFrame(
            columns=(
                CHARGED_COLUMNS
                + DISCHARGED_COLUMNS
                + [f"pH range metastability criterion {stability_criterion}"]
                + ["V vs SHE (V)"]
                + [
                    "Total metastability area fraction",
                    "Below V metastability area fraction",
                    "Above V metastability area fraction",
                ]
            )
        )

    # Recursion base case: all hull phases accounted for
    phases_on_hull = set(df_on_hull["Formula"].tolist())
    phases_in_plateau = set(
        voltage_plateau_df["Charged Formula"].tolist()
        + voltage_plateau_df["Discharged Formula"].tolist()
    )
    if not (phases_on_hull - phases_in_plateau):
        return voltage_plateau_df

    final_stability_pH = [-100, 100]

    for i in range(len(df_on_hull) - 1):
        charged_row = df_on_hull.iloc[i]
        discharged_row = df_on_hull.iloc[i + 1]

        # Skip if already computed
        if discharged_row["MP id"] in voltage_plateau_df["Discharged MP id"].tolist():
            continue

        voltage_vs_ion = calculate_intercalation_voltage(
            discharged_formula=discharged_row["Formula"],
            discharged_energy_per_formula=discharged_row["E (eV/formula)"],
            charged_formula=charged_row["Formula"],
            charged_energy_per_formula=charged_row["E (eV/formula)"],
            functional_string=charged_row["Base XC Functional"],
            working_ion=working_ion,
            working_ion_valency=working_ion_valency,
            working_ion_energies=working_ion_energies,
        )
        voltage_vs_SHE = voltage_vs_ion + working_ion_voltage_vs_SHE

        # Get metastability coordinates
        sc_col = f"Seawater metastability (Criterion {stability_criterion})"
        charged_coords = charged_row.get(sc_col, [])
        discharged_coords = discharged_row.get(sc_col, [])

        if isinstance(charged_coords, str):
            charged_coords = ast.literal_eval(charged_coords)
        if isinstance(discharged_coords, str):
            discharged_coords = ast.literal_eval(discharged_coords)

        # Handle empty/missing coordinates
        if not charged_coords or not discharged_coords:
            stability_pH = ()
            total_area, area_below, area_above = 0.0, 0.0, 0.0
        else:
            plot_path = None
            if save_plot:
                plot_path = os.path.join(
                    save_plot_path,
                    f"{charged_row['MP id']}-{discharged_row['MP id']}"
                    f"_{charged_row['Formula']}-{discharged_row['Formula']}.pdf",
                )

            stability_pH = find_ph_range_at_voltage(
                charged_coords,
                discharged_coords,
                voltage_vs_SHE,
                save_plot=save_plot,
                save_plot_path=plot_path,
                title=f"Seawater metastability (Criterion {stability_criterion})",
                A_phase_name=f"{charged_row['Formula']} {charged_row['MP id']}",
                B_phase_name=f"{discharged_row['Formula']} {discharged_row['MP id']}",
            )
            total_area, area_below, area_above = find_stability_area_around_voltage(
                charged_coords,
                discharged_coords,
                voltage_vs_SHE,
            )

        # Update final stability pH window
        if len(stability_pH) == 0:
            final_stability_pH = []
        elif len(final_stability_pH) > 0:
            if stability_pH[0] > final_stability_pH[0]:
                final_stability_pH[0] = stability_pH[0]
            if stability_pH[1] < final_stability_pH[1]:
                final_stability_pH[1] = stability_pH[1]

        if len(final_stability_pH) != 0:
            new_idx = len(voltage_plateau_df)

            for col in TO_CHARGED:
                voltage_plateau_df.at[new_idx, TO_CHARGED[col]] = charged_row.get(col)
            for col in TO_DISCHARGED:
                voltage_plateau_df.at[new_idx, TO_DISCHARGED[col]] = discharged_row.get(
                    col
                )

            voltage_plateau_df.at[
                new_idx, f"pH range metastability criterion {stability_criterion}"
            ] = stability_pH
            voltage_plateau_df.at[new_idx, "V vs SHE (V)"] = voltage_vs_SHE
            voltage_plateau_df.at[new_idx, "Total metastability area fraction"] = (
                total_area
            )
            voltage_plateau_df.at[new_idx, "Below V metastability area fraction"] = (
                area_below
            )
            voltage_plateau_df.at[new_idx, "Above V metastability area fraction"] = (
                area_above
            )
        else:
            # Remove unstable discharged phase and recurse
            df_trimmed = df[df["MP id"] != discharged_row["MP id"]]
            df_on_hull_trimmed = filter_df_to_get_on_hull_phases(
                df_trimmed,
                working_ion,
            )
            return _get_voltage_plateau_df(
                df_trimmed,
                df_on_hull_trimmed,
                working_ion,
                working_ion_valency,
                working_ion_voltage_vs_SHE,
                working_ion_energies,
                stability_criterion=stability_criterion,
                save_plot=save_plot,
                save_plot_path=save_plot_path,
                voltage_plateau_df=voltage_plateau_df,
            )

    cols_order = get_columns_order(stability_criterion)
    available = [c for c in cols_order if c in voltage_plateau_df.columns]
    return voltage_plateau_df[available]


def obtain_voltage_plateau_data_based_on_convex_hull(
    df,
    selection_criteria,
    working_ion,
    working_ion_valency,
    working_ion_voltage_vs_SHE,
    working_ion_energies,
    cache,
    mp_client,
    element_str,
    use_only_on_hull_phases=True,
    save_plot=False,
    save_plot_path="./",
    decomp_energy_for_metastability=0.3,
):
    """
    Compute voltage plateaus for all selection criteria on a single
    subgroup DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Subgroup data with material properties and metastability columns.
    selection_criteria : list of int
    working_ion : str
    working_ion_valency : int
    working_ion_voltage_vs_SHE : float
    working_ion_energies : dict
    cache : PourbaixCache
    mp_client : MPRester
    element_str : str
    use_only_on_hull_phases : bool
    save_plot : bool
    save_plot_path : str
    decomp_energy_for_metastability : float

    Returns
    -------
    dict
        Mapping ``'sc N'`` to DataFrame (or None if no plateaus).
    """
    logger.debug("Selection criteria: %s", selection_criteria)

    df = populate_seawater_metastability_points_for_seawater_stable_phases(
        df,
        selection_criteria=selection_criteria,
        cache=cache,
        mp_client=mp_client,
        element_str=element_str,
        decomp_energy_for_metastability=decomp_energy_for_metastability,
    )

    df_on_hull = filter_df_to_get_on_hull_phases(df, working_ion)

    if "index" in df_on_hull.columns:
        df_on_hull = df_on_hull.drop(columns=["index"])

    if len(df_on_hull) < 2:
        return {f"sc {sc}": None for sc in selection_criteria}

    # Verify single XC functional
    if df_on_hull["Base XC Functional"].nunique() != 1:
        raise ValueError(
            f"Multiple XC functionals: "
            f"{df_on_hull['Base XC Functional'].unique().tolist()}"
        )

    result = {f"sc {sc}": None for sc in selection_criteria}

    for sc in selection_criteria:
        max_charged = df_on_hull.iloc[0]["Formula"]
        max_discharged = df_on_hull.iloc[-1]["Formula"]

        df_ = df_on_hull.copy() if use_only_on_hull_phases else df.copy()
        df_on_hull_ = df_on_hull.copy()

        while max_charged != max_discharged:
            plateau_df = _get_voltage_plateau_df(
                df_,
                df_on_hull_,
                working_ion,
                working_ion_valency,
                working_ion_voltage_vs_SHE,
                working_ion_energies,
                stability_criterion=sc,
                save_plot=save_plot,
                save_plot_path=save_plot_path,
            )

            if isinstance(result[f"sc {sc}"], pd.DataFrame):
                result[f"sc {sc}"] = _append_if_new(
                    result[f"sc {sc}"],
                    plateau_df,
                )
            elif len(plateau_df) > 0:
                plateau_df = plateau_df.copy()
                plateau_df.insert(
                    loc=0,
                    column="Plot number",
                    value=[0] * len(plateau_df),
                )
                result[f"sc {sc}"] = plateau_df

            # Remove most-charged terminal and re-compute hull
            df_ = df_[df_["Formula"] != max_charged]
            if len(df_) < 2:
                break
            df_on_hull_ = filter_df_to_get_on_hull_phases(df_, working_ion)
            if len(df_on_hull_) < 1:
                break
            max_charged = df_on_hull_.iloc[0]["Formula"]

    return result


# ====================================================================
# Pipeline steps
# ====================================================================


def step1_create_unique_composition_sheets(
    input_path,
    output_dir,
    element,
):
    """
    Step 1: Filter Stage 2 output and group by unique charged composition.

    Parameters
    ----------
    input_path : str
        Path to Stage 2 output Excel.
    output_dir : str
        Directory to write per-composition Excel files.
    element : str
        Working ion element.

    Returns
    -------
    list of str
        Filenames written.
    """
    from spices.pourbaix.composition import rm_element

    df = pd.read_excel(input_path)
    logger.info("Read %d rows from %s", len(df), input_path)

    # Normalize charged composition to reduced formula
    if "Charged composition" not in df.columns:
        df["Charged composition"] = df["Formula"].apply(
            lambda x: Composition(rm_element(x, element)).reduced_formula
        )
    else:
        df["Charged composition"] = df["Charged composition"].apply(
            lambda x: Composition(x).reduced_formula
        )

    # Create boolean columns from metastability lists
    col = "Seawater metastability (Criterion 4)"
    if col in df.columns:
        df[f"{col}_bool"] = df[col].apply(_convert_metastability_list_to_bool)

    os.makedirs(output_dir, exist_ok=True)

    unique_compositions = df["Charged composition"].unique()
    filenames = []

    for composition in unique_compositions:
        sub_df = df[df["Charged composition"] == composition]

        # Check if metastability criterion 4 or seawater stability is True
        bool_cols = [
            "Seawater metastability (Criterion 4)_bool",
        ]
        bool_cols = [c for c in bool_cols if c in sub_df.columns]

        has_metastable = sub_df[bool_cols].any().any() if bool_cols else False
        has_stable = (
            sub_df["Seawater stability"].any()
            if "Seawater stability" in sub_df.columns
            else False
        )

        if has_metastable or has_stable:
            fname = f"{composition}.xlsx"
            sub_df.to_excel(
                os.path.join(output_dir, fname),
                index=False,
            )
            filenames.append(fname)

    logger.info(
        "Step 1: %d/%d compositions have metastable/stable phases",
        len(filenames),
        len(unique_compositions),
    )
    return filenames


def step2_create_subgroups(
    composition_dir,
    working_ion,
    mp_client,
    cache,
    element_str,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    structure_cache_dir="All_charged_structures",
    decomp_energy_for_metastability=0.3,
):
    """
    Step 2: Group structures by similarity and add charged phase info.

    Processes all composition Excel files in *composition_dir*.

    Parameters
    ----------
    composition_dir : str
    working_ion : str
    mp_client : MPRester
    cache : PourbaixCache
    element_str : str
    charged_struct_dir : str
    struct_dir : str
    structure_cache_dir : str
    decomp_energy_for_metastability : float
    """
    excel_files = [f for f in os.listdir(composition_dir) if f.endswith(".xlsx")]
    total = len(excel_files)
    cache_dict = {}

    for idx, fname in enumerate(excel_files):
        logger.info("[%d/%d] Step 2: %s", idx + 1, total, fname)
        path = os.path.join(composition_dir, fname)

        df = pd.read_excel(path)

        # Create subgroups
        subgroups = create_subgroups(
            df,
            working_ion,
            mp_client,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            cache_dict=cache_dict,
        )
        sheets = subgroups_to_excel_sheets(df, subgroups)

        # Add fully-charged phase info
        sheets = add_charged_struct_info(
            sheets,
            working_ion,
            mp_client,
            cache,
            element_str,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            structure_cache_dir=structure_cache_dir,
            cache_dict=cache_dict,
            decomp_energy_for_metastability=decomp_energy_for_metastability,
        )

        # Write back
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, sheet_df in sheets.items():
                sheet_df.to_excel(writer, sheet_name=name, index=False)


def step3_create_voltage_plateau_sheets(
    composition_dir,
    output_dir,
    working_ion,
    working_ion_valency,
    working_ion_voltage_vs_SHE,
    working_ion_energies,
    cache,
    mp_client,
    element_str,
    selection_criteria=None,
    use_only_on_hull_phases=True,
    save_plots=False,
    decomp_energy_for_metastability=0.3,
):
    """
    Step 3: Compute voltage plateaus for all compositions.

    Parameters
    ----------
    composition_dir : str
        Directory with per-composition Excel files (from Steps 1-2).
    output_dir : str
        Directory for voltage plateau output.
    working_ion : str
    working_ion_valency : int
    working_ion_voltage_vs_SHE : float
    working_ion_energies : dict
    cache : PourbaixCache
    mp_client : MPRester
    element_str : str
    selection_criteria : list of int or None
    use_only_on_hull_phases : bool
    save_plots : bool
    decomp_energy_for_metastability : float
    """
    if selection_criteria is None:
        selection_criteria = DEFAULT_SELECTION_CRITERIA

    os.makedirs(output_dir, exist_ok=True)

    excel_files = [f for f in os.listdir(composition_dir) if f.endswith(".xlsx")]
    total = len(excel_files)

    for idx, fname in enumerate(excel_files):
        logger.info("[%d/%d] Step 3: %s", idx + 1, total, fname)

        framework_name = fname.replace(".xlsx", "")
        framework_dir = os.path.join(output_dir, framework_name)
        os.makedirs(framework_dir, exist_ok=True)

        path = os.path.join(composition_dir, fname)
        all_sheets = pd.read_excel(path, sheet_name=None)

        # Merge subgroups
        all_sheets = merge_subgroups(all_sheets)

        # Get merged subgroup sheet names
        merged_names = [name for name in all_sheets if "merged subgroup" in name]

        voltage_results = {sc: {} for sc in selection_criteria}

        for sheet_name in merged_names:
            df_sheet = all_sheets[sheet_name]
            logger.debug("Processing %s/%s", fname, sheet_name)

            for sc in selection_criteria:
                sc_dir = os.path.join(framework_dir, f"sc_{sc}")
                os.makedirs(sc_dir, exist_ok=True)

                plateau_dict = obtain_voltage_plateau_data_based_on_convex_hull(
                    df_sheet,
                    selection_criteria=[sc],
                    working_ion=working_ion,
                    working_ion_valency=working_ion_valency,
                    working_ion_voltage_vs_SHE=working_ion_voltage_vs_SHE,
                    working_ion_energies=working_ion_energies,
                    cache=cache,
                    mp_client=mp_client,
                    element_str=element_str,
                    use_only_on_hull_phases=use_only_on_hull_phases,
                    save_plot=save_plots,
                    save_plot_path=sc_dir,
                    decomp_energy_for_metastability=decomp_energy_for_metastability,
                )

                if isinstance(plateau_dict.get(f"sc {sc}"), pd.DataFrame):
                    voltage_results[sc][f"voltage_{sheet_name}"] = plateau_dict[
                        f"sc {sc}"
                    ]

        # Save voltage plateau sheets per criterion
        for sc in selection_criteria:
            if not voltage_results[sc]:
                continue

            sc_dir = os.path.join(framework_dir, f"sc_{sc}")
            sheet_path = os.path.join(sc_dir, fname)

            with pd.ExcelWriter(sheet_path, engine="openpyxl") as writer:
                for name, vdf in voltage_results[sc].items():
                    vdf.to_excel(writer, sheet_name=name, index=False)


def step4_create_combined_sheet(
    voltage_plateau_dir,
    output_path,
    selection_criteria=None,
):
    """
    Step 4: Consolidate all per-composition results into a single
    multi-sheet Excel file.

    Parameters
    ----------
    voltage_plateau_dir : str
        Root directory containing per-composition voltage plateau data.
    output_path : str
        Path for the combined ``All_voltage_plateaus.xlsx``.
    selection_criteria : list of int or None

    Returns
    -------
    dict
        Mapping ``'criterion N'`` to DataFrame.
    """
    if selection_criteria is None:
        selection_criteria = DEFAULT_SELECTION_CRITERIA

    combined = {f"criterion {sc}": None for sc in selection_criteria}

    comp_dirs = os.listdir(voltage_plateau_dir)
    total = len(comp_dirs)

    for idx, comp_dir in enumerate(comp_dirs):
        comp_path = os.path.join(voltage_plateau_dir, comp_dir)
        if not os.path.isdir(comp_path):
            continue

        logger.debug("[%d/%d] Combining: %s", idx + 1, total, comp_dir)

        for sc_dir_name in os.listdir(comp_path):
            sc_path = os.path.join(comp_path, sc_dir_name)
            if not os.path.isdir(sc_path):
                continue

            # Extract selection criterion from dir name
            try:
                sc = int(sc_dir_name.split("_")[-1])
            except ValueError:
                continue

            if sc not in selection_criteria:
                continue

            # Find Excel file in sc directory
            xlsx_files = [f for f in os.listdir(sc_path) if f.endswith(".xlsx")]
            if not xlsx_files:
                continue

            xlsx_path = os.path.join(sc_path, xlsx_files[0])
            sheets = pd.read_excel(xlsx_path, sheet_name=None)

            # Determine current max plot number
            max_plot = 0
            if isinstance(combined[f"criterion {sc}"], pd.DataFrame):
                max_plot = int(combined[f"criterion {sc}"]["Plot number"].max())
                if np.isnan(max_plot):
                    max_plot = 0

            for sheet_df in sheets.values():
                if len(sheet_df) == 0:
                    continue

                sheet_df = sheet_df.copy()
                sheet_df["Framework formula"] = comp_dir

                # Reorder columns
                cols_order = get_columns_order(sc)
                available_cols = ["Plot number", "Framework formula"] + [
                    c for c in cols_order if c in sheet_df.columns
                ]
                sheet_df = sheet_df[available_cols]

                if not isinstance(combined[f"criterion {sc}"], pd.DataFrame):
                    combined[f"criterion {sc}"] = sheet_df.copy()
                    continue

                sheet_df["Plot number"] += max_plot + 1
                combined[f"criterion {sc}"] = pd.concat(
                    [combined[f"criterion {sc}"], sheet_df],
                    ignore_index=True,
                )
                max_plot = int(combined[f"criterion {sc}"]["Plot number"].max())

    # Save combined file
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, cdf in combined.items():
            if isinstance(cdf, pd.DataFrame):
                cdf.to_excel(writer, sheet_name=name, index=False)

    logger.info("Combined results saved to %s", output_path)
    return combined


# ====================================================================
# Top-level runner
# ====================================================================


def run_intercalation_screening(
    element,
    input_path,
    output_dir=None,
    selection_criteria=None,
    decomp_energy=None,
    use_only_on_hull_phases=True,
    save_plots=False,
    cache_dir=".",
    api_key=None,
):
    """
    Run the full intercalation voltage screening pipeline (Steps 1-4).

    Parameters
    ----------
    element : str
        Working ion (``"Ca"``, ``"K"``, ``"Li"``, ``"Mg"``, or ``"Na"``).
    input_path : str
        Path to Stage 2 output Excel file.
    output_dir : str or None
        Root output directory.  Defaults to
        ``"{element}_intercalation/"``.
    selection_criteria : list of int or None
        Selection criteria to evaluate (default: 4).
    decomp_energy : float or None
        Decomposition energy threshold.  Default 0.3 eV.
    use_only_on_hull_phases : bool
    save_plots : bool
        If True, save pH-overlap PDF plots.
    cache_dir : str
        Directory for Pourbaix caches.
    api_key : str or None

    Returns
    -------
    dict
        Combined voltage plateau data (``'criterion N'`` -> DataFrame).
    """
    if output_dir is None:
        output_dir = f"{element}_intercalation"
    if selection_criteria is None:
        selection_criteria = DEFAULT_SELECTION_CRITERIA
    if decomp_energy is None:
        decomp_energy = INTERCALATION_DECOMP_ENERGY_THRESHOLD

    params = WORKING_ION_PARAMS[element]
    working_ion_valency = params["valency"]
    working_ion_voltage_vs_SHE = params["voltage_vs_SHE"]

    mp_client = get_client(api_key)
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )

    # Get working ion reference energies (once)
    logger.info("Querying %s reference energies from MP", element)
    working_ion_energies = get_energy_of_working_ion(element, mp_client)

    # Directory structure
    composition_dir = os.path.join(output_dir, "unique_charged_composition_sheets")
    plateau_dir = os.path.join(output_dir, "Voltage_plateau_data")
    combined_path = os.path.join(output_dir, "All_voltage_plateaus.xlsx")
    struct_dir = os.path.join(output_dir, "structs")
    charged_struct_dir = os.path.join(output_dir, "charged_structs")
    structure_cache_dir = os.path.join(output_dir, "All_charged_structures")

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # Step 1
    logger.info("=== Step 1: Creating unique composition sheets ===")
    filenames = step1_create_unique_composition_sheets(
        input_path,
        composition_dir,
        element,
    )

    # Step 2
    logger.info("=== Step 2: Creating structure subgroups ===")
    step2_create_subgroups(
        composition_dir,
        element,
        mp_client,
        cache,
        element,
        charged_struct_dir=charged_struct_dir,
        struct_dir=struct_dir,
        structure_cache_dir=structure_cache_dir,
        decomp_energy_for_metastability=decomp_energy,
    )

    # Step 3
    logger.info("=== Step 3: Computing voltage plateaus ===")
    step3_create_voltage_plateau_sheets(
        composition_dir,
        plateau_dir,
        working_ion=element,
        working_ion_valency=working_ion_valency,
        working_ion_voltage_vs_SHE=working_ion_voltage_vs_SHE,
        working_ion_energies=working_ion_energies,
        cache=cache,
        mp_client=mp_client,
        element_str=element,
        selection_criteria=selection_criteria,
        use_only_on_hull_phases=use_only_on_hull_phases,
        save_plots=save_plots,
        decomp_energy_for_metastability=decomp_energy,
    )

    # Step 4
    logger.info("=== Step 4: Creating combined sheet ===")
    combined = step4_create_combined_sheet(
        plateau_dir,
        combined_path,
        selection_criteria,
    )

    elapsed = time.time() - start_time
    logger.info(
        "Intercalation screening completed in %.1f sec. Output: %s",
        elapsed,
        output_dir,
    )

    return combined

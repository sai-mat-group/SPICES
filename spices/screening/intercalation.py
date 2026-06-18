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
import glob as _glob
import logging
import multiprocessing as mp
import os
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from spices.config import (
    WORKING_ION_PARAMS,
    DEFAULT_SELECTION_CRITERIA,
    INTERCALATION_DECOMP_ENERGY_THRESHOLD,
    SEAWATER_DECOMP_ENERGY_THRESHOLD,
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
from spices.intercalation.grouping import (
    enrich_sheet_with_charged_mp_phases,
    discover_new_charged_rows_for_file,
    add_seawater_metastability_bool_columns,
    build_graph_components_from_sheet,
    save_component_graphs,
)
from spices.dataframe_ops import update_seawater_metastability_columns_by_mp_id
from spices.screening.seawater import (
    EST_PER_WORKER_GB,
    _available_memory_gb,
    _dispatch_phase1_tasks,
    _split_dataframe,
    _load_checkpoints,
    _recover_chunk_for_core,
)

logger = logging.getLogger(__name__)

# Tag column added to flattened new-row DataFrame so Phase 3 can re-split
# results back to their source composition xlsx.  Must survive every
# checkpoint write/load cycle.
SOURCE_FILE_COLUMN = "_source_file"

# Cadence (seconds) of the parent-side aggregate "M/N files complete"
# log emitted while Phase 1 workers run.  Per-worker file logs in
# process_logs/phase1_core{cid}.txt give the granular trace.
PHASE1_PROGRESS_INTERVAL = 30.0

# Per-worker shared progress counter, bound by _phase1_init().  Workers
# atomically increment after each file; the parent's progress thread
# reads it every PHASE1_PROGRESS_INTERVAL seconds.
_PHASE1_COUNTER = None

# ====================================================================
# Column name mappings
# ====================================================================

TO_CHARGED = {
    "MP id": "Charged MP id",
    "Formula": "Charged Formula",
    "Pbx elements per formula": "Pbx elements per charged formula",
    "XC Functional": "Charged XC Functional",
    "E (eV/formula)": "Charged Structure E (eV/formula)",
    "Energy Above Hull (eV)": "Charged E above hull (eV)",
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
    "XC Functional": "Discharged XC Functional",
    "E (eV/formula)": "Discharged Structure E (eV/formula)",
    "Energy Above Hull (eV)": "Discharged E above hull (eV)",
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
        "Charged XC Functional",
        "Pbx elements per charged formula",
        "Charged E above hull (eV)",
        "Charged Crystal Structure",
        "Charged Space Group",
        "Charged structure stable in water",
        "Charged structure metastable in water",
        "No of working ions per charged framework",
        "No of working ions per discharged framework",
        "Discharged Structure E (eV/formula)",
        "Discharged XC Functional",
        "Pbx elements per discharged formula",
        "Discharged E above hull (eV)",
        "Discharged Crystal Structure",
        "Discharged Space Group",
        "Discharged structure stable in water",
        "Discharged structure metastable in water",
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
    working_ion_energy,
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
            working_ion=working_ion,
            working_ion_valency=working_ion_valency,
            working_ion_energy=working_ion_energy,
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
                A_phase_name=f"{charged_row['Formula']}\n{charged_row['MP id']}",
                B_phase_name=f"{discharged_row['Formula']}\n{discharged_row['MP id']}",
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
                working_ion_energy,
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
    working_ion_energy,
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
    working_ion_energy : float
        GGA(+U) energy per atom of the stable elemental working-ion phase.
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

    # Verify single XC functional.
    # TODO: remove this guard once trial runs confirm that
    # thermo_type=GGA_GGA+U always yields a single XC Functional value
    # per chemsys.  Until then, surface mixed values rather than silently
    # mixing GGA and GGA+U entries on the hull.
    if df_on_hull["XC Functional"].nunique() != 1:
        raise ValueError(
            f"Multiple XC functionals on hull: "
            f"{df_on_hull['XC Functional'].unique().tolist()}"
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
                working_ion_energy,
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


def _atomic_write_xlsx(df, path):
    """Write *df* to *path* atomically via tmp + ``os.replace``.

    Prevents a SIGKILLed worker from leaving a half-written xlsx that
    a later read would interpret as corrupt.  PID-tagged tmp filename
    avoids two writers clobbering each other.  Tmp suffix is
    ``.tmp.<pid>.xlsx`` so pandas/openpyxl extension validation
    accepts it.
    """
    tmp = f"{path}.tmp.{os.getpid()}.xlsx"
    df.to_excel(tmp, index=False)
    os.replace(tmp, path)


def _cleanup_orphan_tmp_xlsx(directory):
    """Remove ``*.tmp.<pid>.xlsx`` left by SIGKILLed workers."""
    if not os.path.isdir(directory):
        return
    for orphan in _glob.glob(os.path.join(directory, "*.xlsx.tmp.*.xlsx")):
        try:
            os.remove(orphan)
            logger.info("Removed orphan tmp file: %s", orphan)
        except OSError:
            pass


def _phase1_init(counter):
    """ProcessPoolExecutor initializer — bind shared progress counter
    in the worker's module namespace so :func:`_phase1_process_files`
    can increment it without an explicit argument.
    """
    global _PHASE1_COUNTER
    _PHASE1_COUNTER = counter


def _phase1_process_files(
    file_chunk,
    chunk_id,
    composition_dir,
    working_ion,
    element_str,
    charged_struct_dir,
    struct_dir,
    structure_cache_dir,
    log_dir,
):
    """Per-worker Phase 1 body.

    Re-reads each assigned xlsx, runs
    :func:`discover_new_charged_rows_for_file` with a per-worker
    ``cache_dict`` and per-worker ``mp_client`` (resolved post-fork),
    tags new rows with :data:`SOURCE_FILE_COLUMN`, increments the
    shared progress counter after each file, and returns the concat.
    Per-worker logs go to ``log_dir/phase1_core{chunk_id}.txt``
    (mirrors Phase 2).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"phase1_core{chunk_id}.txt")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    worker_logger = logging.getLogger(
        f"spices.intercalation.phase1.{chunk_id}"
    )
    worker_logger.addHandler(file_handler)
    worker_logger.setLevel(logging.INFO)

    mp_client = get_client()
    cache_dict = {}
    new_row_frames = []

    for fname in file_chunk:
        worker_logger.info(
            "[core %d] Phase 1 discovery: %s", chunk_id, fname,
        )
        path = os.path.join(composition_dir, fname)
        df = pd.read_excel(path)
        new_rows_df = discover_new_charged_rows_for_file(
            df,
            working_ion=working_ion,
            mp_client=mp_client,
            element_str=element_str,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            structure_cache_dir=structure_cache_dir,
            cache_dict=cache_dict,
        )
        if len(new_rows_df) > 0:
            new_rows_df = new_rows_df.copy()
            new_rows_df[SOURCE_FILE_COLUMN] = fname
            new_row_frames.append(new_rows_df)

        if _PHASE1_COUNTER is not None:
            with _PHASE1_COUNTER.get_lock():
                _PHASE1_COUNTER.value += 1

    worker_logger.removeHandler(file_handler)
    file_handler.close()

    if new_row_frames:
        return pd.concat(new_row_frames, ignore_index=True)
    return pd.DataFrame()


def _phase1_sequential_discover(
    excel_files,
    originals_by_file,
    working_ion,
    mp_client,
    element_str,
    charged_struct_dir,
    struct_dir,
    structure_cache_dir,
    log_progress=True,
):
    """Sequential Phase 1 discovery loop.

    Used for the ``n_workers <= 1`` path and as the fallback after a
    :class:`BrokenProcessPool`.  Returns the list of tagged new-row
    DataFrames; caller concats.
    """
    total = len(excel_files)
    cache_dict = {}
    new_row_frames = []
    for idx, fname in enumerate(excel_files):
        if log_progress:
            logger.info(
                "[%d/%d] Phase 1 discovery: %s", idx + 1, total, fname,
            )
        else:
            logger.info("Phase 1 sequential recovery: %s", fname)
        df = originals_by_file[fname]
        new_rows_df = discover_new_charged_rows_for_file(
            df,
            working_ion=working_ion,
            mp_client=mp_client,
            element_str=element_str,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            structure_cache_dir=structure_cache_dir,
            cache_dict=cache_dict,
        )
        if len(new_rows_df) > 0:
            new_rows_df = new_rows_df.copy()
            new_rows_df[SOURCE_FILE_COLUMN] = fname
            new_row_frames.append(new_rows_df)
    return new_row_frames


def _phase1_discover_all_files(
    composition_dir,
    working_ion,
    mp_client,
    element_str,
    charged_struct_dir,
    struct_dir,
    structure_cache_dir,
    n_workers=1,
    log_dir=None,
):
    """Phase 1: scan every composition xlsx and tag new charged rows
    with their source file.

    ``n_workers <= 1`` runs the original sequential loop unchanged.
    ``n_workers > 1`` splits the file list round-robin and dispatches
    discovery to a :class:`ProcessPoolExecutor`; each worker has its
    own ``cache_dict`` and ``mp_client`` (resolved post-fork via
    :func:`spices.mp_client.get_client`).  A parent-side
    :class:`threading.Thread` logs aggregate "M/N files complete"
    every :data:`PHASE1_PROGRESS_INTERVAL` seconds while workers run;
    per-worker per-file logs go to
    ``{log_dir}/phase1_core{cid}.txt``.

    On :class:`BrokenProcessPool`, any chunk whose worker died is
    retried sequentially in the parent — Phase 1 has no checkpointing
    by design, but the on-disk struct caches make sequential retry
    fast.

    Returns
    -------
    originals_by_file : dict[str, pd.DataFrame]
        Original (unenriched) df keyed by filename.
    all_new_rows_df : pd.DataFrame
        Concatenation of every new row across all files, with a
        :data:`SOURCE_FILE_COLUMN` column.  Empty DataFrame if no file
        had any new mp ids.
    """
    excel_files = sorted(
        f for f in os.listdir(composition_dir) if f.endswith(".xlsx")
    )
    total = len(excel_files)

    # Sequential pre-pass: parent reads every xlsx for originals_by_file.
    # Cheap (<1 s for ~500 small files) and avoids shipping DataFrames
    # back from workers via IPC.
    originals_by_file = {
        fname: pd.read_excel(os.path.join(composition_dir, fname))
        for fname in excel_files
    }

    t0 = time.time()

    if n_workers <= 1 or total == 0:
        new_row_frames = _phase1_sequential_discover(
            excel_files,
            originals_by_file,
            working_ion=working_ion,
            mp_client=mp_client,
            element_str=element_str,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            structure_cache_dir=structure_cache_dir,
        )
        all_new_rows_df = (
            pd.concat(new_row_frames, ignore_index=True)
            if new_row_frames else pd.DataFrame()
        )
        logger.info(
            "Phase 1 discovery: %.1fs, %d new rows across %d files "
            "(n_workers=1)",
            time.time() - t0, len(all_new_rows_df), total,
        )
        return originals_by_file, all_new_rows_df

    # Multi-worker path.
    if log_dir is None:
        log_dir = os.path.join(
            os.path.dirname(composition_dir.rstrip(os.sep)) or ".",
            "process_logs",
        )

    # Round-robin file split so per-worker chunks are roughly balanced
    # in size.  Drop empty chunks when n_workers > total.
    chunks = [excel_files[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]
    n_active_workers = len(chunks)

    worker_kwargs = dict(
        composition_dir=composition_dir,
        working_ion=working_ion,
        element_str=element_str,
        charged_struct_dir=charged_struct_dir,
        struct_dir=struct_dir,
        structure_cache_dir=structure_cache_dir,
        log_dir=log_dir,
    )

    progress_counter = mp.Value("i", 0)
    stop_progress = threading.Event()

    def _progress_logger():
        while not stop_progress.wait(PHASE1_PROGRESS_INTERVAL):
            with progress_counter.get_lock():
                done = progress_counter.value
            logger.info(
                "Phase 1 progress: %d/%d files complete (%.1fs elapsed)",
                done, total, time.time() - t0,
            )

    progress_thread = threading.Thread(
        target=_progress_logger, daemon=True,
    )
    progress_thread.start()

    new_row_frames = []
    recover_chunks = []
    future_to_chunk = {}

    try:
        with ProcessPoolExecutor(
            max_workers=n_active_workers,
            initializer=_phase1_init,
            initargs=(progress_counter,),
        ) as pool:
            future_to_chunk = {
                pool.submit(
                    _phase1_process_files,
                    chunk, cid, **worker_kwargs,
                ): (cid, chunk)
                for cid, chunk in enumerate(chunks)
            }
            try:
                for fut in as_completed(future_to_chunk):
                    cid, chunk = future_to_chunk[fut]
                    try:
                        result = fut.result()
                        if len(result) > 0:
                            new_row_frames.append(result)
                    except BrokenProcessPool:
                        logger.error(
                            "Phase 1 worker %d died (likely OOM-killed). "
                            "Will retry sequentially.",
                            cid,
                        )
                        recover_chunks.append(chunk)
                    except Exception:
                        logger.exception("Phase 1 worker %d raised", cid)
                        raise
            except BrokenProcessPool:
                logger.error(
                    "Phase 1 ProcessPool broken — retrying remaining "
                    "chunks sequentially."
                )
    finally:
        stop_progress.set()
        progress_thread.join(timeout=2.0)

    # Sweep any chunks whose futures never returned (broken-pool kills
    # all pending futures).
    for fut, (cid, chunk) in future_to_chunk.items():
        if not fut.done() and chunk not in recover_chunks:
            recover_chunks.append(chunk)

    if recover_chunks:
        flat_recover = [fname for chunk in recover_chunks for fname in chunk]
        logger.info(
            "Phase 1 sequential recovery: %d files across %d failed chunks",
            len(flat_recover), len(recover_chunks),
        )
        recovered_frames = _phase1_sequential_discover(
            flat_recover,
            originals_by_file,
            working_ion=working_ion,
            mp_client=mp_client,
            element_str=element_str,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            structure_cache_dir=structure_cache_dir,
            log_progress=False,
        )
        new_row_frames.extend(recovered_frames)

    all_new_rows_df = (
        pd.concat(new_row_frames, ignore_index=True)
        if new_row_frames else pd.DataFrame()
    )
    logger.info(
        "Phase 1 discovery: %.1fs, %d new rows across %d files "
        "(n_workers=%d)",
        time.time() - t0, len(all_new_rows_df), total, n_workers,
    )
    return originals_by_file, all_new_rows_df


def _seal_seawater_columns_for_evaluation(df):
    """Insert empty Seawater stability / Criterion 4 columns if missing.

    Worker logic uses NaN in ``Seawater stability`` to detect "not yet
    evaluated".  Establish that placeholder up front so the restart
    merge has somewhere to write.  Both columns are object dtype so
    they can hold a mix of NaN, bool, and list cells.
    """
    if "Seawater stability" not in df.columns:
        df["Seawater stability"] = np.nan
    if "Seawater metastability (Criterion 4)" not in df.columns:
        df["Seawater metastability (Criterion 4)"] = None
    df["Seawater stability"] = df["Seawater stability"].astype(object)
    df["Seawater metastability (Criterion 4)"] = df[
        "Seawater metastability (Criterion 4)"
    ].astype(object)
    return df


def _merge_evaluated_new_rows(target_df, source_df):
    """Copy seawater columns from *source_df* into *target_df*.

    Match on ``(SOURCE_FILE_COLUMN, MP id)``.  Only rows still NaN in
    ``Seawater stability`` get updated.  Modifies *target_df* in place.
    """
    if source_df is None or len(source_df) == 0:
        return
    if SOURCE_FILE_COLUMN not in source_df.columns:
        return
    if "Seawater stability" not in source_df.columns:
        return

    eval_mask = source_df["Seawater stability"].notna()
    if not eval_mask.any():
        return

    available = [
        c for c in
        ["Seawater stability", "Seawater metastability (Criterion 4)"]
        if c in source_df.columns
    ]
    # Object dtype lets us write bool/list cells into the same column.
    for col in available:
        if col in target_df.columns:
            target_df[col] = target_df[col].astype(object)

    src_eval = source_df.loc[eval_mask].set_index(
        [SOURCE_FILE_COLUMN, "MP id"],
    )

    for i in target_df.index:
        if pd.notna(target_df.at[i, "Seawater stability"]):
            continue
        key = (
            target_df.at[i, SOURCE_FILE_COLUMN],
            target_df.at[i, "MP id"],
        )
        if key not in src_eval.index:
            continue
        for col in available:
            target_df.at[i, col] = src_eval.at[key, col]


def _process_chunk_wrapper_intercalation(chunk_df, chunk_id, kwargs):
    """ProcessPoolExecutor entry point — unpacks kwargs."""
    return _process_chunk_intercalation(chunk_df, chunk_id, **kwargs)


def _process_chunk_intercalation(
    chunk_df,
    chunk_id,
    element_str,
    cache_dir,
    runtime_dir,
    log_dir,
    decomp_energy_for_metastability,
    save_interval,
    max_retries,
    retry_delay,
    restart=False,
):
    """Per-worker body for the parallel sea-Pourbaix evaluation.

    Iterates rows of *chunk_df* and calls
    :func:`update_seawater_metastability_columns_by_mp_id` on each.
    Skips rows with ``Seawater stability`` already non-NaN (restart
    safety).  Writes ``runtime_dir/{element}_core{cid}_part{N}.xlsx``
    every ``save_interval`` rows and a ``_final.xlsx`` at the end.

    The :data:`SOURCE_FILE_COLUMN` column is preserved on every
    checkpoint write so Phase 3 can re-split results back to the
    source composition file.
    """
    if SOURCE_FILE_COLUMN not in chunk_df.columns:
        raise AssertionError(
            f"_source_file column missing from chunk {chunk_id} "
            "— Phase 3 cannot re-split results without it."
        )

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"process_log_core{chunk_id}.txt")
    file_handler = logging.FileHandler(log_path, mode="a" if restart else "w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    worker_logger = logging.getLogger(f"spices.intercalation.worker.{chunk_id}")
    worker_logger.addHandler(file_handler)
    worker_logger.setLevel(logging.INFO)

    mp_client = get_client()
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )
    os.makedirs(runtime_dir, exist_ok=True)

    for index in range(len(chunk_df)):
        row = chunk_df.iloc[index]

        if pd.notna(row.get("Seawater stability")):
            worker_logger.info(
                "[core %d] [%d/%d] %s (%s) — skipping (already evaluated)",
                chunk_id, index + 1, len(chunk_df),
                row.get("Formula", "?"), row.get("MP id", "?"),
            )
            continue

        worker_logger.info(
            "[core %d] [%d/%d] %s (%s) [src=%s]",
            chunk_id, index + 1, len(chunk_df),
            row.get("Formula", "?"), row.get("MP id", "?"),
            row.get(SOURCE_FILE_COLUMN, "?"),
        )

        for attempt in range(max_retries):
            try:
                updated_row = update_seawater_metastability_columns_by_mp_id(
                    row.copy(),
                    cache=cache,
                    mp_client=mp_client,
                    element_str=element_str,
                    metastability_points=True,
                    decomp_energy_for_metastability=(
                        decomp_energy_for_metastability
                    ),
                )
                chunk_df.iloc[index] = updated_row
                break
            except (AttributeError, ImportError, TypeError) as e:
                worker_logger.error(
                    "[core %d] Non-retryable error for %s: %s",
                    chunk_id, row.get("MP id", "?"), e,
                )
                break
            except Exception as e:
                worker_logger.warning(
                    "[core %d] Attempt %d/%d for %s failed: %s",
                    chunk_id, attempt + 1, max_retries,
                    row.get("MP id", "?"), e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        else:
            worker_logger.error(
                "[core %d] Exhausted retries for %s, skipping.",
                chunk_id, row.get("MP id", "?"),
            )

        if save_interval and (index + 1) % save_interval == 0:
            part = (index // save_interval) + 1
            ckpt_path = os.path.join(
                runtime_dir,
                f"{element_str}_core{chunk_id}_part{part}.xlsx",
            )
            chunk_df.to_excel(ckpt_path, index=False)
            worker_logger.info(
                "[core %d] Checkpoint: %d rows -> %s",
                chunk_id, index + 1, ckpt_path,
            )

    final_path = os.path.join(
        runtime_dir, f"{element_str}_core{chunk_id}_final.xlsx",
    )
    chunk_df.to_excel(final_path, index=False)
    worker_logger.info(
        "[core %d] Done. %d rows -> %s",
        chunk_id, len(chunk_df), final_path,
    )

    worker_logger.removeHandler(file_handler)
    file_handler.close()
    return chunk_df


def _phase3_redistribute_and_write(
    originals_by_file, enriched_new_rows_df, composition_dir,
):
    """Phase 3: split enriched rows by source file, concat with the
    original df, and atomically write each composition xlsx.

    Files in *originals_by_file* with no enriched rows are still
    re-written so the post-condition ("every input file is touched")
    holds — useful for invariants in downstream tests.
    """
    by_source = {}
    if len(enriched_new_rows_df) > 0:
        by_source = dict(tuple(
            enriched_new_rows_df.groupby(SOURCE_FILE_COLUMN, sort=False)
        ))

    for fname, original_df in originals_by_file.items():
        path = os.path.join(composition_dir, fname)
        new_rows = by_source.get(fname)
        if new_rows is None or len(new_rows) == 0:
            _atomic_write_xlsx(original_df, path)
            continue

        new_rows = new_rows.drop(columns=[SOURCE_FILE_COLUMN])
        add_seawater_metastability_bool_columns(new_rows)
        out = pd.concat([original_df, new_rows], ignore_index=True)
        _atomic_write_xlsx(out, path)


def step2_enrich_charged_composition_sheets(
    composition_dir,
    working_ion,
    mp_client,
    cache,
    element_str,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    structure_cache_dir="All_charged_structures",
    decomp_energy_for_metastability=SEAWATER_DECOMP_ENERGY_THRESHOLD,
    n_workers=1,
    cache_dir=".",
    restart=False,
    save_interval=100,
    max_retries=500,
    retry_delay=10.0,
    output_dir=None,
):
    """
    Step 2: Enrich each per-composition sheet with fully-charged MP
    phases and their seawater metastability.

    Three-phase pipeline so multi-worker mode can flatten new charged
    structures across all sheets and keep every worker fed regardless
    of per-sheet imbalance:

    1. **Phase 1 (parallel when ``n_workers > 1``, parent otherwise)**
       — for each xlsx in ``composition_dir``, discover new charged mp
       ids and fetch their MP thermo data.  Tag each new row with a
       ``_source_file`` column.  Multi-worker Phase 1 splits the file
       list round-robin and dispatches discovery to a
       :class:`ProcessPoolExecutor`; each worker has its own
       ``cache_dict`` and ``mp_client`` (resolved post-fork).  No
       checkpointing — on :class:`BrokenProcessPool` the failed
       chunks are retried sequentially in the parent (warm disk
       caches make this fast).
    2. **Phase 2 (parallel, workers)** — flatten all new rows into one
       global DataFrame, chunk row-wise round-robin, and dispatch to a
       :class:`ProcessPoolExecutor`.  Each worker creates its own
       ``mp_client`` and :class:`PourbaixCache` after fork.
       :func:`update_seawater_metastability_columns_by_mp_id` is the
       per-row work unit (same expensive sea-Pourbaix bottleneck as
       Stage 2).
    3. **Phase 3 (sequential, parent)** — split enriched rows by
       ``_source_file``, drop the tag, add ``_bool`` companion columns,
       concat with the original df, and atomically write each
       composition xlsx.

    When ``n_workers <= 1`` Phase 2 runs sequentially in the parent;
    no checkpoints are written and ``restart`` is a no-op.

    Parameters
    ----------
    composition_dir : str
        Directory of per-composition xlsx files (output of step 1).
    working_ion : str
    mp_client : MPRester
        Used by Phase 1 (parent) for structure discovery.  In
        multi-worker mode workers re-resolve their own client via
        :func:`spices.mp_client.get_client` and rely on env vars.
    cache : PourbaixCache
        Used by the sequential ``n_workers <= 1`` path.  Workers in
        multi-worker mode build their own caches.
    element_str : str
    charged_struct_dir, struct_dir, structure_cache_dir : str
    decomp_energy_for_metastability : float
    n_workers : int
        ``1`` (default) runs Phase 2 sequentially; ``>1`` uses
        :class:`ProcessPoolExecutor` with ``BrokenProcessPool``
        recovery.
    cache_dir : str
        Directory containing ``saved_pbx/`` and
        ``saved_pbx_entries/`` (workers build their own
        :class:`PourbaixCache` rooted here).
    restart : bool
        If True (and ``n_workers > 1``), load per-row checkpoints from
        ``runtime_sheets/`` and dispatch only un-evaluated rows.
    save_interval : int
        Worker writes a part-checkpoint every N rows.  ``0`` disables
        intermediate checkpoints (final-only).
    max_retries : int
    retry_delay : float
    output_dir : str or None
        Root directory for ``runtime_sheets/`` and ``process_logs/``.
        Defaults to the parent of ``composition_dir``.
    """
    if output_dir is None:
        output_dir = os.path.dirname(composition_dir.rstrip(os.sep)) or "."
    runtime_dir = os.path.join(output_dir, "runtime_sheets")
    log_dir = os.path.join(output_dir, "process_logs")

    _cleanup_orphan_tmp_xlsx(composition_dir)

    originals_by_file, all_new_rows_df = _phase1_discover_all_files(
        composition_dir,
        working_ion=working_ion,
        mp_client=mp_client,
        element_str=element_str,
        charged_struct_dir=charged_struct_dir,
        struct_dir=struct_dir,
        structure_cache_dir=structure_cache_dir,
        n_workers=n_workers,
        log_dir=log_dir,
    )

    if len(all_new_rows_df) == 0:
        logger.info("No new charged structures to evaluate; rewriting originals.")
        _phase3_redistribute_and_write(
            originals_by_file, all_new_rows_df, composition_dir,
        )
        if os.path.isdir(runtime_dir):
            shutil.rmtree(runtime_dir)
        return

    _seal_seawater_columns_for_evaluation(all_new_rows_df)

    if restart and n_workers > 1 and os.path.isdir(runtime_dir):
        ckpt_dfs = _load_checkpoints(runtime_dir, element_str)
        if ckpt_dfs:
            combined = pd.concat(ckpt_dfs, ignore_index=True)
            _merge_evaluated_new_rows(all_new_rows_df, combined)
            logger.info(
                "Restart: merged %d evaluated rows from checkpoints",
                combined["Seawater stability"].notna().sum()
                if "Seawater stability" in combined.columns else 0,
            )
        if os.path.isdir(runtime_dir):
            shutil.rmtree(runtime_dir)

    pending_mask = all_new_rows_df["Seawater stability"].isna()
    pending_df = all_new_rows_df.loc[pending_mask].reset_index(drop=True)
    evaluated_df = all_new_rows_df.loc[~pending_mask].reset_index(drop=True)
    logger.info(
        "Phase 2 dispatch: %d pending, %d already evaluated",
        len(pending_df), len(evaluated_df),
    )

    if n_workers <= 1:
        # Sequential Phase 2 — runs in parent, no subprocess pool.
        for index in range(len(pending_df)):
            row = pending_df.iloc[index]
            logger.info(
                "[seq] [%d/%d] %s (%s) [src=%s]",
                index + 1, len(pending_df),
                row.get("Formula", "?"), row.get("MP id", "?"),
                row.get(SOURCE_FILE_COLUMN, "?"),
            )
            for attempt in range(max_retries):
                try:
                    updated_row = update_seawater_metastability_columns_by_mp_id(
                        row.copy(),
                        cache=cache,
                        mp_client=mp_client,
                        element_str=element_str,
                        metastability_points=True,
                        decomp_energy_for_metastability=(
                            decomp_energy_for_metastability
                        ),
                    )
                    pending_df.iloc[index] = updated_row
                    break
                except (AttributeError, ImportError, TypeError) as e:
                    logger.error(
                        "[seq] Non-retryable error for %s: %s",
                        row.get("MP id", "?"), e,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "[seq] Attempt %d/%d for %s failed: %s",
                        attempt + 1, max_retries,
                        row.get("MP id", "?"), e,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            else:
                logger.error(
                    "[seq] Exhausted retries for %s, skipping.",
                    row.get("MP id", "?"),
                )
        enriched_new_rows_df = pd.concat(
            [evaluated_df, pending_df], ignore_index=True,
        )
    else:
        avail_gb = _available_memory_gb()
        if avail_gb is not None:
            need_gb = n_workers * EST_PER_WORKER_GB
            if need_gb > 0.8 * avail_gb:
                suggested = max(1, int(0.8 * avail_gb / EST_PER_WORKER_GB))
                logger.warning(
                    "n_workers=%d * ~%.1f GB/worker (~%.0f GB) exceeds 80%% "
                    "of available memory (%.0f GB). Workers may be silently "
                    "OOM-killed by the kernel. Consider n_workers=%d.",
                    n_workers, EST_PER_WORKER_GB, need_gb, avail_gb, suggested,
                )

        worker_kwargs = dict(
            element_str=element_str,
            cache_dir=cache_dir,
            runtime_dir=runtime_dir,
            log_dir=log_dir,
            decomp_energy_for_metastability=decomp_energy_for_metastability,
            save_interval=save_interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
            restart=restart,
        )

        chunks = _split_dataframe(pending_df, n_workers)
        logger.info(
            "Phase 2 dispatch: %d pending rows across %d workers",
            len(pending_df), len(chunks),
        )
        pool_args = [(chunk, cid) for cid, chunk in enumerate(chunks)]
        original_by_cid = {cid: chunk for chunk, cid in pool_args}
        results_by_cid = {}

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_cid = {
                pool.submit(
                    _process_chunk_wrapper_intercalation,
                    chunk, cid, worker_kwargs,
                ): cid
                for chunk, cid in pool_args
            }
            try:
                for fut in as_completed(future_to_cid):
                    cid = future_to_cid[fut]
                    try:
                        results_by_cid[cid] = fut.result()
                    except BrokenProcessPool:
                        logger.error(
                            "Worker %d died (likely OOM-killed). "
                            "Recovering chunk from runtime_sheets/.",
                            cid,
                        )
                        results_by_cid[cid] = _recover_chunk_for_core(
                            runtime_dir, element_str, cid,
                            original_by_cid[cid],
                        )
                    except Exception:
                        logger.exception("Worker %d raised", cid)
                        raise
            except BrokenProcessPool:
                logger.error(
                    "ProcessPool broken — recovering remaining chunks "
                    "from runtime_sheets/."
                )

        for cid, chunk in original_by_cid.items():
            if cid not in results_by_cid:
                logger.error(
                    "Chunk %d never returned. Recovering from checkpoint.",
                    cid,
                )
                results_by_cid[cid] = _recover_chunk_for_core(
                    runtime_dir, element_str, cid, chunk,
                )

        processed_chunks = [
            results_by_cid[cid] for cid in sorted(results_by_cid)
        ]
        enriched_new_rows_df = pd.concat(
            [evaluated_df] + processed_chunks, ignore_index=True,
        )

    _phase3_redistribute_and_write(
        originals_by_file, enriched_new_rows_df, composition_dir,
    )

    if os.path.isdir(runtime_dir):
        shutil.rmtree(runtime_dir)
        logger.info("Cleaned up %s", runtime_dir)


def _step3_process_composition(
    fname,
    composition_dir,
    plateau_dir,
    adjacency_cache_dir,
    working_ion,
    working_ion_valency,
    working_ion_voltage_vs_SHE,
    working_ion_energy,
    element_str,
    selection_criteria,
    use_only_on_hull_phases,
    save_plots,
    save_graphs,
    charged_struct_dir,
    struct_dir,
    structure_cache_dir,
    decomp_energy_for_metastability,
    cache_dir,
    restart,
):
    """Per-composition worker for the parallel step-3 dispatch.

    Module-level so it pickles cleanly for ``ProcessPoolExecutor``.

    Resolves the MP client and ``PourbaixCache`` after fork so each
    worker uses its own bounded in-memory LRU layer; the on-disk
    pickle store under ``cache_dir`` is the shared source of truth.
    Raises on any failure so the parent's
    :func:`_dispatch_phase1_tasks` aborts the run (per user
    direction: fail-fast over silent partial completion).
    """
    framework_name = fname.replace(".xlsx", "")
    framework_dir = os.path.join(plateau_dir, framework_name)

    if restart:
        all_present = all(
            os.path.exists(
                os.path.join(framework_dir, f"sc_{sc}", fname),
            )
            for sc in selection_criteria
        )
        if all_present:
            logger.info(
                "Step 3: %s — all sc xlsxes present, skipping (restart)",
                fname,
            )
            return

    os.makedirs(framework_dir, exist_ok=True)

    mp_client = get_client()
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )

    path = os.path.join(composition_dir, fname)
    df = pd.read_excel(path)

    adjacency_cache_path = os.path.join(
        adjacency_cache_dir, f"{framework_name}.pkl",
    )
    component_dfs, component_graphs = build_graph_components_from_sheet(
        df,
        working_ion=working_ion,
        charged_struct_dir=charged_struct_dir,
        struct_dir=struct_dir,
        structure_cache_dir=structure_cache_dir,
        adjacency_cache_path=adjacency_cache_path,
    )

    if save_graphs:
        save_component_graphs(
            component_graphs,
            os.path.join(framework_dir, "graphs"),
        )

    voltage_results = {sc: {} for sc in selection_criteria}

    for i, comp_df in enumerate(component_dfs):
        logger.debug("Step 3: %s/component_%d", fname, i)

        for sc in selection_criteria:
            sc_dir = os.path.join(framework_dir, f"sc_{sc}")
            os.makedirs(sc_dir, exist_ok=True)

            plateau_dict = obtain_voltage_plateau_data_based_on_convex_hull(
                comp_df,
                selection_criteria=[sc],
                working_ion=working_ion,
                working_ion_valency=working_ion_valency,
                working_ion_voltage_vs_SHE=working_ion_voltage_vs_SHE,
                working_ion_energy=working_ion_energy,
                cache=cache,
                mp_client=mp_client,
                element_str=element_str,
                use_only_on_hull_phases=use_only_on_hull_phases,
                save_plot=save_plots,
                save_plot_path=sc_dir,
                decomp_energy_for_metastability=decomp_energy_for_metastability,
            )

            if isinstance(plateau_dict.get(f"sc {sc}"), pd.DataFrame):
                voltage_results[sc][
                    f"voltage_component_{i}"
                ] = plateau_dict[f"sc {sc}"]

    for sc in selection_criteria:
        if not voltage_results[sc]:
            continue

        sc_dir = os.path.join(framework_dir, f"sc_{sc}")
        sheet_path = os.path.join(sc_dir, fname)

        with pd.ExcelWriter(sheet_path, engine="openpyxl") as writer:
            for name, vdf in voltage_results[sc].items():
                vdf.to_excel(writer, sheet_name=name, index=False)


def _parse_coords_cell(val):
    """Parse a ``Seawater metastability (Criterion N)`` cell into a list
    of ``(pH, V)`` tuples.

    Cells are stored as object lists or their string repr (see
    :func:`_get_voltage_plateau_df`).  Returns ``[]`` for NaN, empty, or
    unparseable cells so the caller can treat "no coords" uniformly.
    """
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    return []


def _rerender_process_composition(
    fname,
    composition_dir,
    plateau_dir,
    adjacency_cache_dir,
    working_ion,
    selection_criteria,
    save_voltage_plots,
    save_graphs,
    charged_struct_dir,
    struct_dir,
    structure_cache_dir,
    skipped_dir,
):
    """Offline per-composition plot re-render worker (no MP/Pourbaix).

    Regenerates, purely from a completed run's persisted data:

    * **component-graph PDFs** (``graphs/component_{i}.pdf``) via
      :func:`build_graph_components_from_sheet` (reuses
      ``adjacency_cache/{framework}.pkl`` and loads structs from disk
      CIFs — no MP fetch) + :func:`save_component_graphs`; and
    * **voltage/pH-overlap PDFs** (``sc_{N}/{cMP}-{dMP}_{cF}-{dF}.pdf``)
      via :func:`find_ph_range_at_voltage` with ``save_plot=True``.  The
      charged/discharged metastability coords come from the composition
      xlsx, the voltage from the ``sc_{N}`` plateau sheet.

    No ``mp_client``, no :class:`PourbaixCache`, no xlsx writes.

    Best-effort: any per-composition failure is logged and recorded,
    never raised, so one bad framework can't abort the whole rerender
    (this is a cosmetic regeneration, not the scientific pipeline).
    Voltage plots whose coords are absent from the composition sheet
    (the documented offline limitation) are recorded; all skips are
    written to ``{skipped_dir}/{framework}.csv`` for the parent to
    aggregate.
    """
    framework_name = fname.replace(".xlsx", "")
    framework_dir = os.path.join(plateau_dir, framework_name)
    # (framework, criterion, charged_mp, discharged_mp, reason)
    skipped = []

    try:
        df = pd.read_excel(os.path.join(composition_dir, fname))
    except Exception as e:
        logger.warning("Rerender: cannot read %s (%s); skipping", fname, e)
        return

    if save_graphs:
        try:
            adjacency_cache_path = os.path.join(
                adjacency_cache_dir, f"{framework_name}.pkl",
            )
            _, component_graphs = build_graph_components_from_sheet(
                df,
                working_ion=working_ion,
                charged_struct_dir=charged_struct_dir,
                struct_dir=struct_dir,
                structure_cache_dir=structure_cache_dir,
                adjacency_cache_path=adjacency_cache_path,
            )
            save_component_graphs(
                component_graphs, os.path.join(framework_dir, "graphs"),
            )
        except Exception as e:
            logger.warning(
                "Rerender: graph render failed for %s (%s); skipping graphs",
                framework_name, e,
            )
            skipped.append((framework_name, "graphs", "", "", repr(e)))

    if save_voltage_plots:
        for sc in selection_criteria:
            sc_dir = os.path.join(framework_dir, f"sc_{sc}")
            sheet_path = os.path.join(sc_dir, fname)
            if not os.path.exists(sheet_path):
                continue

            coord_col = f"Seawater metastability (Criterion {sc})"
            if coord_col not in df.columns:
                logger.warning(
                    "Rerender: %s missing column %r; skipping sc_%s "
                    "voltage plots", fname, coord_col, sc,
                )
                continue
            coords_by_mpid = {
                row["MP id"]: _parse_coords_cell(row[coord_col])
                for _, row in df.iterrows()
            }

            try:
                sheets = pd.read_excel(sheet_path, sheet_name=None)
            except Exception as e:
                logger.warning(
                    "Rerender: cannot read %s (%s); skipping", sheet_path, e,
                )
                continue

            for sheet_df in sheets.values():
                for _, prow in sheet_df.iterrows():
                    c_mp = prow.get("Charged MP id")
                    d_mp = prow.get("Discharged MP id")
                    c_f = prow.get("Charged Formula")
                    d_f = prow.get("Discharged Formula")
                    voltage = prow.get("V vs SHE (V)")

                    c_coords = coords_by_mpid.get(c_mp, [])
                    d_coords = coords_by_mpid.get(d_mp, [])

                    if not c_coords or not d_coords or pd.isna(voltage):
                        logger.warning(
                            "Rerender: skipping voltage plot %s-%s "
                            "(%s/sc_%s) — coords/voltage absent from "
                            "composition sheet", c_mp, d_mp,
                            framework_name, sc,
                        )
                        skipped.append((
                            framework_name, f"sc_{sc}", c_mp, d_mp,
                            "coords or voltage absent",
                        ))
                        continue

                    plot_path = os.path.join(
                        sc_dir, f"{c_mp}-{d_mp}_{c_f}-{d_f}.pdf",
                    )
                    try:
                        find_ph_range_at_voltage(
                            c_coords,
                            d_coords,
                            voltage,
                            save_plot=True,
                            save_plot_path=plot_path,
                            A_phase_name=f"{c_f}\n{c_mp}",
                            B_phase_name=f"{d_f}\n{d_mp}",
                        )
                    except Exception as e:
                        logger.warning(
                            "Rerender: voltage plot render failed %s-%s "
                            "(%s); skipping", c_mp, d_mp, e,
                        )
                        skipped.append((
                            framework_name, f"sc_{sc}", c_mp, d_mp, repr(e),
                        ))

    if skipped:
        os.makedirs(skipped_dir, exist_ok=True)
        pd.DataFrame(
            skipped,
            columns=[
                "Framework formula", "Criterion",
                "Charged MP id", "Discharged MP id", "Reason",
            ],
        ).to_csv(
            os.path.join(skipped_dir, f"{framework_name}.csv"), index=False,
        )


def step3_create_voltage_plateau_sheets(
    composition_dir,
    output_dir,
    working_ion,
    working_ion_valency,
    working_ion_voltage_vs_SHE,
    working_ion_energy,
    element_str,
    selection_criteria=None,
    use_only_on_hull_phases=True,
    save_plots=False,
    save_graphs=True,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    structure_cache_dir="All_charged_structures",
    decomp_energy_for_metastability=0.3,
    cache_dir=".",
    n_workers=1,
    restart=False,
):
    """
    Step 3: Partition each enriched sheet into graph-based components
    and compute voltage plateaus per component per selection criterion.

    Workers (or the in-process sequential path) resolve their own
    ``mp_client`` and :class:`PourbaixCache` from ``cache_dir`` after
    fork, mirroring step-2's Phase-2 pattern.  Per-composition
    adjacency dicts are pickled to
    ``{output_dir}/../adjacency_cache/{framework}.pkl`` (sibling of
    ``Voltage_plateau_data/``) so a restart skips the O(n^2)
    StructureMatcher rebuild.

    Parameters
    ----------
    composition_dir : str
        Directory with enriched per-composition Excel files (step 2).
    output_dir : str
        Directory for voltage plateau output (``Voltage_plateau_data/``).
    working_ion : str
    working_ion_valency : int
    working_ion_voltage_vs_SHE : float
    working_ion_energy : float
        GGA(+U) energy per atom of the stable elemental working-ion phase.
    element_str : str
    selection_criteria : list of int or None
    use_only_on_hull_phases : bool
    save_plots : bool
        Save pH-overlap PDFs (passed through to voltage computation).
    save_graphs : bool
        Save one ``component_{idx}.pdf`` per connected component to
        ``{framework_dir}/graphs/``.  Default True.
    charged_struct_dir, struct_dir, structure_cache_dir : str
    decomp_energy_for_metastability : float
    cache_dir : str
        Root dir for the shared Pourbaix on-disk cache
        (``saved_pbx_entries/``, ``saved_pbx/``).  Workers instantiate
        their own :class:`PourbaixCache` against this.
    n_workers : int
        Parallel workers for per-composition dispatch.  ``<=1`` runs
        sequentially in-process.
    restart : bool
        If True, skip a composition whose ``sc_{N}/{composition}.xlsx``
        already exists for every requested selection criterion.  The
        adjacency cache is always-on regardless of this flag.
    """
    if selection_criteria is None:
        selection_criteria = DEFAULT_SELECTION_CRITERIA

    os.makedirs(output_dir, exist_ok=True)
    adjacency_cache_dir = os.path.join(
        os.path.dirname(output_dir.rstrip(os.sep)) or ".",
        "adjacency_cache",
    )
    os.makedirs(adjacency_cache_dir, exist_ok=True)

    excel_files = sorted(
        f for f in os.listdir(composition_dir) if f.endswith(".xlsx")
    )

    def _args_for(fname):
        return (
            fname,
            composition_dir,
            output_dir,
            adjacency_cache_dir,
            working_ion,
            working_ion_valency,
            working_ion_voltage_vs_SHE,
            working_ion_energy,
            element_str,
            list(selection_criteria),
            use_only_on_hull_phases,
            save_plots,
            save_graphs,
            charged_struct_dir,
            struct_dir,
            structure_cache_dir,
            decomp_energy_for_metastability,
            cache_dir,
            restart,
        )

    _dispatch_phase1_tasks(
        "Step 3",
        n_workers,
        excel_files,
        _step3_process_composition,
        _args_for,
    )


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


def rerender_intercalation_plots(
    element,
    output_dir=None,
    selection_criteria=None,
    save_voltage_plots=True,
    save_graphs=True,
    n_workers=1,
):
    """Re-render step-3 plots offline from a completed run's data.

    Mirrors Stage-4's ``--rerender-plots``: reuses a finished run's
    persisted artifacts to redraw plots **without any Materials Project
    / snapshot access** and **without rewriting any xlsx**.  Two plot
    kinds are regenerated (see :func:`_rerender_process_composition`):

    * component-graph PDFs (``Voltage_plateau_data/{fw}/graphs/``), and
    * voltage/pH-overlap PDFs (``Voltage_plateau_data/{fw}/sc_{N}/``).

    Reads ``unique_charged_composition_sheets/`` (metastability coords),
    ``Voltage_plateau_data/{fw}/sc_{N}/{fw}.xlsx`` (accepted plateaus +
    voltages), ``adjacency_cache/``, and the on-disk struct CIFs.

    Limitation
    ----------
    Only voltage plots whose charged/discharged metastability coords are
    persisted in the composition sheet can be regenerated offline.
    Coords computed on-the-fly for seawater-stable phases by
    :func:`populate_seawater_metastability_points_for_seawater_stable_phases`
    during the original run are not written back, so those plateaus'
    plots are skipped (and recorded in ``rerender_skipped_plots.csv``).
    Component graphs are always fully reproducible.

    Parameters
    ----------
    element : str
        Working ion (``"Ca"``, ``"K"``, ``"Li"``, ``"Mg"``, ``"Na"``).
    output_dir : str or None
        Completed run directory.  Defaults to
        ``"{element}_intercalation"``.
    selection_criteria : list of int or None
        Criteria whose ``sc_{N}`` voltage plots to redraw (default: 4).
    save_voltage_plots : bool
        Redraw the per-pair voltage/pH-overlap PDFs.
    save_graphs : bool
        Redraw the component-graph PDFs.
    n_workers : int
        Parallel per-composition workers (``<=1`` = sequential).

    Raises
    ------
    FileNotFoundError
        If ``output_dir`` lacks a completed step-3 run
        (``unique_charged_composition_sheets/`` with xlsx files and
        ``Voltage_plateau_data/``).
    ValueError
        If both plot kinds are disabled (nothing to render).
    """
    if output_dir is None:
        output_dir = f"{element}_intercalation"
    if selection_criteria is None:
        selection_criteria = DEFAULT_SELECTION_CRITERIA

    if not save_voltage_plots and not save_graphs:
        raise ValueError(
            "Nothing to render: both voltage plots and component graphs "
            "are disabled."
        )

    composition_dir = os.path.join(
        output_dir, "unique_charged_composition_sheets",
    )
    plateau_dir = os.path.join(output_dir, "Voltage_plateau_data")
    adjacency_cache_dir = os.path.join(output_dir, "adjacency_cache")
    struct_dir = os.path.join(output_dir, "structs")
    charged_struct_dir = os.path.join(output_dir, "charged_structs")
    structure_cache_dir = os.path.join(output_dir, "All_charged_structures")
    skipped_dir = os.path.join(output_dir, ".rerender_skipped")
    summary_path = os.path.join(output_dir, "rerender_skipped_plots.csv")

    if not os.path.isdir(composition_dir):
        raise FileNotFoundError(
            f"--rerender-plots requires a completed step-3 run, but "
            f"{composition_dir} was not found. Run the full pipeline "
            f"once first."
        )
    excel_files = sorted(
        f for f in os.listdir(composition_dir) if f.endswith(".xlsx")
    )
    if not excel_files:
        raise FileNotFoundError(
            f"--rerender-plots found no composition sheets in "
            f"{composition_dir}. Run the full pipeline once first."
        )
    if save_voltage_plots and not os.path.isdir(plateau_dir):
        raise FileNotFoundError(
            f"--rerender-plots requires {plateau_dir} for voltage plots, "
            f"but it was not found. Run the full pipeline once first."
        )

    # Clear any stale per-worker skip files from an interrupted rerender.
    if os.path.isdir(skipped_dir):
        shutil.rmtree(skipped_dir)

    def _args_for(fname):
        return (
            fname,
            composition_dir,
            plateau_dir,
            adjacency_cache_dir,
            element,
            list(selection_criteria),
            save_voltage_plots,
            save_graphs,
            charged_struct_dir,
            struct_dir,
            structure_cache_dir,
            skipped_dir,
        )

    logger.info(
        "Rerender: regenerating %s for %d compositions (n_workers=%d)",
        "graphs + voltage plots" if (save_graphs and save_voltage_plots)
        else ("graphs" if save_graphs else "voltage plots"),
        len(excel_files), n_workers,
    )

    _dispatch_phase1_tasks(
        "Rerender",
        n_workers,
        excel_files,
        _rerender_process_composition,
        _args_for,
    )

    # Aggregate per-composition skip files into one summary CSV.
    if os.path.isdir(skipped_dir):
        parts = [
            pd.read_csv(os.path.join(skipped_dir, f))
            for f in sorted(os.listdir(skipped_dir))
            if f.endswith(".csv")
        ]
        shutil.rmtree(skipped_dir)
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            combined.to_csv(summary_path, index=False)
            n_volt = int((combined["Criterion"] != "graphs").sum())
            logger.warning(
                "Rerender: %d item(s) could not be reproduced offline "
                "(%d voltage plot(s)); see %s",
                len(combined), n_volt, summary_path,
            )
            return
    # No skips this run — drop a stale summary so it can't mislead.
    if os.path.exists(summary_path):
        os.remove(summary_path)
    logger.info("Rerender: all requested plots regenerated with no skips.")


def run_intercalation_screening(
    element,
    input_path,
    output_dir=None,
    selection_criteria=None,
    decomp_energy=None,
    use_only_on_hull_phases=True,
    save_plots=True,
    save_component_graphs=True,
    cache_dir=".",
    mp_client=None,
    n_workers=1,
    restart=False,
    save_interval=100,
    max_retries=500,
    retry_delay=10.0,
    rerender_plots=False,
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
    save_component_graphs : bool
        If True (default), save one ``component_{idx}.pdf`` per
        connected component under
        ``Voltage_plateau_data/{framework}/graphs/``.
    cache_dir : str
        Directory for Pourbaix caches.
    mp_client : MPRester or SnapshotMPRester or None
        Materials Project client.  If None, resolved via
        :func:`spices.mp_client.get_client` (sticky-default cache,
        ``SPICES_MP_BACKEND``/``MP_API_KEY`` env vars, or live default).
        See **Multiprocessing and backend resolution** below for the
        ``n_workers > 1`` case.
    n_workers : int
        Parallel workers for sub-step 2 (per-row sea-Pourbaix
        evaluation) **and** sub-step 3 (per-composition graph build +
        voltage plateau).  ``1`` (default) runs both sequentially.
        Step 3 workers fork-resolve their own ``mp_client`` and
        :class:`PourbaixCache` against ``cache_dir``.
    restart : bool
        If True, resume sub-step 2 from per-row checkpoints in
        ``output_dir/runtime_sheets/`` **and** skip sub-step 3
        compositions whose ``sc_{N}/{composition}.xlsx`` already
        exists for every requested selection criterion.  The step-3
        adjacency cache (``output_dir/adjacency_cache/``) is always-on
        regardless of this flag.
    save_interval : int
        Sub-step 2 worker writes a part-checkpoint every N rows.
    max_retries : int
        Maximum retries per row on transient failure.
    retry_delay : float
        Seconds to wait between retries.
    rerender_plots : bool
        If True, skip the entire pipeline (steps 1-4) and instead
        re-render step-3's plots **offline** from ``output_dir``'s
        persisted data via :func:`rerender_intercalation_plots`.  No MP
        / snapshot access and no xlsx writes.  ``save_plots`` selects
        the voltage/pH-overlap PDFs and ``save_component_graphs`` the
        component-graph PDFs.  Returns an empty dict.

    Notes
    -----
    **Multiprocessing and backend resolution**

    The ``mp_client`` argument is honored for Phase-1 discovery and
    for Steps 1, 3, 4 (all run in the parent).  In sub-step 2's
    Phase 2 (``n_workers > 1``), each worker is a separate Python
    interpreter and re-resolves its own client via
    :func:`spices.mp_client.get_client` with no arguments.  Workers
    therefore rely on **environment variables inherited from the
    parent process** to choose a backend:

    * ``SPICES_MP_BACKEND`` — ``"snapshot"`` or ``"live"``.
    * ``SPICES_MP_SNAPSHOT_ROOT``, ``SPICES_MP_SNAPSHOT_RELEASE`` —
      snapshot backend.
    * ``MP_API_KEY`` — live backend.

    See :func:`spices.screening.seawater.run_seawater_screening`'s
    docstring for the full discussion (same caveats apply here).

    Returns
    -------
    dict
        Combined voltage plateau data (``'criterion N'`` -> DataFrame).
        Empty dict when ``rerender_plots=True``.
    """
    if rerender_plots:
        rerender_intercalation_plots(
            element=element,
            output_dir=output_dir,
            selection_criteria=selection_criteria,
            save_voltage_plots=save_plots,
            save_graphs=save_component_graphs,
            n_workers=n_workers,
        )
        return {}

    if output_dir is None:
        output_dir = f"{element}_intercalation"
    if selection_criteria is None:
        selection_criteria = DEFAULT_SELECTION_CRITERIA
    if decomp_energy is None:
        decomp_energy = INTERCALATION_DECOMP_ENERGY_THRESHOLD

    params = WORKING_ION_PARAMS[element]
    working_ion_valency = params["valency"]
    working_ion_voltage_vs_SHE = params["voltage_vs_SHE"]

    if mp_client is None:
        mp_client = get_client()
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )

    # Get working ion reference energy (once)
    logger.info("Querying %s reference energy from MP", element)
    working_ion_energy = get_energy_of_working_ion(element, mp_client)

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
    logger.info("=== Step 2: Enriching charged composition sheets ===")
    step2_enrich_charged_composition_sheets(
        composition_dir,
        element,
        mp_client,
        cache,
        element,
        charged_struct_dir=charged_struct_dir,
        struct_dir=struct_dir,
        structure_cache_dir=structure_cache_dir,
        decomp_energy_for_metastability=decomp_energy,
        n_workers=n_workers,
        cache_dir=cache_dir,
        restart=restart,
        save_interval=save_interval,
        max_retries=max_retries,
        retry_delay=retry_delay,
        output_dir=output_dir,
    )

    # Step 3
    logger.info("=== Step 3: Computing voltage plateaus ===")
    step3_create_voltage_plateau_sheets(
        composition_dir,
        plateau_dir,
        working_ion=element,
        working_ion_valency=working_ion_valency,
        working_ion_voltage_vs_SHE=working_ion_voltage_vs_SHE,
        working_ion_energy=working_ion_energy,
        element_str=element,
        selection_criteria=selection_criteria,
        use_only_on_hull_phases=use_only_on_hull_phases,
        save_plots=save_plots,
        save_graphs=save_component_graphs,
        charged_struct_dir=charged_struct_dir,
        struct_dir=struct_dir,
        structure_cache_dir=structure_cache_dir,
        decomp_energy_for_metastability=decomp_energy,
        cache_dir=cache_dir,
        n_workers=n_workers,
        restart=restart,
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

"""
Charged structure generation, caching, and StructureMatcher-based
subgroup creation for intercalation voltage analysis.
"""

import logging
import os

import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)

# Default matcher with attempt_supercell=True to catch supercell matches
# that the original notebooks missed (which used the default False).
DEFAULT_MATCHER = StructureMatcher(attempt_supercell=True)


def generate_charged_struct_from_MP(
    mp_id,
    working_ion,
    mp_client,
    struct_dir=None,
    charged_struct_dir=None,
):
    """
    Download a structure from MP and remove the working ion to produce
    the charged (de-intercalated) framework.

    Parameters
    ----------
    mp_id : str
        Materials Project ID.
    working_ion : str
        Element symbol to remove (e.g. ``"Ca"``).
    mp_client : MPRester
    struct_dir : str or None
        Directory to save the original structure CIF.
    charged_struct_dir : str or None
        Directory to save the charged structure CIF.

    Returns
    -------
    Structure or list
        The charged structure, or an empty list if the MP ID is not
        found.
    """
    logger.info("Generating charged structure for %s", mp_id)

    struct = mp_client.materials.get_structure_by_material_id(mp_id)

    # MP may return empty list for missing IDs
    if isinstance(struct, list) and len(struct) == 0:
        return struct

    # Save original structure
    if struct_dir is not None:
        os.makedirs(struct_dir, exist_ok=True)
        struct.to_file(
            filename=os.path.join(struct_dir, f"{mp_id}.cif"), fmt="cif",
        )

    # Remove working ion to get charged framework
    struct.remove_species([Element(working_ion)])

    # Defensive: ensure Element objects (not charged Species)
    struct.remove_oxidation_states()

    # Save charged structure
    if charged_struct_dir is not None:
        os.makedirs(charged_struct_dir, exist_ok=True)
        struct.to_file(
            filename=os.path.join(charged_struct_dir, f"{mp_id}.cif"),
            fmt="cif",
        )

    return struct


def read_or_generate_charged_struct(
    mp_id,
    working_ion,
    mp_client,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    cache_dict=None,
):
    """
    Retrieve a charged structure from memory cache, disk cache, or MP.

    Parameters
    ----------
    mp_id : str
    working_ion : str
    mp_client : MPRester
    charged_struct_dir : str
        Directory for disk-cached charged structure CIFs.
    struct_dir : str
        Directory for disk-cached original structure CIFs.
    cache_dict : dict or None
        In-memory cache mapping MP ID to Structure. If provided,
        structures are stored here for fast retrieval.

    Returns
    -------
    Structure or list
        The charged structure, or empty list if not found.
    """
    # Check memory cache
    if cache_dict is not None and mp_id in cache_dict:
        return cache_dict[mp_id]

    # Check disk cache
    if os.path.exists(charged_struct_dir):
        cif_path = os.path.join(charged_struct_dir, f"{mp_id}.cif")
        if os.path.exists(cif_path):
            struct = Structure.from_file(cif_path)
            # Defensive: ensure Element objects
            struct.remove_oxidation_states()
            if cache_dict is not None:
                cache_dict[mp_id] = struct
            return struct

    # Download and generate
    struct = generate_charged_struct_from_MP(
        mp_id, working_ion, mp_client,
        struct_dir=struct_dir,
        charged_struct_dir=charged_struct_dir,
    )
    if cache_dict is not None and not isinstance(struct, list):
        cache_dict[mp_id] = struct

    return struct


def find_structures(
    structure,
    mp_client,
    structure_cache_dir="All_charged_structures",
    matcher=None,
):
    """
    Find MP IDs of structures matching the given charged framework.

    Searches a disk cache of CIF files organized by reduced formula.
    If the cache directory for this formula doesn't exist, downloads
    all structures with that formula from MP and saves them.

    Parameters
    ----------
    structure : Structure
        The charged framework structure to match.
    mp_client : MPRester
    structure_cache_dir : str
        Root directory for cached structure CIFs.
    matcher : StructureMatcher or None
        Matcher to use. Defaults to ``DEFAULT_MATCHER``
        (``attempt_supercell=True``).

    Returns
    -------
    list of str
        Matching MP IDs.
    """
    if matcher is None:
        matcher = DEFAULT_MATCHER

    reduced_formula = structure.composition.reduced_formula
    mp_id_list = []

    os.makedirs(structure_cache_dir, exist_ok=True)
    formula_dir = os.path.join(structure_cache_dir, reduced_formula)

    if os.path.exists(formula_dir):
        # Match against saved structures
        for fname in os.listdir(formula_dir):
            if not fname.endswith(".cif"):
                continue
            saved_struct = Structure.from_file(
                os.path.join(formula_dir, fname),
            )
            saved_struct.remove_oxidation_states()
            if matcher.fit(struct1=structure, struct2=saved_struct, symmetric=True):
                mp_id_list.append(fname.split(".")[0])

        return mp_id_list

    # Download and cache all structures with this formula
    downloaded = mp_client.materials.search(
        formula=reduced_formula,
        all_fields=False,
        fields=["material_id", "structure"],
    )

    if not downloaded:
        return mp_id_list

    os.makedirs(formula_dir, exist_ok=True)
    for entry in downloaded:
        entry.structure.to_file(
            filename=os.path.join(formula_dir, f"{entry.material_id}.cif"),
            fmt="cif",
        )

    # Recurse to match against the newly saved structures
    return find_structures(
        structure, mp_client,
        structure_cache_dir=structure_cache_dir,
        matcher=matcher,
    )


def create_subgroups(
    df,
    working_ion,
    mp_client,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    cache_dict=None,
    matcher=None,
):
    """
    Group DataFrame rows by structural similarity of their charged
    (de-intercalated) frameworks using ``StructureMatcher``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``'MP id'`` column.
    working_ion : str
    mp_client : MPRester
    charged_struct_dir : str
    struct_dir : str
    cache_dict : dict or None
        In-memory cache for charged structures.
    matcher : StructureMatcher or None
        Defaults to ``DEFAULT_MATCHER`` (``attempt_supercell=True``).

    Returns
    -------
    dict
        Mapping from subgroup index (int, 1-based) to list of
        ``(pd.Series, Structure)`` tuples.
    """
    if matcher is None:
        matcher = DEFAULT_MATCHER
    if cache_dict is None:
        cache_dict = {}

    subgroups = {}

    for _, row in df.iterrows():
        mp_id = row["MP id"]
        logger.info("Querying structure of %s", mp_id)

        struct = read_or_generate_charged_struct(
            mp_id, working_ion, mp_client,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            cache_dict=cache_dict,
        )

        # Skip if structure not found
        if isinstance(struct, list) and len(struct) == 0:
            logger.warning("%s not found in MP database", mp_id)
            continue

        # Try to fit into an existing subgroup
        matched = False
        for i, subgroup in subgroups.items():
            if matcher.fit(subgroup[0][1], struct):
                subgroups[i].append((row, struct))
                matched = True
                logger.debug("Added %s to subgroup %d", mp_id, i)
                break

        if not matched:
            new_idx = len(subgroups) + 1
            subgroups[new_idx] = [(row, struct)]
            logger.debug("Created subgroup %d for %s", new_idx, mp_id)

    return subgroups


def subgroups_to_excel_sheets(df, subgroups):
    """
    Convert subgroups dict to a dict of DataFrames suitable for
    writing to a multi-sheet Excel file.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame (written as ``'Sheet 1'``).
    subgroups : dict
        As returned by :func:`create_subgroups`.

    Returns
    -------
    dict
        Mapping from sheet name to DataFrame.
    """
    sheets = {"Sheet 1": df}
    for i, subgroup in subgroups.items():
        sheets[f"subgroup {i}"] = pd.DataFrame(
            [item[0] for item in subgroup]
        )
    return sheets


def add_charged_struct_info(
    sheets_dict,
    working_ion,
    mp_client,
    cache,
    element_str,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    structure_cache_dir="All_charged_structures",
    cache_dict=None,
    matcher=None,
    decomp_energy_for_metastability=0.3,
):
    """
    Add fully-charged (de-intercalated) phase data to each subgroup
    sheet.

    For each subgroup, finds the MP IDs of structures matching the
    fully-charged framework, queries their thermo data, computes
    seawater metastability, and appends them as new rows.

    Parameters
    ----------
    sheets_dict : dict
        Mapping from sheet name to DataFrame (as returned by
        :func:`subgroups_to_excel_sheets`).
    working_ion : str
    mp_client : MPRester
    cache : PourbaixCache
    element_str : str
        Working ion element string (e.g. ``"Ca"``).
    charged_struct_dir : str
    struct_dir : str
    structure_cache_dir : str
    cache_dict : dict or None
    matcher : StructureMatcher or None
    decomp_energy_for_metastability : float

    Returns
    -------
    dict
        Updated sheets_dict with charged-phase rows appended.
    """
    from spices.screening.mp_query import collect_mp_data_from_chemsys
    from spices.dataframe_ops import update_seawater_metastability_columns_by_mp_id

    if cache_dict is None:
        cache_dict = {}

    # Cache for seawater metastability to avoid re-computation
    stability_cache = {}

    subgroup_names = [name for name in sheets_dict if "subgroup" in name]

    for sheet_name in subgroup_names:
        df_sheet = sheets_dict[sheet_name]
        existing_mpids = df_sheet["MP id"].tolist()

        completely_charged_mpids = []

        for discharged_mpid in df_sheet["MP id"]:
            charged_struct = read_or_generate_charged_struct(
                discharged_mpid, working_ion, mp_client,
                charged_struct_dir=charged_struct_dir,
                struct_dir=struct_dir,
                cache_dict=cache_dict,
            )

            if isinstance(charged_struct, list) and len(charged_struct) == 0:
                continue

            mpids = find_structures(
                charged_struct, mp_client,
                structure_cache_dir=structure_cache_dir,
                matcher=matcher,
            )

            for mp_id in mpids:
                if mp_id not in existing_mpids and mp_id not in completely_charged_mpids:
                    completely_charged_mpids.append(mp_id)

        if not completely_charged_mpids:
            logger.info("No new charged structures for %s", sheet_name)
            continue

        logger.info(
            "Adding %d charged structures to %s",
            len(completely_charged_mpids), sheet_name,
        )

        # Query thermo data for the new charged MP IDs
        data_dict_list = collect_mp_data_from_chemsys(
            chemsys=None,
            element_str=element_str,
            mp_client=mp_client,
            material_ids=completely_charged_mpids,
        )

        # Add placeholder water stability columns
        for d in data_dict_list:
            d["Structure stable in water"] = np.nan
            d["Structure metastable in water"] = np.nan

        new_rows_df = pd.DataFrame(data_dict_list)

        # Calculate seawater metastability for each new row
        for index, row in new_rows_df.iterrows():
            mp_id = row["MP id"]

            if mp_id in stability_cache:
                new_row = stability_cache[mp_id]
            else:
                new_row = update_seawater_metastability_columns_by_mp_id(
                    row, cache=cache, mp_client=mp_client,
                    element_str=element_str,
                    metastability_points=True,
                    decomp_energy_for_metastability=decomp_energy_for_metastability,
                )
                stability_cache[mp_id] = new_row

            # Add boolean columns for each criterion present in the data
            for col in new_row.index:
                if col.startswith("Seawater metastability (Criterion "):
                    val = new_row[col]
                    bool_val = bool(
                        isinstance(val, (list, tuple)) and len(val) > 0
                    )
                    new_row[f"{col}_bool"] = bool_val

            # Append to the subgroup sheet
            df_sheet = pd.concat(
                [df_sheet, new_row.to_frame().T],
                ignore_index=True,
            )

        sheets_dict[sheet_name] = df_sheet

    return sheets_dict

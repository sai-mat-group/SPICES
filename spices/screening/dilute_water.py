"""
Stage 1: Dilute-limit (pure water) stability screening.

This module replaces the per-element ``query_metastability_stability_data.py``
scripts with a single parameterized implementation.
"""

import logging
import time
from itertools import combinations

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from spices.config import (
    DEFAULT_TRANSITION_METALS,
    DEFAULT_ANIONS,
    DEFAULT_DECOMP_ENERGY_THRESHOLD,
)
from spices.mp_client import get_client
from spices.pourbaix.cache import PourbaixCache
from spices.pourbaix.composition import (
    rm_element,
    make_chemsys_from_combinations,
    get_pourbaix_composition,
)
from spices.pourbaix.stability import is_stable_in_water
from spices.pourbaix.metastability import (
    is_metastable_phase_stable_in_water_search_by_mp_id,
)
from spices.pourbaix.thermo import find_correction

logger = logging.getLogger(__name__)

# DataFrame column names for Stage 1 output
COLUMNS = [
    "MP id",
    "Formula",
    "Pbx elements per formula",
    "Thermo-type",
    "Base XC Functional",
    "Correction XC functional",
    "GGA(+U)/R2SCAN mixing adjustment",
    "E (eV/formula)",
    "Energy Above Hull (eV)",
    "Crystal Structure",
    "Space Group",
    "Structure stable in water",
    "Structure metastable in water",
    "Charged composition",
    "No. of charged structures present",
    "No. of charged structure with same crystal structure",
]


def collect_all_info_from_chemsys(
    chemsys,
    element_str,
    mp_client,
    cache,
    material_ids=None,
    calculate_water_stability=True,
    decomp_energy_for_metastability=0.5,
    read_existing_pbx_diag=False,
    remove_R2SCAN_base=True,
    remove_uncorrected_duplicates=True,
):
    """
    Query the Materials Project for all materials in a chemical system
    and compute their water stability / metastability.

    This is the core per-chemsys function that the Stage 1 runner calls
    in a loop.

    Parameters
    ----------
    chemsys : str
        Chemical system (e.g. ``"Ca-Ti-O"``).
    element_str : str
        Working ion element (e.g. ``"Ca"``).
    mp_client : MPRester
        Materials Project API client.
    cache : PourbaixCache
        Shared cache for entries and diagrams.
    material_ids : str or list or None
        Restrict query to specific MP IDs.
    calculate_water_stability : bool
        Whether to compute water stability columns.
    decomp_energy_for_metastability : float
        Decomposition energy threshold (eV).
    read_existing_pbx_diag : bool
        If True, try reading cached Pourbaix diagrams from disk first.
    remove_R2SCAN_base : bool
        Skip entries whose base XC functional is R2SCAN.
    remove_uncorrected_duplicates : bool
        If an entry has both corrected and uncorrected versions, keep
        only the R2SCAN-corrected one.

    Returns
    -------
    list of dict
        One dict per material, suitable for building a DataFrame.
    """
    thermo_list = mp_client.materials.thermo.search(
        chemsys=chemsys,
        material_ids=material_ids,
        all_fields=False,
        fields=[
            "material_id", "composition", "chemsys", "formula_pretty",
            "thermo_type", "energy_type", "entries", "energy_per_atom",
            "energy_above_hull", "symmetry",
        ],
    )

    return_list = []

    if not thermo_list:
        data = {col: np.nan for col in COLUMNS}
        return [data]

    for entry in thermo_list:
        data = {}
        data["MP id"] = entry.material_id.string
        data["Formula"] = entry.formula_pretty
        data["Pbx elements per formula"] = len(
            get_pourbaix_composition(entry.composition).as_dict()
        )
        data["Thermo-type"] = entry.thermo_type.value
        data["Base XC Functional"] = entry.energy_type
        data["Correction XC functional"] = entry.energy_type
        data["GGA(+U)/R2SCAN mixing adjustment"] = np.nan

        # Determine base and correction XC functionals
        for correction_dict in find_correction(entry):
            if correction_dict["name"] == "MP GGA(+U)/R2SCAN mixing adjustment":
                desc = correction_dict["description"].strip()

                if desc == "Place GGA(+U) energy onto the R2SCAN hull":
                    data["Base XC Functional"] = "R2SCAN"
                    data["Correction XC functional"] = entry.energy_type
                    data["GGA(+U)/R2SCAN mixing adjustment"] = desc

                elif desc in (
                    "Place R2SCAN energy onto the GGA(+U) hull",
                    "Replace R2SCAN energy with GGA(+U) energy",
                ):
                    gga_types = [
                        t.energy_type for t in thermo_list
                        if "GGA" in t.energy_type
                    ]
                    data["Base XC Functional"] = gga_types[0] if gga_types else entry.energy_type
                    data["Correction XC functional"] = entry.energy_type
                    data["GGA(+U)/R2SCAN mixing adjustment"] = desc

                else:
                    data["Base XC Functional"] = entry.energy_type
                    data["Correction XC functional"] = entry.energy_type
                    data["GGA(+U)/R2SCAN mixing adjustment"] = desc

        # Energy per formula unit
        atoms_per_formula = sum(
            Composition(entry.formula_pretty).as_dict()[k]
            for k in entry.composition.as_dict()
        )
        data["E (eV/formula)"] = entry.energy_per_atom * atoms_per_formula
        data["Energy Above Hull (eV)"] = entry.energy_above_hull
        data["Crystal Structure"] = entry.symmetry.crystal_system.value
        data["Space Group"] = entry.symmetry.symbol

        # Skip R2SCAN base entries
        if remove_R2SCAN_base and data["Base XC Functional"] == "R2SCAN":
            continue

        # Handle duplicate entries (same mp_id, different XC)
        if remove_uncorrected_duplicates:
            req_index = -1
            for idx, existing in enumerate(return_list):
                if existing["MP id"] == entry.material_id.string:
                    req_index = idx
                    break

            if req_index > -1:
                existing = return_list[req_index]
                if existing["Base XC Functional"] == data["Base XC Functional"]:
                    if existing["Correction XC functional"] != data["Correction XC functional"]:
                        if ("GGA" in data["Correction XC functional"]
                                and existing["Correction XC functional"] == "R2SCAN"):
                            continue
                        elif (data["Correction XC functional"] == "R2SCAN"
                              and "GGA" in existing["Correction XC functional"]):
                            # Replace existing with current (R2SCAN-corrected)
                            data["Charged composition"] = existing.get("Charged composition")
                            data["No. of charged structures present"] = existing.get(
                                "No. of charged structures present"
                            )
                            data["No. of charged structure with same crystal structure"] = existing.get(
                                "No. of charged structure with same crystal structure"
                            )
                            if calculate_water_stability:
                                data["Structure stable in water"] = existing.get(
                                    "Structure stable in water"
                                )
                                data["Structure metastable in water"] = existing.get(
                                    "Structure metastable in water"
                                )
                            return_list.append(data)
                            return_list.pop(req_index)
                            continue
                    else:
                        continue

        # Water stability analysis
        if calculate_water_stability:
            if read_existing_pbx_diag:
                pbx_diag = cache.read_or_generate_pourbaix_diagram(
                    entry.composition, mp_client,
                )
            else:
                pbx_diag = cache.get_or_generate_pourbaix_diagram(
                    entry.composition, mp_client,
                    mp_id=entry.material_id.string,
                    save_pbx=True, save_entries=True,
                )

            if entry.energy_above_hull == 0:
                water_stability = is_stable_in_water(
                    entry.formula_pretty, 0, pbx_diag, range_of_pH=(0, 14),
                )
            else:
                water_stability = False

            data["Structure stable in water"] = water_stability

            if water_stability:
                data["Structure metastable in water"] = np.nan
            else:
                data["Structure metastable in water"] = (
                    is_metastable_phase_stable_in_water_search_by_mp_id(
                        entry.material_id.string, pbx_diag,
                        pH_grids=14, voltage_grids=10,
                        range_of_pH=(0, 14),
                        decomp_energy_for_metastability=decomp_energy_for_metastability,
                    )
                )

        # Charged (de-intercalated) structure search
        charged_formula = rm_element(entry.formula_pretty, element_str)
        data["Charged composition"] = charged_formula

        charged_entries = mp_client.materials.search(
            formula=charged_formula, all_fields=False, fields=["material_id"],
        )
        data["No. of charged structures present"] = len(
            {e.material_id for e in charged_entries}
        )

        same_crystal_entries = mp_client.materials.search(
            formula=charged_formula,
            crystal_system=entry.symmetry.crystal_system,
            all_fields=False, fields=["material_id"],
        )
        data["No. of charged structure with same crystal structure"] = len(
            {e.material_id for e in same_crystal_entries}
        )

        return_list.append(data)

    return return_list


def run_dilute_water_screening(
    element,
    transition_metals=None,
    anions=None,
    min_tm=1,
    max_tm=3,
    min_anions=1,
    max_anions=3,
    decomp_energy=0.5,
    read_existing_pbx=True,
    output_path=None,
    cache_dir=".",
    save_interval=1000,
    max_retries=500,
    retry_delay=10.0,
    api_key=None,
):
    """
    Run the full dilute-limit water stability screening for a single
    working ion element.

    Parameters
    ----------
    element : str
        Working ion (``"Ca"``, ``"K"``, ``"Li"``, ``"Mg"``, or ``"Na"``).
    transition_metals : list of str or None
        Transition metals to include (default: 15 common TMs).
    anions : list of str or None
        Anion elements (default: O, S, Se, F, Cl, Br, I, P, N).
    min_tm, max_tm : int
        Min/max number of transition metals per compound.
    min_anions, max_anions : int
        Min/max number of anions per compound.
    decomp_energy : float
        Decomposition energy threshold (eV) for metastability.
    read_existing_pbx : bool
        Try reading cached Pourbaix diagrams from disk.
    output_path : str or None
        Path for the output Excel file. If None, defaults to
        ``"{element}_cathodes_pure_water_metastability.xlsx"``.
    cache_dir : str
        Base directory containing ``saved_pbx/`` and
        ``saved_pbx_entries/`` subdirectories.
    save_interval : int
        Save intermediate results every N chemsys iterations.
    max_retries : int
        Maximum retries per chemsys on API failure.
    retry_delay : float
        Seconds to wait between retries.
    api_key : str or None
        Materials Project API key.

    Returns
    -------
    pd.DataFrame
        Screening results.
    """
    if transition_metals is None:
        transition_metals = DEFAULT_TRANSITION_METALS
    if anions is None:
        anions = DEFAULT_ANIONS
    if output_path is None:
        output_path = f"{element}_cathodes_pure_water_metastability.xlsx"

    mp_client = get_client(api_key)
    cache = PourbaixCache(
        entries_dir=f"{cache_dir}/saved_pbx_entries",
        pbx_dir=f"{cache_dir}/saved_pbx",
    )

    # Generate all chemical system combinations
    chemsys_list = []
    for n_tm in range(min_tm, max_tm + 1):
        for tm_combo in combinations(transition_metals, n_tm):
            for n_an in range(min_anions, max_anions + 1):
                for an_combo in combinations(anions, n_an):
                    chemsys = make_chemsys_from_combinations(
                        tm_combo, an_combo, element,
                    )
                    chemsys_list.append(chemsys)

    logger.info(
        "Starting dilute-water screening for %s: %d chemsys to process",
        element, len(chemsys_list),
    )

    all_rows = []
    start_time = time.time()

    for index, chemsys in enumerate(chemsys_list):
        logger.info("[%d/%d] %s", index + 1, len(chemsys_list), chemsys)

        for attempt in range(max_retries):
            try:
                rows = collect_all_info_from_chemsys(
                    chemsys=chemsys,
                    element_str=element,
                    mp_client=mp_client,
                    cache=cache,
                    decomp_energy_for_metastability=decomp_energy,
                    read_existing_pbx_diag=read_existing_pbx,
                )
                break
            except Exception as e:
                logger.warning(
                    "Attempt %d/%d for %s failed: %s",
                    attempt + 1, max_retries, chemsys, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        else:
            logger.error("Exhausted retries for %s, skipping.", chemsys)
            continue

        all_rows.extend(rows)

        # Periodic save
        if save_interval and (index + 1) % save_interval == 0:
            _save_intermediate(all_rows, output_path)

    # Final save
    df = pd.DataFrame(all_rows)
    df.dropna(axis=0, how="all", inplace=True)
    df.to_excel(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info("Screening completed in %.1f sec. Saved to %s", elapsed, output_path)

    return df


def _save_intermediate(rows, output_path):
    """Save intermediate DataFrame to Excel."""
    df = pd.DataFrame(rows)
    df.dropna(axis=0, how="all", inplace=True)
    df.to_excel(output_path, index=False)
    logger.info("Intermediate save: %d rows -> %s", len(df), output_path)

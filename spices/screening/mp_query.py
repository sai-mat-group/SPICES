"""
Stage 0: Materials Project data query and metadata collection.

This module queries the Materials Project for structural and thermodynamic
data across chemical systems defined by combinations of transition metals,
anions, and a working ion.  It produces a DataFrame with 14 columns that
serves as the mandatory input for all downstream pipeline stages.
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
)
from spices.mp_client import get_client
from spices.pourbaix.composition import (
    rm_element,
    make_chemsys_from_combinations,
    get_pourbaix_composition,
)
from spices.pourbaix.thermo import find_correction

logger = logging.getLogger(__name__)

# DataFrame column names for Stage 0 output
MP_QUERY_COLUMNS = [
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
    "Charged composition",
    "No. of charged structures present",
    "No. of charged structure with same crystal structure",
]


def collect_mp_data_from_chemsys(
    chemsys,
    element_str,
    mp_client,
    material_ids=None,
    remove_R2SCAN_base=True,
    remove_uncorrected_duplicates=True,
):
    """
    Query the Materials Project for all materials in a chemical system
    and extract structural / thermodynamic metadata.

    Parameters
    ----------
    chemsys : str or None
        Chemical system (e.g. ``"Ca-Ti-O"``).  Pass *None* when
        querying by ``material_ids`` instead.
    element_str : str
        Working ion element (e.g. ``"Ca"``).
    mp_client : MPRester
        Materials Project API client.
    material_ids : str or list or None
        Restrict query to specific MP IDs.
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
            "material_id",
            "composition",
            "chemsys",
            "formula_pretty",
            "thermo_type",
            "energy_type",
            "entries",
            "energy_per_atom",
            "energy_above_hull",
            "symmetry",
        ],
    )

    return_list = []

    if not thermo_list:
        data = {col: np.nan for col in MP_QUERY_COLUMNS}
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
        mixing_correction_found = False
        for correction_dict in find_correction(entry):
            if correction_dict["name"].upper() == "MP GGA(+U)/R2SCAN MIXING ADJUSTMENT":
                desc = correction_dict["description"].strip()
                # Normalize R2SCAN casing in stored description
                desc_normalized = desc.replace("r2SCAN", "R2SCAN")
                desc_upper = desc.upper()
                mixing_correction_found = True

                if desc_upper == "PLACE GGA(+U) ENERGY ONTO THE R2SCAN HULL":
                    data["Base XC Functional"] = "R2SCAN"
                    data["Correction XC functional"] = entry.energy_type
                    data["GGA(+U)/R2SCAN mixing adjustment"] = desc_normalized

                elif desc_upper in (
                    "PLACE R2SCAN ENERGY ONTO THE GGA(+U) HULL",
                    "REPLACE R2SCAN ENERGY WITH GGA(+U) ENERGY",
                ):
                    gga_types = [
                        t.energy_type for t in thermo_list if "GGA" in t.energy_type
                    ]
                    data["Base XC Functional"] = (
                        gga_types[0] if gga_types else entry.energy_type
                    )
                    data["Correction XC functional"] = entry.energy_type
                    data["GGA(+U)/R2SCAN mixing adjustment"] = desc_normalized

                else:
                    data["Base XC Functional"] = entry.energy_type
                    data["Correction XC functional"] = entry.energy_type
                    data["GGA(+U)/R2SCAN mixing adjustment"] = desc_normalized

        # Standalone R2SCAN entries (thermo_type=R2SCAN, no mixing corrections)
        if not mixing_correction_found and entry.energy_type.upper() == "R2SCAN":
            data["Base XC Functional"] = "R2SCAN"

        # Energy per formula unit
        atoms_per_formula = sum(
            Composition(entry.formula_pretty).as_dict()[k]
            for k in entry.composition.as_dict()
        )
        data["E (eV/formula)"] = entry.energy_per_atom * atoms_per_formula
        data["Energy Above Hull (eV)"] = entry.energy_above_hull
        data["Crystal Structure"] = entry.symmetry.crystal_system.value
        data["Space Group"] = entry.symmetry.symbol

        # Normalize R2SCAN casing in stored values for downstream consistency
        if data["Base XC Functional"].upper() == "R2SCAN":
            data["Base XC Functional"] = "R2SCAN"
        if data["Correction XC functional"].upper() == "R2SCAN":
            data["Correction XC functional"] = "R2SCAN"

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
                    if (
                        existing["Correction XC functional"]
                        != data["Correction XC functional"]
                    ):
                        if (
                            "GGA" in data["Correction XC functional"]
                            and existing["Correction XC functional"] == "R2SCAN"
                        ):
                            continue
                        elif (
                            data["Correction XC functional"] == "R2SCAN"
                            and "GGA" in existing["Correction XC functional"]
                        ):
                            # Replace existing with current (R2SCAN-corrected)
                            data["Charged composition"] = existing.get(
                                "Charged composition"
                            )
                            data["No. of charged structures present"] = existing.get(
                                "No. of charged structures present"
                            )
                            data[
                                "No. of charged structure with same crystal structure"
                            ] = existing.get(
                                "No. of charged structure with same crystal structure"
                            )
                            return_list.append(data)
                            return_list.pop(req_index)
                            continue
                    else:
                        continue

        # Charged (de-intercalated) structure search
        charged_formula = rm_element(entry.formula_pretty, element_str)
        data["Charged composition"] = charged_formula

        charged_entries = mp_client.materials.search(
            formula=charged_formula,
            all_fields=False,
            fields=["material_id"],
        )
        data["No. of charged structures present"] = len(
            {e.material_id for e in charged_entries}
        )

        same_crystal_entries = mp_client.materials.search(
            formula=charged_formula,
            crystal_system=entry.symmetry.crystal_system,
            all_fields=False,
            fields=["material_id"],
        )
        data["No. of charged structure with same crystal structure"] = len(
            {e.material_id for e in same_crystal_entries}
        )

        return_list.append(data)

    return return_list


def run_mp_query(
    element,
    transition_metals=None,
    anions=None,
    min_tm=1,
    max_tm=3,
    min_anions=1,
    max_anions=3,
    output_path=None,
    save_interval=1000,
    max_retries=500,
    retry_delay=10.0,
    api_key=None,
):
    """
    Run the Materials Project data query for a single working ion element.

    This is Stage 0 of the SPICES pipeline.  It systematically queries
    all chemical systems formed by combinations of transition metals,
    anions, and the working ion, and collects structural and
    thermodynamic metadata.

    Parameters
    ----------
    element : str
        Working ion (``"Ca"``, ``"K"``, ``"Li"``, ``"Mg"``, or ``"Na"``).
    transition_metals : list of str or None
        Transition metals to include (default: 15 common TMs).
    anions : list of str or None
        Anion elements (default: O, S, Se, F, Cl, Br, I, P, N, C, Si).
    min_tm, max_tm : int
        Min/max number of transition metals per compound.
    min_anions, max_anions : int
        Min/max number of anions per compound.
    output_path : str or None
        Path for the output Excel file.  If None, defaults to
        ``"{element}_cathodes_mp_data.xlsx"``.
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
        Query results with 14 columns (see ``MP_QUERY_COLUMNS``).
    """
    if transition_metals is None:
        transition_metals = DEFAULT_TRANSITION_METALS
    if anions is None:
        anions = DEFAULT_ANIONS
    if output_path is None:
        output_path = f"{element}_cathodes_mp_data.xlsx"

    mp_client = get_client(api_key)

    # Generate all chemical system combinations
    chemsys_list = []
    for n_tm in range(min_tm, max_tm + 1):
        for tm_combo in combinations(transition_metals, n_tm):
            for n_an in range(min_anions, max_anions + 1):
                for an_combo in combinations(anions, n_an):
                    chemsys = make_chemsys_from_combinations(
                        tm_combo,
                        an_combo,
                        element,
                    )
                    chemsys_list.append(chemsys)

    logger.info(
        "Starting MP data query for %s: %d chemsys to process",
        element,
        len(chemsys_list),
    )

    all_rows = []
    start_time = time.time()

    for index, chemsys in enumerate(chemsys_list):
        logger.info("[%d/%d] %s", index + 1, len(chemsys_list), chemsys)

        for attempt in range(max_retries):
            try:
                rows = collect_mp_data_from_chemsys(
                    chemsys=chemsys,
                    element_str=element,
                    mp_client=mp_client,
                )
                break
            except Exception as e:
                logger.warning(
                    "Attempt %d/%d for %s failed: %s",
                    attempt + 1,
                    max_retries,
                    chemsys,
                    e,
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
    logger.info("MP query completed in %.1f sec. Saved to %s", elapsed, output_path)

    return df


def _save_intermediate(rows, output_path):
    """Save intermediate DataFrame to Excel."""
    df = pd.DataFrame(rows)
    df.dropna(axis=0, how="all", inplace=True)
    df.to_excel(output_path, index=False)
    logger.info("Intermediate save: %d rows -> %s", len(df), output_path)

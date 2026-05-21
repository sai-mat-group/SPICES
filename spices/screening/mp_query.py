"""
Stage 0: Materials Project data query and metadata collection.

This module queries the Materials Project for structural and thermodynamic
data across chemical systems defined by combinations of transition metals,
anions, and a working ion.  It produces a DataFrame whose columns serve as
the mandatory input for all downstream pipeline stages.

SPICES restricts every thermo query to ``thermo_type=GGA_GGA+U`` so the
entire pipeline operates on a single, self-consistent XC family
(matching the live MP API's hardcoded Pourbaix path).
"""

import logging
import time
from itertools import combinations

import pandas as pd
from pymatgen.core import Composition

from spices.config import (
    DEFAULT_TRANSITION_METALS,
    DEFAULT_ANIONS,
)
from spices.mp_client import get_client
from spices.mp_snapshot import THERMO_TYPE
from spices.pourbaix.composition import (
    rm_element,
    make_chemsys_from_combinations,
    get_pourbaix_composition,
)

logger = logging.getLogger(__name__)

# DataFrame column names for Stage 0 output
MP_QUERY_COLUMNS = [
    "MP id",
    "Formula",
    "Pbx elements per formula",
    "XC Functional",
    "E (eV/formula)",
    "Energy Above Hull (eV)",
    "Crystal Structure",
    "Space Group",
    "Is experimental",
    "Charged composition",
    "No. of charged structures present",
    "No. of charged structure with same crystal structure",
]


def collect_mp_data_from_chemsys(
    chemsys,
    element_str,
    mp_client,
    material_ids=None,
    max_retries=500,
    retry_delay=10.0,
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
    max_retries : int
        Maximum retries for the summary endpoint query.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    list of dict
        One dict per material, suitable for building a DataFrame.
    """
    thermo_list = mp_client.materials.thermo.search(
        chemsys=chemsys,
        material_ids=material_ids,
        thermo_types=[THERMO_TYPE],
        all_fields=False,
        fields=[
            "material_id",
            "composition",
            "chemsys",
            "formula_pretty",
            "energy_type",
            "energy_per_atom",
            "energy_above_hull",
            "symmetry",
        ],
    )

    # Query summary endpoint for experimental status
    mp_ids = [e.material_id.string for e in thermo_list]
    theoretical_map = {}
    if mp_ids:
        for attempt in range(max_retries):
            try:
                summary_list = mp_client.materials.summary.search(
                    material_ids=mp_ids,
                    all_fields=False,
                    fields=["material_id", "theoretical"],
                )
                theoretical_map = {
                    s.material_id.string: s.theoretical for s in summary_list
                }
                break
            except Exception as e:
                logger.warning(
                    "Summary query attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        else:
            logger.warning(
                "Summary query failed after %d retries; "
                "marking all entries as non-experimental",
                max_retries,
            )

    return_list = []

    if not thermo_list:
        logger.warning(
            "thermo.search returned 0 results for "
            "chemsys=%s, material_ids=%s — returning empty list",
            chemsys, material_ids,
        )
        return []

    for entry in thermo_list:
        data = {}
        data["MP id"] = entry.material_id.string
        data["Formula"] = entry.formula_pretty
        data["Pbx elements per formula"] = len(
            get_pourbaix_composition(entry.composition).as_dict()
        )
        data["XC Functional"] = entry.energy_type

        # Energy per formula unit
        atoms_per_formula = sum(
            Composition(entry.formula_pretty).as_dict()[k]
            for k in entry.composition.as_dict()
        )
        data["E (eV/formula)"] = entry.energy_per_atom * atoms_per_formula
        data["Energy Above Hull (eV)"] = entry.energy_above_hull
        data["Crystal Structure"] = entry.symmetry.crystal_system.value
        data["Space Group"] = entry.symmetry.symbol
        data["Is experimental"] = not theoretical_map.get(
            entry.material_id.string, True
        )

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

    # Per-MP-id dedup with warn-on-shrink: under the single-thermo_type
    # restriction we do not expect duplicates here.  If they appear the
    # warning surfaces it without silently masking the upstream surprise.
    deduped = []
    seen_mp_ids = set()
    for row in return_list:
        mp_id = row["MP id"]
        if mp_id in seen_mp_ids:
            continue
        seen_mp_ids.add(mp_id)
        deduped.append(row)
    if len(deduped) != len(return_list):
        logger.warning(
            "step-0 dedup dropped %d duplicate-MP-id row(s) for "
            "chemsys=%s, material_ids=%s — investigate why duplicates "
            "exist post-thermo_type filter",
            len(return_list) - len(deduped), chemsys, material_ids,
        )

    return deduped


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
    mp_client=None,
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
    mp_client : MPRester or SnapshotMPRester or None
        Materials Project client.  If None, resolved via
        :func:`spices.mp_client.get_client` (sticky-default cache,
        ``SPICES_MP_BACKEND``/``MP_API_KEY`` env vars, or live default).

    Returns
    -------
    pd.DataFrame
        Query results with the columns listed in ``MP_QUERY_COLUMNS``.
    """
    if transition_metals is None:
        transition_metals = DEFAULT_TRANSITION_METALS
    if anions is None:
        anions = DEFAULT_ANIONS
    if output_path is None:
        output_path = f"{element}_cathodes_mp_data.xlsx"

    if mp_client is None:
        mp_client = get_client()

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
                    max_retries=max_retries,
                    retry_delay=retry_delay,
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

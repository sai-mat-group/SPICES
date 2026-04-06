"""
DataFrame-level operations for updating stability and metastability
columns on screening results.
"""

import logging

from pymatgen.core import Composition, Element

from spices.pourbaix.stability import (
    is_stable_in_water,
    get_all_stable_solids_mp_id_from_pourbaix_diagram,
)
from spices.pourbaix.metastability import (
    is_metastable_phase_stable_in_water_search_by_mp_id,
    is_metastable_phase_stable_in_water,
    is_metastable_phase_stable_in_seawater_search_by_mp_id,
)

logger = logging.getLogger(__name__)


def update_stability_columns(row, cache, mp_client):
    """
    Apply water stability checks to a DataFrame row.

    Intended for use with ``df.apply(update_stability_columns, axis=1,
    cache=cache, mp_client=mp_client)``.

    Parameters
    ----------
    row : pd.Series
    cache : PourbaixCache
    mp_client : MPRester

    Returns
    -------
    pd.Series
    """
    if row.get("Is most stable"):
        pourbaix_diagram = cache.get_or_generate_pourbaix_diagram(
            Composition(row["Formula"]), mp_client,
        )
        row["Sodiated Structure stable in water"] = is_stable_in_water(
            row["Formula"], 0, pourbaix_diagram,
        )
        row["De-sodiated structure stable in water"] = is_stable_in_water(
            row["De-sodiated composition"], 0, pourbaix_diagram,
        )
    return row


def update_metastability_columns_by_mp_id(row, cache, mp_client):
    """
    Apply metastability checks (by MP ID) to a DataFrame row.

    Parameters
    ----------
    row : pd.Series
    cache : PourbaixCache
    mp_client : MPRester

    Returns
    -------
    pd.Series
    """
    if row.get("Sodiated Structure stable in water") is True:
        return row

    pourbaix_diagram = cache.get_or_generate_pourbaix_diagram(
        Composition(row["Formula"]), mp_client,
    )
    row["Sodiated Structure metastable in water"] = (
        is_metastable_phase_stable_in_water_search_by_mp_id(
            row["Materials Project ID"], pourbaix_diagram,
        )
    )
    return row


def update_metastability_columns_by_formula(row, cache, mp_client):
    """
    Apply metastability checks (by formula) to a DataFrame row.

    Parameters
    ----------
    row : pd.Series
    cache : PourbaixCache
    mp_client : MPRester

    Returns
    -------
    pd.Series
    """
    if not row.get("Sodiated Structure stable in water"):
        pourbaix_diagram = cache.get_or_generate_pourbaix_diagram(
            Composition(row["Formula"]), mp_client,
        )
        row["Sodiated Structure metastable in water"] = (
            is_metastable_phase_stable_in_water(
                row["Formula"], 0, pourbaix_diagram,
            )
        )

    if not row.get("De-sodiated structure stable in water"):
        pourbaix_diagram = cache.get_or_generate_pourbaix_diagram(
            Composition(row["Formula"]), mp_client,
        )
        row["De-sodiated structure metastable in water"] = (
            is_metastable_phase_stable_in_water(
                row["De-sodiated composition"], 0, pourbaix_diagram,
            )
        )

    return row


def update_seawater_metastability_columns_by_mp_id(
    row,
    cache,
    mp_client,
    element_str,
    check_stability=True,
    all_selection_criteria=True,
    OER_shift=0.5,
    HER_shift=-0.3,
    range_of_pH=(7, 8.5),
    range_of_voltage=(-1.5, 2),
    pH_grids=7,
    voltage_grids=41,
    decomp_energy_for_metastability=0.5,
    metastability_points=False,
):
    """
    Apply seawater stability and metastability checks to a DataFrame row.

    Populates up to 9 columns: ``'Seawater stability'`` and
    ``'Seawater metastability (Criterion 1)'`` through
    ``'(Criterion 8)'``.

    Parameters
    ----------
    row : pd.Series
        Must contain ``'MP id'`` and ``'Formula'`` columns.
    cache : PourbaixCache
    mp_client : MPRester
    element_str : str
        Working ion element (e.g. ``"Ca"``).
    check_stability : bool
        Whether to check thermodynamic stability in seawater.
    all_selection_criteria : bool
        If True, evaluate all 8 selection criteria.
    OER_shift : float
    HER_shift : float
    range_of_pH : tuple of float
    range_of_voltage : tuple of float
    pH_grids : int
    voltage_grids : int
    decomp_energy_for_metastability : float
    metastability_points : bool
        If True, return lists of (pH, V) tuples instead of booleans.

    Returns
    -------
    pd.Series
        Updated row with seawater columns populated.
    """
    mp_id = row["MP id"]

    # Get or generate seawater Pourbaix diagram
    sea_pourbaix_diagram = cache.read_or_generate_sea_pourbaix_diagram(
        Composition(row["Formula"]), mp_client,
    )

    bare_seawater_pbx = cache.read_or_generate_bare_seawater_pbx(
        mp_client, seawater_pbx=sea_pourbaix_diagram,
    )

    # Seawater stability check
    seawater_stability = False
    if check_stability:
        stable_solids = get_all_stable_solids_mp_id_from_pourbaix_diagram(
            sea_pourbaix_diagram,
        )
        if mp_id in stable_solids:
            seawater_stability = is_stable_in_water(
                row["Formula"], 0, sea_pourbaix_diagram,
                range_of_pH=range_of_pH,
            )
        row["Seawater stability"] = seawater_stability
        logger.info(
            "%s (%s) seawater stability: %s",
            row["Formula"], mp_id, seawater_stability,
        )

    # Seawater metastability check
    if not seawater_stability or metastability_points:
        logger.info("Checking %s metastability in seawater", row["Formula"])
        working_ion = Element(element_str)

        if all_selection_criteria:
            sc_dict = is_metastable_phase_stable_in_seawater_search_by_mp_id(
                mp_id=mp_id,
                sea_pourbaix_diagram=sea_pourbaix_diagram,
                output_metastability_area=metastability_points,
                bare_seawater_pbx=bare_seawater_pbx,
                working_ion=working_ion,
                pH_grids=pH_grids,
                voltage_grids=voltage_grids,
                OER_shift=OER_shift,
                HER_shift=HER_shift,
                selection_criterion="all",
                decomp_energy_for_metastability=decomp_energy_for_metastability,
            )
            for i in range(1, 7):
                row[f"Seawater metastability (Criterion {i})"] = sc_dict[str(i)]
            # Criteria 7-8 only when overpotential shifts are applied
            if HER_shift != 0 or OER_shift != 0:
                row["Seawater metastability (Criterion 7)"] = sc_dict["7"]
                row["Seawater metastability (Criterion 8)"] = sc_dict["8"]
        else:
            row["Seawater metastability (Criterion 2)"] = (
                is_metastable_phase_stable_in_seawater_search_by_mp_id(
                    mp_id=mp_id,
                    sea_pourbaix_diagram=sea_pourbaix_diagram,
                    output_metastability_area=metastability_points,
                    bare_seawater_pbx=bare_seawater_pbx,
                    working_ion=working_ion,
                    OER_shift=OER_shift,
                    HER_shift=HER_shift,
                    pH_grids=pH_grids,
                    voltage_grids=voltage_grids,
                    selection_criterion=2,
                    decomp_energy_for_metastability=decomp_energy_for_metastability,
                )
            )

    return row

"""
Grid-based metastability searches within the water stability window.
"""

import logging

import numpy as np
from pymatgen.core import Composition

from spices.pourbaix.geometry import (
    check_overlap_with_water_stability_lines,
)
from spices.pourbaix.decomposition import (
    get_decomposition_energy_from_mp_id,
    get_decomposition_energy_from_formula,
    get_seawater_decomposition_energy_from_mp_id,
)
from spices.pourbaix.stability import (
    check_solid_entry,
    get_all_unique_compounds_from_pourbaix_diagram,
)

logger = logging.getLogger(__name__)


def is_metastable_phase_stable_in_water_search_by_mp_id(
    mp_id,
    pourbaix_diagram,
    pH_grids=30,
    voltage_grids=10,
    range_of_pH=(0, 14),
    decomp_energy_for_metastability=0.5,
):
    """
    Check if a material (by MP ID) is metastable within the water
    stability window.

    Sweeps a (pH, voltage) grid, and at each point checks whether
    the decomposition energy is below the threshold and the stable
    decomposition product contains a solid phase.

    Parameters
    ----------
    mp_id : str
        Materials Project ID.
    pourbaix_diagram : PourbaixDiagram
    pH_grids : int
        Number of pH grid points (default 30).
    voltage_grids : int
        Number of voltage grid points (default 10).
    range_of_pH : tuple of float
        (min, max) pH range.
    decomp_energy_for_metastability : float
        Maximum decomposition energy (eV) threshold for metastability.

    Returns
    -------
    bool
        True if the material is metastable in water.
    """
    # Check that the mp_id belongs to a solid in this diagram
    mp_id_list = [
        entry.entry_id
        for entry in pourbaix_diagram.unprocessed_entries
        if entry.phase_type == "Solid"
    ]
    mp_id_list = ["mp-" + a.rsplit("-")[1] for a in mp_id_list]

    if mp_id not in mp_id_list:
        return False

    pH_range = np.linspace(range_of_pH[0], range_of_pH[1], pH_grids)
    voltage_range = np.linspace(-1, 1.5, voltage_grids)

    for pH in pH_range:
        for voltage in voltage_range:
            if check_overlap_with_water_stability_lines(pH, voltage):
                decomp_e = get_decomposition_energy_from_mp_id(
                    mp_id, pH, voltage, pourbaix_diagram,
                )
                if decomp_e < decomp_energy_for_metastability:
                    stable_entries = pourbaix_diagram.get_stable_entry(
                        pH, voltage
                    )
                    if check_solid_entry(stable_entries):
                        return True

    return False


# Backward-compatible alias (original had typo "seach")
is_metastable_phase_stable_in_water_seach_by_mp_id = (
    is_metastable_phase_stable_in_water_search_by_mp_id
)


def is_metastable_phase_stable_in_water(
    formula_string,
    charge,
    pourbaix_diagram,
    pH_grids=30,
    voltage_grids=10,
    range_of_pH=(0, 14),
    decomp_energy_for_metastability=0.5,
):
    """
    Check if a material (by formula and charge) is metastable within
    the water stability window.

    Parameters
    ----------
    formula_string : str
        Chemical formula.
    charge : int or float
        Charge state.
    pourbaix_diagram : PourbaixDiagram
    pH_grids : int
    voltage_grids : int
    range_of_pH : tuple of float
    decomp_energy_for_metastability : float

    Returns
    -------
    bool
    """
    all_compounds = get_all_unique_compounds_from_pourbaix_diagram(
        pourbaix_diagram
    )
    compound_string = (
        f"{Composition(formula_string).reduced_formula} "
        f"charge: {float(charge)}"
    )

    if compound_string not in all_compounds:
        return False

    pH_range = np.linspace(range_of_pH[0], range_of_pH[1], pH_grids)
    voltage_range = np.linspace(-1, 1.5, voltage_grids)

    for pH in pH_range:
        for voltage in voltage_range:
            decomp_e = get_decomposition_energy_from_formula(
                formula_string, pH, voltage, pourbaix_diagram,
            )
            if decomp_e < decomp_energy_for_metastability:
                if check_overlap_with_water_stability_lines(pH, voltage):
                    stable_entries = pourbaix_diagram.get_stable_entry(
                        pH, voltage
                    )
                    if check_solid_entry(stable_entries):
                        return True

    return False


# ====================================================================
# Seawater metastability
# ====================================================================


def _is_metastable_phase_stable_in_seawater_search_by_mp_id(
    mp_id,
    sea_pourbaix_diagram,
    bare_seawater_pbx,
    working_ion=None,
    pH_grids=16,
    voltage_grids=15,
    range_of_pH=(7, 8.5),
    range_of_voltage=(-1.5, 2),
    OER_shift=0.5,
    HER_shift=-0.3,
    selection_criterion=4,
    decomp_energy_for_metastability=0.5,
):
    """
    Internal helper: check seawater metastability for a single criterion.

    Parameters
    ----------
    mp_id : str
        Materials Project ID.
    sea_pourbaix_diagram : PourbaixDiagram
        Seawater Pourbaix diagram (cathode + seawater elements).
    bare_seawater_pbx : PourbaixDiagram
        Bare seawater Pourbaix diagram (seawater elements only).
    working_ion : Element or None
        The intercalating ion (e.g. ``Element("Ca")``).
    pH_grids : int
    voltage_grids : int
    range_of_pH : tuple of float
    range_of_voltage : tuple of float
    OER_shift : float
    HER_shift : float
    selection_criterion : int
        1-8 for a single criterion (default 4).
    decomp_energy_for_metastability : float

    Returns
    -------
    bool
    """
    # Check mp_id exists in diagram
    mp_id_list = [
        "mp-" + entry.entry_id.rsplit("-")[1]
        for entry in sea_pourbaix_diagram.unprocessed_entries
        if entry.phase_type == "Solid"
    ]

    if mp_id not in mp_id_list:
        return False

    pH_range = np.linspace(range_of_pH[0], range_of_pH[1], pH_grids)
    voltage_range = np.linspace(
        range_of_voltage[0], range_of_voltage[1], voltage_grids,
    )

    for pH in pH_range:
        for voltage in voltage_range:
            if not check_overlap_with_water_stability_lines(
                pH, voltage, OER_shift=OER_shift, HER_shift=HER_shift,
            ):
                continue

            logger.debug("pH = %.2f, V = %.2f", pH, voltage)
            stable_entry = sea_pourbaix_diagram.get_stable_entry(pH, voltage)
            reactant_entry = bare_seawater_pbx.get_stable_entry(pH, voltage)

            if check_solid_entry(
                stable_entry,
                reactant_entry=reactant_entry,
                mp_id=mp_id,
                pbx_diag=sea_pourbaix_diagram,
                working_ion=working_ion,
                selection_criterion=selection_criterion,
            ):
                decomposition_energy = (
                    get_seawater_decomposition_energy_from_mp_id(
                        mp_id=mp_id, pH=pH, voltage=voltage,
                        sea_pourbaix_diagram=sea_pourbaix_diagram,
                        bare_seawater_pbx=bare_seawater_pbx,
                        decomp_energy_for_metastability=decomp_energy_for_metastability,
                    )
                )
                if decomposition_energy < decomp_energy_for_metastability:
                    return True

    return False


def is_metastable_phase_stable_in_seawater_search_by_mp_id(
    mp_id,
    sea_pourbaix_diagram,
    output_metastability_area=False,
    bare_seawater_pbx=None,
    working_ion=None,
    pH_grids=7,
    voltage_grids=43,
    range_of_pH=(7, 8.5),
    range_of_voltage=(-1.5, 2),
    OER_shift=0.5,
    HER_shift=-0.3,
    selection_criterion=4,
    decomp_energy_for_metastability=0.5,
):
    """
    Check if a material is metastable in seawater, optionally returning
    the (pH, voltage) points where metastability holds.

    Parameters
    ----------
    mp_id : str
        Materials Project ID.
    sea_pourbaix_diagram : PourbaixDiagram
        Seawater Pourbaix diagram (cathode + seawater elements).
    output_metastability_area : bool
        If True, return a list of (pH, V) tuples instead of a boolean.
    bare_seawater_pbx : PourbaixDiagram or None
        Bare seawater diagram.  Required — raises ``ValueError`` if
        ``None``.
    working_ion : Element or None
        The intercalating ion.
    pH_grids : int
    voltage_grids : int
    range_of_pH : tuple of float
    range_of_voltage : tuple of float
    OER_shift : float
    HER_shift : float
    selection_criterion : int
        1-8 for a single criterion (default 4).
    decomp_energy_for_metastability : float

    Returns
    -------
    bool or list
        - ``output_metastability_area=False``: ``bool``.
        - ``output_metastability_area=True``: ``list`` of (pH, V) tuples.

    Raises
    ------
    ValueError
        If *bare_seawater_pbx* is None.
    """
    if bare_seawater_pbx is None:
        raise ValueError(
            "bare_seawater_pbx is required for seawater metastability checks"
        )

    # Boolean-only mode: delegate to private helper
    if not output_metastability_area:
        return _is_metastable_phase_stable_in_seawater_search_by_mp_id(
            mp_id=mp_id,
            sea_pourbaix_diagram=sea_pourbaix_diagram,
            bare_seawater_pbx=bare_seawater_pbx,
            working_ion=working_ion,
            pH_grids=pH_grids,
            voltage_grids=voltage_grids,
            range_of_pH=range_of_pH,
            range_of_voltage=range_of_voltage,
            OER_shift=OER_shift,
            HER_shift=HER_shift,
            selection_criterion=selection_criterion,
            decomp_energy_for_metastability=decomp_energy_for_metastability,
        )

    # Area output mode: collect (pH, V) points
    mp_id_list = [
        "mp-" + entry.entry_id.rsplit("-")[1]
        for entry in sea_pourbaix_diagram.unprocessed_entries
        if entry.phase_type == "Solid"
    ]

    metastability_points = []

    if mp_id not in mp_id_list:
        return metastability_points

    pH_range = np.linspace(range_of_pH[0], range_of_pH[1], pH_grids)
    voltage_range = np.linspace(
        range_of_voltage[0], range_of_voltage[1], voltage_grids,
    )

    for pH in pH_range:
        for voltage in voltage_range:
            if not check_overlap_with_water_stability_lines(
                pH, voltage, OER_shift=OER_shift, HER_shift=HER_shift,
            ):
                continue

            stable_entry = sea_pourbaix_diagram.get_stable_entry(pH, voltage)
            reactant_entry = bare_seawater_pbx.get_stable_entry(pH, voltage)

            if check_solid_entry(
                stable_entry,
                reactant_entry=reactant_entry,
                mp_id=mp_id,
                pbx_diag=sea_pourbaix_diagram,
                working_ion=working_ion,
                selection_criterion=selection_criterion,
            ):
                decomposition_energy = (
                    get_seawater_decomposition_energy_from_mp_id(
                        mp_id=mp_id, pH=pH, voltage=voltage,
                        sea_pourbaix_diagram=sea_pourbaix_diagram,
                        bare_seawater_pbx=bare_seawater_pbx,
                        decomp_energy_for_metastability=decomp_energy_for_metastability,
                    )
                )
                if decomposition_energy < decomp_energy_for_metastability:
                    metastability_points.append((float(pH), float(voltage)))

    return metastability_points


# Backward-compatible alias (original had typo "seach")
is_metastable_phase_stable_in_seawater_seach_by_mp_id = (
    is_metastable_phase_stable_in_seawater_search_by_mp_id
)

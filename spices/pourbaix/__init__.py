"""
Core Pourbaix diagram analysis functions.

This subpackage provides tools for constructing, caching, and analysing
Pourbaix diagrams to determine aqueous stability and metastability of
electrode materials.
"""

from spices.pourbaix.composition import (
    rm_element,
    make_chemsys_from_combinations,
    get_pourbaix_composition,
    get_sea_pourbaix_comp_dict,
    get_sea_pourbaix_conc_dict,
)
from spices.pourbaix.stability import (
    is_stable_in_water,
    get_all_stable_compounds_from_pourbaix_diagram,
    get_unique_stable_compounds_from_pourbaix_diagram,
    get_all_unique_compounds_from_pourbaix_diagram,
    get_all_stable_solids_mp_id_from_pourbaix_diagram,
)
from spices.pourbaix.geometry import (
    check_overlap,
    check_overlap_with_water_stability_lines,
    check_overlap_with_extended_water_stability_lines,
    get_alpha_shape,
    find_ph_range_at_voltage,
    find_stability_area_around_voltage,
)
from spices.pourbaix.decomposition import (
    get_decomposition_energy_from_mp_id,
    get_decomposition_energy_from_formula,
    get_mp_id_from_pbx_entry_id,
    get_seawater_decomposition_energy_from_mp_id,
)
from spices.pourbaix.metastability import (
    is_metastable_phase_stable_in_water_search_by_mp_id,
    is_metastable_phase_stable_in_water,
    is_metastable_phase_stable_in_seawater_search_by_mp_id,
)
from spices.pourbaix.thermo import find_correction, is_most_stable
from spices.pourbaix.plotting import CustomPourbaixPlotter

__all__ = [
    "rm_element",
    "make_chemsys_from_combinations",
    "get_pourbaix_composition",
    "get_sea_pourbaix_comp_dict",
    "get_sea_pourbaix_conc_dict",
    "is_stable_in_water",
    "check_overlap",
    "check_overlap_with_water_stability_lines",
    "check_overlap_with_extended_water_stability_lines",
    "get_alpha_shape",
    "find_ph_range_at_voltage",
    "find_stability_area_around_voltage",
    "get_decomposition_energy_from_mp_id",
    "get_decomposition_energy_from_formula",
    "get_mp_id_from_pbx_entry_id",
    "is_metastable_phase_stable_in_water_search_by_mp_id",
    "is_metastable_phase_stable_in_water",
    "get_seawater_decomposition_energy_from_mp_id",
    "is_metastable_phase_stable_in_seawater_search_by_mp_id",
    "find_correction",
    "is_most_stable",
    "get_all_stable_compounds_from_pourbaix_diagram",
    "get_unique_stable_compounds_from_pourbaix_diagram",
    "get_all_unique_compounds_from_pourbaix_diagram",
    "get_all_stable_solids_mp_id_from_pourbaix_diagram",
    "CustomPourbaixPlotter",
]

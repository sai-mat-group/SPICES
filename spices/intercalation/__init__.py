"""
Intercalation voltage analysis for cathode material screening.

This subpackage provides tools for computing intercalation voltages,
constructing pseudo-binary convex hulls, matching charged structures,
and determining pH stability ranges at computed voltages.
"""

from spices.intercalation.voltage import (
    calculate_intercalation_voltage,
    get_energy_of_working_ion,
)
from spices.intercalation.hull import (
    filter_df_to_get_on_hull_phases,
    get_working_ions_per_framework,
    create_pd_entry,
)
from spices.intercalation.structure import (
    generate_charged_struct_from_MP,
    read_or_generate_charged_struct,
    find_structures,
    create_subgroups,
    subgroups_to_excel_sheets,
    add_charged_struct_info,
)

__all__ = [
    "calculate_intercalation_voltage",
    "get_energy_of_working_ion",
    "filter_df_to_get_on_hull_phases",
    "get_working_ions_per_framework",
    "create_pd_entry",
    "generate_charged_struct_from_MP",
    "read_or_generate_charged_struct",
    "find_structures",
    "create_subgroups",
    "subgroups_to_excel_sheets",
    "add_charged_struct_info",
]

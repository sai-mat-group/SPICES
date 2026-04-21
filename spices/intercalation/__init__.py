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
)

# NOTE: ``spices.intercalation.grouping`` depends on
# ``spices.screening.framework_comparison`` which in turn imports
# ``DEFAULT_MATCHER`` from this subpackage.  We therefore do NOT
# re-export grouping symbols here — import them directly from
# ``spices.intercalation.grouping`` to avoid a circular import.

__all__ = [
    "calculate_intercalation_voltage",
    "get_energy_of_working_ion",
    "filter_df_to_get_on_hull_phases",
    "get_working_ions_per_framework",
    "create_pd_entry",
    "generate_charged_struct_from_MP",
    "read_or_generate_charged_struct",
    "find_structures",
]

"""
SPICES Quick Start Example
==========================

This script demonstrates the basic usage of the SPICES package for
screening cathode materials for desalination applications.

Before running, set your Materials Project API key:
    export MP_API_KEY="your_key_here"
or create a .env file in the project root.
"""

from spices.mp_client import get_client
from spices.pourbaix.cache import PourbaixCache
from spices.pourbaix.composition import rm_element, get_pourbaix_composition
from spices.pourbaix.stability import is_stable_in_water
from spices.pourbaix.metastability import is_metastable_phase_stable_in_water_search_by_mp_id
from spices.pourbaix.geometry import check_overlap_with_water_stability_lines
from spices.screening.dilute_water import collect_all_info_from_chemsys
from spices.io import setup_logging

# ---- Setup ----
setup_logging()

# ---- Example 1: Composition utilities (no API needed) ----
print("=== Composition Utilities ===")
print("De-intercalated LiFePO4:", rm_element("LiFePO4", "Li"))
print("De-intercalated CaTiO3:", rm_element("CaTiO3", "Ca"))

# ---- Example 2: Water stability window check (no API needed) ----
print("\n=== Water Stability Window ===")
print(f"pH=7, V=0.5 in water window: {check_overlap_with_water_stability_lines(7, 0.5)}")
print(f"pH=7, V=2.0 in water window: {check_overlap_with_water_stability_lines(7, 2.0)}")

# ---- Example 3: Single chemsys query (requires API key) ----
# Uncomment the lines below to run with a live API connection:
#
# mp_client = get_client()
# cache = PourbaixCache(entries_dir="saved_pbx_entries", pbx_dir="saved_pbx")
#
# results = collect_all_info_from_chemsys(
#     chemsys="Ca-Ti-O",
#     element_str="Ca",
#     mp_client=mp_client,
#     cache=cache,
#     decomp_energy_for_metastability=0.5,
#     read_existing_pbx_diag=True,
# )
#
# import pandas as pd
# df = pd.DataFrame(results)
# print(df[["MP id", "Formula", "Structure stable in water", "Structure metastable in water"]])

# ---- Example 4: Seawater screening (requires Stage 1 output + API key) ----
# Uncomment the lines below to run with Stage 1 output:
#
# from spices.screening.seawater import run_seawater_screening
#
# df_sea = run_seawater_screening(
#     element="Ca",
#     input_path="Ca_cathodes_pure_water_metastability.xlsx",
#     output_path="Ca_seawater_results.xlsx",
#     decomp_energy=0.3,
# )
# print(df_sea[["MP id", "Formula", "Seawater stability",
#               "Seawater metastability (Criterion 2)"]].head())

print("\nQuickstart complete!")

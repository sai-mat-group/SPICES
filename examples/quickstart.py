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
from spices.screening.mp_query import collect_mp_data_from_chemsys
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

# ---- Example 3: Stage 0 — MP data query (requires API key) ----
# Uncomment the lines below to run with a live API connection:
#
# mp_client = get_client()
#
# # Query a single chemical system (Stage 0)
# results = collect_mp_data_from_chemsys(
#     chemsys="Ca-Ti-O",
#     element_str="Ca",
#     mp_client=mp_client,
# )
#
# import pandas as pd
# df = pd.DataFrame(results)
# print(df[["MP id", "Formula", "Energy Above Hull (eV)", "Crystal Structure"]])
#
# # Or run the full Stage 0 query for an element:
# from spices.screening.mp_query import run_mp_query
# df_step0 = run_mp_query(element="Ca", output_path="Ca_cathodes_mp_data.xlsx")

# ---- Example 3b: Stage 1 — Dilute water evaluation (requires Stage 0 output + API key) ----
# Uncomment the lines below to run with Stage 0 output:
#
# from spices.screening.dilute_water import run_dilute_water_evaluation
#
# df_step1 = run_dilute_water_evaluation(
#     element="Ca",
#     input_data="Ca_cathodes_mp_data.xlsx",
#     output_path="Ca_cathodes_pure_water_metastability.xlsx",
#     decomp_energy=0.5,
# )
# print(df_step1[["MP id", "Formula", "Structure stable in water",
#                  "Structure metastable in water"]])

# ---- Example 4: Stage 2 — Seawater screening (requires Stage 0 or Stage 1 output + API key) ----
# Stage 2 can run on Stage 0 output directly (skipping Stage 1):
#
# from spices.screening.seawater import run_seawater_screening
#
# df_sea = run_seawater_screening(
#     element="Ca",
#     input_path="Ca_cathodes_mp_data.xlsx",  # Stage 0 output
#     output_path="Ca_seawater_results.xlsx",
#     decomp_energy=0.3,
# )
# print(df_sea[["MP id", "Formula", "Seawater stability",
#               "Seawater metastability (Criterion 2)"]].head())

print("\nQuickstart complete!")

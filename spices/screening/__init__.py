"""
Screening pipeline modules.

Each module corresponds to one stage of the SPICES pipeline:

0. ``mp_query`` -- Materials Project data query and metadata collection
1. ``dilute_water`` -- Dilute-limit (pure water) stability evaluation
2. ``seawater`` -- Seawater stability screening
3. ``intercalation`` -- Intercalation voltage filtering
4. ``framework_comparison`` -- Cross-element framework comparison
5. ``statistics`` -- Aggregate statistics and visualization
"""

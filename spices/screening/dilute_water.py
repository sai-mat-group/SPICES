"""
Stage 1: Dilute-limit (pure water) stability evaluation.

This module takes Stage 0 output (MP query results) and evaluates each
material's stability and metastability in pure water using Pourbaix
diagram analysis.  Stage 1 is optional — Stage 2 (seawater) can run
directly on Stage 0 output.
"""

import logging
import time

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from spices.config import DEFAULT_DECOMP_ENERGY_THRESHOLD
from spices.mp_client import get_client
from spices.pourbaix.cache import PourbaixCache
from spices.pourbaix.stability import is_stable_in_water
from spices.pourbaix.metastability import (
    is_metastable_phase_stable_in_water_search_by_mp_id,
)
from spices.screening.mp_query import MP_QUERY_COLUMNS

logger = logging.getLogger(__name__)

# DataFrame column names for Stage 1 output (Stage 0 columns + water stability)
COLUMNS = MP_QUERY_COLUMNS + [
    "Structure stable in water",
    "Structure metastable in water",
]


def run_dilute_water_evaluation(
    element,
    input_data,
    output_path=None,
    decomp_energy=0.5,
    read_existing_pbx=True,
    cache_dir=".",
    save_interval=1000,
    max_retries=500,
    retry_delay=10.0,
    mp_client=None,
):
    """
    Evaluate dilute-water stability for materials in a Stage 0 dataset.

    Parameters
    ----------
    element : str
        Working ion (``"Ca"``, ``"K"``, ``"Li"``, ``"Mg"``, or ``"Na"``).
        Used for output file naming.
    input_data : pd.DataFrame or str
        Stage 0 output — either a DataFrame or a path to an Excel file.
    output_path : str or None
        Path for the output Excel file.  If None, defaults to
        ``"{element}_cathodes_pure_water_metastability.xlsx"``.
    decomp_energy : float
        Decomposition energy threshold (eV) for metastability.
    read_existing_pbx : bool
        Try reading cached Pourbaix diagrams from disk.
    cache_dir : str
        Base directory containing ``saved_pbx/`` and
        ``saved_pbx_entries/`` subdirectories.
    save_interval : int
        Save intermediate results every N rows.
    max_retries : int
        Maximum retries per row on API failure.
    retry_delay : float
        Seconds to wait between retries.
    mp_client : MPRester or SnapshotMPRester or None
        Materials Project client.  If None, resolved via
        :func:`spices.mp_client.get_client` (sticky-default cache,
        ``SPICES_MP_BACKEND``/``MP_API_KEY`` env vars, or live default).

    Returns
    -------
    pd.DataFrame
        Stage 1 results with 16 columns (Stage 0 + water stability).
    """
    if output_path is None:
        output_path = f"{element}_cathodes_pure_water_metastability.xlsx"

    if isinstance(input_data, str):
        df = pd.read_excel(input_data)
    else:
        df = input_data.copy()

    if mp_client is None:
        mp_client = get_client()
    cache = PourbaixCache(
        entries_dir=f"{cache_dir}/saved_pbx_entries",
        pbx_dir=f"{cache_dir}/saved_pbx",
    )

    # Initialize water stability columns
    df["Structure stable in water"] = np.nan
    df["Structure metastable in water"] = np.nan

    logger.info(
        "Starting dilute-water evaluation for %s: %d materials to process",
        element, len(df),
    )

    start_time = time.time()

    for index, row in df.iterrows():
        mp_id = row["MP id"]
        formula = row["Formula"]

        # Skip rows with no MP id (NaN placeholder rows)
        if pd.isna(mp_id):
            continue

        logger.info("[%d/%d] %s (%s)", index + 1, len(df), mp_id, formula)

        for attempt in range(max_retries):
            try:
                composition = Composition(formula)

                # Generate or read Pourbaix diagram
                if read_existing_pbx:
                    pbx_diag = cache.read_or_generate_pourbaix_diagram(
                        composition, mp_client,
                    )
                else:
                    pbx_diag = cache.get_or_generate_pourbaix_diagram(
                        composition, mp_client,
                        mp_id=mp_id,
                        save_pbx=True, save_entries=True,
                    )

                # On-hull materials: check direct stability
                energy_above_hull = row["Energy Above Hull (eV)"]
                if energy_above_hull == 0:
                    water_stability = is_stable_in_water(
                        formula, 0, pbx_diag, range_of_pH=(0, 14),
                    )
                else:
                    water_stability = False

                df.at[index, "Structure stable in water"] = water_stability

                if water_stability:
                    df.at[index, "Structure metastable in water"] = np.nan
                else:
                    df.at[index, "Structure metastable in water"] = (
                        is_metastable_phase_stable_in_water_search_by_mp_id(
                            mp_id, pbx_diag,
                            pH_grids=14, voltage_grids=10,
                            range_of_pH=(0, 14),
                            decomp_energy_for_metastability=decomp_energy,
                        )
                    )

                break
            except Exception as e:
                logger.warning(
                    "Attempt %d/%d for %s failed: %s",
                    attempt + 1, max_retries, mp_id, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        else:
            logger.error("Exhausted retries for %s, skipping.", mp_id)

        # Periodic save
        if save_interval and (index + 1) % save_interval == 0:
            df.to_excel(output_path, index=False)
            logger.info("Intermediate save: %d rows -> %s", index + 1, output_path)

    # Final save
    df.to_excel(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info(
        "Dilute-water evaluation completed in %.1f sec. Saved to %s",
        elapsed, output_path,
    )

    return df

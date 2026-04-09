"""
Stage 2: Seawater stability and metastability screening.

This module takes Stage 0 or Stage 1 output as input and applies
seawater Pourbaix analysis to evaluate material stability under
realistic ocean conditions (pH 7-8.5, real ionic concentrations).
Only the "MP id" and "Formula" columns are required from the input;
all other columns are passed through unchanged.

Replaces the per-element ``add_seawater_metastability_multi-core.ipynb``
notebooks with a single parameterized implementation.

Supports optional multiprocessing via ``n_workers`` parameter.
"""

import logging
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from spices.config import (
    SEAWATER_PH_RANGE,
    SEAWATER_VOLTAGE_RANGE,
    SEAWATER_DECOMP_ENERGY_THRESHOLD,
    DEFAULT_OER_SHIFT,
    DEFAULT_HER_SHIFT,
    SEAWATER_PH_GRIDS,
    SEAWATER_VOLTAGE_GRIDS,
)
from spices.mp_client import get_client
from spices.pourbaix.cache import PourbaixCache
from spices.dataframe_ops import update_seawater_metastability_columns_by_mp_id

logger = logging.getLogger(__name__)

# DataFrame column names added by Stage 2
SEAWATER_COLUMNS = [
    "Seawater stability",
    "Seawater metastability (Criterion 1)",
    "Seawater metastability (Criterion 2)",
    "Seawater metastability (Criterion 3)",
    "Seawater metastability (Criterion 4)",
    "Seawater metastability (Criterion 5)",
    "Seawater metastability (Criterion 6)",
    "Seawater metastability (Criterion 7)",
    "Seawater metastability (Criterion 8)",
]


def run_seawater_screening(
    element,
    input_path,
    output_path=None,
    decomp_energy=None,
    pH_range=None,
    voltage_range=None,
    pH_grids=None,
    voltage_grids=None,
    OER_shift=None,
    HER_shift=None,
    metastability_points=False,
    cache_dir=".",
    save_interval=100,
    max_retries=500,
    retry_delay=10.0,
    n_workers=1,
    api_key=None,
):
    """
    Run the seawater stability screening for a single working ion.

    Reads the Stage 1 (dilute-water) output Excel file, applies
    seawater Pourbaix analysis to each material, and saves results
    with 9 additional columns.

    Parameters
    ----------
    element : str
        Working ion (``"Ca"``, ``"K"``, ``"Li"``, ``"Mg"``, or ``"Na"``).
    input_path : str
        Path to Stage 1 output Excel file.
    output_path : str or None
        Path for the output Excel file.  If None, defaults to
        ``"{element}_cathodes_seawater_metastability.xlsx"``.
    decomp_energy : float or None
        Decomposition energy threshold (eV).  Default uses
        ``SEAWATER_DECOMP_ENERGY_THRESHOLD`` (0.3).
    pH_range : tuple of float or None
        (min, max) pH range.  Default ``(7, 8.5)``.
    voltage_range : tuple of float or None
        (min, max) voltage range.  Default ``(-1.5, 2)``.
    pH_grids : int or None
        Number of pH grid points.  Default 7.
    voltage_grids : int or None
        Number of voltage grid points.  Default 41.
    OER_shift : float or None
        OER overpotential shift.  Default 0.5 V.
    HER_shift : float or None
        HER overpotential shift.  Default -0.3 V.
    metastability_points : bool
        If True, store (pH, V) point lists instead of booleans.
    cache_dir : str
        Base directory containing ``saved_pbx/`` and
        ``saved_pbx_entries/`` subdirectories.
    save_interval : int
        Save intermediate results every N rows.
    max_retries : int
        Maximum retries per row on API failure.
    retry_delay : float
        Seconds to wait between retries.
    n_workers : int
        Number of parallel workers.  ``1`` (default) runs
        sequentially.  Values > 1 use ``multiprocessing.Pool``
        with round-robin row distribution.
    api_key : str or None
        Materials Project API key.

    Returns
    -------
    pd.DataFrame
        Screening results with seawater columns added.
    """
    if decomp_energy is None:
        decomp_energy = SEAWATER_DECOMP_ENERGY_THRESHOLD
    if pH_range is None:
        pH_range = SEAWATER_PH_RANGE
    if voltage_range is None:
        voltage_range = SEAWATER_VOLTAGE_RANGE
    if pH_grids is None:
        pH_grids = SEAWATER_PH_GRIDS
    if voltage_grids is None:
        voltage_grids = SEAWATER_VOLTAGE_GRIDS
    if OER_shift is None:
        OER_shift = DEFAULT_OER_SHIFT
    if HER_shift is None:
        HER_shift = DEFAULT_HER_SHIFT
    if output_path is None:
        output_path = f"{element}_cathodes_seawater_metastability.xlsx"

    # Read Stage 1 output
    df = pd.read_excel(input_path)
    df = df.dropna(subset=["MP id"]).reset_index(drop=True)
    logger.info(
        "Read %d materials from %s", len(df), input_path,
    )

    # Remove pre-existing seawater columns (for re-runs)
    for col in SEAWATER_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Initialize seawater columns
    for col in SEAWATER_COLUMNS:
        df[col] = None

    start_time = time.time()
    output_dir = os.path.dirname(output_path) or "."

    # Common keyword args for the worker function
    worker_kwargs = dict(
        element=element,
        cache_dir=cache_dir,
        api_key=api_key,
        output_dir=output_dir,
        OER_shift=OER_shift,
        HER_shift=HER_shift,
        pH_range=pH_range,
        voltage_range=voltage_range,
        pH_grids=pH_grids,
        voltage_grids=voltage_grids,
        decomp_energy=decomp_energy,
        metastability_points=metastability_points,
        save_interval=save_interval,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    if n_workers <= 1:
        # Sequential mode
        df = _process_chunk(df, 0, **worker_kwargs)
    else:
        # Parallel mode
        chunks = _split_dataframe(df, n_workers)
        logger.info(
            "Splitting %d rows across %d workers", len(df), len(chunks),
        )

        pool_args = [
            (chunk, chunk_id) for chunk_id, chunk in enumerate(chunks)
        ]

        with Pool(n_workers) as pool:
            processed_chunks = pool.starmap(
                _process_chunk_wrapper,
                [(chunk, cid, worker_kwargs) for chunk, cid in pool_args],
            )

        df = pd.concat(processed_chunks, ignore_index=True)

    # Final save
    df.to_excel(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info(
        "Seawater screening completed in %.1f sec. Saved to %s",
        elapsed, output_path,
    )

    return df


def _split_dataframe(df, n_workers):
    """
    Split a DataFrame into chunks using round-robin row distribution.

    Worker 0 gets rows [0, n, 2n, ...], worker 1 gets [1, n+1, 2n+1, ...],
    etc.  This provides better load balancing than contiguous chunks.

    Parameters
    ----------
    df : pd.DataFrame
    n_workers : int

    Returns
    -------
    list of pd.DataFrame
        One chunk per worker.
    """
    indices_per_worker = [
        [i + j for j in range(0, len(df), n_workers) if i + j < len(df)]
        for i in range(n_workers)
    ]
    chunks = [
        df.iloc[indices].reset_index(drop=True)
        for indices in indices_per_worker
        if indices  # skip empty chunks when n_workers > len(df)
    ]
    return chunks


def _process_chunk_wrapper(chunk_df, chunk_id, kwargs):
    """Wrapper for Pool.starmap that unpacks kwargs."""
    return _process_chunk(chunk_df, chunk_id, **kwargs)


def _process_chunk(
    chunk_df,
    chunk_id,
    element,
    cache_dir,
    api_key,
    output_dir,
    OER_shift,
    HER_shift,
    pH_range,
    voltage_range,
    pH_grids,
    voltage_grids,
    decomp_energy,
    metastability_points,
    save_interval,
    max_retries,
    retry_delay,
):
    """
    Process a DataFrame chunk for seawater screening.

    Each worker creates its own ``PourbaixCache`` and ``MPRester``
    to avoid shared state across processes.  Progress is logged to
    a per-worker log file and checkpointed periodically.

    Parameters
    ----------
    chunk_df : pd.DataFrame
    chunk_id : int
        Worker index (used in filenames and log messages).
    element : str
    cache_dir : str
    api_key : str or None
    output_dir : str
    OER_shift, HER_shift : float
    pH_range, voltage_range : tuple of float
    pH_grids, voltage_grids : int
    decomp_energy : float
    metastability_points : bool
    save_interval : int
    max_retries : int
    retry_delay : float

    Returns
    -------
    pd.DataFrame
        Processed chunk with seawater columns populated.
    """
    # Set up per-worker logging
    log_path = os.path.join(output_dir, f"process_log_core{chunk_id}.txt")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    worker_logger = logging.getLogger(f"spices.worker.{chunk_id}")
    worker_logger.addHandler(file_handler)
    worker_logger.setLevel(logging.INFO)

    # Each worker gets its own client and cache
    mp_client = get_client(api_key)
    cache = PourbaixCache(
        entries_dir=f"{cache_dir}/saved_pbx_entries",
        pbx_dir=f"{cache_dir}/saved_pbx",
    )

    runtime_dir = os.path.join(output_dir, "runtime_sheets")
    os.makedirs(runtime_dir, exist_ok=True)

    for index in range(len(chunk_df)):
        row = chunk_df.iloc[index]
        worker_logger.info(
            "[core %d] [%d/%d] %s (%s)",
            chunk_id, index + 1, len(chunk_df),
            row.get("Formula", "?"), row.get("MP id", "?"),
        )

        for attempt in range(max_retries):
            try:
                updated_row = update_seawater_metastability_columns_by_mp_id(
                    row.copy(),
                    cache=cache,
                    mp_client=mp_client,
                    element_str=element,
                    OER_shift=OER_shift,
                    HER_shift=HER_shift,
                    range_of_pH=pH_range,
                    range_of_voltage=voltage_range,
                    pH_grids=pH_grids,
                    voltage_grids=voltage_grids,
                    decomp_energy_for_metastability=decomp_energy,
                    metastability_points=metastability_points,
                )
                chunk_df.iloc[index] = updated_row
                break
            except Exception as e:
                worker_logger.warning(
                    "[core %d] Attempt %d/%d for %s failed: %s",
                    chunk_id, attempt + 1, max_retries,
                    row.get("MP id", "?"), e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        else:
            worker_logger.error(
                "[core %d] Exhausted retries for %s, skipping.",
                chunk_id, row.get("MP id", "?"),
            )

        # Periodic checkpoint
        if save_interval and (index + 1) % save_interval == 0:
            part = (index // save_interval) + 1
            ckpt_path = os.path.join(
                runtime_dir,
                f"{element}_core{chunk_id}_part{part}.xlsx",
            )
            chunk_df.to_excel(ckpt_path, index=False)
            worker_logger.info(
                "[core %d] Checkpoint: %d rows -> %s",
                chunk_id, index + 1, ckpt_path,
            )

    # Final per-worker checkpoint
    final_path = os.path.join(
        runtime_dir, f"{element}_core{chunk_id}_final.xlsx",
    )
    chunk_df.to_excel(final_path, index=False)
    worker_logger.info("[core %d] Done. %d rows -> %s", chunk_id, len(chunk_df), final_path)

    # Clean up handler
    worker_logger.removeHandler(file_handler)
    file_handler.close()

    return chunk_df

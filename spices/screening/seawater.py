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

import glob as _glob
import logging
import os
import re
import shutil
import subprocess
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from pymatgen.core import Composition

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
from spices.pourbaix.cache import PourbaixCache, _get_comp_from_pbx_name
from spices.pourbaix.composition import (
    get_pourbaix_composition,
    get_sea_pourbaix_comp_dict,
)
from spices.dataframe_ops import update_seawater_metastability_columns_by_mp_id

logger = logging.getLogger(__name__)

# DataFrame column names added by Stage 2
# For all 8 criteria, see spices.pourbaix.selection_criteria.SEAWATER_COLUMNS_ALL
SEAWATER_COLUMNS = [
    "Seawater stability",
    "Seawater metastability (Criterion 4)",
]

# Per-worker RSS estimate used by the pre-flight RAM check.  Sized for
# the default PourbaixCache caps (8/8/8) plus an 8-shard SnapshotLoader
# cache, snapshot indexes, pymatgen baseline, and a typical
# build_pourbaix_entries transient peak.
EST_PER_WORKER_GB = 2.5


def _available_memory_gb():
    """Return MemAvailable from /proc/meminfo as float GB.

    Returns ``None`` on platforms without ``/proc/meminfo`` (macOS,
    Windows) so callers can skip the warning gracefully.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kib = int(line.split()[1])
                    return kib / (1024 * 1024)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


# Window after the first BrokenProcessPool during which subsequent ones
# are treated as cascade rather than independent worker deaths.  When one
# worker dies, ProcessPoolExecutor cancels all pending futures, and each
# cancellation surfaces as its own BrokenProcessPool — but all share the
# same root cause as the first.  The user's 64-worker Li run produced 33
# "died" lines within 4 seconds; only the first reflects a real death.
_BROKEN_POOL_CASCADE_WINDOW_SEC = 30.0


def _scan_dmesg_for_oom():
    """Return the last few dmesg lines that look like OOM/SIGKILL events.

    Best-effort: returns ``[]`` if ``dmesg`` is missing, denied, or
    times out.  Linux/HPC-only post-mortem helper.
    """
    try:
        out = subprocess.run(
            ["dmesg", "-T"],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    lines = out.stdout.splitlines()
    keep = [
        ln for ln in lines
        if ("oom" in ln.lower() or "killed process" in ln.lower())
        and "python" in ln.lower()
    ]
    return keep[-5:]


def _scan_journal_for_kills():
    """Return the last few journalctl lines that look like SLURM/cgroup kills.

    Best-effort: returns ``[]`` on any failure.
    """
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-p", "err", "-n", "200",
             "--no-pager"],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    lines = out.stdout.splitlines()
    keep = [
        ln for ln in lines
        if any(t in ln.lower() for t in ("slurm", "cgroup", "oom", "killed"))
    ]
    return keep[-5:]


def _log_pool_postmortem(broken_cids, first_cid):
    """Log a brief post-mortem after one or more broken-pool events.

    Includes scans of dmesg and journalctl so the user can tell apart a
    real OOM (kernel SIGKILL), a SLURM cgroup time-limit kill, and a
    process crash that left no kernel trace.
    """
    if not broken_cids:
        return
    cascade = [c for c in broken_cids if c != first_cid]
    logger.error(
        "Pool break post-mortem: 1 root failure (worker %d) + %d cascade "
        "cancellations (%s).",
        first_cid, len(cascade),
        ", ".join(str(c) for c in cascade) if cascade else "none",
    )

    oom_lines = _scan_dmesg_for_oom()
    if oom_lines:
        logger.error(
            "dmesg shows %d kernel OOM/kill line(s) for python — likely a "
            "real OOM:",
            len(oom_lines),
        )
        for ln in oom_lines:
            logger.error("  dmesg: %s", ln)
    else:
        logger.info(
            "dmesg post-mortem: no python OOM/SIGKILL lines found "
            "(real OOM unlikely; suspect segfault, SLURM time-limit, or "
            "process exception).",
        )

    kill_lines = _scan_journal_for_kills()
    if kill_lines:
        logger.error(
            "journalctl shows %d SLURM/cgroup-related err line(s):",
            len(kill_lines),
        )
        for ln in kill_lines:
            logger.error("  journalctl: %s", ln)


# ----------------------------------------------------------------------
# Phase 1 — parent-side dedup + parallel pre-build of unique Pourbaix
# entries (Phase 1a) and unique sea Pourbaix diagrams (Phase 1b).
#
# Rationale: in the seawater screening, many cathode rows share the same
# chemsys (and even the same sea_comp.reduced_composition.formula) so a
# naive per-row dispatch sends N workers off to compute the same
# 17-minute Pourbaix hull.  Phase 1 enumerates the unique build keys in
# the parent, filters out keys whose pickle already exists on disk
# (restart-friendly), and dispatches one task per remaining unique key
# to a ProcessPoolExecutor.  No two workers ever touch the same key.
#
# Phase 2 (the existing chunked per-row dispatch) then runs with all
# diagrams pre-built — workers always hit the on-disk cache.
# ----------------------------------------------------------------------


def _phase0_discover_unique_keys(df, num_of_sea_elements=4):
    """Enumerate unique sea-chemsys and sea-comp keys from the input df.

    For each row, compute the seawater-merged composition; record the
    resulting chemsys and the reduced-formula key.  Insertion-ordered
    dedup so each unique key appears exactly once with a representative
    full ``Composition`` dict (taken from the first row that produced it,
    used later by Phase 1b's ``read_or_generate_sea_pourbaix_diagram``).

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.  Must have a ``Formula`` column.
    num_of_sea_elements : int
        Passed through to :func:`get_sea_pourbaix_comp_dict`.

    Returns
    -------
    (list of str, list of (str, dict))
        ``unique_chemsys`` : sorted-element chemsys strings
        ``unique_sea_keys`` : list of ``(reduced_formula, comp_as_dict)``
    """
    seen_chemsys = OrderedDict()
    seen_sea_keys = OrderedDict()
    for _, row in df.iterrows():
        formula = row.get("Formula")
        if not formula:
            continue
        try:
            comp = Composition(formula)
        except Exception:
            continue
        pbx_comp = get_pourbaix_composition(comp)
        sea_comp_dict = get_sea_pourbaix_comp_dict(pbx_comp, num_of_sea_elements)
        sea_comp = Composition(sea_comp_dict)
        chemsys = sea_comp.chemical_system
        sea_key = sea_comp.reduced_composition.formula
        seen_chemsys.setdefault(chemsys, None)
        seen_sea_keys.setdefault(sea_key, comp.as_dict())
    return list(seen_chemsys.keys()), list(seen_sea_keys.items())


def _filter_pending_chemsys(unique_chemsys, entries_dir):
    """Drop chemsys whose ``{chemsys}_pbxentry.pkl`` already exists."""
    if not os.path.isdir(entries_dir):
        return list(unique_chemsys)
    on_disk = set()
    for fname in os.listdir(entries_dir):
        if fname.endswith("_pbxentry.pkl"):
            on_disk.add(fname[: -len("_pbxentry.pkl")])
    return [c for c in unique_chemsys if c not in on_disk]


def _filter_pending_sea_keys(unique_sea_keys, pbx_dir):
    """Drop sea_keys whose pickle already exists in ``pbx_dir``.

    Cache files are named ``{formula_underscored}_pbx.pkl`` or
    ``{formula_underscored}_{mp_id}_pbx.pkl``.  Match by the underscored
    reduced formula prefix using :func:`_get_comp_from_pbx_name`.
    """
    if not os.path.isdir(pbx_dir):
        return list(unique_sea_keys)
    on_disk_prefixes = set()
    for fname in os.listdir(pbx_dir):
        if fname.endswith("_pbx.pkl"):
            on_disk_prefixes.add(_get_comp_from_pbx_name(fname))
    return [
        (key, comp_dict)
        for key, comp_dict in unique_sea_keys
        if key.replace(" ", "_") not in on_disk_prefixes
    ]


def _phase1a_build_entries(chemsys, cache_dir, max_retries, retry_delay):
    """Worker: build & save PourbaixEntry list for one ``chemsys``.

    Retries up to ``max_retries`` times on any exception; raises
    :class:`RuntimeError` if all retries are exhausted so the parent
    can abort the run.
    """
    mp_client = get_client()
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )
    last_err = None
    t0 = time.time()
    logger.info("Phase 1a starting build for %s", chemsys)
    for attempt in range(max_retries):
        try:
            cache.read_or_download_pourbaix_entries(
                chemsys, mp_client, save_new=True,
            )
            elapsed = time.time() - t0
            logger.info(
                "Phase 1a finished build for %s in %.1f sec (attempt %d)",
                chemsys, elapsed, attempt + 1,
            )
            return chemsys
        except Exception as e:
            last_err = e
            logger.warning(
                "Phase 1a attempt %d/%d failed for %s: %s",
                attempt + 1, max_retries, chemsys, e,
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise RuntimeError(
        f"Phase 1a: all {max_retries} retries failed for {chemsys}: "
        f"{last_err!r}"
    )


def _phase1b_build_sea_diagram(sea_key, comp_dict, num_of_sea_elements,
                               cache_dir, max_retries, retry_delay):
    """Worker: build & save sea Pourbaix diagram for one ``sea_key``.

    Reuses on-disk entries written by Phase 1a (same chemsys → same
    ``{chemsys}_pbxentry.pkl``); the diagram itself is keyed by
    ``sea_comp.reduced_composition.formula``.

    Retries on any exception; raises if all retries exhausted.
    """
    mp_client = get_client()
    cache = PourbaixCache(
        entries_dir=os.path.join(cache_dir, "saved_pbx_entries"),
        pbx_dir=os.path.join(cache_dir, "saved_pbx"),
    )
    composition_obj = Composition.from_dict(comp_dict)
    last_err = None
    t0 = time.time()
    logger.info("Phase 1b starting build for %s", sea_key)
    for attempt in range(max_retries):
        try:
            cache.read_or_generate_sea_pourbaix_diagram(
                composition_obj, mp_client,
                num_of_sea_elements=num_of_sea_elements,
                save_new_pbx=True, save_new_entries=True,
            )
            elapsed = time.time() - t0
            logger.info(
                "Phase 1b finished build for %s in %.1f sec (attempt %d)",
                sea_key, elapsed, attempt + 1,
            )
            return sea_key
        except Exception as e:
            last_err = e
            logger.warning(
                "Phase 1b attempt %d/%d failed for %s: %s",
                attempt + 1, max_retries, sea_key, e,
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise RuntimeError(
        f"Phase 1b: all {max_retries} retries failed for {sea_key}: "
        f"{last_err!r}"
    )


def _dispatch_phase1_tasks(label, n_workers, tasks, worker_fn, args_for_task):
    """Dispatch Phase 1 tasks across workers; abort on first failure.

    Parameters
    ----------
    label : str
        ``"Phase 1a"`` or ``"Phase 1b"`` — used in log messages.
    n_workers : int
        Number of parallel workers.  ``<=1`` runs sequentially.
    tasks : list
        One element per build job.
    worker_fn : callable
        The worker function (e.g. :func:`_phase1a_build_entries`).
    args_for_task : callable
        Maps a task element to the positional args for ``worker_fn``.

    Raises
    ------
    Exception
        Re-raises the first task failure so the parent can abort.
    """
    total = len(tasks)
    if total == 0:
        return

    if n_workers <= 1:
        for i, task in enumerate(tasks, 1):
            try:
                worker_fn(*args_for_task(task))
            except Exception:
                logger.exception(
                    "%s aborting due to failure in task %s", label, task,
                )
                raise
            logger.info("%s [%d/%d] complete (last: %s)", label, i, total, task)
        return

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_task = {
            pool.submit(worker_fn, *args_for_task(t)): t for t in tasks
        }
        try:
            for fut in as_completed(future_to_task):
                t = future_to_task[fut]
                try:
                    fut.result()
                except Exception:
                    logger.exception(
                        "%s aborting due to failure in task %s", label, t,
                    )
                    # cancel anything not yet running
                    for f in future_to_task:
                        if not f.done():
                            f.cancel()
                    raise
                completed += 1
                logger.info(
                    "%s [%d/%d] complete (last: %s)",
                    label, completed, total, t,
                )
        except BrokenProcessPool:
            logger.error(
                "%s: ProcessPool broken — a worker died before its task "
                "completed.  Re-run with --restart to resume from disk "
                "cache.",
                label,
            )
            raise


def _run_phase1(df, n_workers, cache_dir, num_of_sea_elements,
                phase1_max_retries, phase1_retry_delay):
    """Pre-build all unique chemsys entries and sea diagrams in parallel.

    Phase 0 enumerates unique keys; Phase 1a builds entries; Phase 1b
    builds sea diagrams.  On-disk pickles are filtered out before
    dispatch so restart skips already-done work.  Aborts on any
    irrecoverable task failure (after retries inside the worker).
    """
    entries_dir = os.path.join(cache_dir, "saved_pbx_entries")
    pbx_dir = os.path.join(cache_dir, "saved_pbx")

    t0 = time.time()
    unique_chemsys, unique_sea_keys = _phase0_discover_unique_keys(
        df, num_of_sea_elements=num_of_sea_elements,
    )
    logger.info(
        "Phase 0: discovered %d unique chemsys / %d unique sea_comp keys "
        "from %d rows in %.1f sec",
        len(unique_chemsys), len(unique_sea_keys), len(df),
        time.time() - t0,
    )

    # Phase 1a — entries
    pending_chemsys = _filter_pending_chemsys(unique_chemsys, entries_dir)
    if pending_chemsys:
        logger.info(
            "Phase 1a: building %d/%d unique chemsys entries (%d already on disk)",
            len(pending_chemsys), len(unique_chemsys),
            len(unique_chemsys) - len(pending_chemsys),
        )
        _dispatch_phase1_tasks(
            "Phase 1a", n_workers, pending_chemsys,
            _phase1a_build_entries,
            lambda c: (c, cache_dir, phase1_max_retries, phase1_retry_delay),
        )
    else:
        logger.info(
            "Phase 1a: all %d unique chemsys entries already on disk; "
            "skipping",
            len(unique_chemsys),
        )

    # Phase 1b — sea diagrams
    pending_sea = _filter_pending_sea_keys(unique_sea_keys, pbx_dir)
    if pending_sea:
        logger.info(
            "Phase 1b: building %d/%d unique sea diagrams (%d already on disk)",
            len(pending_sea), len(unique_sea_keys),
            len(unique_sea_keys) - len(pending_sea),
        )
        _dispatch_phase1_tasks(
            "Phase 1b", n_workers, pending_sea,
            _phase1b_build_sea_diagram,
            lambda kc: (
                kc[0], kc[1], num_of_sea_elements, cache_dir,
                phase1_max_retries, phase1_retry_delay,
            ),
        )
    else:
        logger.info(
            "Phase 1b: all %d unique sea diagrams already on disk; "
            "skipping",
            len(unique_sea_keys),
        )


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
    metastability_points=True,
    cache_dir=".",
    save_interval=100,
    max_retries=500,
    retry_delay=10.0,
    n_workers=1,
    restart=False,
    phase1_max_retries=5,
    phase1_retry_delay=30.0,
    num_of_sea_elements=4,
    mp_client=None,
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
    restart : bool
        If True, resume a previous interrupted run by loading
        checkpoint files from ``runtime_sheets/`` and any backup
        from a prior restart.  ``n_workers`` may differ from the
        original run.  Phase 1 (Pourbaix pre-build) automatically
        skips chemsys/sea_comp keys whose pickles already exist on
        disk, so restart distributes only the remaining unique builds
        across all workers.
    phase1_max_retries : int
        Per-task retry budget for Phase 1 (pre-build of unique
        chemsys entries and sea Pourbaix diagrams).  After this many
        failed attempts on a single chemsys/sea_comp build, the run
        aborts.  Default 5.
    phase1_retry_delay : float
        Seconds to wait between Phase 1 task retries.  Default 30.0.
    num_of_sea_elements : int
        Number of seawater ions (Cl, Na, Mg, S, Ca, K) included when
        building seawater Pourbaix diagrams.  Default 4 (Cl/Na/Mg/S);
        must match the value used downstream by the metastability
        evaluator so that Phase 1 pre-builds the right diagrams.
    mp_client : MPRester or SnapshotMPRester or None
        Materials Project client.  If None, resolved via
        :func:`spices.mp_client.get_client` (sticky-default cache,
        ``SPICES_MP_BACKEND``/``MP_API_KEY`` env vars, or live default).
        See **Multiprocessing and backend resolution** below for the
        ``n_workers > 1`` case.

    Returns
    -------
    pd.DataFrame
        Screening results with seawater columns added.

    Notes
    -----
    **Multiprocessing and backend resolution**

    The ``mp_client`` argument is honored *only* when ``n_workers == 1``
    (single-process run).  When ``n_workers > 1``, work is dispatched to
    a :class:`multiprocessing.Pool`; each worker is a separate Python
    interpreter that does **not** share the parent's
    :data:`spices.mp_client._last_client` cache.  Pickling a live
    ``MPRester`` (HTTP session state) or a :class:`SnapshotMPRester`
    (in-memory pickle indexes) across the process boundary is fragile,
    so workers re-resolve their own client by calling
    :func:`spices.mp_client.get_client` with no arguments.

    Each worker therefore relies on **environment variables inherited
    from the parent process** (Linux ``fork`` start method copies the
    parent's ``os.environ``) to choose a backend:

    * ``SPICES_MP_BACKEND`` — ``"snapshot"`` or ``"live"``
      (default ``"live"``).
    * ``SPICES_MP_SNAPSHOT_ROOT`` — path to the local snapshot tree
      (snapshot backend only).
    * ``SPICES_MP_SNAPSHOT_RELEASE`` — release label, e.g.
      ``"2025-06-09"`` (snapshot backend only).
    * ``MP_API_KEY`` — Materials Project API key (live backend only).

    Concrete recipes:

    * Snapshot multi-worker run::

          export SPICES_MP_BACKEND=snapshot
          export SPICES_MP_SNAPSHOT_ROOT=/path/to/MP_snapshots
          export SPICES_MP_SNAPSHOT_RELEASE=2025-06-09
          # then in Python:
          run_seawater_screening(..., n_workers=8)

      Passing ``mp_client=`` is also accepted (used for the parent's
      pre-Pool work) but the env vars above are still required for
      the workers.

    * Live multi-worker run::

          export MP_API_KEY=...
          run_seawater_screening(..., n_workers=8)

    Common pitfall: if you only pass ``mp_client=snapshot_client`` and
    do *not* set ``SPICES_MP_BACKEND=snapshot`` in the environment,
    workers silently fall back to the live backend and start hitting
    ``api.materialsproject.org``.  The parent uses your snapshot
    client, but every worker uses a live ``MPRester`` — and
    inconsistent backends across workers will scramble downstream
    results.
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
        "Read %d materials from %s",
        len(df),
        input_path,
    )

    start_time = time.time()
    output_dir = os.path.dirname(output_path) or "."
    runtime_dir = os.path.join(output_dir, "runtime_sheets")

    # Hard guard: with restart=False, refuse to overwrite an output file
    # that already has evaluated rows.  Silently dropping the seawater
    # columns destroys hours of compute when the user re-runs the same
    # script forgetting --restart.
    if not restart and os.path.isfile(output_path):
        try:
            existing = pd.read_excel(output_path)
        except Exception:
            existing = None
        if (
            existing is not None
            and "Seawater stability" in existing.columns
            and existing["Seawater stability"].notna().any()
        ):
            n_evaluated = int(existing["Seawater stability"].notna().sum())
            raise RuntimeError(
                f"Output file {output_path!r} already has {n_evaluated} "
                f"evaluated row(s). Pass restart=True (CLI: --restart) to "
                f"resume, or delete the file to start over."
            )

    if restart:
        # Keep existing seawater columns for merging
        for col in SEAWATER_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Load backup from a previous restart (if any)
        backup_path = os.path.join(
            output_dir, f"{element}_evaluated_backup.xlsx",
        )
        if os.path.isfile(backup_path):
            backup_df = pd.read_excel(backup_path)
            _merge_evaluated_rows(df, backup_df)
            logger.info(
                "Loaded %d evaluated rows from backup %s",
                backup_df["Seawater stability"].notna().sum(),
                backup_path,
            )

        # Load checkpoint files from runtime_sheets/
        if os.path.isdir(runtime_dir):
            ckpt_dfs = _load_checkpoints(runtime_dir, element)
            if ckpt_dfs:
                combined = pd.concat(ckpt_dfs, ignore_index=True)
                _merge_evaluated_rows(df, combined)
                logger.info(
                    "Loaded %d evaluated rows from checkpoints",
                    combined["Seawater stability"].notna().sum(),
                )

        evaluated = df["Seawater stability"].notna().sum()
        pending = df["Seawater stability"].isna().sum()

        # Save cumulative backup before deleting old checkpoints
        if evaluated > 0:
            backup_path = os.path.join(
                output_dir, f"{element}_evaluated_backup.xlsx",
            )
            df[df["Seawater stability"].notna()].to_excel(
                backup_path, index=False,
            )
            logger.info(
                "Saved backup: %d evaluated rows to %s",
                evaluated, backup_path,
            )

        # Clean slate for new checkpoints
        if os.path.isdir(runtime_dir):
            shutil.rmtree(runtime_dir)

        logger.info("Restart: %d evaluated, %d pending", evaluated, pending)
    else:
        # Fresh run: reset seawater columns
        for col in SEAWATER_COLUMNS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        for col in SEAWATER_COLUMNS:
            df[col] = None

    # Resolve the parent's client up front so backend errors surface
    # here rather than deep inside a worker process.  Workers in
    # multi-worker mode rebuild their own clients from inherited env
    # vars (see the docstring's "Multiprocessing and backend
    # resolution" section).
    if mp_client is None:
        mp_client = get_client()

    # Pre-flight RAM check.  Each worker is sized for ~EST_PER_WORKER_GB;
    # warn if the requested fan-out doesn't fit in 80% of available RAM,
    # since OS OOM-kills here are silent (no Python traceback).
    if n_workers > 1:
        avail_gb = _available_memory_gb()
        if avail_gb is not None:
            need_gb = n_workers * EST_PER_WORKER_GB
            if need_gb > 0.8 * avail_gb:
                suggested = max(1, int(0.8 * avail_gb / EST_PER_WORKER_GB))
                logger.warning(
                    "n_workers=%d * ~%.1f GB/worker (~%.0f GB) exceeds 80%% "
                    "of available memory (%.0f GB). Workers may be silently "
                    "OOM-killed by the kernel. Consider n_workers=%d.",
                    n_workers, EST_PER_WORKER_GB, need_gb,
                    avail_gb, suggested,
                )

    # Phase 1 — pre-build all unique Pourbaix entries (Phase 1a) and sea
    # diagrams (Phase 1b) in parallel.  Restart-safe: keys whose pickle
    # is already on disk are skipped.  Phase 2 (the per-row evaluation
    # below) will then always hit the on-disk cache, eliminating the
    # duplicate-build problem where N workers race on the same chemsys.
    # Only run Phase 1 over rows that still need evaluation.
    if restart:
        phase1_df = df[df["Seawater stability"].isna()].reset_index(drop=True)
    else:
        phase1_df = df
    if len(phase1_df) > 0:
        _run_phase1(
            phase1_df,
            n_workers=max(1, n_workers),
            cache_dir=cache_dir,
            num_of_sea_elements=num_of_sea_elements,
            phase1_max_retries=phase1_max_retries,
            phase1_retry_delay=phase1_retry_delay,
        )
    else:
        logger.info(
            "Phase 1 skipped: no pending rows after restart merge",
        )

    # Common keyword args for the worker function
    worker_kwargs = dict(
        element=element,
        cache_dir=cache_dir,
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
        restart=restart,
    )

    if n_workers <= 1:
        # Sequential mode
        df = _process_chunk(df, 0, **worker_kwargs)
    else:
        # Parallel mode
        chunks = _split_dataframe(df, n_workers)
        logger.info(
            "Splitting %d rows across %d workers",
            len(df),
            len(chunks),
        )

        pool_args = [(chunk, chunk_id) for chunk_id, chunk in enumerate(chunks)]
        original_by_cid = {cid: chunk for chunk, cid in pool_args}
        results_by_cid = {}

        # When a worker dies, ProcessPoolExecutor.fut.result() raises
        # BrokenProcessPool not just for the dead worker but also for
        # every pending future on the same pool.  Track the first break
        # to distinguish the actual root-cause failure from cascade
        # cancellations — historically all 33 cancellations were logged
        # as "Worker N died (likely OOM-killed)" within ~4 seconds, even
        # though only one worker actually died.
        first_break_time = None
        first_break_cid = None
        broken_cids = []

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_cid = {
                pool.submit(
                    _process_chunk_wrapper, chunk, cid, worker_kwargs,
                ): cid
                for chunk, cid in pool_args
            }
            try:
                for fut in as_completed(future_to_cid):
                    cid = future_to_cid[fut]
                    try:
                        results_by_cid[cid] = fut.result()
                    except BrokenProcessPool:
                        now = time.time()
                        if first_break_time is None:
                            first_break_time = now
                            first_break_cid = cid
                            logger.error(
                                "Worker %d: chunk failed (process pool "
                                "broken — root cause to investigate). "
                                "Recovering from runtime_sheets/.",
                                cid,
                            )
                        else:
                            elapsed = now - first_break_time
                            label = (
                                "cascade %.1f sec after worker %d's "
                                "failure" % (elapsed, first_break_cid)
                                if elapsed < _BROKEN_POOL_CASCADE_WINDOW_SEC
                                else "later failure %.1f sec after worker %d"
                                % (elapsed, first_break_cid)
                            )
                            logger.error(
                                "Chunk %d cancelled by broken pool (%s). "
                                "Recovering from runtime_sheets/.",
                                cid, label,
                            )
                        broken_cids.append(cid)
                        results_by_cid[cid] = _recover_chunk_for_core(
                            runtime_dir, element, cid,
                            original_by_cid[cid],
                        )
                    except MemoryError:
                        logger.error(
                            "Worker %d: MemoryError raised inside chunk. "
                            "Recovering from runtime_sheets/.",
                            cid,
                        )
                        broken_cids.append(cid)
                        results_by_cid[cid] = _recover_chunk_for_core(
                            runtime_dir, element, cid,
                            original_by_cid[cid],
                        )
                    except Exception:
                        logger.exception("Worker %d raised", cid)
                        raise
            except BrokenProcessPool:
                # The whole pool just broke; any pending futures will
                # not return.  Fall through to the recover-from-disk
                # block after the pool context manager exits.
                logger.error(
                    "ProcessPool broken — pending chunks will be "
                    "recovered from runtime_sheets/."
                )

        if broken_cids:
            _log_pool_postmortem(broken_cids, first_break_cid)

        # Anything that never returned: pull from checkpoints, else
        # keep the original chunk so restart=True can finish it later.
        for cid, chunk in original_by_cid.items():
            if cid not in results_by_cid:
                logger.error(
                    "Chunk %d never returned. Recovering from checkpoint.",
                    cid,
                )
                results_by_cid[cid] = _recover_chunk_for_core(
                    runtime_dir, element, cid, chunk,
                )

        processed_chunks = [results_by_cid[cid] for cid in sorted(results_by_cid)]
        df = pd.concat(processed_chunks, ignore_index=True)

    # Final save — atomic via tmp + os.replace so a kill mid-write
    # leaves either the old file or the new file, never a corrupt one.
    # The tmp filename keeps a trailing .xlsx so pandas can infer the
    # engine; the PID-tagged middle prevents two parents from writing
    # the same tmp file concurrently.
    tmp_path = f"{output_path}.tmp.{os.getpid()}.xlsx"
    try:
        df.to_excel(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    except Exception:
        logger.exception(
            "Final-write failed for %s — runtime_sheets/ left intact for "
            "restart.", output_path,
        )
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    # Clean up runtime sheets ONLY after the atomic replace succeeded.
    if os.path.isdir(runtime_dir):
        shutil.rmtree(runtime_dir)
        logger.info("Cleaned up %s", runtime_dir)

    elapsed = time.time() - start_time
    logger.info(
        "Seawater screening completed in %.1f sec. Saved to %s",
        elapsed,
        output_path,
    )

    return df


def _merge_evaluated_rows(df, source_df):
    """Copy evaluated seawater values from *source_df* into *df* by MP id.

    Only rows whose seawater columns are still NaN in *df* are updated.
    Modifies *df* in place.
    """
    evaluated = source_df[source_df["Seawater stability"].notna()]
    if evaluated.empty:
        return
    eval_map = evaluated.set_index("MP id")[SEAWATER_COLUMNS]
    needs_update = df["Seawater stability"].isna() & df["MP id"].isin(eval_map.index)
    if not needs_update.any():
        return
    mp_ids = df.loc[needs_update, "MP id"]
    for col in SEAWATER_COLUMNS:
        df.loc[needs_update, col] = mp_ids.map(eval_map[col]).values


def _load_checkpoints(runtime_dir, element):
    """Load the best checkpoint DataFrame for each core.

    For each core, picks ``_final.xlsx`` if it exists, otherwise the
    latest ``_partN.xlsx``.

    Returns a list of DataFrames (one per core that had checkpoints).
    """
    all_files = _glob.glob(
        os.path.join(runtime_dir, f"{element}_core*_*.xlsx"),
    )
    if not all_files:
        return []

    # Group files by core id
    cores = {}
    for f in all_files:
        base = os.path.basename(f)
        after_core = base.split("_core")[1]
        core_id = int(after_core.split("_")[0])
        cores.setdefault(core_id, []).append(f)

    dfs = []
    for core_id in sorted(cores):
        files = cores[core_id]
        final = [f for f in files if "_final." in f]
        if final:
            dfs.append(pd.read_excel(final[0]))
            continue
        parts = [f for f in files if "_part" in f]
        if parts:
            parts.sort(
                key=lambda f: int(re.search(r"_part(\d+)", f).group(1)),
            )
            dfs.append(pd.read_excel(parts[-1]))

    return dfs


def _recover_chunk_for_core(runtime_dir, element, core_id, original_chunk):
    """Return the best available DataFrame for *core_id* after a worker death.

    Prefers ``{element}_core{core_id}_final.xlsx``, falls back to the
    highest-numbered ``_partN.xlsx``, and finally returns *original_chunk*
    unchanged (NaN seawater columns) so a subsequent ``restart=True`` run
    can pick up where this one left off.
    """
    if not os.path.isdir(runtime_dir):
        return original_chunk

    pattern = os.path.join(
        runtime_dir, f"{element}_core{core_id}_*.xlsx",
    )
    matches = _glob.glob(pattern)
    if not matches:
        return original_chunk

    final = [f for f in matches if "_final." in f]
    if final:
        return pd.read_excel(final[0])

    parts = [f for f in matches if "_part" in f]
    if parts:
        parts.sort(key=lambda f: int(re.search(r"_part(\d+)", f).group(1)))
        return pd.read_excel(parts[-1])

    return original_chunk


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
    restart=False,
):
    """
    Process a DataFrame chunk for seawater screening.

    Each worker creates its own ``PourbaixCache`` and MP client to
    avoid shared state across processes.  The MP client is rebuilt
    from environment variables inherited from the parent process —
    see the ``run_seawater_screening`` docstring's "Multiprocessing
    and backend resolution" section.  Progress is logged to a
    per-worker log file and checkpointed periodically.

    Parameters
    ----------
    chunk_df : pd.DataFrame
    chunk_id : int
        Worker index (used in filenames and log messages).
    element : str
    cache_dir : str
    output_dir : str
    OER_shift, HER_shift : float
    pH_range, voltage_range : tuple of float
    pH_grids, voltage_grids : int
    decomp_energy : float
    metastability_points : bool
    save_interval : int
    max_retries : int
    retry_delay : float
    restart : bool
        If True, skip already-evaluated rows and append to logs.

    Returns
    -------
    pd.DataFrame
        Processed chunk with seawater columns populated.
    """
    # Set up per-worker logging
    log_dir = os.path.join(output_dir, "process_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"process_log_core{chunk_id}.txt")
    file_handler = logging.FileHandler(log_path, mode="a" if restart else "w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    worker_logger = logging.getLogger(f"spices.worker.{chunk_id}")
    worker_logger.addHandler(file_handler)
    worker_logger.setLevel(logging.INFO)

    # Each worker gets its own client and cache.  The no-arg
    # get_client() resolves the backend from env vars inherited from
    # the parent process — see run_seawater_screening's docstring.
    mp_client = get_client()
    cache = PourbaixCache(
        entries_dir=f"{cache_dir}/saved_pbx_entries",
        pbx_dir=f"{cache_dir}/saved_pbx",
    )

    runtime_dir = os.path.join(output_dir, "runtime_sheets")
    os.makedirs(runtime_dir, exist_ok=True)

    for index in range(len(chunk_df)):
        row = chunk_df.iloc[index]

        # Skip rows already evaluated (restart mode)
        if pd.notna(row.get("Seawater stability")):
            worker_logger.info(
                "[core %d] [%d/%d] %s (%s) — skipping (already evaluated)",
                chunk_id, index + 1, len(chunk_df),
                row.get("Formula", "?"), row.get("MP id", "?"),
            )
            continue

        worker_logger.info(
            "[core %d] [%d/%d] %s (%s)",
            chunk_id,
            index + 1,
            len(chunk_df),
            row.get("Formula", "?"),
            row.get("MP id", "?"),
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
            except (AttributeError, ImportError, TypeError) as e:
                # Non-retryable errors (e.g. missing mpcontribs-client)
                worker_logger.error(
                    "[core %d] Non-retryable error for %s: %s",
                    chunk_id,
                    row.get("MP id", "?"),
                    e,
                )
                break
            except MemoryError:
                # Don't try to recover — propagate so the parent labels
                # the failure as a real OOM instead of treating it as a
                # silent BrokenProcessPool / cascade.
                worker_logger.error(
                    "[core %d] MemoryError for %s — propagating to parent",
                    chunk_id, row.get("MP id", "?"),
                )
                raise
            except Exception as e:
                worker_logger.warning(
                    "[core %d] Attempt %d/%d for %s failed: %s",
                    chunk_id,
                    attempt + 1,
                    max_retries,
                    row.get("MP id", "?"),
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        else:
            worker_logger.error(
                "[core %d] Exhausted retries for %s, skipping.",
                chunk_id,
                row.get("MP id", "?"),
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
                chunk_id,
                index + 1,
                ckpt_path,
            )

    # Final per-worker checkpoint
    final_path = os.path.join(
        runtime_dir,
        f"{element}_core{chunk_id}_final.xlsx",
    )
    chunk_df.to_excel(final_path, index=False)
    worker_logger.info(
        "[core %d] Done. %d rows -> %s", chunk_id, len(chunk_df), final_path
    )

    # Clean up handler
    worker_logger.removeHandler(file_handler)
    file_handler.close()

    return chunk_df

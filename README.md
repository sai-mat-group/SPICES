# SPICES

**S**eawater **P**ourbaix **I**ntercalation **C**ation **E**lectrode **S**creening

A Python package for high-throughput screening of cathode materials for desalination applications using Pourbaix diagram analysis and Materials Project data.

## Overview

SPICES implements a 6-stage computational pipeline (Stages 0--5) for identifying promising cathode materials for electrochemical desalination:

0. **MP data query** -- Query Materials Project for structural and thermodynamic metadata (mandatory)
1. **Dilute-limit filter** -- Evaluate stability in pure water using Pourbaix diagram analysis (optional)
2. **Seawater filter** -- Evaluate stability under seawater ionic conditions (accepts Stage 0 or Stage 1 output)
3. **Intercalation voltage filter** -- Filter by electrochemical voltage profiles
4. **Framework comparison** -- Compare material frameworks across different working ions (Ca, K, Li, Mg, Na)
5. **Statistical analysis** -- Aggregate per-ion counts into a summary table and publication figures: a per-ion pipeline bar chart, per-ion filter funnels (nested area-proportional circles, not true Venn diagrams), the screened periodic table, and the ion-selective + multi-ion structure-voltage plots

### XC functional restriction

Every MP query in SPICES is restricted to **`thermo_type=GGA_GGA+U`** (a single source of truth: `spices.mp_snapshot.THERMO_TYPE`). This matches the live MP API's hardcoded `get_pourbaix_entries` behavior, so step 0 / step 2 / step 3 use the same XC family that the Pourbaix diagrams in steps 1-2 are built from. R2SCAN-corrected entries are filtered out at the source — both backends — so they cannot leak into hull construction or voltage calculation. Step-0 output therefore carries a single `XC Functional` column (values: `"GGA"` or `"GGA+U"`) instead of the older Base/Correction/mixing-adjustment trio.

## Installation

### Prerequisites

- Python >= 3.9
- A [Materials Project API key](https://materialsproject.org/api)

### Setup

```bash
# Create and activate a conda environment
conda create -n MP_api_2026 python=3.11 -y
conda activate MP_api_2026

# Install SPICES
git clone <repo-url>
cd spices
pip install -e ".[dev]"

# Set your Materials Project API key
echo "MP_API_KEY=your_key_here" > .env
```

## Quick Start

```python
from spices.screening.mp_query import run_mp_query
from spices.screening.dilute_water import run_dilute_water_evaluation
from spices.screening.seawater import run_seawater_screening

# API key is read from MP_API_KEY (env var or .env file).  Each
# pipeline entry point also accepts an explicit mp_client= built via
# spices.mp_client.get_client() — see "Snapshot backend" below.

# Stage 0: Query Materials Project data (mandatory)
df_mp = run_mp_query(
    element="Ca",
    output_path="Ca_cathodes_mp_data.xlsx",
)

# Stage 1: Evaluate dilute-water stability (optional)
df_water = run_dilute_water_evaluation(
    element="Ca",
    input_data="Ca_cathodes_mp_data.xlsx",
    output_path="Ca_cathodes_pure_water_metastability.xlsx",
)

# Stage 2: Seawater screening — accepts Stage 0 or Stage 1 output
df_sea = run_seawater_screening(
    element="Ca",
    input_path="Ca_cathodes_mp_data.xlsx",  # or Stage 1 output
    output_path="Ca_seawater_results.xlsx",
)

# Stage 2 with multiprocessing (48 cores)
df_sea = run_seawater_screening(
    element="Ca",
    input_path="Ca_cathodes_mp_data.xlsx",
    output_path="Ca_seawater_results.xlsx",
    n_workers=48,
)

# Stage 2: restart an interrupted run (n_workers may differ from original)
df_sea = run_seawater_screening(
    element="Ca",
    input_path="Ca_cathodes_mp_data.xlsx",
    output_path="Ca_seawater_results.xlsx",
    n_workers=24,
    restart=True,
)

# Stage 3: Run intercalation voltage screening on Stage 2 output
# Per-composition sheets are partitioned into framework-matching
# components via a graph built over StructureMatcher-equivalence
# (spices.intercalation.grouping).  One PDF per connected component
# is saved to {output_dir}/Voltage_plateau_data/{framework}/graphs/
# unless save_component_graphs=False.  ``n_workers`` and ``restart``
# apply to both sub-step 2 (per-row sea-Pourbaix evaluation) and
# sub-step 3 (per-composition graph build + voltage plateau).
from spices.screening.intercalation import run_intercalation_screening

combined = run_intercalation_screening(
    element="Ca",
    input_path="Ca_seawater_results.xlsx",
    output_dir="Ca_intercalation/",
)

# Stage 3 with multiprocessing.  Adjacency dicts are pickled to
# Ca_intercalation/adjacency_cache/{composition}.pkl so a restart
# skips the O(n^2) StructureMatcher rebuild.
combined = run_intercalation_screening(
    element="Ca",
    input_path="Ca_seawater_results.xlsx",
    output_dir="Ca_intercalation/",
    n_workers=24,
)

# Stage 3: resume after an interrupted run
combined = run_intercalation_screening(
    element="Ca",
    input_path="Ca_seawater_results.xlsx",
    output_dir="Ca_intercalation/",
    n_workers=24,
    restart=True,
)

# Stage 3: re-render BOTH plot kinds (component graphs + per-plateau
# pH-overlap PDFs) from a completed run, fully OFFLINE (no MP/snapshot
# access) and without rewriting any xlsx — mirrors Stage 4's
# --rerender-plots.  Useful to add plots to a run done without
# save_plots, or to refresh plot styling.  input_path is not needed.
from spices.screening.intercalation import rerender_intercalation_plots

rerender_intercalation_plots(
    element="Ca",
    output_dir="Ca_intercalation/",
    n_workers=24,
)

# Stage 4: Compare frameworks across working ions.  The graph is built ONCE
# over every ion in stage3_dirs; `elements` only restricts the per-ion Excel
# output (default None → all ions present).  It also writes
# structure_voltage_records.json — the graph-derived single/multi-ion selection
# that drives the Stage 5 structure-voltage figures.  Set-based comparison (Venn
# diagrams, voltage overlays) is a separate entry point, run_set_based_comparison.
from spices.screening.framework_comparison import run_framework_comparison

stage3_dirs = {e: f"{e}_intercalation/" for e in ["Ca", "K", "Li", "Mg", "Na"]}
result = run_framework_comparison(
    stage3_dirs=stage3_dirs,
    output_dir="framework_comparison/",
)
```

### Using individual analysis functions

```python
from spices.pourbaix.composition import rm_element, get_pourbaix_composition
from spices.pourbaix.stability import is_stable_in_water
from spices.pourbaix.geometry import check_overlap_with_water_stability_lines

# Remove working ion from formula
charged = rm_element("LiFePO4", "Li")  # -> "FePO4"

# Check if a (pH, voltage) point is within the water stability window
in_water = check_overlap_with_water_stability_lines(pH=7, voltage=0.5)
# True
```

## Selection Criteria

Stages 2--3 evaluate whether decomposition products in seawater preserve the cathode structure. This is controlled by a **selection criterion** that defines what "preserved" means.

By default, the pipeline uses **Criterion 4**: at least one solid decomposition product must contain a non-working-ion cation *and* an anionic element. This strikes a balance between being too permissive (e.g. Criterion 1, which accepts any cation) and too strict (e.g. Criterion 6, which requires all framework elements retained as compounds).

### Advanced: exploring other criteria

For users who want to explore all 8 selection criteria, an optional module is available:

```python
from spices.pourbaix.selection_criteria import (
    check_solid_entry_all_criteria,
    ALL_SELECTION_CRITERIA,       # [1, 2, 3, 4, 5, 6, 7, 8]
    SEAWATER_COLUMNS_ALL,         # all 9 column names
)

# Evaluate a single alternative criterion
result = check_solid_entry_all_criteria(entry, reactant, mp_id, pbx, ion,
                                        selection_criterion=3)

# Evaluate all 8 at once (returns dict of bools)
result = check_solid_entry_all_criteria(entry, reactant, mp_id, pbx, ion,
                                        selection_criterion=9)
```

The main pipeline functions also accept a `selection_criterion` parameter if you want to override the default:

```bash
# Stage 3 CLI: evaluate criteria 1, 2, 3, and 4
python scripts/run_intercalation_screening.py Ca input.xlsx --selection-criteria 1,2,3,4
```

See the docstring in `spices/pourbaix/selection_criteria.py` for a full description of all 8 criteria.

## Package Structure

```
spices/
    __init__.py
    config.py                  # Constants and default parameters
    mp_client.py               # Materials Project API client (live + snapshot backends)
    dataframe_ops.py           # DataFrame-level stability update functions
    io.py                      # Logging configuration
    plotting.py                # Publication-style matplotlib rcParams (Helvetica, embedded fonts)

    pourbaix/                  # Core Pourbaix diagram analysis
        composition.py         # Chemical composition manipulation
        geometry.py            # Geometric overlap calculations
        stability.py           # Thermodynamic stability checks (Criterion 4 default)
        decomposition.py       # Decomposition energy calculations
        metastability.py       # Grid-based metastability searches
        selection_criteria.py  # Optional: all 8 selection criteria with docs
        cache.py               # Pourbaix entry/diagram caching (disk + memory)
        plotting.py            # Heat maps and phase region plots
        thermo.py              # find_correction (manual analysis helper) + is_most_stable

    intercalation/             # Intercalation voltage analysis
        voltage.py             # Voltage calculation from DFT energies
        hull.py                # Pseudo-binary convex hull filtering
        structure.py           # Charged-structure caching (3-level: memory/disk/MP)
        grouping.py            # Graph-based framework partitioning and component plotting

    screening/                 # Pipeline stage modules
        mp_query.py            # Stage 0: Materials Project data query
        dilute_water.py        # Stage 1: pure water stability evaluation
        seawater.py            # Stage 2: seawater stability screening
        intercalation.py       # Stage 3: intercalation voltage filtering
        framework_comparison.py # Stage 4: cross-element framework comparison (graph-based)
        framework_set_comparison.py # Stage 4 companion: set-based (mp_id) comparison — Venn + voltage overlays
        statistics.py          # Stage 5: aggregate statistics and figures

    mp_snapshot/               # Offline MP backend for reproducing frozen releases
        __init__.py            # THERMO_TYPE = "GGA_GGA+U" (single source of truth)
        download.py            # AWS OpenData S3 puller (threaded, ~0.6 GB for 2025-06-09)
        loader.py              # Indexed reader over gz-JSONL shards (GGA_GGA+U subroot only)
        pourbaix.py            # Offline get_pourbaix_entries via MP's documented algorithm
        client.py              # SnapshotMPRester — strict drop-in for MPRester

scripts/                       # CLI wrappers and HPC submission templates
    run_seawater_screening.py  # CLI for Stage 2 (supports multiprocessing)
    run_intercalation_screening.py  # CLI for Stage 3 (supports multiprocessing in sub-steps 2 and 3)
    run_framework_comparison.py # CLI for Stage 4 (graph-based framework comparison)
    run_framework_set_comparison.py # CLI for Stage 4 set-based comparison (Venn + voltage overlays)
    run_statistics.py          # CLI for Stage 5
    submit_seawater.sh         # SLURM submission template for Stage 2
    submit_intercalation.sh    # SLURM submission template for Stage 3
```

## HPC / Multiprocessing

Stage 2 and Stage 3 sub-steps 2 and 3 all parallelize via `-n N`.

**Stage 2** is internally a two-phase pipeline:

* **Phase 1** — pre-build all unique Pourbaix entries (Phase 1a, keyed by chemsys) and unique seawater Pourbaix diagrams (Phase 1b, keyed by reduced sea-comp formula).  The parent enumerates the unique build keys from the input dataframe, filters out keys whose pickle already exists on disk, and dispatches **one task per remaining unique key** to the worker pool.  No two workers ever build the same diagram, and no worker waits idle on a lock — every active worker is computing something different.  Each Phase 1 task retries up to `--phase1-max-retries` (default 5) on transient errors; if a task ultimately fails, the run aborts so the user can investigate.
* **Phase 2** — per-row metastability evaluation, chunked round-robin across workers (existing behavior).  Phase 2 always hits the on-disk cache built in Phase 1, so duplicate Pourbaix builds across workers are eliminated.

**Stage 3 sub-step 2** is a 3-phase pipeline: Phase 1 (per-file discovery of new charged MP structures, including disk-cached CIF lookup and `find_structures` formula matching) and Phase 2 (per-row sea-Pourbaix evaluation of those new rows) both use the worker pool — only Phase 3 (atomic xlsx rewrite) is single-threaded in the parent.

**Stage 3 sub-step 3** dispatches one task per unique-charged-composition file to the same worker pool.  Each worker resolves its own `mp_client` and `PourbaixCache` after fork, builds the per-composition framework adjacency dict (`O(n^2)` pairwise `StructureMatcher.fit`), runs BFS to extract connected components, renders component-graph PDFs, and computes voltage plateaus per component per selection criterion.  The adjacency dict is pickled to `{output_dir}/adjacency_cache/{composition}.pkl` — keyed by sorted MP-id tuple + matcher signature — so a restart skips the rebuild.  The cache is always-on regardless of `--restart`.  Sub-steps 1 and 4 still run sequentially in the parent (cheap); Stages 4 and 5 are single-process and should be launched directly on a workstation.

> **Linux/Unix only.**  The multi-worker code uses `fcntl.flock` (POSIX advisory file locks) for the cross-worker `find_structures` race fix and `os.fork`-based `ProcessPoolExecutor` for the worker pool.  Run multi-worker pipelines on Linux (workstation or HPC); on Windows / macOS use `n_workers=1`.

For multi-worker runs against the snapshot backend, export `SPICES_MP_BACKEND=snapshot` (and `SPICES_MP_SNAPSHOT_ROOT`, `SPICES_MP_SNAPSHOT_RELEASE`) so each forked worker resolves the same backend — see the `run_seawater_screening` docstring for the full caveat.

### Restart and crash safety

* **`--restart`** (Stage 2) resumes an interrupted run.  In Phase 1, only chemsys/sea-comp keys whose Pourbaix pickle is missing from disk are re-dispatched — if 8000 of 10000 unique diagrams already exist, the remaining 2000 are split across all workers.  In Phase 2, only rows whose `Seawater stability` is still NaN are re-evaluated, with checkpoint files from `runtime_sheets/` and `{element}_evaluated_backup.xlsx` merged in.
* **`--restart`** (Stage 3) resumes an interrupted intercalation run.  Sub-step 2 resumes from per-row checkpoints in `runtime_sheets/`.  Sub-step 3 skips a composition whose `Voltage_plateau_data/{composition}/sc_{N}/{composition}.xlsx` already exists for every requested selection criterion; partial compositions are re-built from the on-disk adjacency cache (`{output_dir}/adjacency_cache/{composition}.pkl`), so the O(n²) `StructureMatcher` work is paid only once.
* **Stage 4 framework cache** — `run_framework_comparison` bundles the loaded framework pool *and* the O(n²) cross-element `StructureMatcher` adjacency into a single pickle (`{output_dir}/step4_cache.pkl`, or `step4_cache_reconnected.pkl` with `--reconnect-isolated`) plus a `.meta.json` sidecar.  Every run validates the sidecar (schema version, Stage-3 ion dirs and their order, sheet name, reconnect flag, and the full mp-id list); any mismatch — or a truncated / corrupt pickle caught while unpickling — is treated as a cache miss and rebuilt, so a kill mid-write is self-healing on the next run even though the Stage 4 write is *not* atomic (unlike the Stage 2/3 PID-temp + `os.replace` pattern).  `--rerender-plots` requires a valid cache and re-draws the PNGs *and* refreshes `structure_voltage_records.json` (the Stage-5 figure data), skipping the framework load, the O(n²) build, and the per-ion Excel writes; `--force-rebuild` ignores any existing cache and rebuilds from scratch.  Stage 4 is otherwise single-process, and Stage 5 has no cache or restart path.
* **Stage 3 `--rerender-plots`** — re-renders both step-3 plot kinds (component-graph PDFs under `Voltage_plateau_data/{framework}/graphs/` and the per-plateau pH-overlap PDFs under `sc_{N}/`) from a completed run's persisted data, **fully offline** (no MP/snapshot access — graphs reuse `adjacency_cache/` + on-disk struct CIFs, voltage plots read coords from the composition sheets and voltages from the `sc_{N}` plateau sheets) and **plots-only** (no xlsx writes, no step-4 combine).  It skips steps 1–4 entirely and errors with a clear `FileNotFoundError` if no completed run is found in the output dir.  Voltage plots whose metastability coords were computed on-the-fly for seawater-stable phases during the original run (and never written back to the composition sheet) cannot be reproduced offline; they are skipped and listed in `{output_dir}/rerender_skipped_plots.csv`.  Component graphs are always fully reproducible.  Parallelizes per-composition via `-n N` (same worker pool as the full run); mutually exclusive with `--restart`; `--no-voltage-plots` / `--no-component-graphs` restrict to one kind.
* **Hard guard on `restart=False`** — if the Stage 2 output xlsx already contains evaluated rows and `--restart` is *not* passed, the run aborts with a clear error rather than silently overwriting prior progress.  Pass `--restart` to resume, or delete the file to start over.
* **Atomic final write** — the output xlsx is written via `to_excel(<tmp>); os.replace(<tmp>, <final>)`.  A kill mid-write leaves either the previous file or the new one, never a corrupt half-written xlsx.  `runtime_sheets/` is cleaned up only after `os.replace` succeeds.  Stage 3's adjacency pickles use the same PID-tagged temp + `os.replace` pattern so two workers can never leave a truncated pickle on disk.
* **Honest worker-death diagnostics** — when one worker dies, `ProcessPoolExecutor` cancels every pending future and each cancellation surfaces as `BrokenProcessPool`.  Stage 2 distinguishes the *first* (root-cause) failure from subsequent cascade cancellations in the log, then runs a post-mortem scan of `dmesg -T` and `journalctl --user` so the user can tell apart a real OOM (kernel SIGKILL of `python`), a SLURM cgroup time-limit kill, and a process-internal exception.  Stage 3's sub-step 3 dispatcher aborts on the first per-composition worker exception so partial-state corruption cannot accrete silently — re-run with `--restart` to resume.

```bash
# Stage 2: seawater screening (parallel, 8 cores)
python scripts/run_seawater_screening.py Ca input.xlsx -o output.xlsx -n 8

# Stage 2: resume after an interrupted multi-worker run
python scripts/run_seawater_screening.py Ca input.xlsx -o output.xlsx -n 8 --restart

# Stage 2: tune Phase 1 retry budget for flaky MP backends
python scripts/run_seawater_screening.py Ca input.xlsx -o output.xlsx -n 8 \
    --phase1-max-retries 10 --phase1-retry-delay 60

# Stage 3: intercalation voltage screening (parallel sub-steps 2 and 3, 8 cores)
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ -n 8

# Stage 3 with per-plateau pH-overlap PDFs, without component graphs
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ \
    --save-plots --no-component-graphs

# Stage 3 — resume an interrupted multi-worker run.  Sub-step 2 resumes
# from per-row checkpoints; sub-step 3 skips completed compositions and
# reuses the on-disk adjacency cache so the O(n^2) StructureMatcher work
# is paid only once per composition across reruns.
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ \
    -n 8 --restart

# Stage 3 — re-render plots offline from a completed run (no MP/snapshot
# access, no xlsx writes).  input_path is optional here.  Use
# --no-voltage-plots / --no-component-graphs to render only one kind.
python scripts/run_intercalation_screening.py Ca --rerender-plots -o Ca_intercalation/ -n 8

# Stage 4: framework comparison (graph-based; one build over all ions present)
python scripts/run_framework_comparison.py --stage3-dir ./intercalation_results/
# (add --elements K to restrict the per-ion Excel output, e.g. for a smoke test)

# Stage 4: set-based comparison (separate script — Venn diagrams + voltage overlays)
python scripts/run_framework_set_comparison.py --stage3-dir ./intercalation_results/

# Stage 5: aggregate statistics + publication figures (all 5 ions)
python scripts/run_statistics.py \
    --stage2-dir ./2.Seawater-metastability \
    --stage3-dir ./3.Intercalation-voltage-filter \
    --stage4-dir ./4.compare_frameworks \
    -o ./statistics_output --criterion 4
# (--stage4-dir is optional; it enables the structure-voltage figures, which read
#  <stage4-dir>/structure_voltage_records.json written by Stage 4 — pass
#  --stage4-records to point at that JSON explicitly.  --faded-partner-plateaus
#  {all,ph_covering} sets which faded partner-ion markers appear in the multi-ion
#  plot; --bar-color-by {stage,ion}, --no-periodic-table, --criterion N too.)

# HPC submission (edit the templates first)
sbatch scripts/submit_seawater.sh
sbatch scripts/submit_intercalation.sh
```

See `scripts/README.md` for details.

### Stage 3 structure caches

Sub-step 2 of Stage 3 builds three CIF caches under `{output_dir}/` (e.g. `Ca_intercalation/`).  All three are populated lazily on first use and re-read on subsequent runs to skip MP queries.

| Directory | Keying | Stores | Used by |
|---|---|---|---|
| `structs/` | `{mp_id}.cif` | The original (discharged) MP structure as downloaded — written by `generate_charged_struct_from_MP` *before* the working ion is removed. One CIF per discharged MP id processed. | Sub-step 3 graph rendering (`build_graph_components_from_sheet` → `_load_row_struct` primary path); Stage 4 framework comparison (`load_discharged_frameworks`). |
| `charged_structs/` | `{mp_id}.cif` | The charged framework (working-ion-removed) for each discharged MP id, written *after* `Structure.remove_species`.  Acts as the per-mp-id memoization layer for `read_or_generate_charged_struct`. | Sub-step 2 Phase 1 only (`read_or_generate_charged_struct` disk-cache hit path).  Not consumed downstream — purely a re-run accelerator. |
| `All_charged_structures/` | `{reduced_formula}/{mp_id}.cif` | All MP structures that share a given reduced formula, downloaded in bulk by `find_structures` the first time that formula is encountered.  Used to map a charged framework to every MP id whose composition + structure matches.  A `.populated` sentinel inside each subdir flags "fully populated" for the multi-worker fast path. | Sub-step 2 Phase 1 (`find_structures`); sub-step 3 fallback for newly-added charged rows (`_load_row_struct` fallback path); Stage 4 framework comparison (`load_charged_frameworks`). |

`structs/` and `All_charged_structures/` are inputs to Stage 4; `charged_structs/` is internal to Stage 3.  All three are safe to delete to force a re-download on the next run, but doing so wastes time — they are reusable across iterations as long as the MP release pinned by `SPICES_MP_SNAPSHOT_RELEASE` is the same.

In addition, sub-step 3 maintains one **adjacency pickle** per unique-charged-composition file under `{output_dir}/adjacency_cache/{composition}.pkl`.  Each pickle stores the row-pairwise `StructureMatcher`-equivalence adjacency dict in sorted-MP-id-index form, keyed by `(sorted MP-id tuple, matcher signature, schema version)`.  On a re-run the pickle is loaded back into the row order of the current sheet, skipping the O(n²) rebuild — for compositions with a few hundred frameworks this turns a multi-hour rebuild into a sub-second load.  The cache is always-on and invalidates automatically if either the MP-id set or the matcher kwargs in `spices.intercalation.structure.DEFAULT_MATCHER` change.  Delete the directory to force a full rebuild.

## Running Tests

```bash
conda activate MP_api_2026
pytest tests/ -v
```

## Reproducing earlier results against a frozen MP release

The live Materials Project database updates continuously. To reproduce results from an earlier iteration (e.g. a submitted paper), SPICES ships an offline backend that serves queries from a bulk snapshot of a specific MP release on AWS OpenData (`s3://materialsproject-build/collections/<release>/`). The MP API itself does **not** support pinning to a historical database version, so the bulk snapshot is the only route.

### One-time setup

```bash
export MP_API_KEY=your_key_here

# ~0.6 GB; threaded, takes a few minutes. Default release: 2025-06-09.
python -m spices.mp_snapshot.download /path/to/MP_snapshots

# List every release published on the public bucket
python -m spices.mp_snapshot.download --list-releases

# Pin to an older release (e.g. to reproduce a run that predates 2025-06-09)
python -m spices.mp_snapshot.download /path/to/MP_snapshots --release 2024-11-14
```

This downloads the `summary/`, `thermo/thermo_type=GGA_GGA+U/`, and `materials/` collections, freezes the MPContribs `ion_ref_data` project (needed for offline Pourbaix rebuild), and writes a `manifest.json` with provenance. The `GGA_GGA+U_R2SCAN` subroot is **not** pulled — SPICES restricts every thermo query to `GGA_GGA+U` so the R2SCAN-mixed shards would be unused weight. If you have an older snapshot that pre-dates this restriction and want to reclaim disk, `rm -rf <release>/thermo/thermo_type=GGA_GGA+U_R2SCAN/` after upgrading.

The `--release` value is validated against the public bucket before any download begins — a typo exits immediately with a `ValueError` listing the valid releases, rather than silently writing an empty snapshot. If a chosen older release does not publish the `GGA_GGA+U` prefix, `download_collection` logs a warning and returns instead of failing the whole run.

### Using the snapshot backend

Point SPICES at the snapshot either via the `get_client` kwargs or via environment variables — downstream pipeline code is unchanged because every SPICES function takes `mp_client` as a parameter.

```python
from spices.mp_client import get_client
from spices.screening.mp_query import run_mp_query

mp = get_client(
    backend="snapshot",
    snapshot_root="/path/to/MP_snapshots",
    release="2025-06-09",
)
df_step0 = run_mp_query(element="Ca", mp_client=mp)

# or, equivalently, drop the mp_client= kwarg and let the inner
# get_client() pick up the snapshot client via the sticky-default cache:
run_mp_query(element="Ca")

# or set env vars and skip the explicit get_client() call entirely:
#   export SPICES_MP_BACKEND=snapshot
#   export SPICES_MP_SNAPSHOT_ROOT=/path/to/MP_snapshots
#   export SPICES_MP_SNAPSHOT_RELEASE=2025-06-09
```

`SnapshotMPRester` is **strict**: unsupported methods, unsupported thermo types (anything other than `GGA_GGA+U` — including `GGA_GGA+U_R2SCAN` and pure `R2SCAN`), and any network attempt raise `SnapshotBackendError` — preventing silent fallback that would mix data from two database eras in one run.

**Snapshot + multiprocessing** (`run_seawater_screening` with `n_workers > 1`): worker processes do not inherit the parent's sticky-default cache, so the snapshot env vars (`SPICES_MP_BACKEND`, `SPICES_MP_SNAPSHOT_ROOT`, `SPICES_MP_SNAPSHOT_RELEASE`) **must** be exported before launching the run. Passing `mp_client=` alone is not enough — see the `run_seawater_screening` docstring for the full recipe.

The default `materials.thermo.search(...)` call returns one doc per mp-id under `thermo_type=GGA_GGA+U`. SPICES' upstream filter (`thermo_types=[THERMO_TYPE]`) and the snapshot loader's single-subroot indexing keep R2SCAN-corrected energies out of the entry stream entirely, so step-0 output is one row per mp-id with a single `XC Functional` value (`"GGA"` or `"GGA+U"`).

### Reusing earlier Pourbaix caches

The disk-backed `PourbaixCache` works identically with either backend. To pick up entries pickled during an earlier iteration, point it at the old cache dirs:

```python
from spices.pourbaix.cache import PourbaixCache

cache = PourbaixCache(
    entries_dir="/path/to/prior_iteration/saved_pbx_entries",
    pbx_dir="/path/to/prior_iteration/saved_pbx",
)
```

New chemsys queries (e.g. after adding anions like Si or C to `DEFAULT_ANIONS`) cache-miss and are rebuilt offline through `spices.mp_snapshot.pourbaix.build_pourbaix_entries`, then written to the cache alongside the existing pickles.

### Caveat: ion_ref_data freeze date

The MPContribs `ion_ref_data` project is not versioned alongside the bulk S3 dumps — it is a single live document. `download.py` freezes it at download time and records that date in `manifest.json`. In practice aqueous ion thermodynamics are empirical and rarely revised, but peer reviewers should be told the freeze date. If MPContribs was updated since the target release, offline Pourbaix results will differ slightly from live results on the same release.

## Data

Cached Pourbaix diagram data files (pickle format, ~1.3 GB per stage) are archived on Zenodo for reproducibility. They are excluded from the git repository via `.gitignore`. To use existing caches, point the `PourbaixCache` to the directory containing your `saved_pbx/` and `saved_pbx_entries/` folders (see the snapshot section above for the `backend="snapshot"` workflow).

## TODO

### Duplicate `(Charged MP id, Discharged MP id)` rows in Stage 3 `All_voltage_plateaus.xlsx`

Stage 3's per-element `All_voltage_plateaus.xlsx` can contain multiple rows that share the same `(Charged MP id, Discharged MP id)` key with identical `V vs SHE (V)`, `Total metastability area fraction`, and `pH range metastability criterion 4` values — they differ only in `Plot number` (and appear at consecutive plot numbers, e.g. 721/722, 188/189/190).

Observed on the Li run (`cached_MP/Li/2025-06-09/GGA_GGA+U/step3/Li_intercalation/All_voltage_plateaus.xlsx`, 948 raw rows / 940 unique pair keys → 8 extra rows from 7 duplicated keys):

| Pair | Framework | Copies | V (V vs SHE) |
|---|---|---|---|
| mp-1177031 → mp-849623 | FeP2O7 | 2 | +0.271 |
| mp-1177781 → mp-1177512 | VFe(P2O7)2 | 2 | −0.084 |
| mp-752954 → mp-759149 | CrCo3O8 | 2 | +0.231 |
| **mp-756480 → mp-19419** | **FeO2** | **3** | +0.155 |
| mp-760033 → mp-772945 | Mn2(PO4)3 | 2 | +0.751 |
| mp-762265 → mp-762292 | VO2 | 2 | −0.787 |
| mp-768531 → mp-768584 | MnV(PO4)3 | 2 | +0.271 |

Likely cause: during pseudo-binary convex-hull plateau extraction, multiple equivalent two-phase segments between the same charged/discharged endpoints emit separate plateau rows with the same voltage. Decide whether dedup belongs in the plateau-extraction code (`spices.intercalation.hull` / `spices.intercalation.voltage`) or in the per-composition Excel writer, and apply the same logic to other working ions.

## License

MIT

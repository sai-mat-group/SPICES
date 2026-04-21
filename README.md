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
5. **Statistical analysis** -- Generate aggregate statistics, Venn diagrams, and publication-ready figures

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
from spices.mp_client import set_api_key
from spices.screening.mp_query import run_mp_query
from spices.screening.dilute_water import run_dilute_water_evaluation
from spices.screening.seawater import run_seawater_screening

# Set your API key
set_api_key("your_api_key_here")

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
# unless save_component_graphs=False.
from spices.screening.intercalation import run_intercalation_screening

combined = run_intercalation_screening(
    element="Ca",
    input_path="Ca_seawater_results.xlsx",
    output_dir="Ca_intercalation/",
)

# Stage 4: Compare frameworks across working ions
from spices.screening.framework_comparison import run_framework_comparison

stage3_dirs = {e: f"{e}_intercalation/" for e in ["Ca", "K", "Li", "Mg", "Na"]}
result = run_framework_comparison(
    element="Ca",
    stage3_dirs=stage3_dirs,
    output_dir="Ca_framework_comparison/",
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

    pourbaix/                  # Core Pourbaix diagram analysis
        composition.py         # Chemical composition manipulation
        geometry.py            # Geometric overlap calculations
        stability.py           # Thermodynamic stability checks (Criterion 4 default)
        decomposition.py       # Decomposition energy calculations
        metastability.py       # Grid-based metastability searches
        selection_criteria.py  # Optional: all 8 selection criteria with docs
        cache.py               # Pourbaix entry/diagram caching (disk + memory)
        plotting.py            # Heat maps and phase region plots
        thermo.py              # XC functional and energy correction utilities

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
        framework_comparison.py # Stage 4: cross-element framework comparison
        statistics.py          # Stage 5: aggregate statistics and figures

    mp_snapshot/               # Offline MP backend for reproducing frozen releases
        download.py            # AWS OpenData S3 puller (threaded, ~1.14 GB for 2025-06-09)
        loader.py              # Indexed reader over gz-JSONL shards
        pourbaix.py            # Offline get_pourbaix_entries via MP's documented algorithm
        client.py              # SnapshotMPRester — strict drop-in for MPRester

scripts/                       # CLI wrappers and HPC submission templates
    run_seawater_screening.py  # CLI for Stage 2 (supports multiprocessing)
    run_intercalation_screening.py  # CLI for Stage 3
    run_framework_comparison.py # CLI for Stage 4 (graph + optional set comparison)
    run_statistics.py          # CLI for Stage 5
    submit_seawater.sh         # SLURM submission template
    submit_intercalation.sh    # SLURM submission template
```

## HPC / Multiprocessing

Stages 2-3 support HPC execution via CLI scripts:

```bash
# Stage 2: seawater screening (parallel, 8 cores)
python scripts/run_seawater_screening.py Ca input.xlsx -o output.xlsx -n 8

# Stage 3: intercalation voltage screening
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/

# Stage 3 with per-plateau pH-overlap PDFs, without component graphs
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ \
    --save-plots --no-component-graphs

# Stage 4: framework comparison (graph-based by default)
python scripts/run_framework_comparison.py Ca --stage3-dir ./intercalation_results/

# Stage 4 with optional set-based comparison
python scripts/run_framework_comparison.py Ca --stage3-dir ./intercalation_results/ --sets

# HPC submission (edit the .sh files first)
sbatch scripts/submit_seawater.sh
sbatch scripts/submit_intercalation.sh
```

See `scripts/README.md` for details.

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

# ~1.14 GB; threaded, takes a few minutes. Default release: 2025-06-09.
python -m spices.mp_snapshot.download /path/to/MP_snapshots
```

This downloads the `summary/`, `thermo/thermo_type=GGA_GGA+U/`, and `materials/` collections, freezes the MPContribs `ion_ref_data` project (needed for offline Pourbaix rebuild), and writes a `manifest.json` with provenance.

### Using the snapshot backend

Point SPICES at the snapshot either via the `get_client` kwargs or via environment variables — downstream pipeline code is unchanged because every SPICES function takes `mp_client` as a parameter.

```python
from spices.mp_client import get_client

mp = get_client(
    backend="snapshot",
    snapshot_root="/path/to/MP_snapshots",
    release="2025-06-09",
)
# or: export SPICES_MP_BACKEND=snapshot SPICES_MP_SNAPSHOT_ROOT=/path/to/MP_snapshots
```

`SnapshotMPRester` is **strict**: unsupported methods, unsupported thermo types (e.g. `R2SCAN`), and any network attempt raise `SnapshotBackendError` — preventing silent fallback that would mix data from two database eras in one run.

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

## License

MIT

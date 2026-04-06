# SPICES

**S**eawater **P**ourbaix **I**ntercalation **C**ation **E**lectrode **S**creening

A Python package for high-throughput screening of cathode materials for desalination applications using Pourbaix diagram analysis and Materials Project data.

## Overview

SPICES implements a 5-stage computational pipeline for identifying promising cathode materials for electrochemical desalination:

1. **Dilute-limit filter** -- Screen materials for stability in pure water using Pourbaix diagram analysis
2. **Seawater filter** -- Evaluate stability under seawater ionic conditions
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
from spices.screening.dilute_water import run_dilute_water_screening
from spices.screening.seawater import run_seawater_screening

# Set your API key
set_api_key("your_api_key_here")

# Stage 1: Run dilute-water screening for calcium cathodes
df = run_dilute_water_screening(
    element="Ca",
    output_path="Ca_screening_results.xlsx",
)

# Stage 2: Run seawater screening on Stage 1 output
df_sea = run_seawater_screening(
    element="Ca",
    input_path="Ca_screening_results.xlsx",
    output_path="Ca_seawater_results.xlsx",
)

# Stage 2 with multiprocessing (48 cores)
df_sea = run_seawater_screening(
    element="Ca",
    input_path="Ca_screening_results.xlsx",
    output_path="Ca_seawater_results.xlsx",
    n_workers=48,
)

# Stage 3: Run intercalation voltage screening on Stage 2 output
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

## Package Structure

```
spices/
    __init__.py
    config.py                  # Constants and default parameters
    mp_client.py               # Materials Project API client management
    dataframe_ops.py           # DataFrame-level stability update functions
    io.py                      # Logging configuration

    pourbaix/                  # Core Pourbaix diagram analysis
        composition.py         # Chemical composition manipulation
        geometry.py            # Geometric overlap calculations
        stability.py           # Thermodynamic stability checks
        decomposition.py       # Decomposition energy calculations
        metastability.py       # Grid-based metastability searches
        cache.py               # Pourbaix entry/diagram caching (disk + memory)
        plotting.py            # Heat maps and phase region plots
        thermo.py              # XC functional and energy correction utilities

    intercalation/             # Intercalation voltage analysis
        voltage.py             # Voltage calculation from DFT energies
        hull.py                # Pseudo-binary convex hull filtering
        structure.py           # Charged structure gen, StructureMatcher grouping

    screening/                 # Pipeline stage modules
        dilute_water.py        # Stage 1: pure water screening
        seawater.py            # Stage 2: seawater stability screening
        intercalation.py       # Stage 3: intercalation voltage filtering
        framework_comparison.py # Stage 4: cross-element framework comparison
        statistics.py          # Stage 5: aggregate statistics and figures

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

## Data

Cached Pourbaix diagram data files (pickle format, ~1.3 GB per stage) are archived on Zenodo for reproducibility. They are excluded from the git repository via `.gitignore`. To use existing caches, point the `PourbaixCache` to the directory containing your `saved_pbx/` and `saved_pbx_entries/` folders.

## License

MIT

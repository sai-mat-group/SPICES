# SPICES Scripts

CLI wrappers and HPC submission templates for running SPICES pipeline stages.

## Files

| Script | Purpose |
|--------|---------|
| `run_seawater_screening.py` | CLI wrapper for Stage 2 seawater screening |
| `run_intercalation_screening.py` | CLI wrapper for Stage 3 intercalation voltage screening |
| `submit_seawater.sh` | SLURM template for Stage 2 |
| `submit_intercalation.sh` | SLURM template for Stage 3 |

## Local Usage

```bash
# Sequential (single core)
python scripts/run_seawater_screening.py Ca stage1_results.xlsx -o seawater_results.xlsx

# Parallel (8 cores)
python scripts/run_seawater_screening.py Ca stage1_results.xlsx -o seawater_results.xlsx -n 8

# See all options
python scripts/run_seawater_screening.py --help

# Stage 3: Intercalation voltage screening
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/

# With plot saving
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ --save-plots

# See all options
python scripts/run_intercalation_screening.py --help
```

## HPC Usage (SLURM)

1. Edit `submit_seawater.sh`:
   - Set `CONDA_PATH` to your conda installation
   - Set `ENV_NAME` to the environment with SPICES installed
   - Set `ELEMENT`, `INPUT`, `OUTPUT`, `NCORES`, etc.

2. Submit:
   ```bash
   sbatch scripts/submit_seawater.sh
   ```

3. Monitor:
   ```bash
   squeue -u $USER
   tail -f spices_sea_*.log
   ```

## Output Files

When running with multiple workers, intermediate files are saved to:

- `runtime_sheets/{element}_core{n}_part{k}.xlsx` — per-worker checkpoints (every 100 rows)
- `runtime_sheets/{element}_core{n}_final.xlsx` — per-worker final output
- `process_log_core{n}.txt` — per-worker log files
- Final merged output at the path specified by `-o`

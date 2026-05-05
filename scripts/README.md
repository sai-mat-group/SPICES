# SPICES Scripts

CLI wrappers and HPC submission templates for running SPICES pipeline stages.

## Files

| Script | Purpose |
|--------|---------|
| `run_seawater_screening.py` | CLI wrapper for Stage 2 seawater screening |
| `run_intercalation_screening.py` | CLI wrapper for Stage 3 intercalation voltage screening |
| `submit_seawater.sh` | SLURM template for Stage 2 |
| `submit_intercalation.sh` | SLURM template for Stage 3 |

Stage 2 and Stage 3 sub-step 2 (sea-Pourbaix evaluation of newly discovered charged structures) parallelize via `-n N`.  Stage 3 sub-steps 1, 3, and 4 run sequentially in the parent.

## Local Usage

```bash
# Stage 2 — sequential (single core)
python scripts/run_seawater_screening.py Ca stage1_results.xlsx -o seawater_results.xlsx

# Stage 2 — parallel (8 cores).  Internally runs as Phase 1 (parallel
# pre-build of unique Pourbaix entries + sea diagrams, no two workers
# build the same diagram) + Phase 2 (per-row metastability eval, hits
# the on-disk cache).
python scripts/run_seawater_screening.py Ca stage1_results.xlsx -o seawater_results.xlsx -n 8

# Stage 2 — resume an interrupted run.  Phase 1 skips chemsys/sea-comp
# keys whose pickle already exists; Phase 2 skips already-evaluated rows.
# Without --restart, the script refuses to overwrite an output xlsx
# that already has evaluated rows.
python scripts/run_seawater_screening.py Ca stage1_results.xlsx -o seawater_results.xlsx -n 8 --restart

# See all options (incl. --phase1-max-retries / --phase1-retry-delay)
python scripts/run_seawater_screening.py --help

# Stage 3 — sequential
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/

# Stage 3 — parallel sub-step 2 (8 cores), resume from checkpoint
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ -n 8
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ -n 8 --restart

# With plot saving
python scripts/run_intercalation_screening.py Ca seawater_results.xlsx -o Ca_intercalation/ --save-plots

# See all options
python scripts/run_intercalation_screening.py --help
```

## HPC Usage (SLURM)

1. Edit `submit_seawater.sh` or `submit_intercalation.sh`:
   - Set `CONDA_PATH` to your conda installation
   - Set `ENV_NAME` to the environment with SPICES installed
   - Set `ELEMENT`, `INPUT`, `OUTPUT`/`OUTPUT_DIR`, `NCORES`, etc.

2. Submit:
   ```bash
   sbatch scripts/submit_seawater.sh
   sbatch scripts/submit_intercalation.sh
   ```

3. Monitor:
   ```bash
   squeue -u $USER
   tail -f spices_sea_*.log
   tail -f spices_int_*.log
   ```

## Output Files

When running with multiple workers, intermediate files are saved to:

- `runtime_sheets/{element}_core{n}_part{k}.xlsx` — per-worker checkpoints (every 100 rows)
- `runtime_sheets/{element}_core{n}_final.xlsx` — per-worker final output
- `process_logs/process_log_core{n}.txt` — per-worker log files
- `{element}_evaluated_backup.xlsx` — backup of evaluated rows (created on restart)
- Final merged output at the path specified by `-o`

Runtime sheets are automatically cleaned up after a successful run.

For Stage 3 sub-step 2, the same `runtime_sheets/` and `process_logs/` directories are used (rooted at the intercalation `output_dir`).  The `--restart` flag resumes from per-row checkpoints.

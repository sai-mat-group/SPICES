#!/bin/bash
#SBATCH --job-name=spices_sea
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --qos=normal
#SBATCH --time=23:59:00
#SBATCH --output=spices_sea_%j.log
#SBATCH --export=ALL
#SBATCH --no-requeue

# ============================================================
# SPICES Stage 2: Seawater Screening — SLURM Submission Script
# ============================================================
#
# Usage:
#   1. Edit the "User configuration" section below
#   2. Submit: sbatch scripts/submit_seawater.sh
#
# Resuming an interrupted run:
#   Add --restart to the python invocation below.  This skips
#   chemsys/sea_comp keys whose Pourbaix pickles already exist on disk
#   in Phase 1 and skips already-evaluated rows in Phase 2 — only the
#   remaining unique builds + pending rows are distributed across all
#   workers.  Without --restart, the script refuses to overwrite an
#   output file that already has evaluated rows.
#
# Available QOS options (match --time accordingly):
#   debug:  < 1 hour
#   short:  < 6 hours
#   normal: < 1 day
#   medium: < 2 days
#   long:   < 1 week
# ============================================================

# ---- User configuration ----
CONDA_PATH="/path/to/anaconda3"         # e.g. /opt/apps/anaconda3 or ~/anaconda3
ENV_NAME="MP_api_2026"                  # conda environment with SPICES installed
ELEMENT="Ca"                            # Working ion: Ca, K, Li, Mg, or Na
INPUT="Ca_cathodes_pure_water_metastability.xlsx"
OUTPUT="Ca_cathodes_seawater_metastability.xlsx"
NCORES=48                               # Should match --ntasks above
CACHE_DIR="."                           # Directory with saved_pbx/ and saved_pbx_entries/
DECOMP_ENERGY=0.3                       # Decomposition energy threshold (eV)
SPICES_DIR="."                          # Project root (where scripts/ lives)
# -----------------------------

# Activate conda environment
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "============================================"
echo "SPICES Seawater Screening"
echo "Element:    ${ELEMENT}"
echo "Input:      ${INPUT}"
echo "Output:     ${OUTPUT}"
echo "Workers:    ${NCORES}"
echo "Cache dir:  ${CACHE_DIR}"
echo "Started at: $(date)"
echo "============================================"

python "${SPICES_DIR}/scripts/run_seawater_screening.py" \
    "${ELEMENT}" \
    "${INPUT}" \
    -o "${OUTPUT}" \
    -n "${NCORES}" \
    --decomp-energy "${DECOMP_ENERGY}" \
    --cache-dir "${CACHE_DIR}" \
    --metastability-points

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

#!/bin/bash
#SBATCH --job-name=spices_int
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --qos=normal
#SBATCH --time=23:59:00
#SBATCH --output=spices_int_%j.log
#SBATCH --export=ALL
#SBATCH --no-requeue

# ============================================================
# SPICES Stage 3: Intercalation Voltage Screening — SLURM
# ============================================================
#
# Usage:
#   1. Edit the "User configuration" section below
#   2. Submit: sbatch scripts/submit_intercalation.sh
#
# Multi-worker note: only sub-step 2 (sea-Pourbaix evaluation of
# newly discovered charged structures) parallelizes; steps 1, 3, and 4
# run sequentially in the parent.  Use NCORES=12-48 depending on
# available RAM (~2.5 GB per worker — see seawater.py docstring).
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
INPUT="Ca_cathodes_seawater_metastability.xlsx"
OUTPUT_DIR="Ca_intercalation"
NCORES=48                               # Should match --ntasks above
CACHE_DIR="."                           # Directory with saved_pbx/ and saved_pbx_entries/
DECOMP_ENERGY=0.3                       # Decomposition energy threshold (eV)
SELECTION_CRITERIA="4"                  # e.g. "4" or "1,2,3,4"
SPICES_DIR="."                          # Project root (where scripts/ lives)
# -----------------------------

# Activate conda environment
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "============================================"
echo "SPICES Intercalation Voltage Screening"
echo "Element:    ${ELEMENT}"
echo "Input:      ${INPUT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Workers:    ${NCORES}"
echo "Cache dir:  ${CACHE_DIR}"
echo "Started at: $(date)"
echo "============================================"

python "${SPICES_DIR}/scripts/run_intercalation_screening.py" \
    "${ELEMENT}" \
    "${INPUT}" \
    -o "${OUTPUT_DIR}" \
    -n "${NCORES}" \
    --decomp-energy "${DECOMP_ENERGY}" \
    --cache-dir "${CACHE_DIR}" \
    --selection-criteria "${SELECTION_CRITERIA}"
    --save-plots

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

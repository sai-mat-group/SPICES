#!/bin/bash
#SBATCH --job-name=spices_int
#SBATCH --nodes=1
#SBATCH --ntasks=12
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
# ============================================================

# ---- User configuration ----
CONDA_PATH="/path/to/anaconda3"
ENV_NAME="MP_api_2026"
ELEMENT="Ca"
INPUT="Ca_cathodes_seawater_metastability.xlsx"
OUTPUT_DIR="Ca_intercalation"
CACHE_DIR="."
SPICES_DIR="."
# -----------------------------

source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "============================================"
echo "SPICES Intercalation Voltage Screening"
echo "Element:    ${ELEMENT}"
echo "Input:      ${INPUT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Cache dir:  ${CACHE_DIR}"
echo "Started at: $(date)"
echo "============================================"

python "${SPICES_DIR}/scripts/run_intercalation_screening.py" \
    "${ELEMENT}" \
    "${INPUT}" \
    -o "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --save-plots

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

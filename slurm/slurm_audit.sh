#!/bin/bash
#SBATCH --job-name=stn-audit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
cd "$HOME/scratch/stn"
mkdir -p logs

PY="/home/haizhe/conda/envs/SSN/bin/python"
BIDS_ROOT="$HOME/scratch/stn/raw"
OUT_DIR="$HOME/scratch/stn/audit"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PY=$PY"
ls -l "$PY"
"$PY" -c "import sys, mne; print('sys.executable=', sys.executable); print('mne=', mne.__version__)"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "BIDS_ROOT = $BIDS_ROOT"
echo "OUT_DIR   = $OUT_DIR"

"$PY" -u scripts/01_audit_cohort.py \
    --bids-root "$BIDS_ROOT" \
    --out "$OUT_DIR"

echo "Audit complete."
ls -lh "$OUT_DIR"/*.tsv

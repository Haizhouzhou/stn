#!/bin/bash
#SBATCH --job-name=stn-bursts
#SBATCH --output=logs/stn-bursts_%j.out
#SBATCH --error=logs/stn-bursts_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
mkdir -p logs
source /home/haizhe/conda/envs/SSN/bin/activate
cd "$HOME/scratch/stn"

python scripts/03_extract_bursts.py \
    --extracted-root "$HOME/scratch/stn/extracted" \
    --audit-tsv "$HOME/scratch/stn/audit/cohort_summary.tsv" \
    --updrs-tsv "$HOME/scratch/stn/raw/participants_updrs_off.tsv" \
    --out "$HOME/scratch/stn/results/bursts" \
    --band-mode both \
    --jobs 10 \
    "$@"

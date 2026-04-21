#!/bin/bash
#SBATCH --job-name=stn-adm
#SBATCH --output=logs/stn-adm_%j.out
#SBATCH --error=logs/stn-adm_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
cd "$HOME/scratch/stn"
mkdir -p logs results/adm results/figures/04_adm results/tables/04_adm

source activate SSN
which python

python scripts/04_adm_sweep.py \
    --extracted "$HOME/scratch/stn/extracted" \
    --bursts "$HOME/scratch/stn/results/bursts" \
    --cohort-summary "$HOME/scratch/stn/audit/cohort_summary.tsv" \
    --out "$HOME/scratch/stn/results" \
    --jobs 12

#!/bin/bash
#SBATCH --job-name=stn-bursts-dryrun
#SBATCH --output=logs/stn-bursts-dryrun_%j.out
#SBATCH --error=logs/stn-bursts-dryrun_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
mkdir -p logs
source activate SSN
cd "$HOME/scratch/stn"

python scripts/03_extract_bursts.py \
    --extracted-root "$HOME/scratch/stn/extracted" \
    --audit-tsv "$HOME/scratch/stn/audit/cohort_summary.tsv" \
    --out "$HOME/scratch/stn/results/bursts_dryrun" \
    --band-mode both \
    --subject sub-0cGdk9 \
    --jobs 1

#!/bin/bash
#SBATCH --job-name=stn-attach-ann
#SBATCH --output=logs/stn-attach-ann_%j.out
#SBATCH --error=logs/stn-attach-ann_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
mkdir -p logs
source /home/haizhe/conda/envs/SSN/bin/activate
cd "$HOME/scratch/stn"

python scripts/06_attach_annotations.py \
    --extracted-root "$HOME/scratch/stn/extracted" \
    --audit-tsv "$HOME/scratch/stn/audit/cohort_summary.tsv" \
    "$@"

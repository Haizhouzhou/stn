#!/bin/bash
#SBATCH --job-name=stn-bursts
#SBATCH --output=logs/stn-bursts_%j.out
#SBATCH --error=logs/stn-bursts_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
cd "$HOME/scratch/stn"
mkdir -p logs results/bursts results/figures/03_bursts results/tables/03_bursts

source activate SSN
which python
python -c "import mne, scipy, numpy; print('env OK')"

python scripts/03_extract_bursts.py \
    --extracted "$HOME/scratch/stn/extracted" \
    --cohort-summary "$HOME/scratch/stn/audit/cohort_summary.tsv" \
    --out "$HOME/scratch/stn/results"

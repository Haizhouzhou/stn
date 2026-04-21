#!/bin/bash
#SBATCH --job-name=stn-snn
#SBATCH --output=logs/stn-snn_%j.out
#SBATCH --error=logs/stn-snn_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
cd "$HOME/scratch/stn"
mkdir -p logs results/snn results/figures/05_snn results/tables/05_snn

source activate SSN
which python
python -c "import torch, rockpool; print('torch', torch.__version__, '| rockpool', rockpool.__version__)"

python scripts/05_snn_sweep.py \
    --extracted "$HOME/scratch/stn/extracted" \
    --bursts "$HOME/scratch/stn/results/bursts" \
    --adm "$HOME/scratch/stn/results/adm" \
    --cohort-summary "$HOME/scratch/stn/audit/cohort_summary.tsv" \
    --out "$HOME/scratch/stn/results" \
    --jobs 12

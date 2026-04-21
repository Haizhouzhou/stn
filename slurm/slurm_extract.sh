#!/bin/bash
#SBATCH --job-name=stn-extract
#SBATCH --output=logs/stn-extract_%j.out
#SBATCH --error=logs/stn-extract_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal

set -euo pipefail
cd "$HOME/scratch/stn"
mkdir -p logs extracted

source activate SSN
which python
python -c "import mne, joblib, psutil; print('mne', mne.__version__, '| joblib', joblib.__version__)"

python scripts/02_extract_lfp.py \
    --bids-root "$HOME/scratch/stn/raw" \
    --cohort-summary "$HOME/scratch/stn/audit/cohort_summary.tsv" \
    --out "$HOME/scratch/stn/extracted" \
    --resample-hz 1000 \
    --jobs 4

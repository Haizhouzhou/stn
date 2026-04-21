#!/usr/bin/env bash
# Transfer extracted LFP data from HPC to local WSL.
# Run this FROM YOUR LOCAL WSL shell, not from the HPC.
#
# Usage: edit HPC_USER, HPC_HOST, HPC_PATH below, then:
#    bash 03_transfer_to_local.sh
#
# rsync is resumable. If the connection drops, just re-run.

set -euo pipefail

# ---- EDIT THESE ----
HPC_USER="haizhe"
HPC_HOST="u24-login-2"           # or the actual login node address
HPC_PATH="/home/$HPC_USER/scratch/stn/extracted"
LOCAL_DEST="$HOME/stn/data"      # inside WSL -> \\wsl.localhost\Ubuntu\home\linux\stn\data
# --------------------

mkdir -p "$LOCAL_DEST"

echo "Transferring from $HPC_USER@$HPC_HOST:$HPC_PATH"
echo "                -> $LOCAL_DEST"

# --partial  = keep partial files on interrupt (so re-run resumes)
# --progress = show per-file progress
# -a         = archive mode (preserves perms, times, symlinks)
# -z         = compress on the wire (useful for .tsv/.json; minimal effect on .fif which is already binary)
# -h         = human-readable sizes
# --exclude  = leave behind raw .fif and per-subject temp logs we don't need locally
rsync -avhz --partial --progress \
    --exclude 'extract_logs/*.log' \
    "$HPC_USER@$HPC_HOST:$HPC_PATH/" \
    "$LOCAL_DEST/"

echo
echo "Transfer complete. Local layout:"
du -sh "$LOCAL_DEST" || true
ls -lh "$LOCAL_DEST" | head

echo
echo "Verification (run inside the WSL env):"
echo "  python -c \"import mne; raw=mne.io.read_raw_fif('$LOCAL_DEST/sub-0cGdk9/ses-PeriOp/meg/sub-0cGdk9_ses-PeriOp_task-HoldL_acq-MedOff_run-1_lfp.fif', preload=False); print(raw.info)\""

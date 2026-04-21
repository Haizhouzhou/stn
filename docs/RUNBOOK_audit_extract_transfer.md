# AUDIT → EXTRACT → TRANSFER Runbook

This is the practical workflow for going from 162 GB of raw `ds004998` on the HPC
to a lean ~3-5 GB working set on your local WSL machine.

## Why do it this way

- The raw `.fif` files contain 306 MEG channels you don't need for the SNN project;
  keeping only the 8 LFP + 4 EMG + 2 EOG channels and downsampling from ~2.4 kHz to
  1 kHz yields a ~40× size reduction with zero information loss for beta-burst work.
- HPC scratch is fast and cheap; your laptop is neither. Do the heavy I/O where the
  data already lives.
- Raw stays on scratch. If you later want MEG-LFP coupling (Paper 2), it's there.

## Phase 2A — Audit (HPC, login or SLURM short partition)

Files: `01_audit_cohort.py`, `slurm_audit.sh`

Audit reads **only file headers and TSV sidecars** — no signal is loaded. So it's
fast and safe to run even on a login node, though SLURM is cleaner.

```bash
# From ~/scratch/stn
# (Ensure 01_audit_cohort.py is here or adjust path in the SLURM script.)

# Option A: SLURM (recommended; adjust partition for your cluster)
sbatch slurm_audit.sh

# Option B: interactive (also fine; takes <15 min)
python 01_audit_cohort.py --bids-root ~/scratch/stn/raw --out ~/scratch/stn/audit
```

After it completes, inspect:

```bash
cd ~/scratch/stn/audit
column -ts $'\t' cohort_summary.tsv | less -S       # cohort-level view
column -ts $'\t' exclusion_log.tsv                  # excluded subjects
ls per_subject/                                     # per-subject JSON
```

**What to look for in `cohort_summary.tsv`:**

| Column | What you want |
|---|---|
| `include` | Count how many are `True`. Target: 15-20. |
| `exclusion_reasons` | Plausible and honest (e.g., "medoff_duration<180s") |
| `has_rest_medoff` | ≥ 10 subjects with True (baseline beta stats need this) |
| `duration_*_medoff_s` | At least 3 min per included subject |
| `lfp_contacts_left/right` | Typically 4-8 per side for directional leads |
| `sfreq_inconsistent` | Should be False for every subject |
| `updrs_off_total` / `updrs_on_total` | Present for all subjects |

If `include` is TRUE for more than ~15 subjects, you're in good shape. If it's below 10,
your thresholds are too strict (relax `MIN_MEDOFF_DURATION_S` or `MAX_BAD_FRACTION` in
`01_audit_cohort.py`) — but document the change.

**Commit** `cohort_summary.tsv` and `exclusion_log.tsv` to your project git **now**, before
running extraction. Never retroactively change these without a note in CHANGELOG.

## Phase 2B — Extraction (HPC, SLURM array job)

Files: `02_extract_lfp.py`, `slurm_extract.sh`

Extraction **does** load signals (needs ~8-16 GB RAM per fif, and reads ~1.5 GB per file).
Use a SLURM array so subjects run in parallel.

```bash
# From ~/scratch/stn
mkdir -p logs

# Submit array job — one task per subject
sbatch slurm_extract.sh

# Check progress
squeue -u $USER
tail -f logs/stn-extract_*.out
```

Expected wall-clock: ~30-90 min per subject; in parallel across the array, total
~2-3 h from queue submission.

When it finishes, verify:

```bash
cd ~/scratch/stn/extracted
find . -name '*_lfp.fif' | wc -l                   # expect ~50-80 files
du -sh .                                             # expect 3-5 GB
du -sh sub-*/  | sort -h                             # per-subject size distribution
cat extract_logs/extract_all.json | jq '.[] | {subject_id, n_runs: (.runs|length), n_errors: (.errors|length)}'
```

Any subject with `n_errors > 0` needs investigation — check the per-subject log.

## Phase 2C — Transfer to local WSL

File: `03_transfer_to_local.sh`

Edit the HPC_USER/HOST lines, then run **from your local WSL shell**:

```bash
# On WSL
cd ~/stn                      # or wherever your project lives
bash 03_transfer_to_local.sh  # edit host/user first
```

`rsync` is resumable — if it dies halfway, just re-run and it picks up.

Expected transfer time depends on your uplink; at 50 MB/s you'll move 5 GB in ~2 min.

## Final local layout

```
/home/linux/stn/data/
├── sub-0cGdk9/
│   └── ses-PeriOp/meg/
│       ├── sub-0cGdk9_ses-PeriOp_task-HoldL_acq-MedOff_run-1_lfp.fif
│       ├── sub-0cGdk9_ses-PeriOp_task-HoldL_acq-MedOff_run-1_events.tsv
│       └── ...
├── sub-2IhVOz/
├── ...
└── extract_logs/
    ├── extract_all.json       # provenance of every extraction
    └── sub-*.json
```

Each `*_lfp.fif` contains: monopolar LFP (renamed), bipolar LFP pairs (LFP-left-01,
LFP-left-12, etc.), EMG, EOG, STIM — at 1 kHz, with bad channels marked, annotations
and events preserved.

## Sanity check (run in WSL)

```python
import mne
raw = mne.io.read_raw_fif(
    "/home/linux/stn/data/sub-0cGdk9/ses-PeriOp/meg/"
    "sub-0cGdk9_ses-PeriOp_task-HoldL_acq-MedOff_run-1_lfp.fif",
    preload=True,
)
print("sfreq:", raw.info["sfreq"])              # expect 1000.0
print("ch_names:", raw.ch_names)                 # should include LFP-*, EMG*, EOG*
print("channel types:", set(raw.get_channel_types()))  # no 'mag' or 'grad'
print("duration:", raw.times[-1], "s")
print("bads:", raw.info["bads"])
```

If this all looks right, Phase 2 is done and you can start Phase 3 (ground-truth
beta-burst labeling) from the master plan using the local extracted data.

## Failure modes & recovery

| Problem | Cause | Fix |
|---|---|---|
| `allow_maxshield=True` warning | Normal for this dataset | Ignore |
| Subject not in `extracted/` | Marked `include=False` in audit | Rerun extraction for that subject explicitly: `python 02_extract_lfp.py --subject sub-XXX --cohort-summary /dev/null ...` |
| `No adjacent bipolar pairs recognized` warning | Montage TSV column names don't match regex | Inspect the subject's montage.tsv; adjust the regex in `make_bipolar_pairs()` |
| Extraction OOM | Resampling large fif on 8 GB node | Bump `--mem` to 24G in slurm_extract.sh |
| Split files skipped | Correct — MNE handles split-02 via split-01 | No action |
| rsync stalls | HPC firewall / idle timeout | Add `-e "ssh -o ServerAliveInterval=60"` to rsync call |

## When to re-audit

Re-run the audit if:
- You adjust inclusion thresholds
- Dataset version on OpenNeuro is bumped
- You realize a subject should be excluded after Phase 3 (document it, re-run, commit
  the new `cohort_summary.tsv`)

Never silently edit the file by hand.

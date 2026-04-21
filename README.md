# stnbeta

STN beta-burst detection pipeline for ds004998, targeting eventual deployment on the
DYNAP-SE1 neuromorphic chip. The pipeline runs on the UZH S3IT teaching cluster and
produces ground-truth burst labels, ADM spike encodings, and SNN classifier results
that feed into the hardware bring-up phase.

Full context and rationale: [STN_BetaBurst_DynapSE1_MasterPlan.md](STN_BetaBurst_DynapSE1_MasterPlan.md)

---

## Phase status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset acquisition + BIDS audit | ✅ Complete |
| 2 | LFP extraction (monopolar + bipolar, 1 kHz) | ✅ Complete — 20 subjects, 108 runs, 7.2 GB |
| 3 | Tinkhauser beta-burst ground truth | 🔲 Next |
| 4 | ADM eventization sweep | 🔲 Pending |
| 5 | SNN architecture exploration (simulation) | 🔲 Pending |
| 6 | DYNAP-SE1 hardware bring-up | 🔲 Pending |

---

## Repo layout

```
stn/
├── src/stnbeta/          # importable library (pip install -e .)
│   ├── io/               # BIDS loading, cohort audit
│   ├── preprocessing/    # LFP extraction
│   ├── ground_truth/     # burst labeling (Phase 3)
│   ├── encoding/         # ADM eventization (Phase 4)
│   ├── snn/              # SNN architectures (Phase 5)
│   └── analysis/         # metrics, figures
├── scripts/              # thin CLI wrappers (call src/ functions)
├── slurm/                # SLURM job scripts
├── configs/              # Hydra-style YAMLs
├── results/              # all scientific outputs
├── audit/                # BIDS audit outputs (cohort_summary.tsv etc.)
├── extracted/            # extracted LFP .fif files (7.2 GB, sync-included)
├── raw/                  # full BIDS tree (162 GB, sync-EXCLUDED)
├── docs/                 # decisions.md and other docs
└── tests/                # pytest
```

---

## How to run from scratch

### 0. Install the package

```bash
cd ~/scratch/stn
/home/haizhe/conda/envs/SSN/bin/python -m pip install -e .
```

### 1. Cohort audit

```bash
# Login node is fine — no data loading
python scripts/01_audit_cohort.py \
    --bids-root ~/scratch/stn/raw \
    --out ~/scratch/stn/audit
# or via SLURM:
sbatch slurm/slurm_audit.sh
```

Output: `audit/cohort_summary.tsv`, `audit/runs_detail.tsv`

### 2. LFP extraction

```bash
# Must run on compute node — heavy CPU/RAM
sbatch slurm/slurm_extract.sh
# Resources: 6 CPU / 64 GB / 4 parallel workers / ~14 min wall-clock
```

Output: `extracted/sub-*/ses-PeriOp/meg/*_lfp.fif` (7.2 GB)

### 3. Beta-burst ground truth (Phase 3)

```bash
sbatch slurm/slurm_bursts.sh
```

Output: `results/bursts/`

### 4. ADM sweep (Phase 4)

```bash
sbatch slurm/slurm_adm.sh
```

Output: `results/adm/`

### 5. SNN sweep (Phase 5)

```bash
sbatch slurm/slurm_snn.sh
```

Output: `results/snn/`

---

## Sync to local (rsync recipe)

To mirror everything except the raw data to a laptop for local analysis or
DYNAP-SE1 hardware bring-up:

```bash
rsync -avz --progress \
  --exclude='raw/' \
  --exclude='extracted.broken_*/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='wandb/' \
  --exclude='logs/*.out' \
  --exclude='logs/*.err' \
  --exclude='.git/objects/pack/' \
  haizhe@cluster:/home/haizhe/scratch/stn/ \
  ~/projects/stn/
```

After sync, on the laptop:

```bash
cd ~/projects/stn
conda env create -f environment.yml   # or: pip install -e .
```

---

## Key references

- Tinkhauser et al. (2017) *Brain* — beta burst definition and threshold method
- Little et al. (2019) *npj Parkinson's Disease* — clinical relevance
- ds004998 on OpenNeuro — dataset

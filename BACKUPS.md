# Backup Files

All `.bak*` files are stored under `backups/` as surgical checkpoints. Do not
delete without verifying the live script contains the intended state.

| File | Created | Source file | Reason |
|------|---------|-------------|--------|
| `backups/01_audit_cohort.py.bak_lfp` | 2026-04-21 08:15 | `scripts/01_audit_cohort.py` | Before adding LFP channel counting and montage-coverage checks to the audit script |
| `backups/02_extract_lfp.py.bak_rename_fix` | 2026-04-21 09:15 | `scripts/02_extract_lfp.py` | Before fixing `build_lfp_rename_map` to correctly parse the ds004998 explicit schema (left/right_contacts_old → new columns) |
| `backups/02_extract_lfp.py.bak_joblib` | 2026-04-21 10:36 | `scripts/02_extract_lfp.py` | Before adding `--jobs N` / `joblib.Parallel` in-process parallelism and the psutil memory guard |
| `backups/02_extract_lfp.py.bak_pick_fix` | 2026-04-21 12:12 | `scripts/02_extract_lfp.py` | Before tightening `lfp_names` derivation to montage-mapped channels only (Option B fix for sub-8RgPiG EEG005/EEG037 survivors) and adding the hardening assertion |
| `backups/slurm_audit.sh.bak` | 2026-04-20 23:31 | `slurm/slurm_audit.sh` | Before modifying audit SLURM script |
| `backups/slurm_extract.sh.bak` | 2026-04-20 23:30 | `slurm/slurm_extract.sh` | Before converting from SLURM array job (per-subject) to fat single-job workflow |
| `backups/slurm_extract.sh.bak_fatjob` | 2026-04-21 11:14 | `slurm/slurm_extract.sh` | Before reducing resources from 256 GB / 12 CPU / --jobs 10 to 64 GB / 6 CPU / --jobs 4 to reduce queue wait |

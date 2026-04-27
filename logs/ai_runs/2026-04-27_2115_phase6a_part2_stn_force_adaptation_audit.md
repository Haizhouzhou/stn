# Phase 6A Part 2 STN Force Adaptation Audit

Date: 2026-04-27 21:15 CEST

Task: Dataset/code/data audit for the Herz/Groppa/Brown force adaptation package.

Commit before task: `49b1cdefe83907e44098960df3b837bed7b0de6a`

## User Prompt

```text
Implement Phase 6A Part 2: dataset/code/data audit for the Herz/Groppa/Brown force adaptation package under cambium/Force_Scripts.

The full prompt specified: use the existing stn_env only; do not create or install environments; do not modify or copy files in cambium/Force_Scripts; do not execute MATLAB, FieldTrip, LME, permutation statistics, or preprocessing; inventory package files, module folders, expected components, subject/example coverage, MAT and tabular schemas, MATLAB code/dependencies, ZIP members, privacy/governance risks, and Phase 6B recommendations; create the requested reports under reports/phase6_stn_force_adaptation_herz_2023_audit; create scripts/phase6_audit_stn_force_adaptation_herz_2023.py; create this AI run log; validate with py_compile, report inspection, git diff --check, >5 MB report check, and git status; stage only explicit safe paths and commit if validation passes.
```

## Operational Plan

1. Run the requested startup checks: location, repo root, git state, stn_env Python, repo tree, and Force_Scripts tree.
2. Inspect existing Phase 6A PPN audit conventions and representative Force_Scripts documentation/MATLAB code.
3. Implement a robust audit-only Python script with optional scipy/h5py/openpyxl fallbacks and no writes inside `cambium/Force_Scripts`.
4. Generate all requested CSV/JSON/Markdown report artifacts.
5. Validate the script and report outputs, then checkpoint only explicit audit/report/log paths.

## Files Inspected

- `scripts/phase6_audit_ppn_he_tan_2021.py`
- `reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md`
- `logs/ai_runs/2026-04-27_1715_phase6a_ppn_dataset_audit.md`
- `logs/ai_runs/INDEX.md`
- `cambium/Force_Scripts/Force_Scripts/Description.rtf`
- Representative MATLAB files in `1BehavioralData`, `2LocalFieldPotentialData`, and `3DBSEffectsBehavior`
- Generated report artifacts under `reports/phase6_stn_force_adaptation_herz_2023_audit/`

## Files Modified Or Created

- Created `scripts/phase6_audit_stn_force_adaptation_herz_2023.py`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/README_dataset_audit.md`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/dataset_manifest.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/module_folder_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/expected_component_matrix.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/subject_file_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/mat_file_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/tabular_file_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/matlab_code_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/matlab_dependency_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/data_schema_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/privacy_governance_inventory.csv`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/audit_findings.json`
- Created `reports/phase6_stn_force_adaptation_herz_2023_audit/phase6a_part2_commands_run.txt`
- Created this AI run log
- Updated `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`
- `git rev-parse --show-toplevel`
- `git status --short`
- `conda run -n stn_env python -V`
- `source /scratch/haizhe/stn/start_stn.sh && python -V`
- `find . -maxdepth 3 -type d | sort | head -200`
- `find . -maxdepth 3 -type f | sort | head -200`
- `find cambium/Force_Scripts -maxdepth 4 -type d | sort | head -300`
- `find cambium/Force_Scripts -maxdepth 4 -type f | sort | head -300`
- `rg --files scripts | rg 'audit|phase6|ppn|PPN|dataset'`
- `find reports -maxdepth 3 -type d | sort | head -200`
- `find reports -maxdepth 3 -type f | sort | head -300`
- `sed -n '1,220p' scripts/phase6_audit_ppn_he_tan_2021.py`
- `sed -n '1,220p' reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md`
- `sed -n '1,200p' logs/ai_runs/2026-04-27_1715_phase6a_ppn_dataset_audit.md`
- `sed -n '1,120p' logs/ai_runs/INDEX.md`
- `sed -n '1,160p' cambium/Force_Scripts/Force_Scripts/1BehavioralData/ExtractData.m`
- `sed -n '1,160p' cambium/Force_Scripts/Force_Scripts/2LocalFieldPotentialData/GetLFP_FirstLevel.m`
- `sed -n '1,180p' cambium/Force_Scripts/Force_Scripts/3DBSEffectsBehavior/GetEvents_Stim.m`
- `python - <<'PY' ... inspect Description.rtf text ... PY`
- `source /scratch/haizhe/stn/start_stn.sh && python -m py_compile scripts/phase6_audit_stn_force_adaptation_herz_2023.py`
- `source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6_audit_stn_force_adaptation_herz_2023.py --data-root cambium/Force_Scripts --out-dir reports/phase6_stn_force_adaptation_herz_2023_audit --paper-fs 2048 --lfp-analysis-fs 200 --stim-binary-fs 1000`
- `ls -lh reports/phase6_stn_force_adaptation_herz_2023_audit`
- `head -80 reports/phase6_stn_force_adaptation_herz_2023_audit/README_dataset_audit.md`
- `git diff --check`
- `find reports/phase6_stn_force_adaptation_herz_2023_audit -type f -size +5M -print`
- `git status --short`
- Short Python JSON/CSV summary checks over `audit_findings.json`, `mat_file_inventory.csv`, `subject_file_inventory.csv`, and `expected_component_matrix.csv`

## Validation Results

- `conda run -n stn_env python -V` failed because `stn_env` is not registered as a named Conda environment on this host.
- `source /scratch/haizhe/stn/start_stn.sh && python -V` confirmed repo-local `/scratch/haizhe/stn/stn_env/bin/python` Python 3.12.3.
- `py_compile` passed for `scripts/phase6_audit_stn_force_adaptation_herz_2023.py`.
- The audit script completed and wrote all requested report artifacts.
- `git diff --check` passed.
- `find reports/phase6_stn_force_adaptation_herz_2023_audit -type f -size +5M -print` produced no output.
- No files were modified under `cambium/Force_Scripts`.

## Key Counts

- Total physical files under audited root: 94
- MATLAB `.m` path entries: 78
- Real MATLAB code files excluding macOS metadata stubs: 39
- `.mat` files: 4
- Tabular/text files: 2
- ZIP files under audited root: 0
- Subject-like IDs found: 32
- Expected components found: 32
- Expected components missing: 3 (`computeCohen_d`, `jblill`, `shadedErrorBar`)
- Example subjects found: `Kont01`, `Kont02`

## Findings

- The package contains the expected module folders, MATLAB code, `Description.rtf`, and example behavioral data for `Kont01` and `Kont02`.
- The full original 16-patient/15-control cohort and full raw STN LFP participant dataset are not present, consistent with the paper data-availability caveat.
- Two example force MAT files (`Kont01_RL_Force.mat`, `Kont02_RL_Force.mat`) are MATLAB v7.3/HDF5 files. `scipy.io.loadmat` cannot read them and `h5py` is missing in `stn_env`; they are recorded as unreadable for this audit.
- No scan/imaging-like files were detected.
- FieldTrip, MATLAB Statistics/LME functions, permutation/cluster code, and downloaded helper references are present in the MATLAB code inventory.
- Downloaded helper scripts `computeCohen_d`, `jblill`, and `shadedErrorBar` were referenced/expected but not found as files in the extracted package.

## Limitations

- The audit did not execute MATLAB or validate FieldTrip runtime readiness.
- The v7.3 force MAT payloads were not deeply inspected because no approved HDF5 MAT reader is available in `stn_env`.
- Dependency parsing is heuristic, not a full MATLAB parser.
- macOS metadata files under `__MACOSX` and `._*` are inventoried but excluded from real-code interpretation.

## Recommended Phase 6B Next Step

Validate behavioral event parsing on the readable `Kont01/Kont02` `*_Newevents.mat` files and resolve approved HDF5/v7.3 reading for the `*_RL_Force.mat` files before force-trace parsing. Treat the package as a code/methods reference plus example behavioral data, not as a full raw STN LFP dataset unless another data release is provided.

## Final Codex Output

The final response will report the created script/report paths, key counts, critical issues, exact commands, recommended Phase 6B action, and commit/push status.

Commit after task: reported in final response to avoid self-referential commit hash churn.

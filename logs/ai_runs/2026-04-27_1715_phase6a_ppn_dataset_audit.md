# Phase 6A PPN Dataset Audit

Date: 2026-04-27 17:15 CEST

Task: Phase 6A dataset and data audit for the He/Tan 2021 PPN gait dataset.

Commit before task: `ec3a80d`

## User Prompt

```text
You are working in the existing STN repo. Start Phase 6A: dataset and data audit for the PPN gait dataset:

Dataset path:
~/scratch/stn/cambium/Data_Code_PPN_JNeurosci_2021

Repo:
~/scratch/stn

Environment constraint:
Do NOT create a new conda env, venv, poetry env, or install a new environment.
Use the existing environment `stn_env` only.
Prefer:
  conda run -n stn_env python ...
or, if already active:
  python ...
Do not pip/conda install anything unless absolutely necessary; if a dependency is missing, first implement a fallback or document the missing dependency.

Dataset:
"LFPs and EEGs from patients with Parkinson's disease or multiple system atrophy during gait"
Authors: He S, Tan H
Paper: He et al., 2021, Gait-Phase Modulates Alpha and Beta Oscillations in the Pedunculopontine Nucleus.

Goal:
Perform Phase 6A dataset and data audit only. Do not do full preprocessing, modeling, PSD, coherence, gait-phase analysis, or figure reproduction. The audit should establish what files exist, what protocols are present, what channels/modalities are available, whether markers are valid, and what issues need to be resolved before Phase 6B.

Required deliverables:
1. reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md
2. reports/phase6_ppn_he_tan_2021_audit/dataset_manifest.csv
3. reports/phase6_ppn_he_tan_2021_audit/patient_task_matrix.csv
4. reports/phase6_ppn_he_tan_2021_audit/mat_file_inventory.csv
5. reports/phase6_ppn_he_tan_2021_audit/channel_inventory.csv
6. reports/phase6_ppn_he_tan_2021_audit/marker_inventory.csv
7. reports/phase6_ppn_he_tan_2021_audit/signal_sanity_summary.csv
8. reports/phase6_ppn_he_tan_2021_audit/audit_findings.json
9. reports/phase6_ppn_he_tan_2021_audit/phase6a_commands_run.txt

Final response should include:
- A concise summary of what was created.
- Counts: patients found, total files, mat files, raw-like mat files, txt files, code files.
- Any critical issues: missing referenced MAT files, marker problems, missing channels, privacy flags.
- The exact report directory.
- The exact script path.
- The exact commands run.
```

The full prompt also specified expected patients/protocols, raw-data safety rules, MATLAB field/marker conventions, sampling-rate handling, channel/modality heuristics, privacy flags, and the requested startup commands.

## Operational Plan

1. Run the requested repo, git, environment, and file-layout checks.
2. Inspect dataset structure and optional Python MAT readers without installing dependencies.
3. Create a metadata-first Phase 6A audit script under `scripts/`.
4. Run the audit with the existing repo-local `stn_env` Python, because `conda run -n stn_env` is not registered.
5. Validate generated report artifacts, compile the script, and summarize the audit findings.
6. Create this AI run log, update the index, and checkpoint only explicit safe paths.

## Files Inspected

- `start_stn.sh`
- `logs/ai_runs/INDEX.md`
- Dataset directory tree under `cambium/Data_Code_PPN_JNeurosci_2021`
- Dataset `Description.txt`
- Representative protocol TXT files under `RawData/PD01` and `RawData/MSA01`
- Generated Phase 6A report files under `reports/phase6_ppn_he_tan_2021_audit/`

## Files Modified Or Created

- Created `scripts/phase6_audit_ppn_he_tan_2021.py`
- Created `reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md`
- Created `reports/phase6_ppn_he_tan_2021_audit/dataset_manifest.csv`
- Created `reports/phase6_ppn_he_tan_2021_audit/patient_task_matrix.csv`
- Created `reports/phase6_ppn_he_tan_2021_audit/mat_file_inventory.csv`
- Created `reports/phase6_ppn_he_tan_2021_audit/channel_inventory.csv`
- Created `reports/phase6_ppn_he_tan_2021_audit/marker_inventory.csv`
- Created `reports/phase6_ppn_he_tan_2021_audit/signal_sanity_summary.csv`
- Created `reports/phase6_ppn_he_tan_2021_audit/audit_findings.json`
- Created `reports/phase6_ppn_he_tan_2021_audit/phase6a_commands_run.txt`
- Created this AI run log
- Updated `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`
- `git rev-parse --show-toplevel`
- `git status --short`
- `conda run -n stn_env python -V`
- `find . -maxdepth 3 -type d | sort | head -200`
- `find . -maxdepth 3 -type f | sort | head -200`
- `ls -la`
- `ls -la cambium`
- `ls -ld cambium/Data_Code_PPN_JNeurosci_2021 cambium/Data_Code_PPN_JNeurosci_2021.zip`
- `source /scratch/haizhe/stn/start_stn.sh && python -V`
- `find scripts -maxdepth 2 -type d | sort | head -100`
- `find reports audit outputs results/tables -maxdepth 2 -type d 2>/dev/null | sort | head -200`
- `sed -n '1,220p' start_stn.sh`
- `find cambium/Data_Code_PPN_JNeurosci_2021/Data_Code_PPN_JNeurosci_2021 -maxdepth 4 -type d | sort | head -200`
- `find cambium/Data_Code_PPN_JNeurosci_2021/Data_Code_PPN_JNeurosci_2021 -maxdepth 4 -type f | sort | head -200`
- `/scratch/haizhe/stn/stn_env/bin/python - <<'PY' ... optional dependency check ... PY`
- `find cambium/Data_Code_PPN_JNeurosci_2021/Data_Code_PPN_JNeurosci_2021 -type f | sed 's/.*//' | wc -l`
- `find cambium/Data_Code_PPN_JNeurosci_2021/Data_Code_PPN_JNeurosci_2021 -type f -printf '%s\t%p\n' | sort -nr | head -30`
- `sed -n '1,40p' .../RawData/PD01/RestSitting.txt`
- `sed -n '1,40p' .../RawData/PD01/Walk.txt`
- `sed -n '1,40p' .../RawData/MSA01/StepSitting.txt`
- `sed -n '1,80p' .../Description.txt`
- `find .../RawData -type f | sed 's#^.*/##' | sort | head -100`
- `/scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6_audit_ppn_he_tan_2021.py`
- `/scratch/haizhe/stn/stn_env/bin/python scripts/phase6_audit_ppn_he_tan_2021.py --data-root cambium/Data_Code_PPN_JNeurosci_2021 --out-dir reports/phase6_ppn_he_tan_2021_audit --paper-fs 2048`
- `perl -pi -e 's/\r$//' reports/phase6_ppn_he_tan_2021_audit/*.csv && /scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6_audit_ppn_he_tan_2021.py`
- `ls -lh reports/phase6_ppn_he_tan_2021_audit`
- `head -50 reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md`
- Several targeted `grep`, `head`, and short Python JSON/CSV summary checks over generated report files.
- `git diff --check -- scripts/phase6_audit_ppn_he_tan_2021.py reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md reports/phase6_ppn_he_tan_2021_audit/phase6a_commands_run.txt`

## Validation Results

- `conda run -n stn_env python -V` failed with `EnvironmentLocationNotFound`; no new environment was created and no packages were installed.
- `source /scratch/haizhe/stn/start_stn.sh && python -V` confirmed repo-local `stn_env` Python 3.12.3.
- Optional dependency check: scipy available, numpy available, pandas available, h5py missing.
- The audit ran successfully with `/scratch/haizhe/stn/stn_env/bin/python`.
- `py_compile` passed for `scripts/phase6_audit_ppn_he_tan_2021.py`.
- `git diff --check` passed for the audit script and edited report command files.
- Generated reports are small text/CSV/JSON artifacts; no raw patient folders were modified.

## Audit Summary

- Patients found: 11 of 11 expected (`PD01`-`PD07`, `MSA01`-`MSA04`)
- Total files: 461
- MAT files: 126
- Raw-like MAT files: 52
- TXT files: 57
- Code files: 240
- Missing referenced MAT files: none
- Privacy/scan-like files: none flagged
- h5py missing: 8 v7.3/HDF5 result MAT files were classified as unreadable via scipy and documented; raw-like classic MAT files were still audited.
- Channel issues: 18, mainly MSA stepping `ChannelType` count mismatches, duplicate `Ligh` channel names in MSA03 stepping, PD05 missing Cz/Fz guess, and unexpected PD03 walk-like files without accelerometer/MarkerWalk.
- Marker issues: 2, both PD03 walk-like MAT files missing `MarkerWalk`.
- Signal issues: 0 recorded in structured findings.

## Remaining Risks

- The audit uses top-level sampling-rate key search and records `2048 Hz` as paper default/assumed when no file-level scalar sampling-rate key is found.
- h5py is not installed in `stn_env`, so large v7.3 generated result files are inventoried but not deeply inspected.
- PD03 has walk-like MAT files despite FreeWalking being expected only for PD01-PD02; Phase 6B should decide whether to exclude, reclassify, or document them.
- Many orphan MAT files are likely generated results; Phase 6B should separate raw recordings from derived outputs before preprocessing.

## Final Codex Output

The final response will report the created script and report directory, the key counts/issues, the exact commands run, and commit/push status.

Commit after task: reported in final response to avoid self-referential commit hash churn.

# Phase 6A Dataset Audit: PPN He/Tan 2021

## Scope

Dataset: LFPs and EEGs from patients with Parkinson's disease or multiple system atrophy during gait, associated with He et al. 2021, Gait-Phase Modulates Alpha and Beta Oscillations in the Pedunculopontine Nucleus.

This is a Phase 6A audit only. It inventories files, expected protocol coverage, MATLAB schemas, channel/modalities, markers, lightweight signal sanity, Matlab code, and privacy/governance risks. It does not run preprocessing, modeling, PSD, coherence, gait-phase analysis, or figure reproduction.

## Dataset Root And Environment

- Dataset root audited: `/scratch/haizhe/stn/cambium/Data_Code_PPN_JNeurosci_2021`
- Audit timestamp: `2026-04-27T17:13:01+02:00`
- Environment: existing repo `stn_env` only. `conda run -n stn_env python -V` was attempted but the named conda env was not registered; the repo-local `/scratch/haizhe/stn/stn_env` Python was used via `start_stn.sh` or direct path.
- Optional readers: scipy `available`, h5py `missing`.

## Patients And Protocol Coverage

- Expected patients: `PD01, PD02, PD03, PD04, PD05, PD06, PD07, MSA01, MSA02, MSA03, MSA04`
- Observed patients: `MSA01, MSA02, MSA03, MSA04, PD01, PD02, PD03, PD04, PD05, PD06, PD07`
- Missing expected patient folders: `none`

| patient | group | RestSitting | RestStanding | StepSitting | StepStanding | FreeWalking | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PD01 | PD | Y | Y | N | N | Y | 6_orphan_mat_files |
| PD02 | PD | Y | Y | N | N | Y | 8_orphan_mat_files |
| PD03 | PD | Y | Y | N | N | Y | 4_orphan_mat_files;observed_unexpected_FreeWalking |
| PD04 | PD | Y | Y | N | N | N | 4_orphan_mat_files |
| PD05 | PD | Y | Y | N | N | N | 2_orphan_mat_files |
| PD06 | PD | Y | Y | N | N | N | 4_orphan_mat_files |
| PD07 | PD | Y | Y | N | N | N | 4_orphan_mat_files |
| MSA01 | MSA | Y | N | Y | Y | N | 10_orphan_mat_files |
| MSA02 | MSA | Y | N | Y | Y | N | 10_orphan_mat_files |
| MSA03 | MSA | Y | N | Y | N | N | 8_orphan_mat_files |
| MSA04 | MSA | Y | N | Y | N | N | 8_orphan_mat_files |

## File Inventory Summary

- Total files: `461`
- MAT files: `126`
- TXT files: `57`
- Code files: `240`
- Files >= 100000000 bytes: `27`

| extension | count |
| --- | --- |
| .cpp | 2 |
| .doxy | 1 |
| .hpp | 2 |
| .html | 2 |
| .jpg | 3 |
| .m | 240 |
| .mat | 126 |
| .md | 3 |
| .png | 15 |
| .rtf | 1 |
| .txt | 57 |
| [none] | 9 |

Patient-level file counts:

| patient | file_count |
| --- | --- |
| MSA01 | 22 |
| MSA02 | 22 |
| MSA03 | 16 |
| MSA04 | 16 |
| PD01 | 18 |
| PD02 | 20 |
| PD03 | 12 |
| PD04 | 12 |
| PD05 | 10 |
| PD06 | 12 |
| PD07 | 12 |

## Raw-Like MAT Schema Summary

- Total MAT files inspected: `126`
- Raw-like MAT files with `ChannelName`, `ChannelType`, and `data`: `52`
- Unreadable MAT files: `8`
- Explicit sampling-rate variables found: `0`
- When no file-level sampling-rate variable was found, `2048 Hz` was recorded as `paper default / assumed` only.

## Channel And Modality Summary

- Channel rows inventoried: `1332`

| modality_guess | channel_rows |
| --- | --- |
| EEG | 286 |
| PPN_LFP | 766 |
| accelerometer | 34 |
| force_pressure | 52 |
| unknown | 194 |

- Channel issues recorded: `18`

## Marker Summary

- Marker objects inventoried: `60`
- Marker rows with issues: `0`
- Rest recordings may have no meaningful markers; missing movement markers are flagged only for stepping/free-walking raw-like files.

## Signal Sanity Summary

- Raw-like files with signal sanity rows: `52`
- Signal rows with notes/issues: `2`
- Metrics are aggregate/bounded sanity checks only: shapes, missingness, Inf/NaN counts, all-zero/constant channel flags, and aggregate finite-value ranges. No raw time-series samples are written.

## Matlab Code And Reproducibility Audit

- Matlab scripts/functions: `240`
- Obvious main scripts: `Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/ppn_analysis_JNeurosci.m`
- Readme/description files: `Data_Code_PPN_JNeurosci_2021/Description.txt;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/HMM-MAR-master/LICENSE;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/HMM-MAR-master/README.md;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/HMM-MAR-master/utils/hidden_state_inference/README.md;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/README.html;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/README.m;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/README.md;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_01.png;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_02.png;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_03.png;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_04.png;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_05.png;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_06.png;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_07.png;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a/readmeExtras/README_08.png`
- Toolbox-like directories: `Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/HMM-MAR-master;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/a_wavelet_pow_pha.m;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/cbrewer;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/fun_Factorization.m;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/fun_clusterMassTest.m;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/fun_coh_fft.m;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/fun_coh_wavelet.m;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/getSignifClusters.m;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/kakearney-boundedline-pkg-8179f9a;Data_Code_PPN_JNeurosci_2021/PreprocessedData&Code/Toolbox/modulationIndex.m`

Feature keywords observed in Matlab code:

| feature | observed |
| --- | --- |
| preprocessing | Y |
| psd | Y |
| coherence | Y |
| gait_phase_detection | Y |
| modulation_index | Y |
| permutation_statistics | Y |
| figure_generation | Y |

## Privacy And Governance Findings

- DICOM/NIfTI/MRI/CT-like files flagged: `0`
- Flagged files: `none`
- No scan-like files were opened or processed deeply by this audit.
- Patient identifiers observed in paths are summarized at the folder/filename level only; this audit does not inspect private imaging metadata.

## Known Issues And Ambiguities

- Missing referenced MAT files: `none`
- Orphan MAT files not referenced by protocol TXT files: `68`. Many are expected generated results or toolbox outputs; review `patient_task_matrix.csv` before using them.
- Channel issues: `18`; see `channel_inventory.csv` and `audit_findings.json`.
- Marker issues: `2`; see `marker_inventory.csv` and `audit_findings.json`.
- Signal issues: `0`; see `signal_sanity_summary.csv` and `audit_findings.json`.
- Overall status: `completed_with_audit_issues`

## Recommendations For Phase 6B

- Use only raw-like MAT files with ChannelName, ChannelType, data, and valid movement markers as Phase 6B candidates.
- Resolve orphan MAT files by separating raw recordings from generated PSD, gait modulation, permutation, HMM, and IC2019 outputs.
- Treat 2048 Hz as paper default only until a file-level sampling-rate source is confirmed or the original acquisition metadata is located.
- Review channel/modality flags before choosing PPN LFP, EEG Cz/Fz, force, and accelerometer channels for Phase 6B.
- Validate Marker and MarkerWalk timing against data duration before any gait-phase segmentation.
- Port/reuse Matlab preprocessing only after Phase 6B defines a reproducible Python/Brian2-compatible input contract.

## Exact Commands Run

See also `phase6a_commands_run.txt`.

- `pwd`
- `git rev-parse --show-toplevel`
- `git status --short`
- `conda run -n stn_env python -V`
- `find . -maxdepth 3 -type d | sort | head -200`
- `find . -maxdepth 3 -type f | sort | head -200`
- `source /scratch/haizhe/stn/start_stn.sh && python -V`
- `/scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6_audit_ppn_he_tan_2021.py`
- `/scratch/haizhe/stn/stn_env/bin/python scripts/phase6_audit_ppn_he_tan_2021.py --data-root cambium/Data_Code_PPN_JNeurosci_2021 --out-dir reports/phase6_ppn_he_tan_2021_audit --paper-fs 2048`
- `/scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6_audit_ppn_he_tan_2021.py`
- `perl -pi -e 's/\r$//' reports/phase6_ppn_he_tan_2021_audit/*.csv && /scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6_audit_ppn_he_tan_2021.py`
- `ls -lh reports/phase6_ppn_he_tan_2021_audit`
- `head -50 reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md`
- `git status --short`

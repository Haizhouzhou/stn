# Phase 6A Part 2 Dataset Audit: Herz/Groppa/Brown Force Adaptation

## Scope

Dataset/package: Subthalamic nucleus correlates of force adaptation.
Paper context: Dynamic modulation of subthalamic nucleus activity facilitates adaptive behavior.

This audit inventories the shared Force_Scripts package and maps files, code, schemas, and visible dependencies to the dataset description. It does not run MATLAB, FieldTrip, preprocessing, LME models, permutation tests, or analysis reproduction.

## Dataset Root And Environment

- Dataset root audited: `/scratch/haizhe/stn/cambium/Force_Scripts`
- Repository root: `/scratch/haizhe/stn`
- Audit timestamp: `2026-04-27T21:16:32+02:00`
- Python executable: `/scratch/haizhe/stn/stn_env/bin/python`
- Python version: `3.12.3`
- Environment: existing repo `stn_env` only. `conda run -n stn_env python -V` was attempted first and was not registered on this host; subsequent Python commands used `source /scratch/haizhe/stn/start_stn.sh && python ...`.
- Optional readers: scipy `available`, h5py `missing: ModuleNotFoundError: No module named 'h5py'`, openpyxl `missing: ModuleNotFoundError: No module named 'openpyxl'`.

## Data Availability Caveat

The paper states that participant consent did not allow depositing the full original participant dataset. This audit therefore does not treat absence of the full 16-patient/15-control raw cohort as a critical failure. The shared package is assessed as code plus minimum/example data unless the package itself claims that full raw data are present.

## File And Folder Summary

- Physical files under audited root: `94`
- MATLAB `.m` path entries: `78`
- Real MATLAB code files excluding macOS metadata stubs: `39`
- `.mat` files: `4`
- Tabular/text files (`.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, `.rtf`): `2`
- ZIP files under audited root: `0`
- macOS metadata stubs: `50`

Extension counts:

| extension | count |
| --- | --- |
| .ds_store | 5 |
| .m | 78 |
| .mat | 4 |
| .rtf | 2 |
| [none] | 5 |

Analysis-part counts:

| analysis_part | count |
| --- | --- |
| behavioral | 14 |
| dbs_behavior | 15 |
| dbs_lfp | 6 |
| lfp_off_stim | 8 |
| source_data | 4 |
| unknown | 47 |

## Module Mapping

| module_guess | relative_dir | files | m_files | mat_files | tabular_text | notes |
| --- | --- | --- | --- | --- | --- | --- |
| unknown | . | 94 | 78 | 4 | 2 | includes macOS metadata files |
| unknown | Force_Scripts | 49 | 39 | 4 | 1 | includes macOS metadata files |
| Behavioral force analysis / Figure 1 | Force_Scripts/1BehavioralData | 14 | 13 | 0 | 0 | includes macOS metadata files |
| STN LFP first/second-level analysis / Figures 2 and 3 | Force_Scripts/2LocalFieldPotentialData | 8 | 7 | 0 | 0 | includes macOS metadata files |
| DBS timing effects on behavior / Figures 4 and 5 | Force_Scripts/3DBSEffectsBehavior | 15 | 14 | 0 | 0 | includes macOS metadata files |
| DBS timing effects on STN LFP / Figures 4 and 5 | Force_Scripts/4DBSEffectsLocalFieldPotential | 6 | 5 | 0 | 0 | includes macOS metadata files |
| Example behavioral data | Force_Scripts/ExampleData | 4 | 0 | 4 | 0 |  |
| macos_metadata | __MACOSX | 45 | 39 | 0 | 1 | includes macOS metadata files |

## Expected Component Coverage

- Expected components covered by exact or strong near-match file names: `32`
- Expected components missing as files: `3`

| component | covered | path_or_match | notes |
| --- | --- | --- | --- |
| CompareLevodopaDemographicsMVC | Y | Force_Scripts/1BehavioralData/CompareDemographicsLevodopaMVC.m | covered_by_near_or_normalized_filename_match |
| GetEvents_PD | Y | Force_Scripts/1BehavioralData/GetEvents_PD.m |  |
| GetEvents_HC | Y | Force_Scripts/1BehavioralData/GetEvents_HC.m |  |
| ExtractData | Y | Force_Scripts/1BehavioralData/ExtractData.m;Force_Scripts/3DBSEffectsBehavior/ExtractData.m |  |
| ExtractData_HC | Y | Force_Scripts/1BehavioralData/ExtractData_HC.m |  |
| GetForce_PD | Y | Force_Scripts/1BehavioralData/GetForce_PD.m |  |
| GetForce_HC | Y | Force_Scripts/1BehavioralData/GetForce_HC.m |  |
| Forceparameters | Y | Force_Scripts/1BehavioralData/Forceparameters.m;Force_Scripts/3DBSEffectsBehavior/Forceparameters.m |  |
| Forces_within | Y | Force_Scripts/1BehavioralData/Forces_within.m;Force_Scripts/3DBSEffectsBehavior/Forces_within.m |  |
| Stat_within | Y | Force_Scripts/1BehavioralData/Stat_within.m |  |
| Forces_across | Y | Force_Scripts/1BehavioralData/Forces_across.m;Force_Scripts/3DBSEffectsBehavior/Forces_across.m |  |
| stat_across | Y | Force_Scripts/1BehavioralData/stat_across.m |  |
| Plot_Stats | Y | Force_Scripts/1BehavioralData/PlotStats.m | covered_by_near_or_normalized_filename_match |
| GetLFP_FirstLevel | Y | Force_Scripts/2LocalFieldPotentialData/GetLFP_FirstLevel.m |  |
| MakeMontage_AllBipolar | Y | Force_Scripts/2LocalFieldPotentialData/MakeMontage_AllBipolar.m |  |
| EpochData_TF | Y | Force_Scripts/2LocalFieldPotentialData/EpochData_TF.m |  |
| GetLFP_SecondLevel_PlotSpectra | Y | Force_Scripts/2LocalFieldPotentialData/GetLFP_SecondLevel_PlotSpectra.m |  |
| GetLFP_SecondLevel_LME | Y | Force_Scripts/2LocalFieldPotentialData/GetLFP_SecondLevel_LME.m |  |
| PermTests_LME | Y | Force_Scripts/2LocalFieldPotentialData/PermTests_LME.m |  |
| GetLFP_SecondLevel_controlLME | Y | Force_Scripts/2LocalFieldPotentialData/GetLFP_SecondLevel_controlLME.m |  |
| GetEvents_Stim | Y | Force_Scripts/3DBSEffectsBehavior/GetEvents_Stim.m |  |
| GetForce_Stim | Y | Force_Scripts/3DBSEffectsBehavior/GetForce_Stim.m |  |
| GetToS | Y | Force_Scripts/3DBSEffectsBehavior/GetToS.m |  |
| ToS_DownsampleBinaryRemoveRamp | Y | Force_Scripts/3DBSEffectsBehavior/ToS_DownsampleBinaryRemoveRamp.m |  |
| ToS_WindowedStim | Y | Force_Scripts/3DBSEffectsBehavior/ToS_WindowedStim.m |  |
| ToS_Windowed_nexttrial | Y | Force_Scripts/3DBSEffectsBehavior/ToS_Windowed_nexttrial.m |  |
| Plot_ToS | Y | Force_Scripts/3DBSEffectsBehavior/PlotToS.m | covered_by_near_or_normalized_filename_match |
| PermTests_ToS | Y | Force_Scripts/3DBSEffectsBehavior/PermTests_ToS.m;Force_Scripts/4DBSEffectsLocalFieldPotential/PermTests_ToS.m |  |
| GetLFP_FirstLevel_Stim | Y | Force_Scripts/4DBSEffectsLocalFieldPotential/GetLFP_FirstLevel_Stim.m |  |
| GetLFP_FirstLevel_Stim_TrigOnset | Y | Force_Scripts/4DBSEffectsLocalFieldPotential/GetLFP_FirstLevel_Stim_TrigOnset.m |  |
| GetLFP_SecondLevel_Stim | Y | Force_Scripts/4DBSEffectsLocalFieldPotential/GetLFP_SecondLevel_Stim.m |  |
| GetLFP_SecondLevel_TrigOnset | Y | Force_Scripts/4DBSEffectsLocalFieldPotential/GetLFP_SecondLevel_Stim_TrigOnset.m | covered_by_near_or_normalized_filename_match |
| computeCohen_d | N |  | not_found_in_package |
| jblill | N |  | not_found_in_package |
| shadedErrorBar | N |  | not_found_in_package |

## Subject And Example Data Coverage

- Expected shared-package example subjects: `Kont01, Kont02`
- Observed example subjects: `Kont01, Kont02`
- Missing example subjects: `none`
- Study-level cohort context: 16 PD patients, 15 analyzed healthy controls after one control exclusion, and 14 patients in the stimulation session.
- Observed full cohort raw data: not present in this package based on file inventory; code references PD and HC IDs, but raw participant LFP/behavioral cohort files were not shared under the audited root.

| subject_id | group | source | n_files | file_types | notes |
| --- | --- | --- | --- | --- | --- |
| Kont01 | example_control | code_reference;filename;path | 2 | .mat | expected example behavioral subject observed |
| Kont02 | example_control | code_reference;filename;path | 2 | .mat | expected example behavioral subject observed |
| Kont03 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont04 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont06 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont07 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont08 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont09 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont10 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont11 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont12 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont13 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont14 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont15 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |
| Kont16 | example_control | code_reference | 0 |  | subject_id_referenced_in_code_only_no_shared_file_observed |

## MAT-File Schema Summary

- MAT files inventoried: `4`
- MAT files loaded via scipy/h5py: `2`
- MAT read failures/stubs: `2`
- FieldTrip-like MAT structs detected: `0`
- Force-like MAT files detected: `4`
- Event-like MAT files detected: `2`

The four real example MAT files are lightweight behavioral/force/event examples; no full raw STN LFP cohort MAT files were observed under the audited root.

## Tabular/Text Schema Summary

- Tabular/text files inventoried: `2`
- The main readable text file is `Description.rtf`, which documents package structure, MATLAB/FieldTrip requirements, and example-data instructions.
- No subject-level CSV/TSV/XLS/XLSX data tables were observed under the audited root.

## MATLAB Code And Dependency Summary

- MATLAB `.m` path entries inventoried: `78`
- Real MATLAB code files excluding macOS metadata stubs: `39`
- MATLAB dependency/call rows: `787`

| dependency_type | count |
| --- | --- |
| FieldTrip | 28 |
| MATLAB_builtin_or_toolbox | 270 |
| downloaded_helper | 11 |
| local_function | 26 |
| statistics_toolbox | 8 |
| unknown | 444 |

FieldTrip calls are present in the LFP modules. LME/statistics and permutation/cluster code are present. This audit records those requirements but does not test MATLAB or FieldTrip availability.

## Sampling-Rate Observations

- Paper default STN LFP acquisition rate: `2048 Hz` (recorded only as paper context unless explicit in file/code).
- LFP analysis/downsampled rate: `200 Hz` appears in MATLAB code as `NewSR=200`/FieldTrip resampling context.
- Stimulation binary/downsampled rate: `1000 Hz` appears in the package description and stimulation processing code context.
- No explicit `fsample` value was detected in the shared example MAT payloads.

## Privacy And Governance Findings

- Privacy/governance rows: `7`
- Scan/imaging-like files flagged: `0`
- Imaging files were not observed under the audited root. Subject-like IDs in filenames/code are anonymized IDs and are reported only at path/count level.

## Known Issues And Ambiguities

- Critical: 2 non-metadata MAT file(s) were unreadable by available readers (Force_Scripts/ExampleData/Kont01_RL_Force.mat, Force_Scripts/ExampleData/Kont02_RL_Force.mat); these are HDF5/v7.3 example force files and h5py is missing in stn_env.
- Noncritical limitation: Expected helper/component files not observed: computeCohen_d, jblill, shadedErrorBar.
- Noncritical limitation: Full original 16-patient/15-control raw cohort data are not present, which is expected from the paper data-availability statement unless another package claims to contain them.
- Noncritical limitation: MATLAB/FieldTrip/statistics-toolbox runtime readiness was not tested because this audit does not run MATLAB.
- Noncritical limitation: The extracted package contains macOS metadata stubs under __MACOSX and ._* files; these are inventoried but not treated as analysis files.
- Noncritical limitation: Some MATLAB calls resolve to external MATLAB/FieldTrip/toolbox/helper dependencies and must be resolved before reproduction.

## Recommendations For Phase 6B

- Phase 6B should validate behavioral force/event parsing on the Kont01/Kont02 example data only, without assuming full cohort raw data.
- Treat this package primarily as a code/reproducibility reference plus example behavioral data, not as a full raw STN LFP dataset.
- Before Phase 6B behavioral force parsing, resolve MATLAB v7.3/HDF5 reading for the Kont01/Kont02 force files by using an approved h5py-capable stn_env or a MATLAB-side schema export.
- Document MATLAB, FieldTrip, Statistics and Machine Learning Toolbox, and missing helper-script requirements before attempting reproduction.
- Create a code-methods map from the MATLAB modules to the STN detector project before porting any analysis logic.

## Exact Commands Run

See also `phase6a_part2_commands_run.txt`.

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
- `sed -n '1,160p' cambium/Force_Scripts/Force_Scripts/1BehavioralData/ExtractData.m`
- `sed -n '1,160p' cambium/Force_Scripts/Force_Scripts/2LocalFieldPotentialData/GetLFP_FirstLevel.m`
- `sed -n '1,180p' cambium/Force_Scripts/Force_Scripts/3DBSEffectsBehavior/GetEvents_Stim.m`
- `python - <<'PY' ... read first bytes of Description.rtf ... PY`
- `source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6_audit_stn_force_adaptation_herz_2023.py --data-root cambium/Force_Scripts --out-dir reports/phase6_stn_force_adaptation_herz_2023_audit --paper-fs 2048 --lfp-analysis-fs 200 --stim-binary-fs 1000`
- `source /scratch/haizhe/stn/start_stn.sh && python -m py_compile scripts/phase6_audit_stn_force_adaptation_herz_2023.py`
- `ls -lh reports/phase6_stn_force_adaptation_herz_2023_audit`
- `head -80 reports/phase6_stn_force_adaptation_herz_2023_audit/README_dataset_audit.md`
- `git diff --check`
- `find reports/phase6_stn_force_adaptation_herz_2023_audit -type f -size +5M -print`
- `git status --short`

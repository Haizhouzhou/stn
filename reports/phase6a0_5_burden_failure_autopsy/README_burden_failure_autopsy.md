# Phase 6A.0.5 Burden Gate Failure Autopsy

## Purpose

Phase 6A.0.5 diagnoses why the Phase 6A.0 internal STN annotation-derived burden gate failed. It does not authorize Brian2, DYNAP, or SNN simulation.

## Why The Phase 6A.0 FAIL Was Meaningful But Not Decisive

The failed gate showed that the lower-bound LOSO burden estimator did not reach the predeclared threshold. This autopsy checks whether that reflects substrate limits, proxy limits, estimator limits, table continuity limits, or subject heterogeneity.

## Inputs Used

- Input table: `/scratch/haizhe/stn/results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- External PPN/Herz datasets and their audit reports were excluded from model inputs.

## Environment And Commands

- Python: `/scratch/haizhe/stn/stn_env/bin/python (3.12.3)`
- Environment route: `source /scratch/haizhe/stn/start_stn.sh && python ...`

## Previous Phase 6A.0 Summary

- `selected_input`: `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- `subject_count`: `20`
- `valid_subject_count`: `20`
- `allowed_feature_count`: `354`
- `best_tau_ms`: `3000.0`
- `median_loso_pearson`: `0.24896977374123425`
- `median_loso_spearman`: `0.2602476064282385`
- `class_prior_baseline`: `0.10720896000980693`
- `shuffled_baseline`: `0.03811309853555812`
- `beta_feature_baseline`: `0.22797459976862877`
- `proxy_summary`: `{"rows": [{"cohens_d": 0.26620585799971574, "n_units": 111, "negative_condition": "MedOn", "negative_mean": 0.4353580560200478, "notes": "subject-condition units; supportive proxy only", "positive_condition": "MedOff", "positive_mean": 0.44567586997621716, "proxy_type": "medication_state", "rank_biserial_or_auc": 0.5957792207792207, "value_column": "mean_predicted_burden"}, {"cohens_d": 0.02957793763893011, "n_units": 71, "negative_condition": "Move", "negative_mean": 0.43687307347642196, "notes": "subject-condition units; supportive proxy only", "positive_condition": "Hold", "positive_mean": 0.4377956218616303, "proxy_type": "task_state", "rank_biserial_or_auc": 0.49047619047619045, "value_column": "mean_predicted_burden"}, {"cohens_d": 0.22874187904695215, "n_units": 75, "negative_condition": "Move", "negative_mean": 0.43687307347642196, "notes": "subject-condition units; supportive proxy only", "positive_condition": "Rest", "positive_mean": 0.4462835460269331, "proxy_type": "task_st`
- `gate_status`: `FAIL`
- `gate_rationale`: `Gate thresholds for PASS/CONDITIONAL_PASS were not met.`

## Data Contract And Continuity Findings

- Overall continuity class: `event_or_candidate_matrix`
- Continuity class counts: `{'event_or_candidate_matrix': 598, 'continuous_medium': 28}`
- If classified as `event_or_candidate_matrix`, causal burden integration over row order is diagnostic pseudo-burden only, not definitive continuous tracking.

- Contract issue: `table is a definitive continuous burden stream` -> `fail` (Phase 6A.0.5 may compute pseudo-burden for diagnostics, but event/candidate matrices are not definitive burden streams.)

## Leakage Reaudit

- Columns reaudit count: `373`
- Allowed feature count after reaudit: `297`
- High-risk allowed features: `0`

## Feature Sets Tested

- `all_allowed_phase6a0`: `297` features; Full allowed feature pool; expensive models may use SGD or fold-local top-k.
- `beta_like`: `118` features; Single-family beta-like diagnostic feature set.
- `non_beta_allowed`: `179` features; Tests whether non-beta dynamics add burden information.
- `phase6a0_selected96_surrogate`: `78` features; Reproduction/surrogate for prior lower-bound estimator.
- `compact_deployable`: `24` features; Exploratory compact set; inferred, not locked by prior ADR.
- `phase5_2b_refined_if_found`: `0` features; Unavailable if Phase 5_2B only names base features not exact Phase 5_2C columns.
- `phase5_2c_minimum_sufficient_if_found`: `2` features; Extra diagnostic exact refined subset found in Phase 5_2C outputs.
- `phase5_2c_refined_candidates_if_found`: `11` features; Extra diagnostic refined candidate set capped at 80 columns for runtime.
- `top_k_train_only_25`: `25` features; Dynamic top-25; selected features differ by held-out subject.
- `top_k_train_only_50`: `50` features; Dynamic top-50; selected features differ by held-out subject.
- `top_k_train_only_100`: `100` features; Dynamic top-100; selected features differ by held-out subject.

## Estimator Ceiling Results

- Best LOSO annotation-burden ceiling: `Pearson 0.9279920389018084, Spearman 0.9162633458684624` from `sklearn_gradient_boosting_ceiling` / `top_k_train_only_25` at tau `5000.0`.
- Best deployable/simple LOSO ceiling: `Pearson 0.8996638222320061, Spearman 0.9011889156151458` from `numpy_ridge_fallback` / `top_k_train_only_100`.
- Best within-subject calibrated ceiling: `Pearson 0.7123009509806877, Spearman 0.645771009412702` from `sklearn_gradient_boosting_ceiling` / `top_k_train_only_25`.
- Best oracle descriptive ceiling: `Pearson 0.8961740633816817, Spearman 0.8973748088893959` from `oracle_descriptive_subject_fit` / `top_k_train_only_50`.

High-capacity models, when present, are offline ceilings only and are not deployable model claims.

## Tau Heterogeneity Results

- Best tau summary: `{'median_metric_by_tau': {'200.0': 0.5860685868652049, '500.0': 0.5990450907884681, '800.0': 0.6067517314897216, '1500.0': 0.6180716847964074, '3000.0': 0.6204160321282977, '5000.0': 0.5956827534547908}, 'best_tau_ms': '3000.0', 'best_tau_median_metric': 0.6204160321282977}`
- Oracle tau rows are descriptive only and are not validation.

## Subject Distribution / Responder Analysis

- Heterogeneity summary: `{'n_subjects': 20, 'n_high_trackable': 20, 'n_moderate_trackable': 0, 'max_adjacent_correlation_gap': 0.04018148545859357, 'median_best_subject_metric': 0.9304700915594498, 'heterogeneity_interpretation': 'uniformly_high_on_diagnostic_metric'}`

## Proxy Ceiling Results

- Proxy summary: `{'status': 'available', 'best_metric': 0.5141324727483771, 'interpretation': 'weak_proxy_ceiling'}`
- Clinical/task proxies remain supportive only, not the primary gate.

## Beta-Baseline Delta

- Best delta row: `{'model_name': 'sklearn_gradient_boosting_ceiling', 'feature_set_name': 'top_k_train_only_25', 'validation_mode': 'LOSO_cross_subject', 'tau_ms': 5000.0, 'median_pearson': 0.9279920389018084, 'median_spearman': 0.9162633458684624, 'beta_baseline_median_pearson': 0.2418910685290822, 'delta_vs_beta': 0.6861009703727261, 'delta_vs_shuffled': 0.7206742697135204, 'delta_vs_class_prior': 0.722249824033488, 'architecture_value_interpretation': 'meaningful high-capacity improvement; not deployable by itself'}`

## Prior-Phase Comparison

- Prior comparison rows: `69`
- Claims not found in source files are recorded as not found rather than repeated as fact.

## Decision Classification

- Primary classification: `table_limited_not_definitive`
- Secondary classifications: `['proxy_limited']`
- Rationale: Continuity audit classified the selected table as event/candidate or ambiguous.; Available medication/task proxies remain weak supportive evidence.

## Recommended Next Action

- Build a true continuous window-level internal STN feature table before final burden decisions.
- Do not proceed to Brian2/DYNAP Phase 6A from this autopsy alone.

## What Not To Do Next

- Do not proceed to Brian2/DYNAP yet.
- Do not use PPN as primary STN burden validation.
- Do not use Herz/Groppa/Brown as primary STN LFP architecture validation.
- Do not claim clinical validation, clinical efficacy, or commercial-sensing equivalence.

## Role Of PPN And Herz Datasets

He/Tan PPN remains an optional future cross-target extension only after the primary internal STN architecture passes. Herz/Groppa/Brown remains a methods/code reference only.

## Limitations

- Selected Phase 5_2C table contains event/candidate-centered rows; burden traces are pseudo-burden when continuity is not continuous.
- High-capacity sklearn ceilings use deterministic training caps for runtime and are offline descriptive only.
- Window rows are autocorrelated; summaries use subject-level medians.
- Phase 6A.0.5 does not authorize Brian2/DYNAP work.
- continuity_summary.png
- estimator_ceiling_comparison.png
- tau_heterogeneity_heatmap.png
- per_subject_distribution.png
- feature_subset_ablation.png
- proxy_ceiling_summary.png
- beta_baseline_delta.png

## Exact Commands Run

- `pwd`
- `git rev-parse --show-toplevel`
- `git status --short`
- `source /scratch/haizhe/stn/start_stn.sh && python -V`
- `ls -lh reports/phase6a0_burden_viability || true`
- `python - <<'PY' ... Phase 6A.0 findings JSON ... PY`
- `find results/tables/05_phase5 -maxdepth 4 -type f | sort | head -300 || true`
- `find reports -maxdepth 5 -type f | sort | grep -E 'phase5|5_2|5Y|5_2C|burden|feature|ADR|phase6a0' | head -500 || true`
- `rg -n "Phase 5_2B|Phase 5_2C|Phase 5Y|burden|AUROC|LOSO|MedOff|MedOn|Hold|Move|UPDRS|bradykinesia|causal_feature_matrix|refined feature|feature subset|beta-feature" reports scripts results config configs data 2>/dev/null | head -800 || true`
- `source /scratch/haizhe/stn/start_stn.sh && python -m py_compile scripts/phase6a0_5_burden_failure_autopsy.py`
- `source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --stop-after-continuity`
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --stop-after-continuity"`
- `source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --tau-ms 200,500,800,1500,3000,5000`
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --tau-ms 200,500,800,1500,3000,5000"`
- `sacct -j 2579671 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `sacct -j 2579717 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `sacct -j 2580330 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `seff 2579671 2>/dev/null || true`
- `seff 2579717 2>/dev/null || true`
- `seff 2580330 2>/dev/null || true`
- `ls -lh reports/phase6a0_5_burden_failure_autopsy`
- `head -120 reports/phase6a0_5_burden_failure_autopsy/README_burden_failure_autopsy.md`
- `python - <<'PY' ... print burden_failure_autopsy_findings.json ... PY`
- `git diff --check`
- `find reports/phase6a0_5_burden_failure_autopsy -type f -size +5M -print`
- `git status --short`
- `actual script argv: /scratch/haizhe/stn/stn_env/bin/python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --tau-ms 200,500,800,1500,3000,5000`

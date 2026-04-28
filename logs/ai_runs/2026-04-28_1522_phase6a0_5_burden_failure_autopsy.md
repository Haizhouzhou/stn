# Phase 6A.0.5 Burden Failure Autopsy

Date: 2026-04-28 15:22 CEST

Task: Implement Phase 6A.0.5, a burden gate failure autopsy and ceiling analysis on the internal STN Phase 5_2C causal feature matrix.

Commit before task: `361576422b42ab62871e1c34a35b1fe5b29eff64`

## Original User Prompt

The user requested implementation of Phase 6A.0.5 in the existing STN repo: create `scripts/phase6a0_5_burden_failure_autopsy.py`, generate reports under `reports/phase6a0_5_burden_failure_autopsy/`, use the existing `start_stn.sh` environment only, do not install dependencies, do not modify raw data, do not use external PPN/Herz datasets as inputs, run continuity-only first, then run a full autopsy on `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`, produce the required CSV/JSON/PNG/README outputs, add an AI run log, and commit/push explicit safe paths only. The prompt explicitly required no Brian2/DYNAP/SNN scaffold creation and final classification among table/proxy/substrate/estimator/personalization/heterogeneity/regate/abandon/blocker classes.

## Operational Plan

1. Inspect the Phase 6A.0 outputs and selected table schema.
2. Implement a standalone autopsy script with continuity audit, data contract audit, leakage reaudit, feature-set construction, LOSO/within/oracle ceiling diagnostics, tau heterogeneity, proxy ceiling, beta delta, prior-phase comparison, decision matrix, figures, and structured findings.
3. Run `py_compile`, continuity-only, then full analysis under immediate CPU Slurm resources because the selected table is 4.1 GB.
4. Inspect outputs and file sizes, record Slurm accounting, create the run log and index entry.
5. Commit and push only the new script, report directory, and this run log.

## Files Inspected

- `reports/phase6a0_burden_viability/burden_viability_findings.json`
- `reports/phase6a0_burden_viability/burden_metric_by_subject.csv`
- `reports/phase6a0_burden_viability/burden_tau_sweep.csv`
- `reports/phase6a0_burden_viability/feature_column_audit.csv`
- `reports/phase6a0_burden_viability/leakage_risk_audit.csv`
- `reports/phase6a0_burden_viability/baseline_comparison.csv`
- `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- `results/tables/05_phase5/phase5_2c/causal_feature_matrix_feature_metadata.tsv`
- `results/tables/05_phase5/phase5_2c/causal_minimum_sufficient_subset.tsv`
- `results/tables/05_phase5/phase5_2c/causal_refined_candidate_features.tsv`
- `results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv`
- `results/tables/05_phase5/feature_atlas_2b/minimum_sufficient_feature_set.tsv`

## Files Created

- `scripts/phase6a0_5_burden_failure_autopsy.py`
- `reports/phase6a0_5_burden_failure_autopsy/README_burden_failure_autopsy.md`
- `reports/phase6a0_5_burden_failure_autopsy/previous_gate_summary.csv`
- `reports/phase6a0_5_burden_failure_autopsy/table_continuity_audit.csv`
- `reports/phase6a0_5_burden_failure_autopsy/data_contract_audit.csv`
- `reports/phase6a0_5_burden_failure_autopsy/leakage_reaudit.csv`
- `reports/phase6a0_5_burden_failure_autopsy/selected_feature_sets.csv`
- `reports/phase6a0_5_burden_failure_autopsy/feature_subset_ablation.csv`
- `reports/phase6a0_5_burden_failure_autopsy/estimator_ceiling_comparison.csv`
- `reports/phase6a0_5_burden_failure_autopsy/tau_heterogeneity.csv`
- `reports/phase6a0_5_burden_failure_autopsy/subject_distribution_check.csv`
- `reports/phase6a0_5_burden_failure_autopsy/proxy_ceiling_comparison.csv`
- `reports/phase6a0_5_burden_failure_autopsy/beta_baseline_delta.csv`
- `reports/phase6a0_5_burden_failure_autopsy/prior_phase_comparison.csv`
- `reports/phase6a0_5_burden_failure_autopsy/decision_matrix.csv`
- `reports/phase6a0_5_burden_failure_autopsy/burden_failure_autopsy_findings.json`
- `reports/phase6a0_5_burden_failure_autopsy/phase6a0_5_commands_run.txt`
- `reports/phase6a0_5_burden_failure_autopsy/phase6a0_prior_log_integrity_note.md`
- PNG figures under `reports/phase6a0_5_burden_failure_autopsy/figures/`
- This AI run log

No Phase 6A Brian2/DYNAP/SNN simulation scaffold was created.

## Commands Run

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
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --stop-after-continuity"`
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

## Validation Results

- `py_compile` passed for `scripts/phase6a0_5_burden_failure_autopsy.py`.
- Continuity-only Slurm job `2579671` completed in `00:04:41`, exit `0:0`, MaxRSS `8069044K`.
- Initial full Slurm job `2579717` completed in `01:08:00`, exit `0:0`, MaxRSS `8750244K`; it exposed a decision-rule wording issue that was corrected.
- Final full Slurm job `2580330` completed in `00:47:41`, exit `0:0`, MaxRSS `14433004K`.
- No report file under `reports/phase6a0_5_burden_failure_autopsy` exceeds 5 MB.
- External PPN/Herz datasets were not used as model inputs.

## Prior Phase 6A.0 Result Summary

- Status: `FAIL`
- Selected input: `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- Subjects found/valid: `20 / 20`
- Allowed causal features: `354`
- Best tau: `3000 ms`
- Median LOSO Pearson/Spearman: `0.249 / 0.260`
- Baselines at best tau: class-prior `0.107`, shuffled `0.038`, beta-feature `0.228`

## Phase 6A.0.5 Results

- Input table: `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- Subjects: `20`; valid subjects: `20`
- Rows/columns: `690088 / 373`
- Continuity conclusion: `event_or_candidate_matrix`
- Continuity counts: `598` event/candidate groups, `28` continuous-medium groups
- Allowed features after reaudit: `297`
- Best LOSO annotation-burden ceiling: gradient boosting top-k-25, Pearson `0.928`, Spearman `0.916`, tau `5000 ms`
- Best deployable/simple ceiling: numpy ridge top-k-100, Pearson `0.900`, Spearman `0.901`, tau `5000 ms`
- Best within-subject calibrated ceiling: gradient boosting top-k-25, Pearson `0.712`, Spearman `0.646`, tau `5000 ms`
- Best tau summary across diagnostic LOSO rows: `3000 ms` by median metric, with high-ceiling rows peaking at `5000 ms`
- Beta baseline delta: high-capacity best delta `0.686`; this is not deployable by itself
- Proxy ceiling: weak; best AUROC-like metric `0.514`
- Subject distribution: uniformly high on the non-definitive diagnostic metric; not evidence of real continuous burden trackability

## Final Classification

Primary classification: `table_limited_not_definitive`

Secondary classification: `proxy_limited`

Decision rationale: the selected Phase 5_2C table is dominated by event/candidate-centered rows, so row-order burden integration can only be treated as pseudo-burden. The high diagnostic ceilings show that the candidate matrix is highly learnable, but they do not validate continuous causal burden tracking. Available medication/task proxies are weak and clinical score validation remains unavailable.

## Limitations

- The selected Phase 5_2C matrix is not a definitive continuous stream.
- High-capacity ceilings are offline diagnostics with deterministic training caps.
- Logistic ceiling emitted convergence warnings and is not a deployable claim.
- Window rows are autocorrelated; subject-level medians are used for summaries.
- The run log notes the prior Phase 6A.0 log issue in `phase6a0_prior_log_integrity_note.md`; old logs were not edited.

## Recommended Next Step

Build a true continuous window-level internal STN feature table, then rerun a formal Phase 6A.0-style gate on that table. Do not proceed to Brian2/DYNAP Phase 6A from this autopsy alone.

## Commit Status

Commit after task: will be reported in the final response; embedding the hash in this tracked file would change the commit hash.

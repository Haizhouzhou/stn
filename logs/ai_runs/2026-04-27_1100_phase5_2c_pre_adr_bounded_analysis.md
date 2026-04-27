# Phase 5_2C Pre-ADR Bounded Analysis

Date: 2026-04-27 11:00 CEST

Task: Run the owner-required bounded pre-ADR analytical sprint before any formal architecture ADR reopening. The sprint covered alarm reconstruction headroom, scoring tolerance sensitivity, and burden/state ceiling comparison under causal Phase 5_2C constraints.

Commit before task: `adfe6e58233bed4d7cf64b5beeeadb4db0ad122f`

## User Constraints

- Do not reopen the architecture ADR yet.
- Do not create `docs/PHASE5_2C_ARCHITECTURE_ADR_REOPENED.md`.
- Do not execute Stage G, H, or I.
- Do not run Brian2.
- Do not create closeout outcome files or frozen Brian2 specs.
- Do not use MEG.
- Do not touch, rewrite, or relabel Phase 3 labels.
- Do not train or claim a final detector.
- Do not add new physiological feature families.
- Do not replace LOSO with random splits.
- Do not fabricate unsupported metrics.
- Do not create fake downstream outputs.
- Do not duplicate huge prediction/alarm-trace tables.
- Run exactly three scientific subtasks before any ADR reopening decision: alarm reconstruction headroom, scoring tolerance sensitivity, and burden/state ceiling comparison.

## Operational Plan

1. Inspect existing Phase 5_2C event scoring, reassessment outputs, Stage F metrics, and burden/state context.
2. Implement a bounded pre-ADR analysis module, CLI runner, and focused tests.
3. Run local compile, focused pytest, and diff hygiene before production.
4. Run the production analysis once under Slurm CPU resources with the immediate-resource launcher.
5. Collect Slurm accounting, validate outputs, update readiness, write docs/tables, create this AI run log, then commit and push explicit paths only.

## Files Inspected

- `src/stnbeta/phase5_2c/stage_f_event_metrics.py`
- `src/stnbeta/phase5_2c/stage_f_event_metrics_fast.py`
- `src/stnbeta/phase5_2c/event_target_reassessment.py`
- `src/stnbeta/phase5_2c/io.py`
- `configs/phase5_2c.yaml`
- `docs/PHASE5_2C_ARCHITECTURE_ADR.md`
- `docs/PHASE5_2C_PIPELINE_DESIGN.md`
- `docs/PHASE5_2C_EVENT_SCORING_TARGET_ADR.md`
- `docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE_REVISED.md`
- `docs/PHASE5_2C_EVENT_DETECTION_LIMITATION_ANALYSIS.md`
- `docs/PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md`
- `docs/PHASE5Z_BURDEN_STATE.md`
- `docs/PHASE5_RUNBOOK.md`
- `docs/decisions.md`
- `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- `results/tables/05_phase5/phase5_2c/causal_minimum_sufficient_subset.tsv`
- `results/tables/05_phase5/phase5_2c/causal_refined_candidate_features.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_three_tier_event_summary.tsv`
- `results/tables/05_phase5/phase5_2c/event_recall_empirical_ceiling.tsv`
- `results/tables/05_phase5/phase5_2c/event_recall_gap_to_ceiling.tsv`
- `results/tables/05_phase5/phase5_2c/event_per_subject_recall_distribution.tsv`

## Files Modified Or Created

- Created `src/stnbeta/phase5_2c/pre_adr_bounded_analysis.py`
- Created `scripts/05_2c_pre_adr_bounded_analysis.py`
- Created `tests/test_phase5_2c_pre_adr_bounded_analysis.py`
- Created `docs/PHASE5_2C_PRE_ADR_BOUNDED_REMEDIATION_ANALYSIS.md`
- Created `docs/PHASE5_2C_EVENT_ALARM_RECONSTRUCTION_HEADROOM.md`
- Created `docs/PHASE5_2C_SCORING_TOLERANCE_SENSITIVITY.md`
- Created `docs/PHASE5_2C_BURDEN_STATE_CEILING_COMPARISON.md`
- Created `docs/PHASE5_2C_PRE_ADR_RECOMMENDATION.md`
- Created `results/tables/05_phase5/phase5_2c/pre_adr_owner_requirement.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_alarm_reconstruction_sweep.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_alarm_reconstruction_best.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_scoring_tolerance_sweep.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_scoring_tolerance_summary.tsv`
- Created `results/tables/05_phase5/phase5_2c/burden_state_target_availability_pre_adr.tsv`
- Created `results/tables/05_phase5/phase5_2c/burden_state_ceiling_metrics_pre_adr.tsv`
- Created `results/tables/05_phase5/phase5_2c/burden_state_gap_to_ceiling_pre_adr.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_vs_burden_pre_adr_comparison.tsv`
- Created `results/tables/05_phase5/phase5_2c/pre_adr_recommendation.tsv`
- Created `results/tables/05_phase5/phase5_2c/pre_adr_bounded_analysis_validation.tsv`
- Updated `results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv`
- Created this AI run log and updated `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`, `git log -1 --oneline`, targeted `git status --short`
- `rg --files`, `sed`, `head`, and compact TSV inspections for Stage F/event/burden context
- `/scratch/haizhe/stn/stn_env/bin/python -m py_compile src/stnbeta/phase5_2c/pre_adr_bounded_analysis.py scripts/05_2c_pre_adr_bounded_analysis.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest tests/test_phase5_2c_pre_adr_bounded_analysis.py -q -p no:cacheprovider`
- `git diff --check`
- `sinfo -p teaching -o '%P %a %l %D %t %C %m'`
- Production command:

```bash
~/bin/claim_best_immediate_resource.sh --mode cpu --log-dir results/logs/05_phase5/phase5_2c --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=80 --mem=750G --time=04:00:00" "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/05_2c_pre_adr_bounded_analysis.py --config configs/phase5_2c.yaml"
```

- `sacct -j 2566757 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW -P`
- `seff 2566757` when available; no output was returned in this environment.
- Compact regeneration of tolerance summary/docs after enforcing achieved-FP cap feasibility in the summary selector.
- TSV row/column and field-count checks.

## Slurm And Resource Results

- Job ID: `2566757`
- State: `COMPLETED`
- Elapsed: `00:31:32`
- Exit code: `0:0`
- Selected resource: teaching partition, 80 CPUs, 750 GB RAM, 1 node, 4-hour wall time
- MaxRSS: `7672928K`
- AveCPU: `00:29:19`
- CPUTimeRAW: `151360`
- No GPU was requested or used.

## Validation Results

- `py_compile` passed.
- Focused pytest passed: `4 passed`.
- `git diff --check` passed.
- Generated validation table passed: `25/25` checks.
- TSV field-count checks passed for all required pre-ADR output tables.
- No formal reopened ADR file was created.
- No Stage G/H/I execution outputs, Brian2 outputs, closeout files, frozen Brian2 specs, Phase 3 label edits, MEG outputs, or huge duplicate trace tables were created.

## Scientific Results

- Owner requirement recorded: current low-recall event gates remain rejected; `CAND_2C_BALANCED_FP2` is not approved; ADR reopening is deferred until this bounded analysis is reviewed.
- Alarm reconstruction headroom: best Tier 3 proxy bounded strategy was the 150 ms leaky evidence integrator.
  - FP/min 2.0 recall: `0.072810`
  - FP/min 5.0 recall: `0.145639`
  - Interpretation: bounded alarm reconstruction improves over Stage F but remains below 0.10 recall at 2 FP/min and below 0.25 at 5 FP/min.
- Scoring tolerance sensitivity: widened onset tolerance `S1_onset_tolerance_pm600ms` produced large recall increases without changing causal scores.
  - FP/min 2.0 best recall: `0.661713`
  - FP/min 5.0 best recall: `0.856359`
  - Interpretation: event recall is materially sensitive to scoring tolerance, and owner review is needed before ADR reopening.
- Burden/state ceiling comparison: burden/state targets are available as frozen Phase 3 evaluation targets, but deployable causal estimates were modest.
  - Best deployable burden/state AUROC: `0.584675`
  - Best diagnostic burden/state AUROC: `0.561672`
  - Interpretation: burden/state did not clearly supersede the event branch in this bounded causal sprint.

## Recommendation

Recommended option: `EVENT_TARGET_METRIC_SENSITIVE`.

Final task state: `pre_adr_recommends_event_target_metric_sensitive`.

Rationale: event recall improves materially only under widened scoring definitions; target metric requires owner review before any ADR reopening.

Readiness remains `not_ready_scientific_gate_failed`; `can_execute_stage_g_h_i_now=False`.

## Remaining Risks

- Widened scoring is sensitivity analysis only; it is not a deployment target and does not approve G/H/I or Brian2.
- Burden/state targets are sampled-window evaluation targets derived from frozen Phase 3 labels and remain diagnostic.
- Alarm reconstruction strategy choice is posthoc within the bounded analysis; it is not a trained or approved detector.
- Prior Phase 5Z burden/state evidence remains historical context only and does not override causal Phase 5_2C results.

## Final Codex Output

The final response will report the bounded pre-ADR outputs, Slurm accounting, validation results, recommendation `EVENT_TARGET_METRIC_SENSITIVE`, commit/push status, and final task state `pre_adr_recommends_event_target_metric_sensitive`.

Commit after task: reported in final response to avoid self-referential commit hash churn.

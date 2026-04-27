# Phase 5_2C Event-Target Decision Package

Date: 2026-04-27 06:43 CEST

Task: Prepare Phase 5_2C event-target ADR decision package for owner approval.

Commit before task: `7e63aec59d80678197e7bf3fa474ed7696315aa4`

## Original User Prompt

The user approved implementation mode for Phase 5_2C event-target ADR preparation only, with these controlling constraints:

- Do not execute Stage G, H, or I.
- Do not run Brian2.
- Do not create closeout outcome files or frozen Brian2 specs.
- Do not use MEG.
- Do not touch or relabel Phase 3 labels.
- Do not train or claim a final detector.
- Do not fabricate unsupported metrics.
- Do not change the accepted `hybrid_early_warning` architecture ADR.
- Work in `/scratch/haizhe/stn` on branch `main`.
- Source `/scratch/haizhe/stn/start_stn.sh` for environment-sensitive validation.
- Follow `AGENTS.md`.

The requested deliverable was a professional Phase 5_2C event-target decision package so the project owner can approve an explicit engineering gate. The package had to use existing Phase 5_2C docs, configs, and Stage F event summary tables; avoid reading or duplicating huge prediction/alarm-trace files unless strictly necessary; summarize current Tier 1/2/3 event metrics; identify historical target references; build 2-4 candidate engineering gates; recommend exactly one candidate with `recommendation_status = requires_owner_approval`; include owner approval language; and create validation output checking that no G/H/I, Brian2, Phase 3 label, MEG, or huge duplicate trace outputs were created.

Required outputs:

- `docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE.md`
- `results/tables/05_phase5/phase5_2c/event_target_candidate_gates.tsv`
- `results/tables/05_phase5/phase5_2c/event_target_metric_support.tsv`
- `results/tables/05_phase5/phase5_2c/event_target_recommendation.tsv`
- `results/tables/05_phase5/phase5_2c/event_target_decision_package_validation.tsv`

Final required state: `target_decision_package_ready_for_owner_approval`.

## Operational Plan

1. Inspect the readiness table and Stage F event output summaries.
2. Inspect Phase 5_2C architecture, pipeline, event-target, and G/H/I plan docs.
3. Search README, runbook, decisions, and relevant configs for historical FP/min, recall, latency, early-warning, one-alarm-per-burst, closed-loop, and DBS target references.
4. Build candidate gate tables from loaded metrics only.
5. Recommend exactly one gate as requiring owner approval.
6. Validate package scope and table integrity without executing G/H/I.
7. Record provenance and checkpoint the explicit package paths.

## Files Inspected

- `README.md`
- `AGENTS.md`
- `docs/PHASE5_RUNBOOK.md`
- `docs/decisions.md`
- `docs/PHASE5_2C_ARCHITECTURE_ADR.md`
- `docs/PHASE5_2C_PIPELINE_DESIGN.md`
- `docs/PHASE5_2C_EVENT_SCORING_TARGET_ADR.md`
- `docs/PHASE5_2C_STAGE_G_H_I_EXECUTION_PLAN.md`
- `docs/PHASE5Y_TARGET_RECONCILIATION.md`
- `configs/phase5_2c.yaml`
- `configs/phase5_training_teacher.yaml`
- `configs/phase5_dense_teacher.yaml`
- `configs/phase5_ht_hardware_aware.yaml`
- `configs/phase5w_hardware_aware.yaml`
- `results/tables/05_phase5/phase5_2c/event_scoring_target_resolution.tsv`
- `results/tables/05_phase5/phase5_2c/event_scoring_policy.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_three_tier_event_summary.tsv`
- `results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv`
- `results/tables/05_phase5/phase5_2c/causal_minimum_sufficient_subset.tsv`
- `results/tables/05_phase5/phase5_2c/causal_stage_f_event_output_validation_summary.tsv`

The subject-level `causal_event_alarm_trace_summary.tsv` was not duplicated. Its checksum was recorded as a reproducibility caveat because the artifact contains pseudonymous subject IDs.

## Files Modified Or Created

- Created `docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE.md`
- Created `results/tables/05_phase5/phase5_2c/event_target_candidate_gates.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_target_metric_support.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_target_recommendation.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_target_decision_package_validation.tsv`
- Created this AI run log.
- Updated `logs/ai_runs/INDEX.md`.

## Commands Run

- `pwd && git status --short && git log -1 --oneline`
- `date '+%Y-%m-%d %H:%M:%S %Z' && git rev-parse HEAD`
- `sed` reads of the Phase 5_2C docs and output tables.
- `rg -n` searches for FP/min, recall, latency, early-warning, one-alarm-per-burst, closed-loop, DBS, and target references.
- Python/CSV summary of Tier 1, Tier 2, Tier 3, and three-tier event tables.
- `sha256sum results/tables/05_phase5/phase5_2c/causal_event_alarm_trace_summary.tsv`
- `awk -F '\t' ...` TSV field-count checks for the new package tables.
- `source /scratch/haizhe/stn/start_stn.sh && python - <<'PY' ...` package validation. The boot script completed after its import checks; no Brian2 simulation was run.
- `find ...` inventory check for Stage G/H/I, closeout, frozen, or Brian2-gate outputs.
- `grep -RIn ... 'MEG' ...` review of the new package files.
- `git diff --check -- <package paths>`
- `stat -c '%n\t%s bytes\t%y' <package paths>`

No Slurm job was submitted. No production compute was required.

## Validation Results

- TSV schema check passed:
  - `event_target_candidate_gates.tsv`: 3 rows, 44 columns, no bad rows.
  - `event_target_metric_support.tsv`: 12 rows, 29 columns, no bad rows.
  - `event_target_recommendation.tsv`: 1 row, 28 columns, no bad rows.
  - `event_target_decision_package_validation.tsv`: 11 rows, 20 columns, no bad rows.
- Candidate count: 3.
- Recommended candidate rows: 1.
- Recommended gate: `CAND_2C_BALANCED_FP2`.
- Recommendation status: `requires_owner_approval`.
- Validation table checks: 11 of 11 passed.
- `git diff --check` passed for the package paths.
- No focused script/module compile was required because no scripts or Python modules were created or modified.
- No Stage G/H/I execution outputs, Brian2 outputs, closeout outputs, or frozen Brian2 specs were created by this task.

## Result Summary

The package recommends `CAND_2C_BALANCED_FP2` for owner approval. It is an engineering gate at FP/min `2.0`, where Stage F reports Tier 1 recall `0.039319018885667266`, Tier 2 recall `0.03955411017945302`, Tier 3 median recall `0.0393679962385393`, and Tier 3 5th-percentile recall `0.03915935271530444`. The recommendation remains blocked on owner approval and does not update readiness to `ready_to_execute_g_h_i`.

Approval text included in the package:

```text
I approve candidate gate CAND_2C_BALANCED_FP2 as the Phase 5_2C engineering target.
This is an engineering gate for G/H/I execution and Brian2 gate evaluation, not a clinical deployment claim.
```

## Remaining Risks

- Absolute event recall remains low even at the recommended FP/min `2.0` operating point.
- Tier 3 is fragile at FP/min `1.0`, so strict low-FP operation is not currently supported.
- Subject-level worst-case rows are not copied into tracked outputs because the alarm-trace summary contains pseudonymous subject IDs.
- The package is not an approval artifact by itself; owner approval and readiness regeneration/revalidation are still required before G/H/I execution.

## Final Codex Output

The final response will report the created decision package, candidate gates, recommended owner-approval text, validations, output inventory, commit/push status, and final state `target_decision_package_ready_for_owner_approval`.

Commit after task: reported in final response to avoid self-referential commit hash churn.

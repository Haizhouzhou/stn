# Phase 5_2C Event-Target Reassessment

Date: 2026-04-27 07:43 CEST

Task: Revise the Phase 5_2C event-target decision package after owner rejection of all low-recall candidate gates, analyze the Tier 1 / Tier 2 FP/min 1.0 anomaly, estimate empirical event-level recall ceilings, summarize per-subject event recall without exposing raw subject IDs, and reassess the architectural target.

Commit before task: `ad3e1ece23a323d0bb6f8d158e4e5f0523c67b9c`

## User Constraints

- Do not execute Stage G, H, or I.
- Do not run Brian2.
- Do not create closeout outcome files or frozen Brian2 specs.
- Do not use MEG.
- Do not touch, rewrite, or relabel Phase 3 labels.
- Do not train or claim a final detector.
- Do not add new feature families.
- Do not replace LOSO with random splits.
- Do not fabricate unsupported metrics or fake downstream outputs.
- Do not duplicate huge prediction/alarm-trace tables.
- Record that the owner rejected `CAND_2C_STRICT_FP1`, `CAND_2C_BALANCED_FP2`, and `CAND_2C_PERMISSIVE_FP5` because event recall was too low to justify G/H/I execution or a Brian2 gate decision.

## Operational Plan

1. Inspect existing Stage F event metric tables, target-package docs, scorer implementation, and compact burden/state context.
2. Add a reproducible reassessment utility and focused tests.
3. Run the score/event analyses under Slurm CPU resources because the causal feature matrix is 4.36 GB.
4. Generate owner-rejection, anomaly, empirical ceiling, gap, per-subject distribution, reassessment options, recommendation, readiness, validation, and revised documentation outputs.
5. Validate that no Stage G/H/I, Brian2, closeout, frozen specs, Phase 3 label edits, MEG outputs, or huge duplicate traces were created.

## Files Inspected

- `src/stnbeta/phase5_2c/stage_f_event_metrics.py`
- `src/stnbeta/phase5_2c/stage_f_event_metrics_fast.py`
- `src/stnbeta/phase5_2c/io.py`
- `configs/phase5_2c.yaml`
- `docs/PHASE5_2C_ARCHITECTURE_ADR.md`
- `docs/PHASE5_2C_PIPELINE_DESIGN.md`
- `docs/PHASE5_2C_EVENT_SCORING_TARGET_ADR.md`
- `docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE.md`
- `docs/PHASE5_2C_STAGE_G_H_I_EXECUTION_PLAN.md`
- `docs/PHASE5Z_BURDEN_STATE.md`
- `results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_three_tier_event_summary.tsv`
- `results/tables/05_phase5/phase5_2c/causal_event_alarm_trace_summary.tsv`
- `results/tables/05_phase5/phase5_2c/causal_minimum_sufficient_subset.tsv`
- `results/tables/05_phase5/phase5_2c/causal_refined_candidate_features.tsv`
- compact Phase 5 burden/state summary tables.

## Files Modified Or Created

- Created `src/stnbeta/phase5_2c/event_target_reassessment.py`
- Created `scripts/05_2c_event_target_reassessment.py`
- Created `tests/test_phase5_2c_event_target_reassessment.py`
- Created `docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE_REVISED.md`
- Created `docs/PHASE5_2C_EVENT_DETECTION_LIMITATION_ANALYSIS.md`
- Created `docs/PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md`
- Created `results/tables/05_phase5/phase5_2c/event_target_owner_rejection.tsv`
- Created `results/tables/05_phase5/phase5_2c/tier1_tier2_fp1_anomaly_analysis.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_recall_empirical_ceiling.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_recall_gap_to_ceiling.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_per_subject_recall_distribution.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_target_reassessment_options.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_target_reassessment_recommendation.tsv`
- Updated `results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv`
- Created `results/tables/05_phase5/phase5_2c/event_target_reassessment_validation.tsv`
- Created this AI run log and updated `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`, `git log -1 --oneline`, targeted `git status --short`
- `stat` and `head`/`sed` inspections of Stage F event outputs and docs
- `sinfo`, `sacctmgr`, `squeue`, and `sacct` resource/accounting checks
- `source /scratch/haizhe/stn/start_stn.sh && python -m py_compile src/stnbeta/phase5_2c/event_target_reassessment.py scripts/05_2c_event_target_reassessment.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest tests/test_phase5_2c_event_target_reassessment.py -q -p no:cacheprovider`
- First CPU launcher attempt failed on invalid `standard` account/partition candidates.
- Successful production command:

```bash
~/bin/claim_best_immediate_resource.sh --mode cpu --log-dir results/logs/05_phase5/phase5_2c --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=80 --mem=750G --time=04:00:00" "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/05_2c_event_target_reassessment.py --config configs/phase5_2c.yaml"
```

- Compact regeneration after validation-scope patch:
  `/scratch/haizhe/stn/stn_env/bin/python - <<'PY' ...`
- `git diff --check`
- TSV field-count checks with `awk -F '\t'`
- `sacct -j 2566156 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW -P`

## Slurm And Resource Results

- Initial default CPU ladder failed because `standard` with account `mlnlp2.pilot.s3it.uzh` was invalid.
- Successful job: `2566156`
- Partition/account: `teaching` / `mlnlp2.pilot.s3it.uzh`
- Requested/allocated: 80 CPUs, 750 GB, 1 node, 4 hours
- State: `COMPLETED`
- Elapsed: `00:32:58`
- Exit code: `0:0`
- MaxRSS: `7640680K`
- AveCPU: `00:30:21`
- CPUTimeRAW: `158240`
- No GPU was requested or used.

## Validation Results

- `py_compile` passed.
- Focused pytest passed: `3 passed`.
- Reassessment validation table: 18/18 checks passed.
- `git diff --check` passed.
- TSV field-count checks passed for all reassessment output tables.
- No Stage G/H/I execution outputs, Brian2 outputs, closeout files, frozen Brian2 specs, Phase 3 label edits, MEG outputs, or huge duplicate trace tables were created.

## Scientific Results

- Owner rejection recorded for all three prior candidate gates.
- Tier 1 / Tier 2 FP/min 1.0 inconsistency was classified as a threshold-grid / quantization discontinuity, not an identified measurement bug.
- Tier 1 FP/min 1.0 observed recall was `0.002145`; dense score-ranking ceiling was `0.037095`.
- Tier 2 FP/min 1.0 observed recall was `0.023029`; dense score-ranking ceiling was `0.037428`.
- Tier 3 median FP/min 1.0 observed recall was `0.002562`; dense score-ranking ceiling was `0.034245`.
- At FP/min 2.0, the best diagnostic score-ranking ceiling from deployable score sources remained below `0.10` recall.
- Per-subject output uses hashed subject keys; raw pseudonymous IDs are not exposed.

## Recommendation

Recommended option: `OPTION_C_REOPEN_ADR_BURDEN_STATE`.

Final task state: `event_target_reassessment_recommends_adr_reopen_burden`.

Rationale: diagnostic score-ranking ceiling from deployable score sources remains below `0.10` recall at FP/min `2.0`, indicating low-FP discrete event detection is likely information-limited. Current readiness is `not_ready_scientific_gate_failed`, and `can_execute_stage_g_h_i_now=False`.

## Remaining Risks

- Ceiling estimates use diagnostic threshold sweeps and are not deployment claims.
- The subset-score ceiling used loaded causal refined variants, not an exhaustive new feature-family search.
- The low-FP event branch may still benefit from bounded remediation, but current evidence does not justify G/H/I execution or Brian2 gate evaluation.
- Prior burden/state evidence is historical context only and does not override causal Stage C/D/E/F results.

## Final Codex Output

The final response will report the revised package, owner rejection, anomaly analysis, empirical ceiling, per-subject distribution, reassessment recommendation, Slurm accounting, validation results, output inventory, and final task state `event_target_reassessment_recommends_adr_reopen_burden`.

Commit after task: reported in final response to avoid self-referential commit hash churn.

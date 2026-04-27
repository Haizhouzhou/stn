# Phase 5_2C Event-Target Decision Package

Status: ready for project-owner target approval.

This package prepares an explicit Phase 5_2C engineering gate for owner decision. It does not approve the gate by itself, does not execute Stage G/H/I, does not run Brian2, does not select a closeout outcome, and does not claim a final detector.

## Current Readiness

`stage_g_h_i_readiness_assessment.tsv` reports:

- `readiness_state`: `ready_to_plan_g_h_i`
- `target_status`: `unavailable`
- `event_metrics_computed`: `True`
- `can_execute_stage_g_h_i_now`: `False`
- `can_plan_stage_g_h_i`: `True`
- `qc_reason`: `Historical FP/min and latency references exist, but no explicit Phase 5_2C event-level recall target/gate was found.`

Therefore, this package is plan/decision preparation only. Readiness must not be changed to `ready_to_execute_g_h_i` until the owner explicitly approves a target and the readiness artifact is regenerated or revalidated.

## Existing Target Evidence

The current Phase 5_2C event-scoring ADR defines a reporting policy, not a target. It reports the FP/min grid `0.5`, `1.0`, `2.0`, and `5.0`, with recall at the FP/min grid as the primary reported event metric, plus AUROC/AUPRC, precision, F1, latency relative to Phase 3 onset, early-warning latency, and one-alarm-per-burst fraction.

Historical references establish useful context but not an approved Phase 5_2C gate:

- Prior Phase 5 documents mention low-FP controlled recall, event F1, latency, and one-alarm-per-burst behavior.
- Older training and dense configs define recall targets at `<=1 FP/min`, but those are prior-stage software/training gates and are not automatically transferable to Phase 5_2C.
- The accepted Phase 5_2C architecture ADR says event-level recall at an FP/min gate is the future Stage F operating metric and negative latency should be treated as early-warning candidate-state evidence, not committed alarm success.
- `configs/phase5_2c.yaml` explicitly has `project_target_available: false`.

The detailed reference classification is in `results/tables/05_phase5/phase5_2c/event_target_metric_support.tsv`.

## Current Stage F Event Metrics

The event outputs remain pre-Brian2 engineering estimates. They are sufficient for owner selection of an engineering gate, not for a clinical or deployment claim.

| FP/min gate | Tier 1 recall | Tier 2 recall | Tier 3 median recall | Tier 3 5th percentile recall | Tier 1 precision | Tier 2 precision | Tier 3 median F1 | Median committed latency | One-alarm/burst summary |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.0021452080557950006 | NA | 0.002561515555207272 | 0.0023695243319489068 | 0.4256120527306968 | NA | 0.005089157139399711 | about 150 ms | Tier 2 has no threshold at this cap; Tier 3 median one-alarm/burst is 0.002561515555207272 |
| 1.0 | 0.0021452080557950006 | 0.023029151320429434 | 0.002561515555207272 | 0.0023695243319489068 | 0.4256120527306968 | 0.3973125414731254 | 0.005089157139399711 | about 150 ms | Tier 3 collapses relative to Tier 2; Tier 3 median one-alarm/burst is 0.002561515555207272 |
| 2.0 | 0.039319018885667266 | 0.03955411017945302 | 0.0393679962385393 | 0.03915935271530444 | 0.38123771182473554 | 0.38088153245304074 | 0.07136841471300837 | about 150 ms | Tier 3 median one-alarm/burst is 0.03882924535694695 |
| 5.0 | 0.13387469634041219 | 0.13733249745317766 | 0.13336043413525586 | 0.13289955724473004 | 0.33672647381911386 | 0.3359612462006079 | 0.19062319351086626 | about 150 ms | Tier 3 median one-alarm/burst is 0.13225844369563516 |

Quantization and mismatch behavior:

- Tier 1 to Tier 2 AUROC degradation is negligible: about `0.0000040653`.
- At FP/min `2.0`, Tier 2 to Tier 3 median recall degradation is `0.00018611394091371664`.
- At FP/min `5.0`, Tier 2 to Tier 3 median recall degradation is `0.003972063317921798`.
- At FP/min `1.0`, Tier 2 to Tier 3 median recall degradation is `0.02046763576522216`, which is the main low-FP fragility.
- Tier 3 median AUROC is `0.8197257700122798`, down from Tier 2 AUROC `0.827925936846802`.

Latency policy:

- Committed median latency is approximately `150 ms` relative to Phase 3 onset at the reported operating points.
- Negative latencies, where present, are early-warning/candidate-state evidence only and do not count as committed alarm success.

Per-subject worst cases:

- The tracked Tier 1/2/3 tables do not include subject identifiers.
- Subject-level rows are available in the untracked `causal_event_alarm_trace_summary.tsv`, but this package does not duplicate them because that artifact contains per-subject pseudonymous IDs.
- The checksum recorded for that artifact during package preparation was `4a1faa0abf182b6bef96bb547948da1668c004235cfa51e619ef07cd936c1112`.

## Candidate Engineering Gates

Detailed candidate definitions are in `results/tables/05_phase5/phase5_2c/event_target_candidate_gates.tsv`.

| Candidate | Profile | FP/min | Existing Stage F satisfies? | Interpretation |
| --- | --- | ---: | --- | --- |
| `CAND_2C_STRICT_FP1` | strict low-FP | 1.0 | False | Desirable low-FP discipline, but current Tier 1 and Tier 3 recall are too low. |
| `CAND_2C_BALANCED_FP2` | balanced | 2.0 | True | Lowest reported operating point where Tier 1, Tier 2, and Tier 3 mismatch summaries remain aligned around 0.039 recall. |
| `CAND_2C_PERMISSIVE_FP5` | permissive | 5.0 | True | Stronger recall but materially higher false-positive allowance; useful as diagnostic fallback but too permissive as the primary recommendation. |

## Recommended Candidate

Recommended gate: `CAND_2C_BALANCED_FP2`.

Recommendation status: `requires_owner_approval`.

Rationale:

- It is the lowest reported FP/min operating point where the continuous, quantized, and mismatch-aware summaries are all stable.
- Tier 3 5th-percentile recall remains close to the median at FP/min `2.0`.
- Quantization degradation is negligible.
- Mismatch degradation is small at FP/min `2.0`, unlike the fragile FP/min `1.0` point.
- It is less permissive than FP/min `5.0`, so it is a better engineering screen for whether Stage G/H/I are meaningful.
- It still has low absolute recall, so owner approval is required before using it as a gate.

This is an engineering gate only. It is meant to decide whether Stage G/H/I execution is worth running and whether later Brian2 gate evaluation can be considered. It is not a clinical target, not a deployment threshold, and not evidence that Brian2 is justified.

## Approval Text

The project owner can approve the recommended gate with:

```text
I approve candidate gate CAND_2C_BALANCED_FP2 as the Phase 5_2C engineering target.
This is an engineering gate for G/H/I execution and Brian2 gate evaluation, not a clinical deployment claim.
```

After approval, readiness should be regenerated or revalidated. G/H/I execution remains blocked until `stage_g_h_i_readiness_assessment.tsv` reports `ready_to_execute_g_h_i`.

## Validation

Validation table: `results/tables/05_phase5/phase5_2c/event_target_decision_package_validation.tsv`.

The validation checks confirm:

- this decision package exists;
- three candidate gates are provided;
- exactly one candidate is recommended;
- recommendation status is `requires_owner_approval`;
- no G/H/I execution outputs were created;
- no Brian2 outputs or frozen Brian2 specs were created;
- no Phase 3 label files were modified by this task;
- no MEG was introduced;
- no huge duplicate prediction or alarm-trace table was created;
- `git diff --check` passed.

## Non-Claims

- This package does not execute Stage G/H/I.
- This package does not approve Outcome 1.
- This package does not justify Brian2 simulation.
- This package does not modify the accepted `hybrid_early_warning` architecture ADR.
- This package does not train or claim a final detector.
- This package does not change Phase 3 labels.

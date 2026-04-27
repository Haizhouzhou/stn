# Phase 5_2C Pre-ADR Bounded Remediation Analysis

Status: bounded analysis complete. The formal architecture ADR was not reopened. Stage G/H/I were not executed. Brian2 was not run.

## Owner Requirement

The project owner rejected the current low-recall event gates and required this bounded analytical sprint before any formal architecture ADR reopening.

## R1 Alarm Reconstruction Headroom

|   fp_per_min_target | tier                             | strategy_id                  | param_id   |    recall |   precision |        F1 |   fp_per_min_achieved | headroom_interpretation                                                                       |
|--------------------:|:---------------------------------|:-----------------------------|:-----------|----------:|------------:|----------:|----------------------:|:----------------------------------------------------------------------------------------------|
|                 0.5 | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.021119  |    0.404196 | 0.0401409 |              0.507621 | bounded reconstruction improves event recall but requires owner review before target approval |
|                 1   | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.0393778 |    0.392969 | 0.0715636 |              0.992952 | bounded reconstruction improves event recall but requires owner review before target approval |
|                 2   | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.0728097 |    0.373901 | 0.121891  |              1.98369  | best bounded event reconstruction remains below 0.10 recall at 2 FP/min                       |
|                 5   | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.145639  |    0.342657 | 0.204358  |              4.54203  | best bounded event reconstruction remains below 0.25 recall at 5 FP/min                       |

## R2 Scoring Tolerance Sensitivity

|   fp_per_min_target |   current_policy_best_recall | best_scoring_definition    |   best_recall |   recall_improvement_factor | tolerance_interpretation                                  |
|--------------------:|-----------------------------:|:---------------------------|--------------:|----------------------------:|:----------------------------------------------------------|
|                 0.5 |                    0.0195518 | S1_onset_tolerance_pm600ms |      0.36014  |                    18.4198  | event recall is materially sensitive to scoring tolerance |
|                 1   |                    0.0378105 | S1_onset_tolerance_pm600ms |      0.508111 |                    13.4383  | event recall is materially sensitive to scoring tolerance |
|                 2   |                    0.0694891 | S1_onset_tolerance_pm600ms |      0.661713 |                     9.52255 | event recall is materially sensitive to scoring tolerance |
|                 5   |                    0.143788  | S1_onset_tolerance_pm600ms |      0.856359 |                     5.95572 | event recall is materially sensitive to scoring tolerance |

## R3 Burden/State Ceiling

| target_id                              |   window_s | score_source                           | deployability_class    |   pearson_correlation |   high_burden_AUROC |   high_burden_AUPRC |   per_subject_metric_median |
|:---------------------------------------|-----------:|:---------------------------------------|:-----------------------|----------------------:|--------------------:|--------------------:|----------------------------:|
| B3_recent_event_density_state          |          5 | rolling_dynamic_confirmation_occupancy | deployable_estimate    |             0.0873445 |            0.584675 |            0.391059 |                   0.0968132 |
| B1_frozen_phase3_rolling_burden        |          5 | rolling_dynamic_confirmation_occupancy | deployable_estimate    |             0.0891244 |            0.575822 |            0.303494 |                   0.132448  |
| B2_high_burden_interval_classification |          5 | rolling_dynamic_confirmation_occupancy | deployable_estimate    |             0.0891244 |            0.575822 |            0.303494 |                   0.132448  |
| B4_long_burst_sustained_state          |         60 | rolling_dynamic_confirmation_occupancy | deployable_estimate    |             0.0442923 |            0.564911 |            0.294841 |                   0.0430471 |
| B1_frozen_phase3_rolling_burden        |         60 | rolling_dynamic_confirmation_score     | deployable_estimate    |             0.0785334 |            0.561926 |            0.278206 |                   0.0653465 |
| B2_high_burden_interval_classification |         60 | rolling_dynamic_confirmation_score     | deployable_estimate    |             0.0785334 |            0.561926 |            0.278206 |                   0.0653465 |
| B1_frozen_phase3_rolling_burden        |          5 | rolling_best_loaded_refined_variants   | diagnostic_upper_bound |             0.0624531 |            0.561672 |            0.294564 |                   0.105264  |
| B2_high_burden_interval_classification |          5 | rolling_best_loaded_refined_variants   | diagnostic_upper_bound |             0.0624531 |            0.561672 |            0.294564 |                   0.105264  |
| B1_frozen_phase3_rolling_burden        |         10 | rolling_dynamic_confirmation_occupancy | deployable_estimate    |             0.0322223 |            0.558939 |            0.294909 |                   0.0894105 |
| B2_high_burden_interval_classification |         10 | rolling_dynamic_confirmation_occupancy | deployable_estimate    |             0.0322223 |            0.558939 |            0.294909 |                   0.0894105 |
| B4_long_burst_sustained_state          |         30 | rolling_dynamic_confirmation_occupancy | deployable_estimate    |             0.0584467 |            0.558273 |            0.292431 |                   0.0663725 |
| B1_frozen_phase3_rolling_burden        |          5 | rolling_dynamic_confirmation_score     | deployable_estimate    |             0.0291759 |            0.555542 |            0.282856 |                   0.0654007 |

## Event vs Burden Comparison

| branch                                   | target_type                        | primary_metric                         |   primary_metric_value | deployability_class    | interpretation                                                          |
|:-----------------------------------------|:-----------------------------------|:---------------------------------------|-----------------------:|:-----------------------|:------------------------------------------------------------------------|
| current_event_detector                   | discrete_event                     | Tier3 median recall                    |               0.039368 | deployable_estimate    | current Stage F event detector remains low recall                       |
| best_bounded_alarm_remediation           | discrete_event                     | A1_leaky_evidence_integrator           |               0.145639 | proxy                  | best bounded event reconstruction remains below 0.25 recall at 5 FP/min |
| best_widened_scoring_sensitivity         | discrete_event_scoring_sensitivity | S1_onset_tolerance_pm600ms             |               0.856359 | proxy                  | event recall is materially sensitive to scoring tolerance               |
| best_deployable_burden_state_estimate    | burden_state                       | rolling_dynamic_confirmation_occupancy |               0.584675 | deployable_estimate    | best deployable causal burden/state estimate                            |
| best_diagnostic_burden_state_upper_bound | burden_state                       | rolling_best_loaded_refined_variants   |               0.561672 | diagnostic_upper_bound | diagnostic upper bound from loaded causal refined variants              |

Recommendation: `EVENT_TARGET_METRIC_SENSITIVE`.

Final task state: `pre_adr_recommends_event_target_metric_sensitive`.

Rationale: event recall improves materially only under widened scoring definitions; target metric requires owner review before ADR reopening

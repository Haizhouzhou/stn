# Phase 5_2C Burden/State Ceiling Comparison

Burden/state targets are derived from frozen Phase 3 labels for evaluation only. Phase 3 labels and thresholds are not used as input features. Prior Phase 5Z evidence is context only and does not override causal Phase 5_2C evidence.

## Target Availability

| target_id                              | target_available   | target_derivation                                                            | target_tautology_risk                       |
|:---------------------------------------|:-------------------|:-----------------------------------------------------------------------------|:--------------------------------------------|
| B1_frozen_phase3_rolling_burden        | True               | frozen Phase 3 event labels aggregated over rolling sampled-window histories | evaluation_target_from_phase3               |
| B2_high_burden_interval_classification | True               | high-burden binary target derived from fold-local burden quantiles           | evaluation_target_from_phase3_quantile      |
| B3_recent_event_density_state          | True               | recent true-event density over rolling sampled-window histories              | evaluation_target_from_phase3               |
| B4_long_burst_sustained_state          | True               | long-burst target from frozen duration/offset metadata when available        | evaluation_target_from_phase3_duration_rule |

## Top Causal Burden/State Metrics

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

## Gap To Diagnostic/Oracle Ceilings

| target_id                              |   window_s | best_deployable_score_source           |   deployable_primary_metric_value |   diagnostic_primary_metric_value |   oracle_primary_metric_value |   gap_deployable_to_diagnostic |
|:---------------------------------------|-----------:|:---------------------------------------|----------------------------------:|----------------------------------:|------------------------------:|-------------------------------:|
| B1_frozen_phase3_rolling_burden        |          5 | rolling_dynamic_confirmation_occupancy |                          0.575822 |                          0.561672 |                      1        |                    -0.0141497  |
| B1_frozen_phase3_rolling_burden        |         10 | rolling_dynamic_confirmation_occupancy |                          0.558939 |                          0.538345 |                      1        |                    -0.0205941  |
| B1_frozen_phase3_rolling_burden        |         30 | rolling_dynamic_confirmation_score     |                          0.525562 |                          0.496352 |                      0.999949 |                    -0.0292096  |
| B1_frozen_phase3_rolling_burden        |         60 | rolling_dynamic_confirmation_score     |                          0.561926 |                          0.524743 |                      0.999955 |                    -0.0371835  |
| B2_high_burden_interval_classification |          5 | rolling_dynamic_confirmation_occupancy |                          0.575822 |                          0.561672 |                      1        |                    -0.0141497  |
| B2_high_burden_interval_classification |         10 | rolling_dynamic_confirmation_occupancy |                          0.558939 |                          0.538345 |                      1        |                    -0.0205941  |
| B2_high_burden_interval_classification |         30 | rolling_dynamic_confirmation_score     |                          0.525562 |                          0.496352 |                      0.999949 |                    -0.0292096  |
| B2_high_burden_interval_classification |         60 | rolling_dynamic_confirmation_score     |                          0.561926 |                          0.524743 |                      0.999955 |                    -0.0371835  |
| B3_recent_event_density_state          |          5 | rolling_dynamic_confirmation_occupancy |                          0.584675 |                          0.55352  |                      1        |                    -0.0311544  |
| B3_recent_event_density_state          |         10 | rolling_dynamic_confirmation_occupancy |                          0.551455 |                          0.527326 |                      0.999023 |                    -0.0241285  |
| B3_recent_event_density_state          |         30 | rolling_dynamic_confirmation_score     |                          0.486725 |                          0.455464 |                      0.999071 |                    -0.0312608  |
| B3_recent_event_density_state          |         60 | rolling_dynamic_confirmation_score     |                          0.506478 |                          0.515761 |                      0.999824 |                     0.00928309 |
| B4_long_burst_sustained_state          |          5 | rolling_dynamic_confirmation_occupancy |                          0.555222 |                          0.531687 |                      1        |                    -0.023535   |
| B4_long_burst_sustained_state          |         10 | rolling_dynamic_confirmation_occupancy |                          0.54726  |                          0.517914 |                      0.999983 |                    -0.0293461  |
| B4_long_burst_sustained_state          |         30 | rolling_dynamic_confirmation_occupancy |                          0.558273 |                          0.491165 |                      0.999961 |                    -0.0671079  |
| B4_long_burst_sustained_state          |         60 | rolling_dynamic_confirmation_occupancy |                          0.564911 |                          0.440866 |                      0.999941 |                    -0.124045   |

# Phase 5_2C Event Detection Limitation Analysis

This analysis tests whether low Stage F event recall is likely a scorer anomaly, alarm-reconstruction limitation, or information limitation. It uses causal Stage F scores and event structure; it does not infer ceilings from AUROC alone.

## Tier 1 / Tier 2 FP/min 1.0 Anomaly

| tier                              |   event_recall |   achieved_fp_per_min |   threshold |   tie_count_at_threshold | anomaly_classification                 |
|:----------------------------------|---------------:|----------------------:|------------:|-------------------------:|:---------------------------------------|
| tier1_continuous                  |     0.00214521 |             0.0482169 |     5.80423 |                        0 | coarse_threshold_grid_artifact         |
| tier2_quantized                   |     0.0230292  |             0.574334  |     1       |                     6966 | quantization_tie_threshold_jump        |
| tier3_quantized_mismatched_median |     0.00256152 |             0.0660018 |     1.04252 |                        0 | mismatch_seed_threshold_grid_fragility |

Interpretation: the same event evaluator is used across tiers. Tier 1 is limited by a coarse continuous threshold grid that jumps from a very conservative point to a point above the 1 FP/min cap. Tier 2 quantization creates a max-bin threshold at 1.0 with many tied scores, yielding a larger feasible operating point. Tier 3 seed aggregation masks that Tier 2 behavior. No measurement bug was identified.

## Empirical Ceiling Gap

| ceiling_type                              |   fp_per_min |   observed_recall |   ceiling_recall |   recall_gap | gap_interpretation                                                          |
|:------------------------------------------|-------------:|------------------:|-----------------:|-------------:|:----------------------------------------------------------------------------|
| score_ranking_ceiling_dense_threshold     |          0.5 |        0.00256152 |        0.0193754 |   0.0168139  | observed close to ceiling; likely information-limited for this score source |
| event_window_aggregation_ceiling          |          0.5 |        0.00256152 |        0.019444  |   0.0168825  | observed close to ceiling; likely information-limited for this score source |
| subset_score_ceiling_best_refined_variant |          0.5 |        0.00256152 |        0.0180628 |   0.0155013  | observed close to ceiling; likely information-limited for this score source |
| score_ranking_ceiling_dense_threshold     |          1   |        0.00256152 |        0.0374285 |   0.034867   | moderate recoverable alarm reconstruction slack                             |
| event_window_aggregation_ceiling          |          1   |        0.00256152 |        0.0383982 |   0.0358367  | moderate recoverable alarm reconstruction slack                             |
| subset_score_ceiling_best_refined_variant |          1   |        0.00256152 |        0.037693  |   0.0351315  | moderate recoverable alarm reconstruction slack                             |
| score_ranking_ceiling_dense_threshold     |          2   |        0.039368   |        0.0695772 |   0.0302092  | moderate recoverable alarm reconstruction slack                             |
| event_window_aggregation_ceiling          |          2   |        0.039368   |        0.0708017 |   0.0314337  | moderate recoverable alarm reconstruction slack                             |
| subset_score_ceiling_best_refined_variant |          2   |        0.039368   |        0.0677651 |   0.0283971  | moderate recoverable alarm reconstruction slack                             |
| score_ranking_ceiling_dense_threshold     |          5   |        0.13336    |        0.150929  |   0.0175682  | observed close to ceiling; likely information-limited for this score source |
| event_window_aggregation_ceiling          |          5   |        0.13336    |        0.15325   |   0.0198897  | observed close to ceiling; likely information-limited for this score source |
| subset_score_ceiling_best_refined_variant |          5   |        0.13336    |        0.137861  |   0.00450102 | observed close to ceiling; likely information-limited for this score source |

## Per-Subject Distribution Summary

| tier             |   fp_per_min | summary_stat   |   event_recall |   near_zero_subject_count |   subjects_recall_gt_0_10 |   subjects_recall_gt_0_25 |
|:-----------------|-------------:|:---------------|---------------:|--------------------------:|--------------------------:|--------------------------:|
| tier1_continuous |          0.5 | median         |    0.000202026 |                        19 |                         0 |                         0 |
| tier1_continuous |          0.5 | p05            |    0           |                        19 |                         0 |                         0 |
| tier1_continuous |          0.5 | p95            |    0.00722889  |                        19 |                         0 |                         0 |
| tier1_continuous |          1   | median         |    0.000202026 |                        19 |                         0 |                         0 |
| tier1_continuous |          1   | p05            |    0           |                        19 |                         0 |                         0 |
| tier1_continuous |          1   | p95            |    0.00722889  |                        19 |                         0 |                         0 |
| tier1_continuous |          2   | median         |    0.0191453   |                         5 |                         3 |                         1 |
| tier1_continuous |          2   | p05            |    0           |                         5 |                         3 |                         1 |
| tier1_continuous |          2   | p95            |    0.174691    |                         5 |                         3 |                         1 |
| tier1_continuous |          5   | median         |    0.13908     |                         0 |                        15 |                         3 |
| tier1_continuous |          5   | p05            |    0.0167398   |                         0 |                        15 |                         3 |
| tier1_continuous |          5   | p95            |    0.330141    |                         0 |                        15 |                         3 |
| tier2_quantized  |          0.5 | median         |  nan           |                       nan |                       nan |                       nan |
| tier2_quantized  |          0.5 | p05            |  nan           |                       nan |                       nan |                       nan |
| tier2_quantized  |          0.5 | p95            |  nan           |                       nan |                       nan |                       nan |
| tier2_quantized  |          1   | median         |    0.00691861  |                         8 |                         3 |                         0 |
| tier2_quantized  |          1   | p05            |    0           |                         8 |                         3 |                         0 |
| tier2_quantized  |          1   | p95            |    0.115136    |                         8 |                         3 |                         0 |
| tier2_quantized  |          2   | median         |    0.0196171   |                         5 |                         3 |                         1 |
| tier2_quantized  |          2   | p05            |    0           |                         5 |                         3 |                         1 |
| tier2_quantized  |          2   | p95            |    0.174916    |                         5 |                         3 |                         1 |
| tier2_quantized  |          5   | median         |    0.144413    |                         0 |                        15 |                         3 |
| tier2_quantized  |          5   | p05            |    0.0177446   |                         0 |                        15 |                         3 |
| tier2_quantized  |          5   | p95            |    0.332693    |                         0 |                        15 |                         3 |

The tracked per-subject table uses hashed subject keys and does not expose raw pseudonymous IDs.

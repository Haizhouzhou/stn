# Phase 5_2C Scoring Tolerance Sensitivity

This analysis changes event scoring windows without changing causal scores. Widened and interval scoring rows are sensitivity analyses only, not deployment claims.

|   fp_per_min_target |   current_policy_best_recall | best_scoring_definition    |   best_recall |   recall_improvement_factor | tolerance_interpretation                                  |
|--------------------:|-----------------------------:|:---------------------------|--------------:|----------------------------:|:----------------------------------------------------------|
|                 0.5 |                    0.0195518 | S1_onset_tolerance_pm600ms |      0.36014  |                    18.4198  | event recall is materially sensitive to scoring tolerance |
|                 1   |                    0.0378105 | S1_onset_tolerance_pm600ms |      0.508111 |                    13.4383  | event recall is materially sensitive to scoring tolerance |
|                 2   |                    0.0694891 | S1_onset_tolerance_pm600ms |      0.661713 |                     9.52255 | event recall is materially sensitive to scoring tolerance |
|                 5   |                    0.143788  | S1_onset_tolerance_pm600ms |      0.856359 |                     5.95572 | event recall is materially sensitive to scoring tolerance |

# Phase 5_2C Architectural Target Reassessment

The accepted Stage B architecture remains `hybrid_early_warning`, but the event-target evidence no longer supports treating current low-FP discrete event detection as a passing engineering gate.

## Strategic Options

| option_id                                 | option_title                                                  | recommended_option   | recommendation_reason                                                                                                                                                            |
|:------------------------------------------|:--------------------------------------------------------------|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OPTION_A_DEFER_TARGET_PENDING_REMEDIATION | Defer target approval pending bounded event-level remediation | False                | not selected under current ceiling/gap interpretation                                                                                                                            |
| OPTION_B_DISCOVERY_ONLY_BRIAN2_FRAMING    | Discovery-only G/H/I and later Brian2 characterization        | False                | not selected under current ceiling/gap interpretation                                                                                                                            |
| OPTION_C_REOPEN_ADR_BURDEN_STATE          | Reopen Stage B ADR toward burden/state tracking               | True                 | diagnostic score-ranking ceiling from deployable score sources remains below 0.10 recall at FP/min 2.0, indicating low-FP discrete event detection is likely information-limited |
| OPTION_D_STOP_EVENT_BRANCH_INSUFFICIENT   | Stop low-FP event-detection branch as insufficient            | False                | not selected under current ceiling/gap interpretation                                                                                                                            |

## Burden / State-Tracking Context

Prior Phase 5 burden-state summaries exist and are used only as historical/diagnostic context. The compact burden estimator summary reports best Pearson correlation about 0.295 and best high-burden AUROC about 0.749. The Phase 5Z closeout did not justify deployment, but it supports treating burden/state tracking as a serious target question rather than ignoring it.

## Recommendation

Recommended option: `OPTION_C_REOPEN_ADR_BURDEN_STATE`.

Rationale: diagnostic score-ranking ceiling from deployable score sources remains below 0.10 recall at FP/min 2.0, indicating low-FP discrete event detection is likely information-limited

This recommendation does not execute Stage G/H/I, does not authorize Brian2, and does not select Outcome 1. It asks the owner to treat burden/state tracking as a serious architectural reassessment path if low-FP event recall remains information-limited after bounded remediation analysis.

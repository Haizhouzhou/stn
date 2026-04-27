# Phase 5_2C Event-Target Decision Package, Revised

Status: current low-recall target candidates rejected by project owner.

The headline finding is low event recall, not procedural approval of a low-recall gate. The previous recommendation of `CAND_2C_BALANCED_FP2` is not owner-approved. That candidate would have accepted about 3.9% Tier 3 median recall at 2 FP/min, which is not adequate as a passing engineering target for Stage G/H/I execution or a Brian2 gate decision.

No Stage G/H/I execution was performed. No Brian2 simulation was run. No closeout or frozen Brian2 specification was created.

## Stage F Event Recall

|   FP/min |     Tier 1 | Tier 2             |   Tier 3 median |
|---------:|-----------:|:-------------------|----------------:|
|      0.5 | 0.00214521 | NA                 |      0.00256152 |
|      1   | 0.00214521 | 0.0230291513204294 |      0.00256152 |
|      2   | 0.039319   | 0.039554110179453  |      0.039368   |
|      5   | 0.133875   | 0.1373324974531776 |      0.13336    |

## Analytical Clarifications

- The Tier 1 / Tier 2 FP/min 1.0 inconsistency was analyzed and is documented as a threshold-grid / quantization discontinuity rather than an identified scorer bug.
- The empirical ceiling analysis was performed from causal scores and event structure, not AUROC alone.
- Per-subject recall distributions were summarized with hashed subject keys only.

## Owner Decision Needed

The project owner must decide whether to remediate the event detector, run only discovery-style characterization, reopen the architecture toward burden/state tracking, or close the event-detection branch as insufficient.

Current reassessment recommendation: `OPTION_C_REOPEN_ADR_BURDEN_STATE`.

Recommendation reason: diagnostic score-ranking ceiling from deployable score sources remains below 0.10 recall at FP/min 2.0, indicating low-FP discrete event detection is likely information-limited

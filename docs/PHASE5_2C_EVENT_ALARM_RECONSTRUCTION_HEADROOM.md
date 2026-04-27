# Phase 5_2C Event Alarm Reconstruction Headroom

This document reports bounded causal score-to-alarm strategies A0-A5. Thresholds were selected on training subjects only and evaluated on held-out subjects. Strategy choice is analytical and posthoc; it is not an approved detector or target gate.

|   fp_per_min_target | tier                             | strategy_id                  | param_id   |    recall |   precision |        F1 |   fp_per_min_achieved | headroom_interpretation                                                                       |
|--------------------:|:---------------------------------|:-----------------------------|:-----------|----------:|------------:|----------:|----------------------:|:----------------------------------------------------------------------------------------------|
|                 0.5 | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.021119  |    0.404196 | 0.0401409 |              0.507621 | bounded reconstruction improves event recall but requires owner review before target approval |
|                 1   | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.0393778 |    0.392969 | 0.0715636 |              0.992952 | bounded reconstruction improves event recall but requires owner review before target approval |
|                 2   | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.0728097 |    0.373901 | 0.121891  |              1.98369  | best bounded event reconstruction remains below 0.10 recall at 2 FP/min                       |
|                 5   | tier3_quantized_mismatched_proxy | A1_leaky_evidence_integrator | tau150ms   | 0.145639  |    0.342657 | 0.204358  |              4.54203  | best bounded event reconstruction remains below 0.25 recall at 5 FP/min                       |

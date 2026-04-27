# Phase 5_2C Stage G/H/I Execution Plan

Status: plan only. Stage G/H/I were not executed.

Final task state for this readiness branch: `readiness_not_execute_plan_only`.

## Readiness Row

`results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv`:

```tsv
readiness_state	target_status	event_metrics_computed	can_execute_stage_g_h_i_now	can_plan_stage_g_h_i	qc_reason	support_status	qc_status	source_table	source_columns	source_lineage	config_hash	random_seed	uses_future_samples	uses_phase3_threshold	uses_phase3_duration_rule	uses_test_participant_statistics	tautology_risk_level	leakage_risk_level	SNN_compatible	DYNAP_candidate
ready_to_plan_g_h_i	unavailable	True	False	True	Historical FP/min and latency references exist, but no explicit Phase 5_2C event-level recall target/gate was found.	proxy	warning	causal_three_tier_event_summary.tsv	target_status,event metrics	Stage G/H/I readiness assessment	b43c5edce66220e2	20260426	False	False	False	False	low	low	True	True
```

Branch decision: `ready_to_plan_g_h_i` means write/update this plan and stop. Do not execute Stage G/H/I from this state.

## Artifact Sanity Check

- Recovered Stage F event metric tables exist:
  - `causal_tier1_event_metrics.tsv`, 4234 bytes, mtime `2026-04-27 06:15:26 +0200`
  - `causal_tier2_event_metrics.tsv`, 4249 bytes, mtime `2026-04-27 06:15:26 +0200`
  - `causal_tier3_event_metrics.tsv`, 112494 bytes, mtime `2026-04-27 06:15:26 +0200`
  - `stage_g_h_i_readiness_assessment.tsv`, 720 bytes, mtime `2026-04-27 06:15:26 +0200`
- Stage F event validation passed according to `causal_stage_f_event_output_validation_summary.tsv`.
- The validation summary contains:
  - `no_g_h_i_or_brian2_execution_tables=True`
  - `no_huge_duplicate_prediction_table=True`
  - `no_meg_in_event_outputs=True`
  - `no_phase3_label_file_modified=True`
- No Phase 5_2C Stage G/H/I execution outputs were found during the narrow output check. Existing non-Phase-5_2C frozen config `configs/nsm_mono_frozen.yaml` is unrelated to this branch.
- No Brian2 outputs were created for this branch.
- No MEG was introduced in Stage F event outputs. Some upstream FIF paths contain a BIDS `/meg/` directory component while naming `_lfp.fif` files; Stage F validation explicitly passed `no_meg_in_event_outputs`.
- No duplicate huge event/alarm trace table was created. The existing huge held-out prediction table is pre-existing and was not duplicated by this branch.
- `causal_event_alarm_trace_summary.tsv` remains untracked because it contains per-subject pseudonymous IDs. It is summarized by tracked aggregate event outputs, and its checksum at this inspection was `4a1faa0abf182b6bef96bb547948da1668c004235cfa51e619ef07cd936c1112`.
- `causal_stage_f_event_output_validation_summary.tsv` remains untracked because blank terminal fields are serialized as trailing tabs. Do not force-commit it as-is only to satisfy Git whitespace checks.

## Source Metrics For Future Execution

Minimum sufficient subset from `causal_minimum_sufficient_subset.tsv`:

- `causal_derivative_on_count__h150__smooth50ms__p95`
- `causal_rise_slope__h100__smooth10ms__p85`

Stage D subset metrics from `causal_minimum_sufficient_subset.tsv`:

- LOSO AUROC: `0.7490318279546455`
- LOSO AUPRC: `0.6346940773803722`
- per-subject IQR AUROC: `0.040531987479782705`
- subset_has_dynamic_confirmation: `True`
- baseline_ratio_alone_for_alarm: `False`

Stage F window/proxy summary from `causal_three_tier_summary.tsv`:

- macro_state: `causal_cdef_completed_but_project_target_unavailable`
- project_target_available: `False`
- Tier 1 AUROC: `0.7490318279546455`
- Tier 2 AUROC: `0.7490208407164056`
- Tier 3 median AUROC: `0.7415369389295998`
- target_dependent_gate_status: `unavailable`

Stage F event target resolution from `event_scoring_policy.tsv`:

- target_status: `unavailable`
- primary_fp_min_grid: `[0.5, 1.0, 2.0, 5.0]`
- primary_event_metric: `recall_at_fp_min_grid`
- secondary_metrics: `AUROC,AUPRC,precision,F1,latency,one_alarm_per_burst_fraction`
- target_dependent_pass_fail: `unavailable`
- qc_reason: `Historical FP/min and latency references exist, but no explicit Phase 5_2C event-level recall target/gate was found.`

## Execution Blocker

The branch is plan-only because the readiness row has `target_status=unavailable` and `can_execute_stage_g_h_i_now=False`.

Required target-definition actions before any future Stage G/H/I execution:

1. Define an explicit Phase 5_2C event-level operating target in config or ADR form.
2. Include at least the FP/min gate, event recall target, committed-alarm latency policy, and one-alarm-per-burst expectation.
3. Regenerate or revalidate Stage F event readiness so `stage_g_h_i_readiness_assessment.tsv` moves to `ready_to_execute_g_h_i`.
4. Keep target-dependent gates unavailable until the target source is explicit. Do not infer a pass/fail target from historical references alone.

## Future Stage G Plan: SNN Approximation Engineering

Run this stage only after the readiness state is `ready_to_execute_g_h_i`.

Inputs:

- `configs/phase5_2c.yaml`
- `configs/phase5_2c_causal_frontend.yaml`
- `causal_feature_matrix.tsv`
- `causal_minimum_sufficient_subset.tsv`
- Stage F event outputs

Feature scope:

- Use only the two-feature causal minimum sufficient subset for alarm commitment.
- Do not add baseline ratio to the commitment path because it is not in the selected minimum sufficient subset and `baseline_ratio_alone_for_alarm=False`.
- Do not include boundary/veto features unless Stage D/F evidence is updated to justify them.

Components to approximate:

- D1 ON-count / event population for `causal_derivative_on_count__h150__smooth50ms__p95`
- D1 rise-slope / short-window integration population for `causal_rise_slope__h100__smooth10ms__p85`
- threshold/coincidence approximation
- refractory/self-inhibition approximation
- score/fusion state needed to preserve the Stage E/Stage F fold-local score path

Implementation outputs, if execution later becomes allowed:

- `src/stnbeta/phase5_2c/snn_approximation.py`
- `scripts/05_2c_snn_approximation.py`
- `scripts/05_2c_validate_stage_g_outputs.py`
- `tests/test_phase5_2c_snn_approximation.py`
- `docs/PHASE5_2C_SNN_APPROXIMATION_DESIGN.md`
- `results/tables/05_phase5/phase5_2c/snn_approximation_fidelity.tsv`
- `results/tables/05_phase5/phase5_2c/population_size_sweep.tsv`
- `results/tables/05_phase5/phase5_2c/refined_tier3.tsv`
- `results/tables/05_phase5/phase5_2c/stage_g_output_validation_summary.tsv`

Parameter plan:

- population sizes: `16`, `32`, `64`, `128`
- tune bounded tau, threshold, population size, and quantized weights
- use stochastic rounding for parallel connection counts
- preserve causality and fold-local isolation
- reuse Stage F mismatch assumptions unless config updates them:
  - time constants sigma/mu = `0.25`
  - synaptic weights sigma/mu = `0.20`
  - refractory periods sigma/mu = `0.20`
  - thresholds sigma/mu = `0.15`

Fidelity metrics:

- component-level correlation, MAE, RMSE where meaningful
- AUROC/AUPRC preservation against the ideal continuous pipeline
- event metric preservation where Stage F event outputs support it
- `NA` plus qc_reason for unsupported target-dependent metrics

Stage G gate:

- approximation fidelity adequate for selected components
- causal status preserved
- quantization-aware approximation does not catastrophically degrade metrics
- refined Tier 3 not catastrophically worse than Stage F Tier 3
- tests and validator pass
- target-dependent gates remain `unavailable` until an explicit target exists

## Future Stage H Plan: DYNAP-SE1 Feasibility Audit

Run this stage only if Stage G executes and passes its gate.

Audit scope:

- neuron count
- synapse count per post-neuron
- CAM slots per post-neuron
- parallel connection count
- required bias groups
- core count
- spike traffic estimate
- mismatch sensitivity
- reliance on impossible precise per-neuron continuous parameters

DYNAP-SE1 constraints:

- binary on/off synapses
- 64 CAM slots per neuron
- parallel connections for graded weights, target `<= 8` per pair
- shared biases per 256-neuron core
- avoid reliance on neuron 0 of core 0
- avoid documented CAM clash patterns where applicable
- spike traffic below expected saturation risk
- target total core count `<= 4` where possible

Implementation outputs, if execution later becomes allowed:

- `src/stnbeta/phase5_2c/dynap_audit.py`
- `scripts/05_2c_dynap_audit.py`
- `scripts/05_2c_validate_stage_h_outputs.py`
- `tests/test_phase5_2c_dynap_audit.py`
- `results/tables/05_phase5/phase5_2c/dynap_resource_audit.tsv`
- `results/tables/05_phase5/phase5_2c/dynap_core_allocation.md`
- `results/figures/05_phase5/phase5_2c/dynap_resource_breakdown.png`
- `results/tables/05_phase5/phase5_2c/stage_h_output_validation_summary.tsv`

Stage H gate:

- resource audit complete
- no hard DYNAP-SE1 constraints exceeded unless explicitly documented as failure
- CAM slot and parallel-connection constraints satisfied
- total core count feasible, preferably `<= 4`
- spike traffic risk not catastrophic
- validator and tests pass
- resource feasibility alone cannot justify Brian2 Outcome 1 while target_status remains `unavailable`

## Future Stage I Plan: Closeout And Brian2 Gate Decision

Run this stage only if Stage H executes and passes its gate.

Implementation outputs, if execution later becomes allowed:

- `src/stnbeta/phase5_2c/closeout.py`
- `scripts/05_2c_closeout.py`
- `scripts/05_2c_validate_stage_i_outputs.py`
- updated `tests/test_phase5_2c_outputs.py`, if needed
- `docs/PHASE5_2C_CLOSEOUT.md`
- `results/tables/05_phase5/phase5_2c/closeout_summary.tsv`
- `results/figures/05_phase5/phase5_2c/closeout_overview.png`
- `results/tables/05_phase5/phase5_2c/stage_i_output_validation_summary.tsv`

Outcome rules:

- Outcome 1 is forbidden while `target_status=unavailable`.
- If G/H later pass but target remains unavailable, choose the appropriate non-Outcome-1 closeout, most likely Outcome 2 unless SNN or DYNAP evidence indicates a stronger failure mode.
- Do not create frozen Brian2 specs unless Outcome 1 is selected under the required rules.
- Do not run Brian2 in Phase 5_2C G/H/I planning or closeout.

## Validation And Compute Rules For Future Execution

If future readiness becomes `ready_to_execute_g_h_i`, use strong immediately available resources according to `AGENTS.md` for production work. Before production jobs:

- `bash -n` for new or modified Slurm scripts
- `python -m py_compile` for new or modified scripts
- `python -m py_compile` for relevant `src/stnbeta/phase5_2c/*.py`
- focused pytest for G/H/I tests
- `git diff --check`
- required Slurm preflight and post-job accounting

After production jobs:

- trust outputs only after `sacct` confirms `COMPLETED`
- run validators for produced stages
- run focused tests
- run `git diff --check`

## Non-Claims

- No Stage G/H/I execution was performed in this branch.
- No Brian2 simulation was run.
- No DYNAP-SE1 deployment claim is made.
- No final detector is claimed.
- No Phase 3 labels were relabeled.
- No MEG is introduced.
- No unsupported metric is filled with zero.

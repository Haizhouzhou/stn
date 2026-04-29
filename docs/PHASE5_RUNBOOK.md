# Phase 5 Runbook

This document covers the implemented Phase 5 stack and the 2026-04-24 recovery closeout:

- clustered Brian2/Brian2CUDA duration-bucket state machine
- topology-only synthetic proof
- end-to-end synthetic validation through the frozen Phase 4 front end
- real-data dev evaluation against frozen Phase 3 labels
- lightweight QC panel and focused multi-GPU sweep orchestration
- Phase 5W/5X/5Y/5Z closeout evidence for representation gap, predictive compensation,
  target reconciliation, and burden-state detector testing

## Scope

Included:
- STN-LFP only
- frozen Phase 4 filter-bank + rectify/amplify + LI&F encoder inputs
- frozen Phase 3 burst labels as the primary reference target
- Brian2 runtime for CPU smoke/debug
- Brian2CUDA for focused sweeps

Explicitly excluded:
- MEG
- Phase 3 relabeling/redefinition
- Phase 6 LOSO cohort evaluation as the main task
- SG-BPTT / feedback-control training as the main path
- hardware deployment

## Environment boot

The current project environment for future GPU / training / Brian2CUDA / JAX / PyTorch
work is `stn_env`, booted through the repository script:

```bash
cd /scratch/haizhe/stn
source /scratch/haizhe/stn/start_stn.sh
python scripts/validate_stn_env.py
```

Every new interactive compute-node session and every new SLURM batch script should source
`/scratch/haizhe/stn/start_stn.sh` after changing into the repository. The script
activates `stn_env`, loads CUDA, sets CUDA/Brian2CUDA library paths, preserves
SLURM-provided `CUDA_VISIBLE_DEVICES`, and sets JAX memory behavior. The legacy
`--xla_cpu_use_numa=false` flag is skipped when the installed JAX rejects it.
`.venv-phase4` remains legacy / archival and should be used only for exact reproduction
of older Phase 4/5 commands that require it.

## Current Phase 5 decision

The current Phase 4/5/5X/5Y/5Z detector path is closed for Phase 6 and hardware deployment.
Phase 5W showed that the offline Phase 3 oracle passes while causal Phase 3-like replay
and R1-R6 causal/DYNAP-compatible representations fail low-FP recovery. Phase 5X tested
SNN-native predictive compensation, including adaptive Smith prediction; the best
predictive branch improved some diagnostics but did not pass. Phase 5Y kept Phase 3
labels fixed and added phenotype-aligned online-control metrics over the same burst
intervals. It found long-burst and burden metrics scientifically useful, but existing
causal detector traces still do not meet the controlled low-FP gate under onset,
interval, long-burst, or burden scoring. Phase 5Z then built explicit burden-state and
long-burst targets from the frozen Phase 3 intervals and tested slow burden integrators,
multi-timescale burden populations, adaptive-baseline burden, sparse burden readouts,
population-coded burden estimators, long-burst state detectors, and hybrid burden/alarm
signals. The best causal burden trace reached only weak correlation, long-burst detectors
still had uncontrolled false alarm time, and hardware feasibility estimates did not rescue
the software gate.

Use Phase 5Y outputs when discussing target validity:

```bash
python scripts/05y_run_all_target_reconciliation.py
```

Primary outputs:

- `results/tables/05_phase5/target_reconciliation/pd_beta_phenotype_audit.md`
- `results/tables/05_phase5/target_reconciliation/label_timing_variance_summary.md`
- `results/tables/05_phase5/target_reconciliation/metric_definitions.md`
- `results/tables/05_phase5/target_reconciliation/detector_rescore_all_metrics.tsv`
- `results/tables/05_phase5/target_reconciliation/phase5y_target_decision.md`

Do not interpret interval, long-burst, or burden readouts as Phase 3 relabeling. The
Phase 3 Tinkhauser burst intervals remain the frozen offline reference; Phase 5Y changes
only the online-control metric used to judge whether an alarm is useful.

Use Phase 5Z outputs when discussing whether the biologically aligned burden/long-burst
target rescues the current causal representation path:

```bash
python scripts/05z_run_all_burden_state.py
```

Primary outputs:

- `results/tables/05_phase5/burden_state/burden_target_summary.tsv`
- `results/tables/05_phase5/burden_state/baseline_burden_score_summary.tsv`
- `results/tables/05_phase5/burden_state/burden_estimator_summary.tsv`
- `results/tables/05_phase5/burden_state/long_burst_detector_summary.tsv`
- `results/tables/05_phase5/burden_state/hybrid_burden_alarm_summary.tsv`
- `results/tables/05_phase5/burden_state/realistic_synthetic_burden_summary.tsv`
- `results/tables/05_phase5/burden_state/full_dev_burden_summary.tsv`
- `results/tables/05_phase5/burden_state/phase5z_burden_state_closeout.md`

Phase 5Z outcome: burden and long-burst state targets remain scientifically meaningful,
but the current causal Phase 4/5/5W/5X traces do not yet provide a deployable
DYNAP-compatible burden or long-burst state detector. Do not proceed to Phase 6 or
hardware from this path.

## Primary architecture

Phase 5 uses a hand-designed monotonic duration-bucket state machine:

- direct beta evidence enters only `D0`
- later duration states are reached only by forward bucket-to-bucket routing
- state clusters are `D_idle`, `D0`, `D1`, `D2`, `D3`, `D4`
- default cluster template: `32` excitatory + `8` inhibitory neurons per state
- reset is driven by beta absence through the quiet/reset path
- readout uses stable occupancy of `D2+`

Current frozen design choice for the interrupted synthetic case:
- a `20 ms` quiet gap is treated as reset-sized, i.e. `60 on / 20 off / 60 on` is allowed to split

## Key configs

- filter bank: `configs/filter_bank.yaml`
- frozen LI&F encoder: `configs/lif_encoder.yaml`
- Phase 5 base config: `configs/nsm_mono.yaml`
- Phase 5 frozen config: `configs/nsm_mono_frozen.yaml`
- focused sweep lattice: `configs/gridsearch_phase5.yaml`
- midpoint / holdoff probes used during real-dev tuning:
  - `configs/gridsearch_phase5_focused.yaml`
  - `configs/gridsearch_phase5_holdoff.yaml`
  - `configs/gridsearch_phase5_midpoint.yaml`

Important closeout distinction:
- `configs/nsm_mono_frozen.yaml` is a provisional engineering freeze.
- It is not a scientific acceptance freeze because real dev still misses AUC/latency targets.
- The recovered interface uses explicit `top2_mean` beta pooling before D0, D2 dwell `80 ms`,
  occupancy threshold `0.04`, and quiet holdoff `100 ms`.

## CPU validation order

Topology-only synthetic proof:

```bash
source /scratch/haizhe/stn/start_stn.sh
python scripts/05a_validate_state_machine_synthetic.py \
  --level topology \
  --backend runtime \
  --no-grid \
  --out results/phase5_synthetic/dev_runtime_topology
```

End-to-end synthetic proof through frozen Phase 4:

```bash
python scripts/05a_validate_state_machine_synthetic.py \
  --level end_to_end \
  --backend runtime \
  --grid-config configs/gridsearch_phase5_focused.yaml \
  --out results/phase5_synthetic/recovery2_end_to_end_runtime_interface_final
```

Primary real-data dev case:

```bash
python scripts/05b_run_phase5_dev.py \
  --subject sub-0cGdk9 \
  --conditions MedOff_Hold \
  --channels LFP-left-01 \
  --band-mode fixed_13_30 \
  --backend runtime \
  --grid-config configs/gridsearch_phase5_focused.yaml \
  --grid-indices 2 \
  --out results/phase5_real_dev/probes/recovery2_one_channel_interface_grid2
```

Full dev architecture attempt:

```bash
python scripts/05b_run_phase5_dev.py \
  --subject sub-0cGdk9 \
  --conditions MedOff_Hold \
  --band-mode fixed_13_30 \
  --backend runtime \
  --grid-config configs/gridsearch_phase5_focused.yaml \
  --grid-indices 2 \
  --architectures consensus,pooled_entry \
  --consensus-k 1,2 \
  --consensus-window-ms 25 \
  --out results/phase5_real_dev/probes/recovery2_full_dev_architecture_grid2
```

Long GPU dev runs also write per-case partial progress files under
`results/phase5_real_dev/progress/`.

QC panel:

```bash
python scripts/05c_phase5_qc_panel.py \
  --backend runtime \
  --nsm-config configs/nsm_mono_frozen.yaml \
  --out results
```

As of this closeout, the QC panel is deferred. The dev subject is specific enough
for engineering iteration, but latency and recall are not stable enough for a
cross-subject readiness check.

## Preferred GPU path

Use the exact validated interactive allocation first:

```bash
srun --partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal \
  --gres=gpu:6 --cpus-per-task=32 --mem=128G --time=04:00:00 --pty bash
```

Inside the allocation:

```bash
cd /scratch/haizhe/stn
source /scratch/haizhe/stn/start_stn.sh
python scripts/validate_stn_env.py
python scripts/05d_phase5_gpu_sweep.py \
  --modes topology,end_to_end,dev,qc \
  --gpus 0,1,2,3,4,5 \
  --backend cuda_standalone \
  --subject sub-0cGdk9 \
  --conditions MedOff_Hold \
  --band-mode fixed_13_30 \
  --out results
```

The exact allocation was validated on job `2552763` with six Tesla V100-SXM2-32GB GPUs.
`scripts/validate_phase4_env.py` completed a true Brian2CUDA `full_run`; logs are under
`results/logs/05_phase5/recovery2_*`. Focused synthetic CUDA sweeps still suffer from
slow sequential standalone rebuilds across synthetic trace lengths; interrupted runs
are preserved under `results/phase5_synthetic/recovery2_end_to_end_gpu_interface_final*`.

Batch fallback:

```bash
sbatch slurm/slurm_phase5_gpu.sh
```

For new batch scripts, use `slurm/slurm_template_stn_env.sh` as the starting point and
keep the `source /scratch/haizhe/stn/start_stn.sh` line. The older Phase 4/5 batch
scripts are retained for historical reproduction.

## Outputs

- synthetic outputs: `results/phase5_synthetic/`
- real dev outputs: `results/phase5_real_dev/`
- front-end attribution outputs: `results/phase5_frontend_ablation/`
- figures: `results/figures/05_phase5/`
- tables: `results/tables/05_phase5/`
- logs: `results/logs/05_phase5/`

Each stage writes a manifest with:

- config hash
- seed
- backend
- git commit if available
- CUDA / GPU visibility

## Current residuals to watch

- topology-only synthetic passes after recovery: short rejection `1.0`, threshold/long detection `1.0`,
  reset `0.9`, skipped transitions `0`, negative FP `0`.
- best recovered end-to-end synthetic short-rejecting point: short rejection `1.0`, true detection `0.5`,
  reset `0.6`, skipped transitions `0`, negative FP `0`.
- full dev per-channel result: median AUC `0.803`, mean FP `0.86/min`, median latency `131 ms`;
  targets `AUC >= 0.85` and latency `<= 50 ms` are still missed.
- latency decomposition shows Phase 4 evidence and D0/D2 often lead the frozen Phase 3 onset, while
  stable readout lags; this is consistent with a label-alignment plus dwell/readout persistence gap.
- full real-data runtime runs are slow on 736k-sample files; use preserved progress files and GPU only
  when the standalone compile cost is justified.

## Front-End Attribution Ablation

This path asks whether the Phase 5 miss/latency problem is caused mainly by the
Phase 4 front end rather than by the monoNSM topology. It keeps Phase 3 labels frozen
and keeps the duration-bucket monoNSM as the primary downstream reference.

Part A baseline audit:

```bash
python scripts/05f_frontend_attribution_audit.py \
  --out results/phase5_frontend_ablation/baseline_audit
```

Part B same-filter encoder comparison:

```bash
python scripts/05g_frontend_ablation_synthetic.py \
  --level end_to_end \
  --parts B \
  --out results/phase5_frontend_ablation/synthetic_part_b

python scripts/05h_frontend_ablation_real_dev.py \
  --parts B \
  --candidates b0_lif_baseline \
  --channels LFP-left-01 \
  --out results/phase5_frontend_ablation/real_dev_part_b_one_channel_b0
```

Run `b1_analog_quasi`, `b2_adm`, and `b3_onoff` similarly when a full-length
single-channel Brian2 runtime comparison is needed.

Part C filtering/evidence comparison:

```bash
python scripts/05g_frontend_ablation_synthetic.py \
  --level end_to_end \
  --parts C \
  --candidates c2_single_beta_lif,c4_mean_pool_lif,c4_max_pool_lif,c6_lowfreq_veto_lif \
  --out results/phase5_frontend_ablation/synthetic_part_c

python scripts/05h_frontend_ablation_real_dev.py \
  --parts C \
  --candidates c2_single_beta_lif \
  --channels LFP-left-01 \
  --out results/phase5_frontend_ablation/real_dev_part_c_one_channel_c2
```

Across-channel pooling checks:

```bash
python scripts/05h_frontend_ablation_real_dev.py \
  --parts C \
  --candidates c5_across_mean_lif \
  --out results/phase5_frontend_ablation/real_dev_part_c_across_mean
```

Repeat with `c5_across_top1_lif` and `c5_across_top2_lif`.

Part D backup comparator:

```bash
python scripts/05g_frontend_ablation_synthetic.py \
  --level end_to_end \
  --parts D \
  --out results/phase5_frontend_ablation/synthetic_part_d

python scripts/05h_frontend_ablation_real_dev.py \
  --parts D \
  --out results/phase5_frontend_ablation/real_dev_part_d_swta_all_channels
```

Current attribution result:
- Duplicate alarms are not the dominant failure mode.
- LI&F is not clearly the main bottleneck; analog and ADM are worse on recall, and ON/OFF is too noisy.
- Single pooled beta helps synthetic specificity but not the real dev channel.
- Across-channel pooled entry does not rescue the dev case.
- Band-sWTA + duration counter reaches high AUC but FP/min remains far above target.
- The next scientific move should focus on interpretable persistence/readout/eventization logic, not a broad black-box classifier.

## Phase 5P Fusion Probe

Phase 5P is a scientific probe stage before any next Phase 5 patch. It keeps Phase 3
labels frozen, keeps the Phase 4 front end as the baseline input, and tests which
evidence streams should serve candidate opening, validation, reset/veto, channel
fusion, and event-level alarm roles.

Login-node-safe preserved-output evidence table:

```bash
python scripts/05j_fusion_probe_evidence_table.py
```

Synthetic role-fusion probe through the frozen Phase 4 front end:

```bash
python scripts/05k_fusion_probe_synthetic.py
```

Real-dev preserved-output/event-proxy comparison:

```bash
python scripts/05l_fusion_probe_real_dev.py
```

Closeout report:

```bash
python scripts/05m_fusion_probe_report.py
```

Outputs:
- `results/phase5_fusion_probe/baseline_index.tsv`
- `results/tables/05_phase5/fusion_probe/evidence_role_table.tsv`
- `results/tables/05_phase5/fusion_probe/synthetic_fusion_summary.tsv`
- `results/tables/05_phase5/fusion_probe/real_dev_fusion_summary.tsv`
- `results/tables/05_phase5/fusion_probe/fusion_probe_closeout.md`
- figures under `results/figures/05_phase5/fusion_probe/`

Current Phase 5P closeout:
- LI&F remains useful sustained beta-rate evidence, but not sufficient eventization.
- ON/OFF and sWTA help timing/spectral validity but are too noisy as final detectors.
- ADM is not supported as a standalone detector; only slope-gate use remains plausible.
- D2-entry validation and compact mixtures were tested, but no synthetic candidate passed
  the full short-rejection, threshold/long-detection, reset, skipped-transition, and
  negative-control FP gate.
- nnNSM/frequency-drift tracking is implemented as a lightweight probe and carried only
  as a comparison, not the main architecture.
- Recommended outcome: document the Phase 5 architecture/readout limitation before
  patching the monoNSM.

## Phase 5Q Decisive Validation and Adjudication

Phase 5Q is a targeted adjudication stage. It does not reopen Phase 3 labels, does
not change the frozen Phase 4 front end, and does not patch the monoNSM detector.

Login-node-safe audits:

```bash
python scripts/05o_audit_d2_occupancy_alignment.py
python scripts/05q_synthetic_oracle_check.py
python scripts/05s_phase5_adjudication_report.py
```

Real LFP/FIF audits must run under SLURM:

```bash
sbatch --partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal \
  --cpus-per-task=4 --mem=24G --time=00:45:00 \
  --chdir=/scratch/haizhe/stn \
  --output=results/logs/05_phase5/adjudication/label_alignment_%j.out \
  --error=results/logs/05_phase5/adjudication/label_alignment_%j.err \
  --wrap=".venv-phase4/bin/python scripts/05p_label_alignment_audit.py"

sbatch --partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal \
  --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --chdir=/scratch/haizhe/stn \
  --output=results/logs/05_phase5/adjudication/fresh_real_dev_%j.out \
  --error=results/logs/05_phase5/adjudication/fresh_real_dev_%j.err \
  --wrap=".venv-phase4/bin/python scripts/05r_validate_best_proxy_real_dev.py"
```

Always check SLURM completion before trusting outputs:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
```

Key outputs:
- `results/tables/05_phase5/adjudication/d2_occupancy_audit.md`
- `results/tables/05_phase5/adjudication/label_alignment_audit.md`
- `results/tables/05_phase5/adjudication/synthetic_oracle_audit.md`
- `results/tables/05_phase5/adjudication/fresh_real_dev_summary.tsv`
- `results/tables/05_phase5/adjudication/phase5_adjudication_closeout.md`

Current Phase 5Q closeout:
- Outcome A: metric bug found.
- The D2+ occupancy metric now anchors at D2 onset when available. Older Phase 5P
  TP-vs-miss D2+ occupancy comparisons are invalid.
- Label alignment is a major latency interpretation issue and future reports should
  use dual Phase 3 and causal/envelope latency references.
- The interval oracle passes the synthetic gate; the gate is achievable by ideal timing.
- Fresh raw all-channel validation does not validate `lif_swta_d2_entry_event_proxy`
  as a patch candidate.

## Phase 5R Corrected D2 Recovery

Phase 5R is a focused recovery step, not a broad architecture search. It keeps Phase 3
labels frozen, keeps the Phase 4 front end as the baseline input, and does not change
the monoNSM topology. The added interface layer is limited to beta-burst-specific
entry/maintenance/reset hysteresis and D2-entry candidate eventization.

Login-node-safe audits:

```bash
python scripts/05t_recompute_corrected_d2_summaries.py
python scripts/05u_audit_phase5_auc_and_matching.py
python scripts/05v_fit_beta_burst_interface.py
python scripts/05x_phase5_recovery_report.py
```

Fresh real LFP/FIF validation must run under SLURM:

```bash
sbatch --partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal \
  --cpus-per-task=8 --mem=64G --time=06:00:00 \
  --chdir=/scratch/haizhe/stn \
  --output=results/logs/05_phase5/recovery/fresh_dev_recovery_%j.out \
  --error=results/logs/05_phase5/recovery/fresh_dev_recovery_%j.err \
  --wrap=".venv-phase4/bin/python scripts/05w_validate_phase5_recovery_dev.py"
```

Always check:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
```

Key outputs:
- `results/tables/05_phase5/recovery/corrected_d2_summary.tsv`
- `results/tables/05_phase5/recovery/auc_matching_audit.md`
- `results/tables/05_phase5/recovery/synthetic_recovery_summary.tsv`
- `results/tables/05_phase5/recovery/fresh_dev_recovery_summary.tsv`
- `results/tables/05_phase5/recovery/phase5_recovery_closeout.md`

Current Phase 5R closeout:
- Outcome 3: D2 state representation is not useful on real data.
- Corrected baseline D2 support is not selective: TP support is 1.0, miss support is
  ~0.97, and FP support is 1.0, with median D2+ occupancy 4 ms across groups.
- Candidate-specific AUC is not fabricated for event-only candidates; `shared_score_auc`
  remains ~0.803 for the baseline score.
- Synthetic recovery did not pass the gate. The high-recall lenient hysteresis candidate
  produced negative-control FPs and very high real-dev FP/min.
- Fresh real-dev validation job `2556207` completed successfully, but no candidate met
  recall, FP/min, latency, and interpretability targets together.

## Phase 5T Trainable SNN / NSM Diagnostic Route

Phase 5T is a gate-driven training diagnostic after Phase 5R. It keeps Phase 3 labels
and the Phase 4 front end frozen, does not start Phase 6 LOSO, and does not deploy to
DYNAP-SE1 hardware. Teacher models are software-only upper bounds, not accepted final
detectors.

Login-node-safe preserved-artifact stages:

```bash
.venv-phase4/bin/python scripts/05t_prepare_training_data.py
.venv-phase4/bin/python scripts/05t_teacher_diagnostic.py
.venv-phase4/bin/python scripts/05t_train_spiking_readout.py
.venv-phase4/bin/python scripts/05t_train_input_state_mono_nsm.py
.venv-phase4/bin/python scripts/05t_train_architecture_hedges.py
.venv-phase4/bin/python scripts/05t_hardware_aware_eval.py
.venv-phase4/bin/python scripts/05t_small_qc_panel.py
.venv-phase4/bin/python scripts/05t_phase5_training_report.py
```

Use SLURM before any rerun that regenerates dense Phase 4 spike/current traces from
raw LFP, extracted FIF, MNE objects, or GPU-heavy training jobs. Always check:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
```

Key outputs:
- `results/phase5_training/data/dev_epoch_index.tsv`
- `results/phase5_training/data/dev_training_data_manifest.json`
- `results/tables/05_phase5/training/e1_teacher_summary.tsv`
- `results/phase5_training/e1_teacher_threshold_sweep.tsv`
- `results/tables/05_phase5/training/phase5t_training_closeout.md`

Current Phase 5T closeout:
- Outcome 1: information-limited preserved traces or unresolved target/evidence mismatch.
- E0 prepared 10-second epoch artifacts from frozen Phase 3 labels plus preserved Phase
  4/5R event summaries; no raw LFP/FIF data were loaded on the login node.
- E1 teacher AUC was ~0.572 and recall at <=1 FP/min was 0.0, so E2-E6 were gate-skipped.
- DYNAP-SE1 quantization, mismatch, CAM, and core audits are implemented, but no trained
  candidate reached E5 in this run.

## Phase 5T-2 Dense-Trace Trainable Audit

Phase 5T-2 resolves the main limitation of Phase 5T: the preserved-artifact teacher used
proposal/event summaries rather than dense regenerated traces. This audit regenerates
dense Phase 4/5 trace-level evidence under SLURM, keeps Phase 3 labels frozen, and then
runs only the gates justified by the evidence.

D0 dense trace regeneration must not run on the login node. The completed dev run used:

```bash
sbatch --partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal \
  --cpus-per-task=8 --mem=64G --time=04:00:00 \
  --chdir=/scratch/haizhe/stn \
  --output=results/logs/05_phase5/training_dense/d0_dense_%j.out \
  --error=results/logs/05_phase5/training_dense/d0_dense_%j.err \
  --wrap=".venv-phase4/bin/python scripts/05y_prepare_dense_training_traces.py"
```

Check every SLURM job before trusting outputs:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
```

Dense-stage scripts:

```bash
.venv-phase4/bin/python scripts/05y_train_dense_teacher.py
.venv-phase4/bin/python scripts/05y_train_constrained_spiking_eventizer.py
.venv-phase4/bin/python scripts/05y_compare_preserved_vs_dense_teacher.py
.venv-phase4/bin/python scripts/05y_train_dense_input_state_mono_nsm.py
.venv-phase4/bin/python scripts/05y_train_dense_architecture_hedges.py
.venv-phase4/bin/python scripts/05y_hardware_aware_dense_eval.py
.venv-phase4/bin/python scripts/05y_small_qc_dense_panel.py
.venv-phase4/bin/python scripts/05y_phase5_dense_report.py
```

Current Phase 5T-2 closeout:
- Outcome 1: dense traces information-limited under the tested causal models.
- D0 SLURM job `2557754` completed and produced 370 ten-second channel-epochs across
  five usable `sub-0cGdk9 MedOff_Hold fixed_13_30` channels.
- D1 dense causal teacher improved sample separability versus preserved artifacts
  (`AUC ~0.834`, `PR-AUC ~0.287`) but had recall `0.0` at <=1 FP/min.
- D2 constrained spiking eventizer passed CAM/fan-in feasibility (`20/64` CAM slots)
  but failed event recovery with recall ~0.051 at <=1 FP/min and best F1 ~0.089.
- D3-D6 were gate-skipped; no trainable input-to-state, hedge, hardware-aware survivor,
  QC panel, Phase 6 LOSO, MEG, or hardware deployment was run.

Key outputs:
- `results/phase5_training_dense/data/dense_training_data_manifest.json`
- `results/tables/05_phase5/training_dense/d1_dense_teacher_summary.tsv`
- `results/tables/05_phase5/training_dense/d2_spiking_eventizer_summary.tsv`
- `results/tables/05_phase5/training_dense/d1_preserved_vs_dense_teacher.md`
- `results/tables/05_phase5/training_dense/phase5t2_dense_closeout.md`

## Phase 5V High-Throughput Dense-Branch Audit

Phase 5V expands the dense-trace audit into a high-throughput branch comparison over
the representation ladder, teacher ladder, constrained spiking eventizer, trainable
input-to-state monoNSM, heterogeneous population monoNSM, onset-gated duration
architecture, and feedback-control/bias optimization. It keeps Phase 3 labels and the
Phase 4 front end frozen. It does not run MEG, Phase 6 LOSO, hardware deployment, or a
generic black-box final detector.

The production continuation used sbatch, not interactive `srun --pty`. The completed
resume job was:

```bash
sbatch results/logs/05_phase5/highthroughput/phase5v_resume_max.sbatch
```

SLURM job `2558332` completed on `2026-04-25` with an exclusive teaching node,
`--gres=gpu:6`, `--cpus-per-task=80`, `--mem=0`, and `--time=08:00:00`. It resumed
after previous job `2557922` had written `feedback_control_iter063_sweep.tsv`.

Resume behavior:
- `scripts/05v_run_all_highthroughput_branches.py` skips completed branch summaries.
- `scripts/05v_feedback_control_bias_opt.py` skips complete
  `feedback_control_iter*_sweep.tsv` files and computes only missing iterations.
- The resume skipped feedback-control iterations `000`-`063` and computed `064`-`071`.

Current Phase 5V closeout:
- Outcome 1: current representations are information-limited.
- Best controlled-FP branch was the spiking eventizer with recall `~0.078` at
  <=1 FP/min and best event F1 `~0.327`.
- Trainable input-to-state reached best event F1 `~0.329` only at high FP/min and
  recall `0.0` at <=1 FP/min.
- Teacher, heterogeneous population, onset-gated, and feedback-control branches did
  not exceed `~0.018` recall at <=1 FP/min.
- Hardware-aware audit checked 34 promising candidates; CAM/core constraints were
  plausible, but quantized/mismatch re-evaluation was not recomputed for summary-only
  candidates and no software event gate passed.

Key outputs:
- `results/tables/05_phase5/highthroughput/branch_comparison_summary.tsv`
- `results/tables/05_phase5/highthroughput/phase5v_highthroughput_closeout.md`
- `results/tables/05_phase5/highthroughput/hardware_aware_summary.tsv`
- `results/phase5_highthroughput/phase5v_branch_status.json`
- `results/logs/05_phase5/highthroughput/phase5v_resume_max_2558332.out`

## Phase 5X Predictive-Compensation Study

Phase 5X is a focused continuation after Phase 5W. It keeps Phase 3 labels frozen and
tests whether causal, SNN-native predictive compensation can bridge the
offline-to-causal representation gap. It does not run MEG, Phase 6 LOSO, DYNAP-SE1
hardware deployment, Phase 3 relabeling, or a generic black-box final detector.

Production execution uses sbatch, not interactive `srun --pty`:

```bash
sbatch results/logs/05_phase5/predictive/phase5x_predictive_max.sbatch
```

The production script requests the strongest immediately available teaching-node V100
configuration first: one exclusive node, 8 V100 GPUs, 80 CPUs, `--mem=0`, and 8 hours.
It logs CUDA visibility and PyTorch availability. Historical Phase 5X execution used
`.venv-phase4`; future learned predictive-compensation or causal-representation reruns
should use `stn_env` via `source /scratch/haizhe/stn/start_stn.sh` so PyTorch, JAX,
CuPy, and Numba availability is checked before any GPU allocation is trusted. If a branch falls
back to deterministic NumPy/SciPy, record that it is not a GPU-accelerated learned SNN
candidate.

Stage scripts:

```bash
source /scratch/haizhe/stn/start_stn.sh
python scripts/05x_latency_budget_audit.py
python scripts/05x_build_predictive_representations.py
python scripts/05x_run_predictive_compensation_branches.py
python scripts/05x_run_realistic_synthetic_predictive.py
python scripts/05x_run_hardware_aware_predictive_audit.py
python scripts/05x_predictive_report.py
```

Key outputs:
- `results/tables/05_phase5/predictive/latency_budget_summary.tsv`
- `results/tables/05_phase5/predictive/delay_model_summary.tsv`
- `results/tables/05_phase5/predictive/imc_parameter_derivation.tsv`
- `results/tables/05_phase5/predictive/predictive_compensation_summary.tsv`
- `results/tables/05_phase5/predictive/smith_predictor_summary.tsv`
- `results/tables/05_phase5/predictive/real_dev_predictive_summary.tsv`
- `results/tables/05_phase5/predictive/hardware_aware_predictive_summary.tsv`
- `results/tables/05_phase5/predictive/phase5x_predictive_closeout.md`

Current Phase 5X closeout:
- Outcome 2: predictive compensation improves but does not pass low-FP event gates.
- Adaptive Smith was the best supported predictive branch (`~0.038` recall at <=1
  FP/min) but did not beat the current R3 baseline (`~0.049`).
- Constant-shift timing realignment partially improves causal evidence (`~0.224` recall
  at <=1 FP/min for Phase 4 beta evidence shifted -200 ms), but it is diagnostic only.
- The job allocated 8 V100 GPUs and CUDA reported all GPUs visible. `.venv-phase4` did
  not provide PyTorch, so Phase 5X did not run GPU-accelerated learned predictive
  compensation.

## Phase 5_2A Ground-Truth-Guided Feature Atlas

Phase 5_2A is a diagnostic atlas stage before any new detector design. It keeps frozen
Phase 3 Tinkhauser intervals unchanged, uses STN-LFP only, and characterizes true
labeled bursts against matched beta-like non-burst windows. It does not relabel Phase 3,
train a final detector, run Phase 6, use MEG, or deploy DYNAP-SE1 hardware.

Production execution must use `stn_env` via the repository boot script:

```bash
cd /scratch/haizhe/stn
source /scratch/haizhe/stn/start_stn.sh
python scripts/validate_stn_env.py --require-gpu --brian2cuda-smoke
sbatch --test-only slurm/slurm_phase5_2a_feature_atlas.sh
sbatch slurm/slurm_phase5_2a_feature_atlas.sh
```

The atlas reconstructs per-file raw provenance by exact matching reproduced fixed-band
Phase 3 event tuples against the frozen Phase 3 parquet events, then discards unmatched
recomputed events. This matching is for locating raw windows only; the frozen Phase 3
labels remain the authority.

Primary outputs:
- `results/tables/05_phase5/feature_atlas/window_index.tsv`
- `results/tables/05_phase5/feature_atlas/time_series_features.tsv`
- `results/tables/05_phase5/feature_atlas/spectral_features.tsv`
- `results/tables/05_phase5/feature_atlas/spatial_features.tsv`
- `results/tables/05_phase5/feature_atlas/feature_separability_summary.tsv`
- `results/tables/05_phase5/feature_atlas/causal_separability_over_time.tsv`
- `results/tables/05_phase5/feature_atlas/low_power_feature_set_ranking.tsv`
- `results/tables/05_phase5/feature_atlas/phase5_2a_feature_atlas_report.md`
- `docs/PHASE5_2A_FEATURE_ATLAS.md`

## Phase 5 Final Decision Package

The final Phase 5 package consolidates Phase 5W and Phase 5X into a supervisor-ready
decision:

- `docs/PHASE5_FINAL_DECISION.md`
- `results/tables/05_phase5/phase5_final_decision_package.md`
- `results/tables/05_phase5/phase5_final_decision_summary.tsv`
- `results/tables/05_phase5/phase5_environment_caveat.md`
- `results/figures/05_phase5/phase5_offline_to_causal_gap_summary.png`
- `results/figures/05_phase5/phase5_predictive_compensation_summary.png`

Decision:
- Do not proceed to Phase 6 LOSO.
- Do not proceed to DYNAP-SE1 hardware deployment.
- Do not run more hand-designed monoNSM patches as the next primary route.
- If continuing scientifically, first create a GPU-capable training environment and run a
  focused learned predictive-compensation or causal-representation branch; otherwise
  write the limitation result.

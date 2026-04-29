# Brian2 SNN State-Machine Handover

This handover is for a Codex agent or engineer porting the legacy Brian2/Brian2CUDA
state-machine implementation into a new repository. It explains what exists in this
repository, what each component does, and the safest order for rebuilding it elsewhere.

For a shorter teammate quick-start focused only on the status-machine simulation files,
see `docs/BRIAN2_STATUS_MACHINE_TEAM_HANDOFF.md`.

This is a porting guide, not a performance claim. The legacy hand-designed onset-alarm
path did not pass the original low-false-positive event-scoring gate and must remain
documented as negative evidence. The Brian2 state machine is still useful as an
engineering reference for STN-LFP-only spiking state logic, duration buckets, reset
behavior, and Brian2CUDA project reuse.

## Scope Boundaries

Port only the STN-LFP state-machine simulation and its lightweight validation harnesses.
Do not add MEG to the detector pipeline, do not modify Phase 3 labels, and do not treat
legacy Phase 3 labels as biological facts or as the final operational reference
standard. In a new beta-state or beta-burden project, Phase 3 artifacts should be
considered frozen legacy reference labels unless a new target review explicitly defines
candidate reference labels.

The old Phase 4/5 Brian2 implementation supports:

- filter-bank plus rectify/amplify conversion from one STN LFP channel to encoder currents;
- LI&F-like encoder neurons driven by those currents;
- a quiet/reset drive derived from beta absence;
- a monotonic duration-bucket state machine;
- Brian2 runtime, C++ standalone, and Brian2CUDA standalone execution paths;
- synthetic validation of duration progression, reset, short-event rejection, and readout plumbing;
- real-data dev evaluation against frozen Phase 3 reference artifacts.

The old implementation does not establish:

- a final STN contact-pair choice;
- a final beta-state or beta-burden label set;
- a deployable DYNAP-SE1 detector;
- successful Phase 6 or hardware readiness;
- a passed low-false-positive onset-alarm detector.

## Source Map

Core Brian2 implementation:

| Legacy path | Role |
| --- | --- |
| `src/stnbeta/snn_brian2/runner.py` | Main runner module. Contains both Phase 4 and Phase 5 network builders, runtime execution, standalone project reuse, quiet-drive derivation, beta pooling, and result dataclasses. |
| `src/stnbeta/snn_brian2/neuron_model.py` | Brian2 equation strings for encoder, quiet/reset, Phase 4 buckets, Phase 4 readout, Phase 5 excitatory clusters, Phase 5 inhibitory clusters, and Phase 5 readout. |
| `src/stnbeta/snn_brian2/synapse_model.py` | Small Brian2 `on_pre` snippets for direct voltage increments and Phase 5 conductance increments. |
| `src/stnbeta/snn_brian2/topologies/nsm_monotonic_duration.py` | Config dataclasses and YAML loaders for the Phase 4 compact model and Phase 5 clustered model. |
| `src/stnbeta/snn_brian2/__init__.py` | Exports the Phase 4 public API only; Phase 5 callers import directly from `runner.py` and the topology module. |

Front-end and readout helpers:

| Legacy path | Role |
| --- | --- |
| `src/stnbeta/preprocessing/filter_bank.py` | Butterworth band-pass filter bank. Supports zero-phase offline filtering and causal filtering. |
| `src/stnbeta/preprocessing/rectify_amplify.py` | Full-wave rectify, optional low-pass smoothing, percentile normalization, power/gain, and clipping. |
| `src/stnbeta/encoding/lif_encoder.py` | Converts filtered bands into rectified currents and contains a small LI&F encoder smoke runner. |
| `src/stnbeta/phase5/readout.py` | Converts Phase 5 occupancy/readout traces into a continuous score, stable mask, and event table. |
| `src/stnbeta/phase5/metrics.py` | Synthetic and real-data scoring utilities. Useful for tests, but do not port as an acceptance claim without reviewing targets. |

Primary configs:

| Legacy path | Role |
| --- | --- |
| `configs/filter_bank.yaml` | Default bands: boundary low, four beta bands from 13 to 30 Hz, boundary high. |
| `configs/lif_encoder.yaml` | LI&F time constant, threshold, refractory period, and rectify/amplify settings. |
| `configs/nsm_mono.yaml` | Base Phase 4 and Phase 5 state-machine parameters. |
| `configs/nsm_mono_frozen.yaml` | Provisional engineering freeze used in later Phase 5 recovery work. Do not interpret it as scientific acceptance. |

Validation and run scripts:

| Legacy path | Role |
| --- | --- |
| `scripts/04b_validate_state_machine_synthetic.py` | Phase 4 compact duration-bucket synthetic validation. |
| `scripts/05a_validate_state_machine_synthetic.py` | Phase 5 clustered duration-bucket synthetic validation, topology-only or end-to-end through the front end. |
| `scripts/05b_run_phase5_dev.py` | Real-data dev runner for one subject/channel/condition path. Loads extracted LFP and Phase 3 reference artifacts. |
| `tests/test_state_machine_synthetic.py` | Lightweight tests for the compact Phase 4 state machine. |
| `tests/test_phase5_state_machine.py` | Lightweight Phase 5 runtime tests plus a guarded Brian2CUDA smoke test. |
| `docs/PHASE4_RUNBOOK.md` | Historical Phase 4 runbook. |
| `docs/PHASE5_RUNBOOK.md` | Historical Phase 5 runbook and closeout context. |

## End-to-End Data Flow

The intended signal path is one STN LFP trace at a time:

1. Load or provide one STN LFP channel as a one-dimensional array with sampling rate.
2. Apply the configured beta filter bank.
3. Full-wave rectify and amplify each filtered band into nonnegative encoder currents.
4. Provide `encoder_currents` with shape `(n_inputs, n_steps)` and a same-length
   `band_roles` sequence.
5. Optionally pool beta evidence before state-machine entry.
6. Derive or provide `quiet_drive`, a one-dimensional reset drive with length `n_steps`.
7. Run the Brian2 state machine.
8. Convert spikes into state occupancy and readout traces.
9. Apply dwell and gap-bridging rules to produce stable readout masks or events.
10. Evaluate only against the target/scoring standard approved for the new project.

Minimal Phase 5-style pseudocode:

```python
from stnbeta.encoding.lif_encoder import currents_from_filtered_bands, load_lif_encoder_config
from stnbeta.phase5.readout import build_readout_summary
from stnbeta.preprocessing.filter_bank import apply_filter_bank, load_filter_bank_config
from stnbeta.preprocessing.rectify_amplify import load_rectify_amplify_config
from stnbeta.snn_brian2.runner import (
    derive_quiet_drive,
    prepare_phase5_entry_currents,
    run_duration_bucket_state_machine,
)
from stnbeta.snn_brian2.topologies.nsm_monotonic_duration import (
    load_duration_bucket_cluster_config,
)

filter_cfg = load_filter_bank_config("configs/filter_bank.yaml")
lif_cfg = load_lif_encoder_config("configs/lif_encoder.yaml")
rectify_cfg = load_rectify_amplify_config("configs/lif_encoder.yaml")
nsm_cfg = load_duration_bucket_cluster_config("configs/nsm_mono_frozen.yaml")

filtered = apply_filter_bank(lfp_1d, sfreq_hz, filter_cfg, causal=False)
band_names = [band.name for band in filter_cfg.bands]
band_roles = [band.role for band in filter_cfg.bands]
currents = currents_from_filtered_bands(
    filtered,
    band_names,
    rectify_config=rectify_cfg,
    sfreq_hz=sfreq_hz,
    causal=False,
)

model_currents, model_roles = prepare_phase5_entry_currents(
    currents,
    band_roles,
    mode=nsm_cfg.evidence_aggregation_mode,
)
quiet = derive_quiet_drive(model_currents, model_roles)
result = run_duration_bucket_state_machine(
    model_currents,
    model_roles,
    nsm_cfg,
    backend="runtime",
    quiet_drive=quiet,
    seed=0,
)
readout = build_readout_summary(result, nsm_cfg)
```

When porting to a new repository, replace the import paths with the new package names.
Keep the array shapes and role semantics unchanged until tests pass.

## Front-End Contract

The filter-bank config defines six rows in this order:

| Name | Band | Role |
| --- | --- | --- |
| `boundary_low` | 9.0-13.0 Hz | `boundary` |
| `beta_1` | 13.0-17.25 Hz | `beta` |
| `beta_2` | 17.25-21.5 Hz | `beta` |
| `beta_3` | 21.5-25.75 Hz | `beta` |
| `beta_4` | 25.75-30.0 Hz | `beta` |
| `boundary_high` | 30.0-40.0 Hz | `boundary` |

`apply_filter_bank` accepts a one-dimensional trace or a two-dimensional
`(n_channels, n_samples)` array. For this state-machine path, use one STN LFP channel at
a time unless the new repository deliberately defines a multi-channel state input.

`rectify_and_amplify` performs:

- absolute value;
- optional low-pass smoothing;
- percentile normalization per row;
- optional power transform;
- gain and offset;
- optional clipping.

The default `configs/lif_encoder.yaml` settings are:

- `dt_ms: 1.0`;
- `tau_ms: 15.0`;
- `refractory_ms: 4.0`;
- `threshold: 1.0`;
- `gain: 1.0`;
- rectify/amplify gain `2.0`, power `1.25`, 95th-percentile normalization, clip max `4.0`,
  and smoothing at `25 Hz`.

Important porting rule: the Brian2 state machines assume the time step in the config is
aligned with the current samples. The legacy front end uses `1 ms` samples in the
synthetic and dev paths.

## Quiet/Reset Drive

`derive_quiet_drive(encoder_currents, band_roles)` computes reset evidence from beta
absence:

1. select rows whose role is `beta`;
2. average those beta rows, or average all rows if no beta role is present;
3. clip to nonnegative beta energy;
4. scale by the 95th percentile plus a small epsilon;
5. return `clip(1.0 - beta_energy / scale, 0.0, 1.5)` as `float32`.

The result has shape `(n_steps,)`. In Brian2 it drives the single quiet/reset neuron
through a `TimedArray`. Quiet spikes reset state activity. In Phase 5, the quiet
refractory period is the maximum of `quiet_refractory_ms` and `quiet_holdoff_ms`, so
holdoff directly changes reset timing.

## Phase 4 Compact State Machine

The compact implementation is the earlier, simpler model. It is mainly useful as a
minimal reference for the monotonic duration-bucket idea.

Public objects:

- config dataclass: `MonotonicStateMachineConfig`;
- loader: `load_nsm_config`;
- result dataclass: `StateMachineResult`;
- runner: `run_state_machine`;
- standalone wrapper: `StandaloneStateMachineProject`.

Default duration bucket thresholds are `100`, `200`, `350`, and `500 ms`. The config
stores these in milliseconds and converts them to dimensionless Brian2 thresholds using
`bucket_threshold_scale`.

Network groups:

| Group | Size | Purpose |
| --- | ---: | --- |
| `encoder` | one neuron per input band | LI&F neuron driven by `encoder_drive(t, i)`. |
| `quiet` | 1 | LI&F reset neuron driven by `quiet_drive(t)`. |
| `buckets` | one neuron per threshold | Monotonic duration buckets. |
| `readout` | 1 | Spikes when the configured bucket and later buckets fire enough. |

Key Phase 4 synapses:

- `beta_syn`: beta encoder rows connect only to bucket `0`;
- `boundary_syn`: non-beta rows inhibit every bucket by negative weight;
- `sustain_syn`: bucket self-recurrence;
- `forward_syn`: bucket `i` excites bucket `i+1`;
- `reset_syn`: quiet neuron resets bucket voltage;
- `readout_syn`: configured readout bucket and later buckets excite the readout neuron;
- `reset_readout`: quiet neuron resets readout voltage.

This compact path returns raw spike times, spike indices, optional bucket voltages,
bucket thresholds, band roles, and the readout bucket index. It does not compute Phase 5
occupancy traces.

## Phase 5 Clustered Duration-Bucket State Machine

The Phase 5 implementation is the main Brian2/Brian2CUDA reference for a spiking state
machine. It uses populations instead of single bucket neurons.

Public objects:

- config dataclass: `DurationBucketClusterConfig`;
- loader: `load_duration_bucket_cluster_config`;
- result dataclass: `DurationBucketRunResult`;
- runner: `run_duration_bucket_state_machine`;
- standalone wrapper: `StandaloneDurationBucketProject`;
- beta pooling helper: `prepare_phase5_entry_currents`;
- effective input weight helper: `phase5_entry_weight`;
- beta evidence helper: `aggregate_beta_evidence`.

Default state names and bounds:

| State | Lower bound | Upper bound | Interpretation |
| --- | ---: | ---: | --- |
| `D_idle` | 0 ms | none | baseline/idle state cluster. |
| `D0` | 0 ms | 50 ms | entry bucket. |
| `D1` | 50 ms | 100 ms | short sustained evidence. |
| `D2` | 100 ms | 200 ms | default readout threshold state. |
| `D3` | 200 ms | 400 ms | longer duration state. |
| `D4` | 400 ms | none | longest duration state. |

Default cluster sizes are `32` excitatory and `8` inhibitory neurons per state.

Phase 5 network groups:

| Group | Size | Purpose |
| --- | ---: | --- |
| `phase5_encoder` | one neuron per model current row | LI&F encoder driven by current rows. |
| `phase5_quiet` | 1 | Reset neuron driven by beta absence. |
| `phase5_exc` | `n_states * cluster_exc_size` | Excitatory state populations. |
| `phase5_inh` | `n_states * cluster_inh_size` | Inhibitory local populations. |
| `phase5_readout` | 1 | Spiking readout neuron driven by threshold state and later. |

Phase 5 excitatory dynamics use conductance-like state variables:

- `g_input` from beta input;
- `g_forward` from earlier duration states;
- `g_recurrent` from same-state recurrence;
- `g_inh` from inhibition;
- `g_reset` from quiet reset;
- optional threshold and membrane-time heterogeneity via `mismatch_cov_pct`.

Phase 5 synapse topology:

| Synapse | Source -> target | Effect |
| --- | --- | --- |
| `phase5_beta_syn` | beta encoder rows -> `D0` excitatory cluster | Adds `g_input`; direct beta evidence enters only `D0`. |
| `phase5_idle_recurrent_syn` | `D_idle` excitatory -> `D_idle` excitatory | Maintains idle activity. |
| `phase5_recurrent_syn` | each `D0`-`D4` excitatory cluster -> itself | Sustains active bucket state. |
| `phase5_e_to_i_syn` | local excitatory -> local inhibitory | Drives local inhibitory population. |
| `phase5_i_to_e_syn` | local inhibitory -> local excitatory | Adds local inhibitory conductance. |
| `phase5_forward_syn` | `D0 -> D1 -> D2 -> D3 -> D4` | Enforces monotonic duration progression. |
| `phase5_lateral_syn` | later states -> earlier states | Suppresses earlier states after progression. |
| `phase5_reset_exc_syn` | quiet -> non-idle excitatory clusters | Adds reset conductance. |
| `phase5_reset_inh_syn` | quiet -> all inhibitory clusters | Adds reset conductance. |
| `phase5_readout_syn` | readout threshold state and later -> readout neuron | Produces readout spikes. |
| `phase5_reset_readout_syn` | quiet -> readout neuron | Resets readout. |

Non-beta rows are retained by `prepare_phase5_entry_currents` for diagnostics and future
extensions, but in the current Phase 5 topology only rows marked `beta` connect to `D0`.
Do not assume boundary rows drive the Phase 5 state machine unless you deliberately add
and test that topology.

## Beta Entry Pooling

`prepare_phase5_entry_currents(currents, roles, mode=...)` is the explicit interface
between front-end rows and the Phase 5 state machine.

Supported modes:

| Mode | Behavior |
| --- | --- |
| `raw` or `none` | Keep every row unchanged. |
| `sum` | Sum beta rows into one pooled beta row. |
| `mean` | Average beta rows into one pooled beta row. |
| `max` | Use the samplewise maximum beta row. |
| `top2_mean` | Use the samplewise mean of the two largest beta rows. |

For pooled modes, the output rows are:

1. one pooled row with role `beta`;
2. all original non-beta rows, in their original order.

`phase5_entry_weight(config, roles)` compensates input weight for aggregation mode. Mean
pooling divides by the number of beta inputs; `top2_mean` divides by two when at least
two beta rows are present. The frozen engineering config uses `top2_mean`.

## Phase 5 Readout

`build_readout_summary(result, config)` converts occupancy traces into a continuous score
and stable readout mask.

The function:

1. finds the readout threshold state index from `config.readout_bucket_index`;
2. sums occupancy from that state and all later states, usually `D2+`;
3. takes the maximum of this state score and the Brian2 readout trace;
4. thresholds the score with `config.occupancy_active_threshold`;
5. applies dwell with `config.readout_dwell_ms`;
6. bridges short gaps using `config.quiet_holdoff_ms`;
7. returns stable mask, onset/offset times, and per-state active masks.

The frozen engineering config uses:

- threshold state `D2`;
- dwell `80 ms`;
- occupancy active threshold `0.040`;
- quiet holdoff `100 ms`;
- beta entry mode `top2_mean`;
- input weight `1.00`.

For a new beta-state or beta-burden project, do not inherit the old event readout as the
final target. Treat `D2+` occupancy, readout trace, and stable masks as candidate signals
to review against the new target definition.

## Brian2 Backend Pattern

Both state-machine runners support:

- `runtime`: ordinary Brian2 runtime with NumPy codegen;
- `cpp_standalone`: compiled standalone project;
- `cuda_standalone`: Brian2CUDA compiled project.

Runtime execution is simplest:

1. import Brian2 lazily;
2. `b2.start_scope()`;
3. seed Brian2;
4. create `TimedArray` inputs;
5. build groups, synapses, and monitors;
6. `network.run(duration)`;
7. collect monitor arrays into dataclasses.

Standalone execution compiles once and reuses the project:

1. create a `StandaloneDurationBucketProject` or `StandaloneStateMachineProject`;
2. build the network with zero placeholder `TimedArray` values;
3. run once to let Brian2 create code objects;
4. call `b2.device.build(..., run=False)`;
5. for each real run, call `b2.device.run(..., run_args={...})`;
6. pass replacement current arrays, quiet arrays, and tunable scalar parameters through
   `run_args`.

Treat these as build-time dimensions that require a new standalone project:

- `n_steps`;
- `n_inputs`;
- `band_roles` if they alter connectivity;
- cluster sizes;
- state names;
- readout threshold state if it changes source connectivity;
- any parameter used to initialize arrays that are not replaced through `run_args`;
- default clock time step.

Treat these as run-time parameters when already wired in `run_args`:

- encoder gain, bias, tau, threshold, reset, refractory;
- quiet gain, tau, threshold, refractory;
- synaptic time constants and weights;
- reset and readout weights;
- readout neuron tau, threshold, refractory;
- mismatch scale.

Readout-only parameters such as dwell and occupancy threshold are applied after Brian2
simulation and can be changed without recompiling, provided the same occupancy traces are
acceptable.

For `cuda_standalone`, the legacy runner calls `ensure_cuda_runtime_libraries()` before
selecting the Brian2CUDA device. In the UZH cluster environment, future compute-node
sessions should start with:

```bash
cd /scratch/haizhe/stn
source /scratch/haizhe/stn/start_stn.sh
python scripts/validate_stn_env.py
```

Do not run full real-data extraction, full FIF processing, or GPU sweeps on the login
node. Use the cluster resource policy of the target repository.

## Synthetic Validation

The Phase 5 synthetic suite lives in `src/stnbeta/phase5/synthetic_suite.py`.

Cases:

- `no_burst`;
- `short_40ms`;
- `near_threshold_90ms`;
- `threshold_crossing_120ms`;
- `sustained_200ms`;
- `long_400ms_plus`;
- `two_bursts_with_quiet_gap`;
- `interrupted_burst_60_on_20_off_60_on`;
- `decaying_burst`;
- `noisy_jittered_burst`.

The suite can generate:

- topology-only direct currents, where the current rows are already prepared synthetic
  band evidence;
- end-to-end synthetic LFP traces, which must pass through the filter bank and
  rectify/amplify front end before Brian2.

The synthetic tests check engineering behavior:

- short events below threshold should not trigger stable readout;
- sustained events should progress monotonically through duration buckets;
- later buckets should not be reached by skipping earlier ones;
- quiet gaps should reset state according to the configured holdoff;
- negative controls should not produce excessive readout;
- timing plumbing should expose encoder, `D0`, `D2`, and stable-readout markers.

These tests are necessary porting smoke tests, but they are not sufficient evidence for
a final real-data detector.

Useful legacy commands:

```bash
source /scratch/haizhe/stn/start_stn.sh

python scripts/04b_validate_state_machine_synthetic.py \
  --backend runtime \
  --no-grid \
  --out results/phase4_synthetic/dev_runtime

python scripts/05a_validate_state_machine_synthetic.py \
  --level topology \
  --backend runtime \
  --no-grid \
  --out results/phase5_synthetic/dev_runtime_topology

python scripts/05a_validate_state_machine_synthetic.py \
  --level end_to_end \
  --backend runtime \
  --grid-config configs/gridsearch_phase5_focused.yaml \
  --out results/phase5_synthetic/recovery2_end_to_end_runtime_interface_final
```

For a new repository, first port the lightweight tests and run them with synthetic data
only. Do not start real-data or GPU execution until the runtime backend passes.

## Real-Data Dev Path

The legacy real-data dev runner is `scripts/05b_run_phase5_dev.py`. It:

1. loads one or more extracted STN LFP condition cases;
2. loads legacy Phase 3 reference event artifacts;
3. reads rest/task annotations from the source FIF metadata;
4. applies the filter bank and rectify/amplify front end;
5. prepares Phase 5 entry currents and quiet drive;
6. runs the duration-bucket state machine;
7. builds the readout summary;
8. computes real-case diagnostics and latency decomposition;
9. writes tables, figures, manifests, and progress files.

Example legacy dev command:

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

Do not port this command as a default new-project workflow without first defining the
new data paths, target/scoring standard, and subject/session metadata contract. In the
new repository, prefer a data adapter that consumes staged STN-LFP-only derivatives and
explicit metadata manifests rather than reading raw mixed-modality FIF files directly.

## Minimal Port Order For A New Repository

1. Create the new package modules for configs, front end, Brian2 equations, synapses,
   runner, readout, synthetic suite, and tests.
2. Port `load_yaml` or replace it with the new repository's config loader.
3. Port `configs/filter_bank.yaml`, `configs/lif_encoder.yaml`, and a reviewed copy of
   the Phase 5 state-machine config. Keep the old frozen config clearly marked as
   legacy engineering, not final science.
4. Port `neuron_model.py` and `synapse_model.py` unchanged except for package names.
5. Port `DurationBucketClusterConfig` and its loader.
6. Port `derive_quiet_drive`, `prepare_phase5_entry_currents`, `phase5_entry_weight`,
   `_exp_filter`, state-block utilities, `_build_duration_bucket_network`,
   `_phase5_collect_result`, and `run_duration_bucket_state_machine`.
7. Port `build_readout_summary` and event-mask helpers.
8. Port only the synthetic suite and lightweight tests needed to prove runtime behavior.
9. Add the standalone project wrapper after runtime behavior matches.
10. Add Brian2CUDA smoke tests guarded on CUDA/Brian2CUDA availability.
11. Add real-data adapters only after the new repo has staged STN-LFP-only data and
    path configs.
12. Add target/scoring review docs before using readout masks as candidate labels or
    evaluation targets.

## Minimal Test Checklist

After porting, tests should cover at least:

- config loading from a mapping and from YAML;
- `derive_quiet_drive` shape, dtype, and clipping;
- raw, mean, max, sum, and `top2_mean` beta pooling;
- `phase5_entry_weight` scaling for pooled beta modes;
- Phase 5 runtime rejects `short_40ms`;
- Phase 5 runtime detects `threshold_crossing_120ms`;
- Phase 5 runtime progresses through buckets without skipping on `long_400ms_plus`;
- readout summary produces stable events only after dwell;
- standalone project reuse can run two synthetic cases with the same compiled shape;
- Brian2CUDA smoke test is skipped cleanly when CUDA is unavailable.

Legacy lightweight pytest command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest \
  tests/test_state_machine_synthetic.py \
  tests/test_phase5_state_machine.py \
  -q -p no:cacheprovider
```

In a new repository, adapt the Python path and keep tests synthetic-only by default.

## Porting Hazards

Do not silently change these details:

- current arrays are `(n_inputs, n_steps)`, while Brian2 `TimedArray` values are
  transposed to `(n_steps, n_inputs)`;
- `band_roles` length must match the number of current rows after any pooling step;
- direct beta evidence enters only `D0`;
- Phase 5 non-beta rows are not connected unless a new topology explicitly uses them;
- quiet drive must be one-dimensional and aligned to the current samples;
- `D_idle` is state index `0`; bucket spike properties subtract one from non-idle state
  indices for compatibility with older bucket metrics;
- readout source clusters are built from `config.readout_bucket_index + 1` in the full
  `state_names` list, because `D_idle` is included at index `0`;
- standalone project reuse is only valid while build-time dimensions and connectivity are
  unchanged;
- changing dwell or occupancy threshold changes postprocessing, not Brian2 spikes;
- changing filter causality changes latency interpretation and must be recorded;
- old Phase 3 reference artifacts must remain frozen unless a new approved target review
  supersedes them.

## What To Carry Forward

Carry forward:

- the Brian2 group/synapse construction pattern;
- the synthetic duration-bucket validation cases;
- the explicit beta pooling interface;
- the quiet/reset drive construction;
- the runtime-first, standalone-second validation order;
- the compile-once/reuse-many Brian2CUDA pattern;
- the evidence that the old onset-alarm path failed the low-false-positive gate.

Do not carry forward as final authority:

- the old event-scoring target;
- the old `D2+` readout as a final beta-state definition;
- the old real-data dev subject/channel choice as a final benchmark;
- the old Phase 5 frozen config as a final accepted detector;
- any claim that the Brian2 state machine is ready for Phase 6 or DYNAP-SE1 deployment.

For a new STN-only beta-state or beta-burden monitor, use this legacy implementation as
a tested spiking-state simulator template. Then define the new operational reference
standard, candidate labels, scoring metrics, and data contract before running real-data
experiments.

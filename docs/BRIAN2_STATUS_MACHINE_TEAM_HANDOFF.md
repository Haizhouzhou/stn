# Brian2 Status-Machine Team Handoff

This note is for a teammate who only needs the Brian2/Brian2CUDA status-machine simulation part of the legacy STN beta repository.

It is not a handoff for the old real-data onset-alarm project. The old Tinkhauser-style onset-alarm detector path failed the low-false-positive event gate and should stay archived as negative evidence. Reuse the state-machine mechanism, not the old target.

## What To Use

Core Brian2 state-machine implementation:

- `src/stnbeta/snn_brian2/runner.py`
- `src/stnbeta/snn_brian2/neuron_model.py`
- `src/stnbeta/snn_brian2/synapse_model.py`
- `src/stnbeta/snn_brian2/topologies/nsm_monotonic_duration.py`

Minimal front-end and synthetic support:

- `src/stnbeta/preprocessing/filter_bank.py`
- `src/stnbeta/preprocessing/rectify_amplify.py`
- `src/stnbeta/encoding/lif_encoder.py`
- `src/stnbeta/synthetic/beta_burst_generator.py`
- `src/stnbeta/phase4/config.py`
- `src/stnbeta/phase4/front_end.py`
- `src/stnbeta/phase4/gpu.py`
- `src/stnbeta/phase4/grid.py`
- `src/stnbeta/phase4/manifests.py`
- `src/stnbeta/phase4/metrics.py`
- `src/stnbeta/phase4/real_data.py`
- `src/stnbeta/phase5/grid.py`
- `src/stnbeta/phase5/metrics.py`
- `src/stnbeta/phase5/readout.py`
- `src/stnbeta/phase5/synthetic_suite.py`

Configs and runnable checks:

- `configs/filter_bank.yaml`
- `configs/lif_encoder.yaml`
- `configs/nsm_mono.yaml`
- `configs/nsm_mono_frozen.yaml`
- `configs/synthetic_beta.yaml`
- `scripts/04b_validate_state_machine_synthetic.py`
- `scripts/05a_validate_state_machine_synthetic.py`
- `tests/test_state_machine_synthetic.py`
- `tests/test_phase5_state_machine.py`

Full historical handover:

- `docs/BRIAN2_SNN_STATE_MACHINE_HANDOVER.md`

## Mental Model

The simulation has two related state machines.

Phase 4 compact model:

- one LI&F encoder neuron per input band;
- one quiet/reset neuron;
- one neuron per duration bucket;
- forward bucket progression;
- recurrent sustain;
- readout from the configured bucket and later buckets.

Phase 5 clustered model:

- encoder/input populations receive prepared beta/boundary currents;
- direct beta evidence enters only `D0`;
- excitatory state clusters represent `D_idle`, `D0`, `D1`, `D2`, `D3`, `D4`;
- inhibitory clusters provide local E/I control;
- forward synapses implement `D0 -> D1 -> D2 -> D3 -> D4`;
- recurrent synapses sustain the current state;
- later states laterally suppress earlier states;
- quiet/reset suppresses non-idle state populations and readout;
- readout is driven by stable `D2+` occupancy.

## Input Contract

Do not feed raw analog LFP directly into the state machine.

Expected input is a spike/current stream derived from STN-LFP features:

1. filter one STN LFP channel into configured bands;
2. rectify/amplify filtered bands into nonnegative currents;
3. label bands with roles such as `beta` and `boundary`;
4. optionally pool beta evidence before D0 entry;
5. derive a quiet/reset drive from beta absence;
6. run the Brian2 state machine.

The core array shape is:

```text
encoder_currents: (n_inputs, n_steps)
band_roles:       length n_inputs
quiet_drive:      (n_steps,)
```

## Minimal Phase 5 API

```python
from stnbeta.phase5.readout import build_readout_summary
from stnbeta.snn_brian2.runner import (
    derive_quiet_drive,
    prepare_phase5_entry_currents,
    run_duration_bucket_state_machine,
)
from stnbeta.snn_brian2.topologies.nsm_monotonic_duration import (
    load_duration_bucket_cluster_config,
)

config = load_duration_bucket_cluster_config("configs/nsm_mono_frozen.yaml")
model_currents, model_roles = prepare_phase5_entry_currents(
    encoder_currents,
    band_roles,
    mode=config.evidence_aggregation_mode,
)
quiet = derive_quiet_drive(model_currents, model_roles)
result = run_duration_bucket_state_machine(
    model_currents,
    model_roles,
    config,
    backend="runtime",
    quiet_drive=quiet,
    seed=0,
)
readout = build_readout_summary(result, config)
```

Use `backend="runtime"` for CPU/local synthetic checks. Use `backend="cuda_standalone"` only inside a real Slurm GPU allocation after verifying CUDA and PyTorch GPU visibility.

## CPU Synthetic Smoke Checks

From the repo root:

```bash
source /scratch/haizhe/stn/start_stn.sh

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest \
  tests/test_state_machine_synthetic.py \
  tests/test_phase5_state_machine.py \
  -q -p no:cacheprovider -k 'not cuda_smoke'
```

Topology-only Phase 5 synthetic run:

```bash
source /scratch/haizhe/stn/start_stn.sh

/scratch/haizhe/stn/stn_env/bin/python scripts/05a_validate_state_machine_synthetic.py \
  --level topology \
  --backend runtime \
  --no-grid \
  --out results/phase5_synthetic/teammate_topology_smoke
```

Older compact Phase 4 synthetic run:

```bash
source /scratch/haizhe/stn/start_stn.sh

/scratch/haizhe/stn/stn_env/bin/python scripts/04b_validate_state_machine_synthetic.py \
  --backend runtime \
  --no-grid \
  --out results/phase4_synthetic/teammate_compact_smoke
```

Do not run full FIF extraction, full real-data sweeps, GPU sweeps, or Brian2CUDA on the login node.

## Porting Checklist

1. Port the config loaders first: `phase4/config.py` and `topologies/nsm_monotonic_duration.py`.
2. Port neuron and synapse equation strings.
3. Port `runner.py` with only the runtime backend first.
4. Port `phase5/synthetic_suite.py`, `phase5/readout.py`, and the two state-machine tests.
5. Make the topology synthetic tests pass on CPU.
6. Add C++ standalone and Brian2CUDA only after CPU runtime behavior is correct.
7. Retune for the new beta-state or beta-burden target before using real data.

## Do Not Copy Blindly

- Do not treat `configs/nsm_mono_frozen.yaml` as a scientifically accepted detector.
- Do not revive the old low-FP onset-alarm task as the active target.
- Do not mix MEG into the primary detector pipeline.
- Do not use raw LFP as direct state-machine input.
- Do not run Brian2CUDA without a verified Slurm GPU allocation.
- Do not claim DYNAP-SE1 readiness from this handoff alone.

## Historical Status

The state-machine mechanism worked best as synthetic/topology engineering substrate. The old real-data onset-alarm detector did not pass the low-false-positive event-scoring path. For the new project, reuse the Brian2 state-machine scaffold and revalidate it against an explicitly defined adaptive STN beta-burden or beta-state target.

# Phase 4 Runbook

This document covers the implemented Phase 4 stack:

- synthetic beta-burst generation
- Brian2/Brian2CUDA duration-bucket validation
- real-data STN-LFP filter-bank + full-wave rectify/amplify + LI&F encoder
- frozen Phase 3 label consumption for dev-time evaluation
- CPU smoke paths and GPU multi-worker sweep path

## Scope

Included:
- STN-LFP only
- Brian2 as the primary simulator
- Brian2CUDA as the primary sweep backend
- CPU runtime fallback for smoke tests
- fixed-band (`fixed_13_30`) Phase 3 labels as the primary reference

Explicitly excluded:
- MEG loading
- Phase 3 redefinition or rerun
- Phase 5 training
- hardware deployment

## Environment

Preferred local setup:

```bash
cd ~/scratch/stn
python3.11 -m venv .venv-phase4
source .venv-phase4/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'numpy<2' -e '.[bursts,phase4,phase4-gpu,dev]'
```

The current Brian2/Brian2CUDA combination resolved by pip is compatible with
`numpy<2`; keep Phase 4 in its own env instead of mutating older Phase 3 envs.

## Validation

Login-node-safe validation:

```bash
module load cuda/12.9.1
python scripts/validate_phase4_env.py --allow-no-gpu \
    --compute-capability 8.0 \
    --cuda-runtime-version 12.9
```

Expected behavior:
- CPU Brian2 runtime smoke test must pass
- `brian2` and `brian2cuda` imports must pass
- if no GPU driver is visible, the script falls back to Brian2CUDA code-generation-only smoke

## Preferred GPU allocation path

Use the exact interactive allocation that was validated for the Phase 4 closeout:

```bash
srun --partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal \
    --gres=gpu:6 --cpus-per-task=32 --mem=128G --time=04:00:00 --pty bash
```

Inside the allocation:

```bash
cd ~/scratch/stn
module load cuda/12.9.1
source .venv-phase4/bin/activate
python scripts/validate_phase4_env.py
```

Notes:
- on S3IT, `module load cuda/12.9.1` exposes `nvcc` but does not always populate the CUDA runtime library path
- the validator and Phase 4 runners now prepend `${CUDA_HOME}/targets/x86_64-linux/lib` automatically before Brian2CUDA execution
- the validated node type in this closeout exposed 6 x `Tesla V100-SXM2-32GB`

## Synthetic benchmark

Generate traces:

```bash
python scripts/04a_generate_synthetic_beta.py
```

Quick validation:

```bash
python scripts/04b_validate_state_machine_synthetic.py \
    --backend runtime \
    --no-grid \
    --out results/phase4_synthetic/dev_runtime
```

Single-GPU debug path:

```bash
python scripts/04b_validate_state_machine_synthetic.py \
    --backend cuda_standalone \
    --grid-indices 0 \
    --out results/phase4_synthetic
```

## Real-data front end

Single-subject dev run:

```bash
python scripts/04_adm_sweep.py \
    --subject sub-0cGdk9 \
    --conditions MedOff_Hold \
    --band-mode fixed_13_30 \
    --backend runtime \
    --no-grid \
    --out results
```

The real-data dev path currently uses single-file conditions for event-level onset/lag
metrics because the frozen Phase 3 per-condition parquet contract does not store source-file
provenance for multi-file conditions.

## Multi-GPU sweep execution

```bash
python scripts/04c_phase4_multigpu_sweep.py \
    --modes synthetic,real \
    --gpus 0,1,2,3,4,5 \
    --backend cuda_standalone \
    --subject sub-0cGdk9 \
    --conditions MedOff_Hold \
    --out results
```

The launcher:
1. shards grid indices across the requested GPU ids
2. pins one worker process per GPU via `CUDA_VISIBLE_DEVICES`
3. reuses one Brian2 standalone build per worker and rebuild-only axes (`dt_ms`, `synaptic_delay_ms`)
4. writes per-worker logs under `results/logs/04_phase4/`
5. merges worker summaries and then reruns the best grid point to emit the final Phase 4 figures/tables

Batch alternative:

```bash
sbatch slurm/slurm_phase4_gpu.sh
```

## Frozen closeout configs

- real-data front end: `configs/lif_encoder.yaml`
- filter bank: `configs/filter_bank.yaml`
- conservative synthetic state-machine freeze: `configs/nsm_mono_frozen.yaml`
- sweep lattice used for closeout: `configs/gridsearch_phase4.yaml`

The synthetic sweep also contains a higher raw-score operating point with earlier readout,
but the frozen synthetic config is intentionally the conservative one because it removes
the short-burst and negative-control pathologies more cleanly.

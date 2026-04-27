# stnbeta

STN-LFP beta-burst and beta-state detection for eventual DYNAP-SE1 evaluation.

The current roadmap is evidence-conditioned after Phase 5_2B. The old onset-alarm
detector path is negative: the current causal Phase 4/5 representation did not pass
onset, interval, long-burst, or burden targets. Phase 5_2B changed the next step by
finding a coherent, causal, SNN/DYNAP-compatible feature family, but the strongest
integrated evidence is still proxy/summary-level and carries some tautology risk.

Current decision: proceed to Phase 5_2C pre-Brian2 refinement and validation. Do not
start Phase 6 or DYNAP-SE1 hardware bring-up from the current evidence.

Primary roadmap: [MASTER_PLAN.md](MASTER_PLAN.md)

## Project Identity

- Primary signal: STN-LFP only.
- MEG: optional future analysis only, not part of the primary detector pipeline.
- Hardware target: DYNAP-SE1, not DYNAP-SE2, Rockpool, or a generic neuromorphic stack.
- Hardware stack when hardware begins: samna / INI DYNAP-SE1 stack.
- Simulation stack: Brian2 / Brian2CUDA is the primary SNN/NSM path.
- PyTorch: allowed for training, feasibility checks, and upper-bound diagnostics only.
- Accepted detector candidates must be replayed or implemented in Brian2,
  Brian2CUDA, or Brian2-equivalent simulation before SNN, hardware, or deployment
  claims.
- Architectural identity: De Luca-style filter bank -> LI&F encoder -> population NSM.

## Current Phase Status

| Phase | Description | Status |
| --- | --- | --- |
| 1 | Dataset acquisition + BIDS audit | Complete |
| 2 | LFP extraction | Complete |
| 3 | Tinkhauser beta-burst ground truth | Complete; frozen labels |
| 4 | Filter-bank / LI&F / Brian2CUDA / front-end setup | Complete |
| 5A-5Z | Hand-designed monoNSM, front-end ablations, fusion probes, trainable recovery, representation-gap audit, predictive compensation, target reconciliation, and burden/long-burst state detection | Complete as negative for the current causal representation path |
| 5_2A | Ground-truth-guided beta-burst feature atlas | Complete |
| 5_2B | Participant-wise derivative/dynamics/hard-negative feature atlas | Complete |
| 5_2B readout | Scientific extraction from completed atlas | Complete; conditional positive |
| 5_2C | Candidate refinement, mechanistic pipeline, quantized/mismatched evaluation, SNN approximation, DYNAP feasibility audit, and Brian2 gate decision | Next |
| 6 | Brian2 population detector simulation | Blocked until Phase 5_2C Outcome 1 |
| Hardware | DYNAP-SE1 bring-up | Blocked until Brian2 detector passes |

## Current Scientific Conclusion

The old onset-alarm path is negative. Phase 5W showed that the offline Phase 3
oracle can pass while causal replay and DYNAP-compatible upstream representations
fail low-false-positive event recovery. Phase 5X predictive compensation improved
some diagnostics but did not pass. Phase 5Y and Phase 5Z showed that interval,
long-burst, and burden targets are meaningful, but the current causal traces still
do not pass the controlled gates.

Phase 5_2B found a promising feature family:

- `beta_local_baseline_ratio`
- D1 / ON-count features
- D2 / launch acceleration features
- beta boundary veto / boundary contrast
- low/high beta context
- sparse channel weighting
- dwell / burden integrator features

The Phase 5_2B readout conclusion is B. Conditional positive: a coherent causal
SNN/DYNAP-compatible feature family exists, but the strongest integrated evidence is
proxy/summary-level and some rows retain tautology risk. The next step is therefore
Phase 5_2C, not Phase 6, not hardware, and not a final limitation note.

## What Went Wrong / What We Learned

Beta is not simply "13-30 Hz energy." A beta burst is an offline thresholded
envelope interval. The previous detector path failed because true labeled bursts
overlap with beta-like imposters:

- high_beta_unlabeled
- short_beta_like
- near_threshold_beta
- burst-adjacent beta
- boundary/broadband artifact-like beta

Phase 5_2B showed that overlap may be reducible with local baseline normalization,
D1 rise / ON-count, D2 launch / acceleration, dwell or burden integration, boundary
vetoes, and spatial weighting. These are design cues, not accepted detector inputs
until direct Phase 5_2C validation passes.

## Phase 5_2C Positioning

Phase 5_2C is pre-Brian2. It should refine and validate the seven candidate feature
families without adding new feature families and without making a final classifier
claim.

Planned stages:

A. Resolve Phase 5_2B residual issues: leakage sentinel classification, safe
candidate set, and LOSO baselines for seven features.

B. Architecture decision: reactive vs predictive detector, with an ADR before
pipeline design.

C. Bounded hyperparameter refinement of the seven candidate feature families. No new
feature families.

D. Multivariate combination analysis: minimum sufficient subset, LOSO held-out
predictions, no final classifier claim.

E. Non-spiking mechanistic pipeline: continuous causal reference implementation only.

F. Three-tier performance estimation: continuous, quantized, and quantized plus
mismatched.

G. SNN approximation engineering: still pre-Brian2.

H. DYNAP-SE1 feasibility audit: CAM use, core mapping, bias groups, spike traffic,
quantization, and mismatch.

I. Closeout: Outcome 1 proceeds to Brian2 simulation; all other outcomes block
Brian2.

## Environment

Use the project boot script:

```bash
cd /scratch/haizhe/stn
source /scratch/haizhe/stn/start_stn.sh
```

The script activates `/scratch/haizhe/stn/stn_env`, loads CUDA 12.9.1, sets CUDA and
Brian2CUDA library paths, preserves Slurm GPU visibility, and sets JAX memory
behavior. `.venv-phase4` is legacy only and should be used only for exact archival
reproduction of old scripts.

Validate on a compute node:

```bash
python scripts/validate_stn_env.py --strict --require-gpu --brian2cuda-smoke
```

Recorded compute-node validation for `stn_env` passed with 8 V100 GPUs visible,
Torch CUDA true, JAX seeing 8 devices, Brian2CUDA import OK, and Brian2CUDA smoke
passing.

## Current Run Guidance

No production Phase 5_2C command should be run until the Phase 5_2C prompt and
resource request are approved. The intended Slurm style for the next gated stage is:

```bash
sbatch slurm/slurm_phase5_2c.sh
```

That command is documentation of the intended production style, not permission to run
Phase 5_2C here. Phase 6 and hardware commands are intentionally blocked and are not
listed as runnable next steps.

## Brian2 And Diagnostic Policy

NumPy, Pandas, and SciPy scripts are allowed for audits, summaries, and deterministic
probes. PyTorch is allowed for training and upper-bound diagnostics. NumPy-only and
PyTorch-only detector proxies are diagnostic only.

Any accepted SNN/NSM detector candidate must be replayed or implemented in Brian2,
Brian2CUDA, or Brian2-equivalent simulation before SNN, hardware, or deployment
claims. Brian2 simulation is the next step only after the Phase 5_2C gate passes.

## DYNAP-SE1 Constraints

Hardware-aware candidates must report:

- binary synapses
- 64 CAM slots per neuron
- parallel connections for graded weights
- shared biases per 256-neuron core
- fan-in and core mapping
- quantization
- mismatch
- spike traffic
- hardware-aware validation before deployment

## Guardrails

- Do not run heavy work on the login node.
- Do not load MEG in the primary detector pipeline.
- Do not modify Phase 3 labels, frozen splits, evaluation boundaries, or benchmark
  definitions.
- Do not start Phase 6 or DYNAP-SE1 hardware without an explicit gate pass.
- Do not treat tautological Phase 3 threshold/duration features as deployable
  detector inputs.
- Do not treat proxy atlas features as final detector features until direct validation
  passes.
- Do not replace the detector with a generic black-box classifier.

## Repository Layout

```text
src/stnbeta/          importable library code
scripts/              CLI wrappers and analysis runners
slurm/                Slurm batch scripts
configs/              phase and model configuration files
docs/                 runbooks, decisions, and phase reports
results/              generated scientific outputs; do not stage blindly
extracted/            extracted LFP files; do not stage
raw/                  raw BIDS data; do not stage
logs/ai_runs/         Codex task provenance logs
tests/                pytest tests
```

## Key Evidence Files

- [docs/PHASE5_RUNBOOK.md](docs/PHASE5_RUNBOOK.md)
- [docs/PHASE5_2A_FEATURE_ATLAS.md](docs/PHASE5_2A_FEATURE_ATLAS.md)
- [docs/PHASE5_2B_DERIVATIVE_DYNAMICS.md](docs/PHASE5_2B_DERIVATIVE_DYNAMICS.md)
- [results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_analysis.md](results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_analysis.md)
- [results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_summary.tsv](results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_summary.tsv)
- [docs/decisions.md](docs/decisions.md)

# STN Beta-Burst Detection On DYNAP-SE1 - Master Plan v3

Status: evidence-conditioned roadmap after Phase 5_2B.

This document replaces the old aspirational roadmap as the execution guide. The older
`STN_BetaBurst_DynapSE1_MasterPlan.md` remains useful only as a project-identity
artifact. Current planning is governed by the completed Phase 5A-5Z negative results,
Phase 5_2A atlas, and Phase 5_2B derivative/dynamics readout.

## 1. Project Identity

The project target is STN-LFP beta-burst / beta-state detection for eventual
DYNAP-SE1 evaluation. The detector remains a neuromorphic SNN/NSM project, not a
generic classifier project.

Preserved identity constraints:

- Primary signal: STN-LFP only.
- MEG: optional future analysis only; never the primary detector pipeline.
- Hardware target: DYNAP-SE1, not DYNAP-SE2 or Rockpool.
- Hardware stack when hardware begins: samna / INI DYNAP-SE1 stack.
- Simulation stack: Brian2 / Brian2CUDA is the primary SNN simulation path.
- PyTorch may be used for training, feasibility checks, and diagnostic upper bounds.
- Accepted SNN/NSM candidates must have Brian2, Brian2CUDA, or Brian2-equivalent
  replay before SNN or hardware claims.
- Architectural identity: De Luca-style filter bank -> LI&F encoder -> population
  NSM.

Hardware-aware work must report DYNAP-SE1 feasibility:

- binary synapses
- 64 CAM slots per neuron
- parallel connections for graded weights
- shared core biases
- fan-in and core mapping
- quantization
- mismatch
- spike traffic
- hardware-aware validation before deployment

## 2. Current Scientific Conclusion

The old onset-alarm detector path is negative. The current causal Phase 4/5
representation did not pass onset, interval, long-burst, or burden targets.

Phase 5W showed that the offline Phase 3 oracle can pass while causal Phase 3-like
replay and DYNAP-compatible upstream representations fail low-false-positive event
recovery. Phase 5X tested predictive compensation and did not pass. Phase 5Y kept
Phase 3 intervals frozen and showed that interval and phenotype-aligned scoring do
not rescue the current causal traces. Phase 5Z showed that long-burst and burden
state targets are meaningful, but do not rescue the same current representation path.

Phase 5_2B changed the picture by identifying a promising, low-power feature family:

- `beta_local_baseline_ratio`
- D1 / ON-count features
- D2 / launch acceleration features
- beta boundary veto / boundary contrast
- low/high beta context
- sparse channel weighting
- dwell / burden integrator features

The Phase 5_2B readout conclusion is B. Conditional positive: a coherent
non-tautological, causal, SNN/DYNAP-compatible feature family is present, but the
recommended set depends on proxy or summary-level evidence and some rows retain
tautology risk.

Therefore the next step is Phase 5_2C. It is not Phase 6, not hardware, and not yet
a final limitation note.

## 3. Evidence-Conditioned Phase Status

| Phase | Description | Status |
| --- | --- | --- |
| 1 | Dataset acquisition + BIDS audit | Complete |
| 2 | LFP extraction | Complete |
| 3 | Tinkhauser beta-burst ground truth | Complete; frozen labels |
| 4 | Filter-bank / LI&F / Brian2CUDA / front-end setup | Complete |
| 5A-5Z | Hand-designed monoNSM, front-end ablations, fusion probes, trainable recovery, representation-gap audit, predictive compensation, target reconciliation, and burden/long-burst state detection | Completed as negative for the current causal representation path |
| 5_2A | Ground-truth-guided beta-burst feature atlas | Complete |
| 5_2B | Participant-wise derivative/dynamics/hard-negative feature atlas | Complete |
| 5_2B readout | Scientific extraction from completed atlas | Complete; conditional positive |
| 5_2C | Candidate feature refinement, mechanistic pipeline, continuous/quantized/mismatched evaluation, SNN approximation, DYNAP feasibility audit, and Brian2 gate decision | Next |
| 6 | Brian2 population detector simulation | Blocked until Phase 5_2C Outcome 1 |
| Hardware | DYNAP-SE1 bring-up | Blocked until Brian2 detector passes |

## 4. What Went Wrong / What We Learned

Beta is not simply "13-30 Hz energy." A beta burst is an offline thresholded
envelope interval. A causal detector must distinguish those label-defining intervals
from beta-like activity that is physiologically or spectrally similar but unlabeled.

The old detector path failed because true bursts overlap with beta-like imposters:

- high_beta_unlabeled
- short_beta_like
- near_threshold_beta
- burst-adjacent beta
- boundary/broadband artifact-like beta

Phase 5_2B showed that this overlap may be reduced by combining:

- local baseline normalization
- D1 rise / ON-count
- D2 launch / acceleration
- dwell / burden integration
- boundary veto
- spatial weighting

The implication is not that Phase 3 labels are wrong, and not that DYNAP-SE1 cannot
detect beta bursts. The implication is narrower: the previously tested causal
representation did not preserve the right evidence, while the Phase 5_2B atlas
points to a different pre-Brian2 feature family that deserves direct validation.

## 5. Phase 5_2C Plan

Phase 5_2C is pre-Brian2. It validates whether the Phase 5_2B candidate family can
survive direct, causal, participant-wise, quantized, and hardware-aware tests. It must
not add new feature families, claim a final classifier, run Phase 6, deploy hardware,
use MEG, or modify Phase 3 labels.

### A. Resolve Phase 5_2B Residual Issues

- Classify and quarantine leakage sentinel behavior.
- Define the safe candidate feature set.
- Run LOSO baselines for the seven candidate feature families.

### B. Architecture Decision

- Decide reactive vs predictive vs hybrid detector.
- Write an ADR before pipeline design.
- Treat early baseline/context shifts as candidate-state evidence unless direct
  validation justifies committed alarms.

### C. Bounded Hyperparameter Refinement

- Refine only the seven candidate feature families.
- Do not add new feature families.
- Keep Phase 3 labels and negative definitions fixed.

### D. Multivariate Combination Analysis

- Identify a minimum sufficient subset.
- Produce LOSO held-out predictions.
- Make no final classifier claim.

### E. Non-Spiking Mechanistic Pipeline

- Build a continuous causal reference implementation only.
- Use it to make the feature-to-state contract explicit before any SNN translation.

### F. Three-Tier Performance Estimation

- Tier 1: continuous.
- Tier 2: quantized.
- Tier 3: quantized plus mismatched.

### G. SNN Approximation Engineering

- Engineer SNN-compatible approximations.
- Remain pre-Brian2 until the closeout gate passes.

### H. DYNAP-SE1 Feasibility Audit

- Audit CAM slots, core allocation, shared bias groups, fan-in, binary synapses,
  parallel connections for graded weights, spike traffic, quantization, and mismatch.

### I. Closeout

- Outcome 1: proceed to Brian2 population detector simulation.
- Any other outcome: block Brian2 and either refine the scientific question within
  approved bounds or write the documented negative result.

## 6. Completion Paths

The project has three valid completion paths:

1. Documented negative result if Phase 5_2C shows that the substrate is insufficient.
2. Brian2 population simulation if Phase 5_2C passes Outcome 1.
3. DYNAP-SE1 hardware bring-up only if the Brian2 detector passes its gate.

Phase 6 and hardware are not automatic next steps. A limitation result is valid if the
candidate substrate fails direct validation.

## 7. Environment Policy

Use the canonical repository path and boot script:

```bash
cd /scratch/haizhe/stn
source /scratch/haizhe/stn/start_stn.sh
```

The boot script activates `/scratch/haizhe/stn/stn_env`, loads CUDA 12.9.1, sets
CUDA/JAX variables, configures Brian2CUDA library paths, and preserves Slurm GPU
visibility. `.venv-phase4` is legacy only.

Validate on a compute node:

```bash
python scripts/validate_stn_env.py --strict --require-gpu --brian2cuda-smoke
```

Recorded compute-node validation passed with 8 V100 GPUs visible, Torch CUDA true,
JAX seeing 8 devices, Brian2CUDA import OK, and Brian2CUDA smoke passing.

Do not run heavy work on the login node. Do not submit Slurm jobs unless the user
explicitly asks for a compute run and the resource policy is followed.

## 8. Brian2 Rule

NumPy, Pandas, and SciPy are allowed for audits, summaries, and deterministic probes.
PyTorch is allowed for training and upper-bound diagnostics. NumPy-only detector
proxies are diagnostic only. PyTorch-only models are diagnostic only unless replayed
or represented in Brian2 / Brian2-equivalent simulation.

Any accepted SNN/NSM detector candidate must be replayed or implemented in Brian2,
Brian2CUDA, or Brian2-equivalent simulation before hardware or SNN claims. Brian2
simulation is the next step only after the Phase 5_2C gate passes.

## 9. Run Guidance

Current boot:

```bash
source /scratch/haizhe/stn/start_stn.sh
```

Current environment validation:

```bash
python scripts/validate_stn_env.py --strict --require-gpu --brian2cuda-smoke
```

Current intended next Slurm style, not a production command until the Phase 5_2C
prompt and script are approved:

```bash
sbatch slurm/slurm_phase5_2c.sh
```

Phase 6 and hardware commands are intentionally blocked.

## 10. Warnings

- Do not run heavy work on the login node.
- Do not load MEG in the primary pipeline.
- Do not modify Phase 3 labels.
- Do not start Phase 6 or DYNAP-SE1 hardware without an explicit gate pass.
- Do not treat tautological Phase 3 threshold/duration features as deployable
  detector inputs.
- Do not treat proxy atlas features as final detector features until direct validation
  passes.
- Do not replace the detector with a generic black-box classifier.

## 11. Evidence Sources

Primary sources inspected for this roadmap:

- `README.md`
- `AGENTS.md`
- `docs/PHASE5_RUNBOOK.md`
- `docs/PHASE5_2A_FEATURE_ATLAS.md`
- `docs/PHASE5_2B_DERIVATIVE_DYNAMICS.md`
- `docs/decisions.md`
- `docs/PHASE5W_WIDEPROBE.md`
- `docs/PHASE5X_PREDICTIVE_COMPENSATION.md`
- `docs/PHASE5Y_TARGET_RECONCILIATION.md`
- `docs/PHASE5Z_BURDEN_STATE.md`
- `results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_analysis.md`
- `results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_summary.tsv`
- `results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv`
- `results/tables/05_phase5/feature_atlas_2b/snn_dynap_candidate_features.tsv`
- `results/tables/05_phase5/feature_atlas_2b/hbu_subtype_diagnosis_summary.tsv`

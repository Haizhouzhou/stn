# 2026-04-27 05:02 CEST - Phase 5_2B Roadmap Rewrite

## Task Name
Rewrite `MASTER_PLAN.md` and `README.md` into an evidence-conditioned roadmap after Phase 5_2B.

## Git Commit Before Task
`90d1050793a23a98cc563921eee32e8cca58a939`

## Original User Prompt
```text
Start in Plan mode first.

You are continuing work inside:

  /scratch/haizhe/stn

Your task is:

  Rewrite MASTER_PLAN.md and README.md into an evidence-conditioned roadmap after Phase 5_2B.

This is documentation only.

Do NOT:
- run data processing,
- train models,
- modify Phase 3 labels,
- use MEG,
- start Phase 6,
- submit SLURM jobs,
- deploy hardware.

The user provided an older master plan v2. It is useful as a project-identity document but outdated as an execution roadmap. Rewrite it into v3.

## Files to inspect

Read first:

- README.md
- AGENTS.md
- docs/PHASE5_RUNBOOK.md
- docs/PHASE5_2A_FEATURE_ATLAS.md
- docs/PHASE5_2B_DERIVATIVE_DYNAMICS.md
- docs/decisions.md
- results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_analysis.md
- results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_summary.tsv
- results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv
- results/tables/05_phase5/feature_atlas_2b/snn_dynap_candidate_features.tsv
- results/tables/05_phase5/feature_atlas_2b/hbu_subtype_diagnosis_summary.tsv
- any existing MASTER_PLAN.md if present

Also inspect previous phase closeout docs if present:
- PHASE5W_WIDEPROBE
- PHASE5X_PREDICTIVE_COMPENSATION
- PHASE5Y_TARGET_RECONCILIATION
- PHASE5Z_BURDEN_STATE

## Main rewrite goal

Create or update:

- MASTER_PLAN.md
- README.md

The new documents must reflect the actual current repo status, not the old aspirational roadmap.

## Preserve these identity constraints

The rewritten plan must preserve:

- Project target: STN-LFP beta-burst / beta-state detection for eventual DYNAP-SE1 deployment.
- Primary signal: STN-LFP only.
- MEG: optional future analysis only, not primary detector pipeline.
- Hardware target: DYNAP-SE1, not SE2, not Rockpool.
- Simulation stack: Brian2 primary for SNN/NSM simulation.
- PyTorch: training / diagnostic use only.
- Hardware stack: samna / INI DYNAP-SE1 stack when hardware begins.
- Architectural identity: De Luca-style filter-bank -> LI&F encoder -> population NSM.
- DYNAP constraints:
  - binary synapses,
  - 64 CAM slots per neuron,
  - parallel connections for graded weights,
  - shared core biases,
  - mismatch,
  - quantization,
  - hardware-aware validation before deployment.

## Update phase status

Replace the old phase table with an evidence-conditioned one:

Phase 1:
Dataset acquisition + BIDS audit - complete.

Phase 2:
LFP extraction - complete.

Phase 3:
Tinkhauser beta-burst ground truth - complete.

Phase 4:
Filter-bank / LI&F / Brian2CUDA / front-end setup - complete.

Phase 5A-5Z:
Hand-designed monoNSM, front-end ablations, fusion probes, trainable recovery, representation-gap audit, predictive compensation, target reconciliation, and burden/long-burst state detection - completed as negative for the current causal representation path.

Phase 5_2A:
Ground-truth-guided beta-burst feature atlas - complete.

Phase 5_2B:
Participant-wise derivative/dynamics/hard-negative feature atlas - complete.

Phase 5_2B readout:
Scientific extraction from completed atlas - complete, conditional positive.

Phase 5_2C:
Candidate feature refinement, mechanistic pipeline, three-tier continuous/quantized/mismatched evaluation, SNN approximation, DYNAP feasibility audit, and Brian2 gate decision - next.

Phase 6:
Brian2 population detector simulation - blocked until Phase 5_2C Outcome 1.

Hardware:
DYNAP-SE1 bring-up - blocked until Brian2 detector passes.

## Required current scientific conclusion

The documents must state:

- The old onset-alarm detector path is negative.
- The current causal Phase 4/5 representation did not pass onset, interval, long-burst, or burden targets.
- Phase 5_2B changed the picture by finding a promising feature family:
  - beta_local_baseline_ratio,
  - D1 / ON-count,
  - D2 / launch acceleration,
  - beta boundary veto,
  - low/high beta context,
  - sparse channel weighting,
  - dwell / burden integrator.
- The readout conclusion was B. Conditional positive:
  a coherent causal SNN/DYNAP-compatible feature family exists, but the strongest integrated evidence is proxy/summary-level and carries some tautology risk.
- Therefore the next step is not Phase 6, not hardware, and not a limitation note yet.
- The next step is Phase 5_2C.

## Update "What went wrong / what we learned"

Add a concise section explaining:

- Beta is not simply "13-30 Hz energy."
- A beta burst is an offline thresholded envelope interval.
- The old detector failed because true bursts overlap with beta-like imposters:
  - high_beta_unlabeled,
  - short_beta_like,
  - near_threshold_beta,
  - burst-adjacent beta,
  - boundary/broadband artifact-like beta.
- Phase 5_2B showed that overlap can potentially be reduced with:
  - local baseline normalization,
  - D1 rise / ON-count,
  - D2 launch / acceleration,
  - dwell / burden integrator,
  - boundary veto,
  - spatial weighting.

## Add Phase 5_2C plan summary

The plan must describe Phase 5_2C as pre-Brian2.

Stages:

A. Resolve Phase 5_2B residual issues:
- leakage sentinel classification,
- safe candidate set,
- LOSO baselines for seven features.

B. Architecture decision:
- reactive vs predictive detector,
- ADR before pipeline design.

C. Bounded hyperparameter refinement of seven candidate features:
- no new feature families.

D. Multivariate combination analysis:
- minimum sufficient subset,
- LOSO held-out predictions,
- no final classifier claim.

E. Non-spiking mechanistic pipeline:
- continuous reference implementation only.

F. Three-tier performance estimation:
- continuous,
- quantized,
- quantized + mismatched.

G. SNN approximation engineering:
- still pre-Brian2.

H. DYNAP-SE1 feasibility audit:
- CAM, cores, bias groups, spike traffic.

I. Closeout:
- Outcome 1 means proceed to Brian2 simulation.
- Other outcomes block Brian2.

## Update environment section

The documents must describe the current environment:

- Use `/scratch/haizhe/stn/start_stn.sh`.
- It activates `/scratch/haizhe/stn/stn_env`.
- It loads CUDA 12.9.1.
- It sets CUDA/JAX variables.
- Compute-node validation passed:
  - 8 V100 visible,
  - Torch CUDA true,
  - JAX sees 8 devices,
  - Brian2CUDA import ok,
  - Brian2CUDA smoke passed.
- `.venv-phase4` is legacy only.

## Add Brian2 rule

Include:

- NumPy/Pandas/SciPy scripts are allowed for audits and summaries.
- PyTorch is allowed for training / upper-bound diagnostics.
- Any accepted SNN/NSM detector candidate must be replayed or implemented in Brian2 / Brian2-equivalent simulation before hardware or SNN claims.
- NumPy-only detector proxies are diagnostic only.
- Brian2 simulation is the next step only after Phase 5_2C gate passes.

## Update run instructions

README should include:

1. How to boot environment:

   source /scratch/haizhe/stn/start_stn.sh

2. How to validate env:

   python scripts/validate_stn_env.py --strict --require-gpu --brian2cuda-smoke

3. Current recommended next command:
   not a production command until Phase 5_2C prompt is approved, but show the intended SLURM style:

   sbatch slurm/slurm_phase5_2c.sh

4. Explain that Phase 6 / hardware commands are intentionally blocked.

## Add negative result completion path

MASTER_PLAN.md should state that a valid project completion could be:

- documented negative result if Phase 5_2C shows the substrate is insufficient;
- Brian2 population simulation if Phase 5_2C passes;
- hardware bring-up only if Brian2 passes.

## Preserve warnings

State clearly:

- Do not run heavy work on login node.
- Do not load MEG in primary pipeline.
- Do not modify Phase 3 labels.
- Do not start Phase 6 or DYNAP-SE1 hardware without explicit gate pass.
- Do not treat tautological Phase 3 threshold/duration features as deployable detector inputs.
- Do not treat proxy atlas features as final detector features until direct validation passes.

## Validation

Run lightweight validation only:

- python -m py_compile any new helper script if created
- git diff --check

Do not run data processing.
Do not submit SLURM.

## Final response format

Return exactly:

1. Summary
2. Files changed
3. MASTER_PLAN.md update
4. README update
5. Current phase status
6. Phase 5_2C positioning
7. Environment / Brian2 policy
8. Commands actually run
9. Validation results
10. Remaining risks

Proceed now: rewrite MASTER_PLAN.md and README.md into an evidence-conditioned roadmap after Phase 5_2B. Do not run scientific computation.
```

## Operational Plan
1. Treat the user request for Plan mode as a request to start with an explicit plan, while continuing in the active Default mode.
2. Inspect the requested documentation and generated summary tables only.
3. Rewrite `MASTER_PLAN.md` as v3 and rewrite `README.md` to match the post-Phase 5_2B roadmap.
4. Add the required AI run log and update `logs/ai_runs/INDEX.md`.
5. Run only lightweight validation: `git diff --check`.
6. Stage explicit safe paths, commit, and push the documentation/log checkpoint unless blocked.

## Files Inspected
- `README.md`
- `AGENTS.md`
- `STN_BetaBurst_DynapSE1_MasterPlan.md`
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

## Files Modified Or Created
- `README.md` - rewritten as an evidence-conditioned post-Phase 5_2B project entry point.
- `MASTER_PLAN.md` - created as v3 evidence-conditioned roadmap.
- `logs/ai_runs/2026-04-27_0502_phase5_2b_roadmap_rewrite.md` - created.
- `logs/ai_runs/INDEX.md` - updated.

## Commands Run
```bash
ls -l README.md AGENTS.md MASTER_PLAN.md docs/PHASE5_RUNBOOK.md docs/PHASE5_2A_FEATURE_ATLAS.md docs/PHASE5_2B_DERIVATIVE_DYNAMICS.md docs/decisions.md results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_analysis.md results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_summary.tsv results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv results/tables/05_phase5/feature_atlas_2b/snn_dynap_candidate_features.tsv results/tables/05_phase5/feature_atlas_2b/hbu_subtype_diagnosis_summary.tsv 2>&1
wc -l README.md AGENTS.md MASTER_PLAN.md docs/PHASE5_RUNBOOK.md docs/PHASE5_2A_FEATURE_ATLAS.md docs/PHASE5_2B_DERIVATIVE_DYNAMICS.md docs/decisions.md results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_analysis.md results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_summary.tsv results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv results/tables/05_phase5/feature_atlas_2b/snn_dynap_candidate_features.tsv results/tables/05_phase5/feature_atlas_2b/hbu_subtype_diagnosis_summary.tsv 2>&1
ls -l docs/PHASE5W_WIDEPROBE.md docs/PHASE5X_PREDICTIVE_COMPENSATION.md docs/PHASE5Y_TARGET_RECONCILIATION.md docs/PHASE5Z_BURDEN_STATE.md 2>&1
git status -sb
git rev-parse HEAD
sed -n '1,260p' README.md
sed -n '261,560p' README.md
sed -n '1,220p' AGENTS.md
sed -n '1,220p' docs/PHASE5_2B_DERIVATIVE_DYNAMICS.md
sed -n '1,220p' results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_analysis.md
sed -n '1,260p' docs/PHASE5_RUNBOOK.md
sed -n '261,540p' docs/PHASE5_RUNBOOK.md
sed -n '541,820p' docs/PHASE5_RUNBOOK.md
sed -n '1,220p' docs/PHASE5_2A_FEATURE_ATLAS.md
sed -n '1,220p' docs/decisions.md
sed -n '221,460p' docs/decisions.md
sed -n '461,720p' docs/decisions.md
cat results/tables/05_phase5/feature_atlas_2b/phase5_2b_readout_summary.tsv
cat results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv
cat results/tables/05_phase5/feature_atlas_2b/snn_dynap_candidate_features.tsv
cat results/tables/05_phase5/feature_atlas_2b/hbu_subtype_diagnosis_summary.tsv
cat docs/PHASE5W_WIDEPROBE.md
cat docs/PHASE5X_PREDICTIVE_COMPENSATION.md
cat docs/PHASE5Y_TARGET_RECONCILIATION.md
cat docs/PHASE5Z_BURDEN_STATE.md
find . -maxdepth 2 -type f -iname '*MASTER*PLAN*' -o -iname '*MasterPlan*'
sed -n '1,260p' STN_BetaBurst_DynapSE1_MasterPlan.md
sed -n '261,560p' STN_BetaBurst_DynapSE1_MasterPlan.md
sed -n '1,220p' scripts/validate_stn_env.py
ls -l slurm/slurm_phase5_2c.sh slurm/slurm_phase5_2c_causal_def.sh slurm/slurm_phase5_2c_causal_feature_matrix.sh slurm/slurm_phase5_2c_causal_frontend.sh slurm/slurm_phase5_2c_causal_stage_c.sh slurm/slurm_phase5_2c_cdef.sh slurm/slurm_phase5_2c_stage_a.sh slurm/slurm_phase5_2c_stage_f_event.sh 2>&1
sed -n '1,220p' docs/PHASE5_2C_ARCHITECTURE_ADR.md 2>/dev/null || true
sed -n '1,220p' docs/PHASE5_2C_PIPELINE_DESIGN.md 2>/dev/null || true
date '+%Y-%m-%d %H:%M %Z'
```

Post-edit validation, staging, commit, and push commands are recorded after they run.

Additional post-edit checks:

```bash
rg -n "old onset|Phase 5_2C|Brian2|DYNAP-SE1|beta_local_baseline_ratio|high_beta_unlabeled|slurm_phase5_2c|start_stn|validate_stn_env|tautolog|proxy atlas|MEG|Phase 6" README.md MASTER_PLAN.md
git diff --check
git diff --stat -- README.md MASTER_PLAN.md logs/ai_runs/INDEX.md logs/ai_runs/2026-04-27_0502_phase5_2b_roadmap_rewrite.md
sed -n '1,260p' MASTER_PLAN.md
sed -n '261,520p' MASTER_PLAN.md
sed -n '1,260p' README.md
git status --short
git diff --stat
git diff --check
git ls-files -s | awk '{print $4}' | while read -r f; do
  [ -f "$f" ] || continue
  size=$(stat -c%s "$f")
  if [ "$size" -ge 100000000 ]; then
    printf '%s\t%s\n' "$size" "$f"
  fi
done
```

## Validation Results
- Content grep confirmed both `README.md` and `MASTER_PLAN.md` mention the required negative onset path conclusion, Phase 5_2C positioning, Brian2/DYNAP-SE1 policy, core feature names, warnings, boot command, validation command, and intended `slurm_phase5_2c` style.
- `git diff --check` produced no output.
- No helper script was created, so no `python -m py_compile` command was applicable.
- The tracked large-file check produced no output.
- No scientific computation, model training, data processing, `sbatch`, or `srun` command was run.

## Final Codex Response Text
The exact final response is written in chat after validation, commit, and push complete. The final response is not embedded with a final commit hash here to avoid self-referential commit-hash churn.

## Risks And Blockers
- The repository had many pre-existing unrelated modified and untracked paths before this task. They are not part of this documentation rewrite.
- `MASTER_PLAN.md` did not exist before this task.
- `slurm/slurm_phase5_2c.sh` was not present; the README documents it only as the intended future Slurm style requested by the user, not as a command to run during this task.
- Existing untracked Phase 5_2C draft files were inspected only to understand repo context but the roadmap follows the requested post-Phase 5_2B positioning with Phase 5_2C as next.

## Git Commit After Task
Pending before staging and commit. See the final Codex response for commit and push status.

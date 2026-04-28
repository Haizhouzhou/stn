# Phase 6A.0 Burden-Target Viability Gate

## Purpose And Corrected Phase 6 Framing

Phase 6 starts with an internal STN-LFP burden/state viability gate. The claim being tested is a neuromorphic approximation of a clinically relevant STN-LFP burden/state-tracking policy, not clinical efficacy or FDA-grade validity.

## Why Not PPN Or Herz First

The He/Tan PPN dataset is an optional cross-target extension only after the primary STN architecture works. The Herz/Groppa/Brown force-adaptation package is a methods/code reference with minimum example data; it is not used as a primary STN burden-validation input. External PPN/Herz paths were explicitly excluded from candidate input selection.

## Dataset/Input Files Discovered And Selected

- Selected input recommendation: `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- Selection confidence: `high`
- Selected input files used: `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- Discovery candidates recorded: `1545`

Top discovery candidates:

| path | score | use | notes |
| --- | --- | --- | --- |
| `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv` | 32 | candidate_primary_input | large table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/causal_frontend_smoke_features.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/event_alarm_reconstruction_best.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/event_scoring_tolerance_sweep.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/causal_event_threshold_grid.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/event_alarm_reconstruction_sweep.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/phase5_2c/causal_feature_matrix_feature_metadata.tsv` | 24 | candidate_primary_input | table sampled by header only during discovery |
| `results/tables/05_phase5/feature_atlas_2b/burden_window_features.tsv` | 22 | candidate_primary_input | large table sampled by header only during discovery |
| `results/tables/05_phase5/feature_atlas_2b/burden_window_index.tsv` | 22 | candidate_primary_input | large table sampled by header only during discovery |

## Environment And Commands

- Python: `/scratch/haizhe/stn/stn_env/bin/python (3.12.3)`
- Environment route: `source /scratch/haizhe/stn/start_stn.sh && python ...`

## Feature And Leakage Audit

- Allowed causal feature count: `354`
- Excluded/leakage feature count: `19`
- Leakage risk summary: `3 excluded/suspicious columns; selected model features restricted to allowed causal feature list.`
- Feature names suggesting labels, targets, future/post windows, split/fold, onset/offset, oracle outputs, file paths, or identity fields were excluded.
- Any precomputed global or subject-level normalization suspicion is documented in `leakage_risk_audit.csv`; training normalization for the model is fitted on training subjects only.

## Burden Target Construction

Annotation-derived burden uses the binary label column and a causal leaky integrator: `burden_t = burden_{t-1} + alpha * (label_t - burden_{t-1})`, `alpha = 1 - exp(-dt / tau)`. Time deltas are estimated from the selected time/order column within subject/session/condition/channel streams. No centered windows or future labels are used.

## Model And LOSO Validation

The primary model is a deterministic subject-held-out ridge least-squares probability scorer on selected allowed causal features. Features are standardized using training subjects only, then applied to the held-out subject. Gate metrics are summarized per subject first, then by subject-level median/IQR.

## Tau Sweep Results

| tau_ms | median_pearson | median_spearman | valid_subjects |
| --- | --- | --- | --- |
| 200.0 | 0.19833266877777161 | 0.19494261756066633 | 20 |
| 500.0 | 0.2156129276828324 | 0.21698217886900945 | 20 |
| 800.0 | 0.22663720287504674 | 0.23080836757185158 | 20 |
| 1500.0 | 0.23588365345703 | 0.24706277086051123 | 20 |
| 3000.0 | 0.24896977374123425 | 0.2602476064282385 | 20 |

## Baseline Comparison

| baseline | tau_ms | median_spearman | notes |
| --- | --- | --- | --- |
| class_prior_baseline | 3000.0 | 0.10720896000980693 | subject-level median; no pooled-window gate |
| shuffled_label_baseline | 3000.0 | 0.03811309853555812 | subject-level median; no pooled-window gate |
| simple_beta_feature_baseline | 3000.0 | 0.22797459976862877 | subject-level median; no pooled-window gate |
| model_based_burden_estimate | 3000.0 | 0.2602476064282385 | subject-level median; no pooled-window gate |

## Clinical/Task Proxy Results

| proxy | positive | negative | AUROC/rank-biserial | Cohen d |
| --- | --- | --- | --- | --- |
| medication_state | MedOff | MedOn | 0.5957792207792207 | 0.26620585799971574 |
| task_state | Hold | Move | 0.49047619047619045 | 0.02957793763893011 |
| task_state | Rest | Move | 0.5235714285714286 | 0.22874187904695215 |
| task_state | Rest | Hold | 0.5236111111111111 | 0.2005076972899538 |

## Gate Decision

- Overall status: `FAIL`
- Gate rationale: Gate thresholds for PASS/CONDITIONAL_PASS were not met.

## Interpretation

This is an internal STN-substrate technical viability result. It is not clinical validation, not PKG validation, and not evidence of clinical efficacy.

## Next Actions

- Stop Phase 6 simulation work and reconsider the burden/state-machine pivot.
- Inspect whether target construction, feature families, or annotation burden definition should be revised.

If the gate passes, proceed to Phase 6A Brian2/state-machine simulation on this internal STN substrate. If it conditionally passes, add ablations and robustness checks before simulation. If it fails, stop and reconsider the burden pivot. If blocked, provide the required internal STN feature/label table columns.

## PPN And Herz Dataset Handling

PPN remains optional cross-target generalization only after the primary STN burden architecture works. Herz remains a methods/code reference only.

## Limitations

- Model used deterministic subset of 96 selected allowed features from 354 allowed causal features to keep Phase 6A.0 lightweight and report-sized.
- allowed features from metadata results/tables/05_phase5/phase5_2c/causal_feature_matrix_feature_metadata.tsv
- Critical blocker: Gate thresholds for PASS/CONDITIONAL_PASS were not met.

## Exact Commands Run

- `pwd`
- `git rev-parse --show-toplevel`
- `git status --short`
- `source /scratch/haizhe/stn/start_stn.sh && python -V`
- `find . -maxdepth 3 -type d | sort | head -300`
- `find . -maxdepth 3 -type f | sort | head -300`
- `find reports -maxdepth 4 -type f | sort | grep -E 'phase3|phase4|phase5|5_2|5Y|burden|burst|ADR|audit' | head -300 || true`
- `find data -maxdepth 5 -type f | sort | grep -E 'phase3|phase4|phase5|burst|burden|feature|label|stn|parquet|csv|json|pkl|npz|mat' | head -300 || true`
- `rg -n "Phase 5_2B|Phase 5_2C|burden|AUROC|268,959|LOSO|MedOff|MedOn|Hold|Move|UPDRS|bradykinesia|burst" reports scripts data config configs pyproject.toml README* 2>/dev/null | head -500 || true`
- `source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_burden_viability.py --out-dir reports/phase6a0_burden_viability --stop-after-discovery`
- `source /scratch/haizhe/stn/start_stn.sh && python -m py_compile scripts/phase6a0_burden_viability.py`
- `~/bin/claim_best_immediate_resource.sh --mode cpu "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_burden_viability.py --out-dir reports/phase6a0_burden_viability --tau-ms 200,500,800,1500,3000"`
- `sinfo -o '%P %a %l %D %c %m %G' | head -80`
- `sacctmgr -nP show assoc user=$USER format=Account,Partition,QOS 2>/dev/null | head -80 || true`
- `sinfo -p teaching -o '%P %a %l %D %c %m %G' || true`
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_burden_viability.py --out-dir reports/phase6a0_burden_viability --tau-ms 200,500,800,1500,3000"`
- `sacct -j 2577074 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `seff 2577074 2>/dev/null || true`
- `ls -lh reports/phase6a0_burden_viability`
- `head -100 reports/phase6a0_burden_viability/README_burden_viability.md`
- `python - <<'PY' ... print burden_viability_findings.json ... PY`
- `git diff --check`
- `find reports/phase6a0_burden_viability -type f -size +5M -print`
- `git status --short`

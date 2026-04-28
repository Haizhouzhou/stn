# Phase 6A.0 Burden Viability Gate

Date: 2026-04-28 02:15 CEST

Task: Corrected Phase 6 start with a burden-target viability gate on the existing internal 20-subject STN LFP substrate.

Commit before task: `b647d08e01f8ab0b20aa0e070afc2d8c206b6e51`

## User Prompt

```text
Implement corrected Phase 6 start: do not make PPN or Herz the primary Phase 6 validation dataset. Phase 6 must begin with Phase 6A.0, a burden-target viability gate on the existing internal STN LFP substrate from prior phases. Use only the existing project environment via start_stn.sh or stn_env; do not install dependencies or create environments. Exclude external PPN/Herz data and reports from candidate input selection. Build a two-pass discovery/analysis workflow, create the requested reports under reports/phase6a0_burden_viability, create scripts/phase6a0_burden_viability.py and reports/phase6_publication_strategy/ADR_phase6_corrected_strategy.md, run validation and quality checks, create this AI run log, and commit/push only explicit safe paths. If Phase 6A.0 is PASS create the simulation scaffold; if CONDITIONAL_PASS create only a next-steps README; if FAIL or BLOCKED create no simulation scaffold.
```

## Operational Plan

1. Run the requested startup checks and inspect prior Phase 5/Phase 6 report conventions.
2. Implement `scripts/phase6a0_burden_viability.py` with two-pass input discovery, external path exclusion, data-contract validation, leakage audit, LOSO burden modeling, causal burden integration, baselines, proxy summaries, figures, and gate decision logic.
3. Run discovery-only first and require a clear internal candidate before analysis.
4. Run the full analysis on Slurm CPU resources because the selected table is 4.1 GB.
5. Write the corrected Phase 6 strategy ADR, report artifacts, run log, and validation results.
6. Commit and push only explicit safe report/script/log paths.

## Files Inspected

- `logs/ai_runs/2026-04-27_2115_phase6a_part2_stn_force_adaptation_audit.md`
- `logs/ai_runs/INDEX.md`
- `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv` header only during discovery, selected columns during analysis
- `results/tables/05_phase5/phase5_2c/causal_feature_matrix_feature_metadata.tsv`
- `results/tables/05_phase5/phase5_2c/causal_feature_matrix_validation.tsv`
- `results/tables/05_phase5/phase5_2c/input_integrity.tsv`
- Prior Phase 5 docs/reports matched by the `prior_phase_comparison.csv` search

## Files Modified Or Created

- Created `scripts/phase6a0_burden_viability.py`
- Created `reports/phase6a0_burden_viability/README_burden_viability.md`
- Created `reports/phase6a0_burden_viability/input_discovery.csv`
- Created `reports/phase6a0_burden_viability/selected_input_recommendation.json`
- Created `reports/phase6a0_burden_viability/feature_column_audit.csv`
- Created `reports/phase6a0_burden_viability/metadata_proxy_audit.csv`
- Created `reports/phase6a0_burden_viability/burden_tau_sweep.csv`
- Created `reports/phase6a0_burden_viability/burden_metric_by_subject.csv`
- Created `reports/phase6a0_burden_viability/burden_metric_by_condition.csv`
- Created `reports/phase6a0_burden_viability/condition_separability.csv`
- Created `reports/phase6a0_burden_viability/clinical_proxy_correlation.csv`
- Created `reports/phase6a0_burden_viability/baseline_comparison.csv`
- Created `reports/phase6a0_burden_viability/leakage_risk_audit.csv`
- Created `reports/phase6a0_burden_viability/prior_phase_comparison.csv`
- Created `reports/phase6a0_burden_viability/burden_viability_findings.json`
- Created `reports/phase6a0_burden_viability/phase6a0_commands_run.txt`
- Created small PNG figures under `reports/phase6a0_burden_viability/figures/`
- Created `reports/phase6_publication_strategy/ADR_phase6_corrected_strategy.md`
- Created this AI run log
- Updated `logs/ai_runs/INDEX.md`

No Phase 6A SNN simulation scaffold was created because the Phase 6A.0 gate returned `FAIL`.

## Commands Run

- `pwd`
- `git rev-parse --show-toplevel`
- `git status --short`
- `source /scratch/haizhe/stn/start_stn.sh && python -V`
- `find . -maxdepth 3 -type d | sort | head -300`
- `find . -maxdepth 3 -type f | sort | head -300`
- `find reports -maxdepth 4 -type f | sort | grep -E 'phase3|phase4|phase5|5_2|5Y|burden|burst|ADR|audit' | head -300 || true`
- `find data -maxdepth 5 -type f | sort | grep -E 'phase3|phase4|phase5|burst|burden|feature|label|stn|parquet|csv|json|pkl|npz|mat' | head -300 || true`
- `rg -n "Phase 5_2B|Phase 5_2C|burden|AUROC|268,959|LOSO|MedOff|MedOn|Hold|Move|UPDRS|bradykinesia|burst" reports scripts data config configs pyproject.toml README* 2>/dev/null | head -500 || true`
- Multiple bounded `sed`, `head`, `ps`, `lsof`, and `find` commands while debugging discovery scope.
- `/scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6a0_burden_viability.py`
- `source /scratch/haizhe/stn/start_stn.sh && python -m py_compile scripts/phase6a0_burden_viability.py`
- `source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_burden_viability.py --out-dir reports/phase6a0_burden_viability --stop-after-discovery`
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

## Validation Results

- `source /scratch/haizhe/stn/start_stn.sh && python -V` confirmed `/scratch/haizhe/stn/stn_env/bin/python` Python 3.12.3.
- `py_compile` passed for `scripts/phase6a0_burden_viability.py`.
- Discovery-only pass completed and selected `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv` with high confidence.
- Default CPU Slurm ladder failed immediately because the configured `standard` account/partition combination was invalid.
- A CPU-only teaching partition candidate succeeded: job `2577074`, 32 CPUs, 128 GB requested memory, completed in 00:09:19 with exit code 0.
- `git diff --check` passed.
- `find reports/phase6a0_burden_viability -type f -size +5M -print` produced no output.
- No external PPN/Herz dataset or audit report was used as a model input.

## Key Counts

- Selected input: `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv`
- Discovery candidates recorded: 1545
- Subjects found: 20
- Valid subjects: 20
- Allowed causal features: 354
- Selected model features: 96
- Excluded/non-feature columns: 19
- Leakage/suspicious excluded columns: 3
- Tau sweep: 200, 500, 800, 1500, 3000 ms
- Best tau: 3000 ms
- Best median LOSO Pearson: 0.24896977374123425
- Best median LOSO Spearman: 0.2602476064282385

## Findings

- Gate status: `FAIL`.
- The model beat class-prior and shuffled baselines but did not reach the `CONDITIONAL_PASS` lower bound of 0.30 median subject-level Pearson/Spearman correlation.
- Best model median Spearman at tau 3000 ms was 0.2602; class-prior baseline was 0.1072, shuffled-label baseline was 0.0381, and simple beta-feature baseline was 0.2280.
- Medication-state separability was weak supportive context only: MedOff vs MedOn AUROC/rank-biserial 0.5958, Cohen's d 0.2662.
- Task-state separability was weak or near chance.
- No UPDRS/bradykinesia/tremor/rigidity columns were present, so clinical proxy validation was not assessable.
- The result is annotation-derived burden validation on the internal STN substrate, not clinical validation.

## Limitations

- The model used a deterministic subset of 96 selected allowed causal features from 354 allowed causal features to keep the Phase 6A.0 run lightweight and report-sized.
- Discovery scoring is heuristic; the selected consolidated Phase 5_2C causal feature matrix had clearly higher score and confidence than alternatives.
- Precomputed feature normalization cannot be undone; leakage audit records available metadata and excludes suspicious columns.
- Window samples are autocorrelated, so the gate is based on subject-level medians, not pooled window statistics.

## Recommended Next Step

Stop Phase 6A Brian2/state-machine simulation work for now and reconsider the burden/state-machine pivot. Inspect target construction, feature families, annotation burden definition, and possible ablations before deciding whether another Phase 6A.0-style run is justified.

PPN remains optional cross-target generalization only after a primary STN architecture passes. Herz remains a methods/code reference only.

## Final Codex Output

The final response will report that Phase 6A.0 analysis completed with `FAIL`, identify the selected input, summarize the metrics and baselines, note that no simulation scaffold was created, and report commit/push status.

Commit after task: reported in final response to avoid self-referential commit hash churn.

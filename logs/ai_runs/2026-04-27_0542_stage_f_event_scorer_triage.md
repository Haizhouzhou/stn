# AI Run Log - Stage F Event Scorer Triage

- Date/local time: 2026-04-27 05:42 CEST
- Task name: Diagnose slow Phase 5_2C Stage F event scorer and prepare safe recovery path
- Git commit hash before task: `6ba1d01599d04d3ed90874e8926e906f0934853c`
- Git commit hash after task: reported in final response if a commit is created

## Complete Original User Prompt

```text
We have an existing Slurm job still running:

Job ID: 2565967
Instruction constraint: do NOT kill/cancel this job and do NOT launch a duplicate production job that writes the same final outputs while it is still running.

Current status:
- Runtime is already >90 minutes.
- It is CPU-active.
- sstat showed:
  JobID            AveCPU     MaxRSS     AveRSS   NTasks
  2565967.batch    01:36:42   5292892K   1001136K        1
- Expected output files still do not exist:
  results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv
  results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv
  results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv
  results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv

Task:
First analyze why the existing Stage F / event scorer is taking so long. Inspect the relevant scripts, logs, input table sizes, loops, joins, groupbys, and output-writing behavior. Identify the bottleneck without interfering with job 2565967.

Allowed:
- Read scripts, logs, and input/output directories.
- Check non-invasive Slurm/accounting/status information.
- Benchmark small samples only.
- Write an optimized/corrected scorer in a new scratch path or new script name.
- Validate logic on a small subset.
- Prepare the exact command that should be run later if the original job times out or fails validation.

Not allowed:
- Do not scancel or kill job 2565967.
- Do not overwrite the expected production output files while job 2565967 is still running.
- Do not submit a second full production job that duplicates job 2565967 while it is still running.
- Do not remove or modify files that the running job may need.
- Do not use the same final output paths except for read-only existence/mtime checks.

Goal:
Produce a diagnosis of the slowdown, a faster corrected implementation if possible, and a ready-to-run recovery command that can be used only after job 2565967 exits, times out, or fails validation.
```

## Operational Plan

1. Inspect Slurm status, command, logs, and expected output paths read-only.
2. Inspect Stage F scripts, validation, config, input table sizes, and scorer loops.
3. Benchmark only small samples from the causal feature matrix.
4. Add an optimized scorer under a new script/module name without modifying the script used by the running job.
5. Validate equivalence on a small in-memory subset and verify the production guard refuses to run while job `2565967` is active.
6. Update `outputs/current_status.md` with the long-run triage report.

## Files Inspected

- `slurm/slurm_phase5_2c_stage_f_event.sh`
- `scripts/05_2c_stage_f_event_completion.py`
- `scripts/05_2c_validate_stage_f_event_outputs.py`
- `src/stnbeta/phase5_2c/stage_f_event_metrics.py`
- `src/stnbeta/phase5_2c/stage_f_event_validation.py`
- `src/stnbeta/phase5_2c/io.py`
- `src/stnbeta/phase5_2c/loso_baselines.py`
- `configs/phase5_2c.yaml`
- `configs/phase5_2c_causal_frontend.yaml`
- `results/logs/05_phase5/phase5_2c/phase5_2c_event_f_2565967.out`
- `results/logs/05_phase5/phase5_2c/phase5_2c_event_f_2565967.err`
- `results/logs/05_phase5/phase5_2c/phase5_2c_event_f_2565956.out`
- `results/logs/05_phase5/phase5_2c/phase5_2c_event_f_2565956.err`
- `results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv` metadata/header/sample only
- `results/tables/05_phase5/phase5_2c/causal_minimum_sufficient_subset.tsv`

## Files Modified Or Created

- Created `src/stnbeta/phase5_2c/stage_f_event_metrics_fast.py`
- Created `scripts/05_2c_stage_f_event_completion_fast.py`
- Rewrote `outputs/current_status.md`
- Created this AI run log
- Updated `logs/ai_runs/INDEX.md`

## Commands Run

- `git status -sb`
- `squeue -j 2565967 ...`
- `sstat -j 2565967.batch ...`
- `sacct -j 2565967 ...`
- `scontrol show job 2565967`
- `sed`/`nl` reads of Stage F scripts, configs, validation, and helper modules
- `find` reads for logs and Stage F output path existence/mtime
- `stat` on causal matrix and related Stage F tables
- Small Python samples from `causal_feature_matrix.tsv` using selected columns only
- Small in-memory equivalence benchmark comparing original and optimized event scorer on 5000 rows
- Small optimized benchmark on 50000 rows and 4 score bundles
- `python -m py_compile scripts/05_2c_stage_f_event_completion_fast.py src/stnbeta/phase5_2c/stage_f_event_metrics_fast.py`
- `python scripts/05_2c_stage_f_event_completion_fast.py ... --require-job-exited 2565967`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest tests/test_phase5_2c_event_metrics.py -q -p no:cacheprovider`
- `git diff --check -- scripts/05_2c_stage_f_event_completion_fast.py src/stnbeta/phase5_2c/stage_f_event_metrics_fast.py outputs/current_status.md`

## Validation Results

- Job `2565967` was observed running and CPU-active; it was not killed or modified.
- No duplicate production job was launched.
- Expected production event output files were only checked for existence/mtime and were not written by Codex.
- `causal_feature_matrix.tsv` size: about `4.36 GB`; sampled first 1000 lines and estimated roughly `692k` rows, consistent with config expectation `690088`.
- Original-vs-fast small-sample equivalence on 5000 rows passed for tier event outputs, threshold grid, and alarm trace summary.
- 5000-row benchmark with 4 bundles: original `3.959 s`; optimized `0.717 s`; about `5.52x` faster.
- 50000-row optimized sample with 4 bundles: read `1.301 s`; fold score `0.023 s`; optimized event scoring `7.938 s`.
- Guarded runner refused to run while `squeue -j 2565967` showed the original job.
- `py_compile` passed.
- Existing event metric unit tests passed: `6 passed in 28.13s`.
- `git diff --check` passed for touched files.

## Diagnosis

The slowdown is caused by serial full-frame event reconstruction in `stage_f_event_metrics.py`. The current implementation evaluates about 60 thresholds for each default score bundle, including 30 mismatch seeds, and repeatedly rebuilds pandas alarm tables, sorts candidate rows, groups by `fif_path/channel`, runs a Python refractory loop, and then repeats alarm reconstruction for subject summaries. This underuses the 32 allocated CPUs and hides progress because final event tables are written only after all bundles and thresholds complete.

## Recovery Command Prepared

Run only after job `2565967` exits, times out, or fails validation:

```bash
~/bin/claim_best_immediate_resource.sh --mode cpu "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/05_2c_stage_f_event_completion_fast.py --config configs/phase5_2c.yaml --causal-config configs/phase5_2c_causal_frontend.yaml --require-job-exited 2565967 && python scripts/05_2c_validate_stage_f_event_outputs.py --config configs/phase5_2c.yaml"
```

## Risks / Blockers

- The optimized scorer was validated on small samples only; full production should still be validated with `scripts/05_2c_validate_stage_f_event_outputs.py`.
- Runtime estimates are based on sample benchmarks and code structure because the running job emits no per-threshold progress.
- The optimized subject-level fast path assumes a `fif_path/channel` group belongs to one subject, consistent with the current matrix path convention.

## Final Codex Output

The final user-facing response is delivered in the chat final answer and reports the commit/push status if a checkpoint is created.

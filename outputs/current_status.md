# Current Long-Running Job Status

- updated: `2026-04-27 05:38 CEST`
- job_id: `2565967`
- job_name: `phase5_2c_event_f`
- command: `/scratch/haizhe/stn/slurm/slurm_phase5_2c_stage_f_event.sh`
- working_directory: `/scratch/haizhe/stn`
- elapsed_time_observed: `01:48:22`
- time_limit: `02:00:00`
- requested_resources: `1 node, 1 task, 32 CPUs/task, 128G memory, no GPU`
- observed_use: `CPU-active; AveCPU about elapsed wall time, MaxRSS about 5.3G, AveRSS about 1.1G`
- output_logs:
  - `/scratch/haizhe/stn/results/logs/05_phase5/phase5_2c/phase5_2c_event_f_2565967.out`
  - `/scratch/haizhe/stn/results/logs/05_phase5/phase5_2c/phase5_2c_event_f_2565967.err`

## Expected Outputs

Missing at the time of inspection:

- `results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv`

Present from the current event-scoring run:

- `results/tables/05_phase5/phase5_2c/event_scoring_target_resolution.tsv`
- `results/tables/05_phase5/phase5_2c/event_scoring_policy.tsv`

Last observed event-scoring output modification time: `2026-04-27 03:53:01 CEST`.

## Diagnosis

The job is not idle; it is CPU-active. The bottleneck is the serial Stage F event reconstruction loop in `src/stnbeta/phase5_2c/stage_f_event_metrics.py`.

The causal feature matrix is about `4.36 GB` with an expected `690088` rows. Stage F loads a 14-column subset, computes one base score, builds 32 score bundles by default, then evaluates about 60 thresholds per bundle. For every threshold, the current implementation rebuilds a pandas alarm table, sorts candidate rows, groups by `fif_path/channel`, applies a Python refractory loop, and evaluates event matches. It then rebuilds alarm tables again for subject-level summaries. This creates roughly `32 * 60 = 1920` repeated full-frame mask/sort/groupby passes plus 128 additional subject-summary alarm reconstructions. The allocation has 32 CPUs, but the scorer is effectively single-process Python/pandas work.

Output-writing behavior also hides progress: after the two target-resolution files, the required event metric tables are written only after all tiers, thresholds, mismatch seeds, and subject summaries complete.

## Progress And ETA

- estimated_progress: target-resolution complete; event metric reconstruction still in progress or stuck inside the serial threshold/bundle loop.
- estimated_remaining_time_for_original: uncertain, but likely close to or beyond the remaining wall-time budget because no final event output exists after about 1:48 elapsed and the job has a 2:00 limit.
- bottleneck_hypothesis: CPU-bound serial pandas/Python loop with repeated sort/groupby/refractory reconstruction; not memory-bound and not GPU-related.
- original_job_left_running: yes.
- original_job_cancelled: no.
- duplicate_production_job_started: no.

## Safe Recovery Path Prepared

Prepared but not launched as production while job `2565967` is running:

- optimized module: `src/stnbeta/phase5_2c/stage_f_event_metrics_fast.py`
- guarded runner: `scripts/05_2c_stage_f_event_completion_fast.py`

The optimized scorer preserves the Stage F output contract but precomputes event/group caches and avoids repeated pandas sort/groupby work. A small in-memory equivalence check on 5000 rows passed against the original implementation. A 50000-row sample with 4 score bundles took about `1.3 s` to read, `0.02 s` to fold-score, and `7.94 s` for optimized event scoring.

Recovery command to run only after job `2565967` exits, times out, or fails validation:

```bash
~/bin/claim_best_immediate_resource.sh --mode cpu "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/05_2c_stage_f_event_completion_fast.py --config configs/phase5_2c.yaml --causal-config configs/phase5_2c_causal_frontend.yaml --require-job-exited 2565967 && python scripts/05_2c_validate_stage_f_event_outputs.py --config configs/phase5_2c.yaml"
```

The runner refuses to start while `squeue -j 2565967` still shows the original job.

## Validation Plan

After any recovery run, validate:

1. `python scripts/05_2c_validate_stage_f_event_outputs.py --config configs/phase5_2c.yaml`
2. Confirm all four required event/readiness outputs exist and are non-empty.
3. Compare row counts and required FP/min grid coverage across tier outputs.
4. Review `causal_stage_f_event_output_validation_summary.tsv` before treating outputs as final.

## Assumptions And Uncertainties

- `fif_path/channel` groups are assumed to belong to one subject, which matches the causal feature matrix path convention and the existing event scorer.
- Runtime estimates are based on small-sample benchmarks and code complexity, not on internal progress counters, because the current scorer does not emit per-threshold progress.
- No production output path was overwritten by Codex during this triage.

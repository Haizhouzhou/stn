# Current Stage F Recovery Status

- updated: `2026-04-27 06:16 CEST`
- repository: `/scratch/haizhe/stn`
- observed_pwd: `/home/haizhe/scratch/stn`
- head_commit: `554b970aa75dd0f68b1e2c397cb7ed33383ebcf6`
- recovery_status: `completed`
- validation_status: `passed`

## Reconstructed Starting State

- `git log -1 --oneline`: `554b970 phase5_2c: add guarded fast event scorer recovery`
- commit `554b970aa75dd0f68b1e2c397cb7ed33383ebcf6` was present.
- worktree had many pre-existing dirty/untracked files; this task avoided unrelated paths.
- prior triage log present: `logs/ai_runs/2026-04-27_0542_stage_f_event_scorer_triage.md`
- prior status file reported job `2565967` still running at `2026-04-27 05:38 CEST` with the four final event outputs missing.

## Original Job 2565967

- job_name: `phase5_2c_event_f`
- command: `/scratch/haizhe/stn/slurm/slurm_phase5_2c_stage_f_event.sh`
- working_directory: `/scratch/haizhe/stn`
- squeue_state_at_reconnect: no active row for `2565967`
- sacct_state: `TIMEOUT`
- elapsed: `02:00:18`
- timelimit: `02:00:00`
- allocated_resources: `32 CPUs, 128G memory, teaching partition`
- batch_step_state: `CANCELLED`, exit code `0:15`
- original_job_left_untouched: yes
- original_job_cancelled_by_codex: no
- duplicate_run_while_original_active: no

## Output State Before Recovery

At reconnect, all four expected production outputs were missing:

- `results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv`
- `results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv`

## Recovery Attempts

The guarded recovery was run only after `squeue -j 2565967` showed no active job and `sacct` reported timeout.

First attempt used the default CPU ladder from `~/bin/claim_best_immediate_resource.sh --mode cpu`. It did not allocate resources and did not enter the command phase because `standard` was invalid for `mlnlp2.pilot.s3it.uzh`:

- failed candidates: `standard` with `32/128G`, `24/96G`, `16/64G`, and `8/32G`
- failure reason: `Invalid account or account/partition combination specified`
- production outputs written by this attempt: no

After inspecting Slurm/account state, `teaching` was identified as the valid partition for this account. The successful recovery used an explicit strongest valid immediate CPU candidate:

```bash
~/bin/claim_best_immediate_resource.sh --mode cpu \
  --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --exclusive --cpus-per-task=80 --mem=0 --time=08:00:00" \
  --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=512G --time=08:00:00" \
  --candidate "--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00" \
  "cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/05_2c_stage_f_event_completion_fast.py --config configs/phase5_2c.yaml --causal-config configs/phase5_2c_causal_frontend.yaml --require-job-exited 2565967 && python scripts/05_2c_validate_stage_f_event_outputs.py --config configs/phase5_2c.yaml"
```

Selected resource:

- recovery_job_id: `2566076`
- selected_resource: `teaching, 1 exclusive node, 80 CPUs, all node memory, 08:00:00`
- host: `u24-chiivm0-603`
- visible_cpu_count: `80`
- visible_memory: about `755 GiB`
- CUDA_VISIBLE_DEVICES: unset
- PyTorch CUDA visible devices: `0`
- CPU-mode GPU visibility result: acceptable
- command_exit: `0`
- sacct_state: `COMPLETED`
- elapsed: `00:19:07`
- MaxRSS: `7753984K`
- AveCPU: `00:18:15`
- seff_status: unavailable; `seff` not found on PATH

Resource logs:

- `outputs/immediate_resource_attempts.log`
- `outputs/resource_probe_report.md`
- `outputs/resource_attempt_20260427T035620Z_1174915_1.log`

## Final Output Files

| Path | Size Bytes | Mtime |
| --- | ---: | --- |
| `results/tables/05_phase5/phase5_2c/causal_tier1_event_metrics.tsv` | 4234 | `2026-04-27 06:15:26.419745753 +0200` |
| `results/tables/05_phase5/phase5_2c/causal_tier2_event_metrics.tsv` | 4249 | `2026-04-27 06:15:26.426745719 +0200` |
| `results/tables/05_phase5/phase5_2c/causal_tier3_event_metrics.tsv` | 112494 | `2026-04-27 06:15:26.440745652 +0200` |
| `results/tables/05_phase5/phase5_2c/stage_g_h_i_readiness_assessment.tsv` | 720 | `2026-04-27 06:15:26.552745116 +0200` |

Line counts:

- `causal_tier1_event_metrics.tsv`: `5`
- `causal_tier2_event_metrics.tsv`: `5`
- `causal_tier3_event_metrics.tsv`: `121`
- `stage_g_h_i_readiness_assessment.tsv`: `2`

Additional Stage F outputs written by the recovery:

- `results/tables/05_phase5/phase5_2c/causal_event_threshold_grid.tsv`
- `results/tables/05_phase5/phase5_2c/causal_event_alarm_trace_summary.tsv`
- `results/tables/05_phase5/phase5_2c/causal_three_tier_event_summary.tsv`
- `results/tables/05_phase5/phase5_2c/causal_stage_f_event_output_validation_summary.tsv`

## Validation

Validation command:

```bash
python scripts/05_2c_validate_stage_f_event_outputs.py --config configs/phase5_2c.yaml
```

Validation result:

- `Phase 5_2C Stage F event output validation passed`
- validation summary: `results/tables/05_phase5/phase5_2c/causal_stage_f_event_output_validation_summary.tsv`
- readiness: `ready_to_plan_g_h_i`
- target_status: `unavailable`
- can_plan_stage_g_h_i: `True`
- can_execute_stage_g_h_i_now: `False`
- readiness_qc_reason: `Historical FP/min and latency references exist, but no explicit Phase 5_2C event-level recall target/gate was found.`

## Assumptions And Notes

- Production outputs were written only after job `2565967` had timed out and left `squeue`.
- The guarded runner kept `--require-job-exited 2565967` enabled.
- No action was taken to cancel, kill, or modify job `2565967`.
- The successful recovery used CPU resources only because the Stage F fast scorer is CPU/pandas work.
- `outputs/current_status.md` was overwritten by the resource launcher during recovery and then replaced with this task-level recovery report.

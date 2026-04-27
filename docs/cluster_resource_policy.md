# UZH Slurm Cluster Resource Policy

This repository inherits the global Codex policy in `~/.codex/AGENTS.md`.

For this policy, V100 is the strongest available GPU type. GPU jobs should use the largest immediately available V100 allocation first, with enough CPU cores, memory, wall time, and node count for the task. Do not wait indefinitely, do not submit waiting jobs unless explicitly requested, and do not use CPU fallback for GPU jobs.

Before running work inside an allocation, verify CUDA, visible GPU count, CPU count, and memory. For GPU jobs, PyTorch must see the requested GPUs before the command runs; if PyTorch sees zero GPUs or fewer GPUs than requested, abort or retry a corrected immediate V100 request.

## Long-Running Job Triage Summary

Do not passively wait for long-running jobs that are slow, stalled, under-resourced, or not producing expected outputs. First triage safely with Slurm status commands, logs, output timestamps/sizes, and observable CPU/GPU/memory usage. Estimate progress and remaining runtime, diagnose the likely bottleneck, and record the assessment in `outputs/current_status.md`.

Do not kill productive jobs without explicit user approval, do not modify raw/research data, and do not launch duplicate jobs that write to the same active output path. Safe acceleration is allowed only when the original job is untouched, the accelerated attempt writes to a separate staging path such as `outputs/<task>/accelerated_<timestamp>/`, the input/output contract is preserved, stronger immediate resources are used when appropriate, and validation is completed before any result is treated as final. Accelerated heavy jobs should use `~/bin/claim_best_immediate_resource.sh --mode gpu` for GPU tasks or `--mode cpu` for CPU/memory-heavy tasks. Use immediate allocations only; do not submit waiting jobs.

## Launchers

GPU jobs:

```bash
~/bin/claim_best_immediate_resource.sh --mode gpu "cd /path/to/repo && <command>"
```

CPU/memory-heavy jobs:

```bash
~/bin/claim_best_immediate_resource.sh --mode cpu "cd /path/to/repo && <command>"
```

The launcher writes:

- `outputs/resource_probe_report.md`
- `outputs/immediate_resource_attempts.log`
- `outputs/current_status.md`

## Preferred Interactive GPU Command

Local Slurm inspection on 2026-04-25 reported typed V100 GRES on the `lowprio` partition with up to 8 V100 GPUs on one node. The preferred interactive request is:

```bash
srun --partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:8 --cpus-per-task=64 --mem=256G --time=04:00:00 --pty bash
```

If partition or account visibility changes, use the verified largest valid V100 request:

```bash
srun --partition=<PARTITION> --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:<MAX_VALID_V100_COUNT> --cpus-per-task=<BALANCED_CPU_COUNT> --mem=<BALANCED_MEMORY> --time=04:00:00 --pty bash
```

## Default Ladders

GPU fallback ladder:

1. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:8 --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=256G --time=04:00:00`
2. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:7 --nodes=1 --ntasks=1 --cpus-per-task=56 --mem=224G --time=04:00:00`
3. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:6 --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=192G --time=04:00:00`
4. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:5 --nodes=1 --ntasks=1 --cpus-per-task=40 --mem=160G --time=04:00:00`
5. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:4 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00`
6. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:3 --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=96G --time=04:00:00`
7. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:2 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=04:00:00`
8. `--partition=lowprio --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00`

CPU/memory fallback ladder:

1. `--partition=standard --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00`
2. `--partition=standard --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=96G --time=04:00:00`
3. `--partition=standard --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=04:00:00`
4. `--partition=standard --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=04:00:00`

For bigger-than-default V100 jobs, inspect `sinfo`, `squeue`, and `scontrol`, then prepend a stronger immediate V100 candidate:

```bash
~/bin/claim_best_immediate_resource.sh --mode gpu --candidate "--partition=<PARTITION> --account=mlnlp2.pilot.s3it.uzh --qos=normal --gres=gpu:V100:<COUNT> --nodes=<N> --ntasks=<N> --cpus-per-task=<CPUS> --mem=<MEM> --time=<TIME>" "cd /path/to/repo && <command>"
```

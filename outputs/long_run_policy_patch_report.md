# Long-Running Job Policy Patch Report

Date: 2026-04-27 05:25 CEST

## 1. Files Changed

- `/home/haizhe/.codex/AGENTS.md`
- `AGENTS.md`
- `docs/cluster_resource_policy.md`
- `outputs/long_run_policy_patch_report.md`
- `logs/ai_runs/2026-04-27_0525_long_run_policy_patch.md`
- `logs/ai_runs/INDEX.md`

`Makefile` was not changed because no `Makefile` exists in this repository.

## 2. Exact New Long-Running Job Policy Section

The complete reusable section added to `/home/haizhe/.codex/AGENTS.md` is:

```markdown
## Long-Running Job Triage and Safe Acceleration Policy

Core principle: the user's time is more expensive than compute. Do not passively wait forever for slow jobs. If a job is running much longer than expected, has no output progress, appears under-resourced, or is likely to miss useful turnaround, Codex must analyze, estimate, and attempt safe acceleration when possible.

Rules:

- Do not kill running productive jobs unless the user explicitly authorizes it.
- Do not modify research/raw data.
- Do not corrupt, overwrite, or race existing output files.
- Do not launch duplicate jobs that write to the same output path as an active job.
- Do not submit jobs that wait indefinitely.
- Do not use CPU fallback for GPU tasks.
- Do not silently downgrade resources.
- Do not passively wait for long-running jobs without periodic ETA and progress analysis.

When a job runs longer than expected or appears stalled, Codex must:

1. Identify the running job, command, working directory, output paths, logs, and current state.
2. Inspect safe status only: `squeue`, `sacct`, `sstat` if available, job logs, output file timestamps and sizes, and CPU/GPU/memory usage when safely observable.
3. Estimate likely remaining runtime from available evidence: processed records/items so far, output growth rate, log progress counters, CPU/GPU utilization, and elapsed time versus expected work.
4. Diagnose the likely bottleneck: CPU-bound, memory-bound, I/O-bound, GPU not used, too few workers, inefficient algorithm, serialization bottleneck, stuck/no progress, or waiting on an external resource.
5. Decide whether safe acceleration is possible.

Safe acceleration is allowed when:

- The original job remains untouched.
- The accelerated attempt writes to a separate staging directory or uniquely named output path.
- The accelerated attempt preserves the same input/output contract.
- The accelerated attempt uses stronger immediate Slurm resources when appropriate.
- The accelerated attempt logs why it was launched.
- The accelerated attempt has validation steps before any result is treated as final.
- Any final replacement or promotion of outputs is atomic or explicitly approved.

Safe acceleration examples:

- Run a corrected implementation that fixes a known bottleneck.
- Increase CPU cores, memory, or V100 GPU count using the immediate-resource launcher.
- Use chunking, multiprocessing, batching, vectorization, or GPU inference when compatible with the task.
- Resume from checkpoints if available.
- Produce outputs in a staging directory such as `outputs/<task>/accelerated_<timestamp>/`.
- Compare accelerated outputs against the expected schema and, where possible, against partial outputs from the original job.

Unsafe acceleration examples:

- Starting the same command again with the same output path.
- Overwriting outputs from the running job.
- Editing raw data.
- Killing the original job without explicit user approval.
- Switching GPU tasks to CPU.
- Reducing resources without logging failed stronger attempts.
- Producing a different output schema or incomplete result while claiming success.

Required long-run triage report: when a job has been running longer than expected, Codex must write or update a short report, preferably `outputs/current_status.md`. The report must include job id, command if known, working directory if known, elapsed time, requested resources, observed CPU/GPU/memory use if available, output files expected, output files currently present, last output modification time, estimated progress, estimated remaining time, bottleneck hypothesis, whether the original job is being left running, whether an accelerated attempt is recommended or started, exact resource request for any accelerated attempt, staging output path for any accelerated attempt, validation plan, and assumptions/uncertainties.

Interaction rule: if the original instruction says "do not kill" or "do not duplicate," interpret that as "do not kill the original productive job" and "do not duplicate the same unsafe write path." Still analyze the job and, when safe, run an accelerated equivalent attempt in an isolated staging path.

If the accelerated attempt finishes first, validate schema, row counts, checksums or deterministic fields where applicable, and compare against partial or expected outputs where possible. Do not delete the original job's outputs. Do not kill the original job unless the user explicitly approves. Clearly report which output is validated and which job produced it.

If the original job finishes first, validate it normally. Cancel or ignore any not-yet-started acceleration attempt if it is safe and useful to do so. Do not delete staged outputs unless explicitly asked.

Resource policy integration: for any accelerated heavy job, use `~/bin/claim_best_immediate_resource.sh` unless the user explicitly requests a different resource strategy. For GPU tasks, use `--mode gpu`. For CPU/memory-heavy tasks, use `--mode cpu`. Use immediate allocations only. Do not submit waiting jobs.
```

The repo-local `AGENTS.md` now adds a short inheritance section with the same operational requirements and points to the global policy to avoid duplicating the full resource ladder.

## 3. How This Changes Behavior For Slow Jobs

Codex should no longer merely wait for hours when a job is slow or not producing expected outputs. It must inspect safe status, estimate progress and remaining runtime, diagnose bottlenecks, update `outputs/current_status.md`, and decide whether a safe accelerated attempt can run in isolation.

The policy keeps the no-corruption constraints: productive jobs stay running unless the user authorizes termination, raw/research data is untouched, and active output paths are not overwritten or raced.

## 4. Handling A Running Job With No Expected Outputs

Codex should identify the job id, command, working directory, logs, expected outputs, current outputs, timestamps, sizes, and resource request. It should inspect safe Slurm/accounting/log information, estimate progress and ETA from any counters or output growth, form a bottleneck hypothesis, and write the current assessment to `outputs/current_status.md`.

If acceleration is safe, Codex can launch a corrected or better-resourced attempt only to a distinct staging path, using the immediate-resource launcher and logging the exact resource request and validation plan. If acceleration is not safe, Codex should say why and keep monitoring with ETA updates rather than passively waiting.

## 5. Safe Acceleration Versus Unsafe Duplication

Safe acceleration preserves the input/output contract while isolating writes. It leaves the original job untouched, writes to a unique staging directory, uses stronger immediate resources where useful, validates schema and deterministic fields, and requires atomic or explicit approval before promoting results.

Unsafe duplication reruns the same command against the same output path, overwrites active outputs, edits raw data, kills the original job without approval, silently downgrades resources, switches GPU work to CPU, or claims success from incomplete or schema-incompatible outputs.

## 6. Assumptions

- The repository-local policy should not duplicate the full global resource ladder; it should inherit the full global triage policy and add project-specific report/staging paths.
- `outputs/long_run_policy_patch_report.md` is a small requested policy report, not a scientific generated artifact from a running job.
- No active output file was overwritten by this patch; this report file did not exist before the task.
- No real Slurm job was submitted, cancelled, or inspected beyond reading policy files and repository state.
- No raw data, extracted data, subject-level files, or active result outputs were modified.

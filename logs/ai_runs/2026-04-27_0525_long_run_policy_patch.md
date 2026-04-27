# 2026-04-27 05:25 CEST - Long-Running Job Policy Patch

## Task Name
Add long-running job triage and safe acceleration policy.

## Git Commit Before Task
`408814e7301852c226a9cf9a0a093393fcc35358`

## Original User Prompt
```text
You are working in this repository on the UZH Slurm cluster.

Patch the resource policy because the current behavior is too passive for long-running jobs. Codex must not kill productive jobs or corrupt outputs, but it also must not simply wait for hours when a job is slow, stuck, under-resourced, or not producing expected outputs.

Goal:
Add a reusable “Long-Running Job Triage and Safe Acceleration Policy” to the relevant AGENTS.md files.

Files to inspect and update safely:
- ~/.codex/AGENTS.md
- ./AGENTS.md if this is inside a repository
- docs/cluster_resource_policy.md if present

Preserve existing useful instructions. Do not duplicate large policy sections unnecessarily.

Add or update a section with the following policy:

Long-Running Job Triage and Safe Acceleration Policy

Core principle:
My time is more expensive than compute. Do not passively wait forever for slow jobs. If a job is running much longer than expected, has no output progress, appears under-resourced, or is likely to miss useful turnaround, Codex must analyze, estimate, and attempt safe acceleration when possible.

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
2. Inspect safe status only:
   - squeue
   - sacct
   - sstat if available
   - job logs
   - output file timestamps and sizes
   - CPU/GPU/memory usage when safely observable
3. Estimate likely remaining runtime from available evidence:
   - processed records/items so far
   - output growth rate
   - log progress counters
   - CPU/GPU utilization
   - elapsed time versus expected work
4. Diagnose likely bottleneck:
   - CPU-bound
   - memory-bound
   - I/O-bound
   - GPU not used
   - too few workers
   - inefficient algorithm
   - serialization bottleneck
   - stuck or no progress
   - waiting on external resource
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
- Produce outputs in a staging directory such as outputs/<task>/accelerated_<timestamp>/.
- Compare accelerated outputs against the expected schema and, where possible, against partial outputs from the original job.

Unsafe acceleration examples:
- Starting the same command again with the same output path.
- Overwriting outputs from the running job.
- Editing raw data.
- Killing the original job without explicit user approval.
- Switching GPU tasks to CPU.
- Reducing resources without logging failed stronger attempts.
- Producing a different output schema or incomplete result while claiming success.

Required long-run triage report:
When a job has been running longer than expected, Codex must write or update a short report, preferably:

outputs/current_status.md

The report must include:
- job id
- command if known
- working directory if known
- elapsed time
- requested resources
- observed CPU/GPU/memory use if available
- output files expected
- output files currently present
- last output modification time
- estimated progress
- estimated remaining time
- bottleneck hypothesis
- whether the original job is being left running
- whether an accelerated attempt is recommended or started
- exact resource request for any accelerated attempt
- staging output path for any accelerated attempt
- validation plan
- assumptions and uncertainties

Interaction rule:
If the original instruction says “do not kill” or “do not duplicate,” interpret that as:
- Do not kill the original productive job.
- Do not duplicate the same unsafe write path.
- But do analyze and, when safe, run an accelerated equivalent attempt in an isolated staging path.

If the accelerated attempt finishes first:
- Validate schema, row counts, checksums or deterministic fields where applicable.
- Compare against partial or expected outputs where possible.
- Do not delete the original job’s outputs.
- Do not kill the original job unless the user explicitly approves.
- Clearly report which output is validated and which job produced it.

If the original job finishes first:
- Validate it normally.
- Cancel or ignore any not-yet-started acceleration attempt if it is safe and useful to do so.
- Do not delete staged outputs unless explicitly asked.

Resource policy integration:
For any accelerated heavy job, use ~/bin/claim_best_immediate_resource.sh unless the user explicitly requests a different resource strategy.
For GPU tasks, use --mode gpu.
For CPU/memory-heavy tasks, use --mode cpu.
Use immediate allocations only. Do not submit waiting jobs.

Also patch docs/cluster_resource_policy.md, if present, with a short summary of this long-running job triage policy.

If a Makefile exists, add a non-breaking helper target only if it does not conflict with existing targets:

long-run-status:
	@echo "Inspect running Slurm jobs with squeue/sacct and update outputs/current_status.md manually or through Codex."

Safe verification only:
- Show the modified AGENTS.md section.
- Do not submit a real job.
- Do not kill any job.
- Do not modify raw data.
- Do not overwrite active outputs.

Write a final report to:

outputs/long_run_policy_patch_report.md

or, if not inside a repo:

long_run_policy_patch_report.md

The report must include:
1. Files changed.
2. The exact new long-running job policy section.
3. How this changes behavior for slow jobs.
4. How Codex should handle a running job that has not produced expected outputs.
5. How safe acceleration differs from unsafe duplication.
6. Any assumptions made.
```

## Operational Plan
1. Inspect global/repo policy files, the cluster policy doc, `Makefile` existence, and the report path.
2. Patch the global AGENTS file with the full reusable long-running job triage policy.
3. Patch repo-local `AGENTS.md` with a shorter inheritance section to avoid duplicating the global resource ladder.
4. Patch `docs/cluster_resource_policy.md` with a short summary.
5. Create `outputs/long_run_policy_patch_report.md` with required details.
6. Run safe verification only; no Slurm job submission, no job killing, no raw-data or active-output modifications.

## Files Inspected
- `/home/haizhe/.codex/AGENTS.md`
- `AGENTS.md`
- `docs/cluster_resource_policy.md`
- `.gitignore`
- `outputs/`
- `logs/ai_runs/`

## Files Modified Or Created
- `/home/haizhe/.codex/AGENTS.md`
- `AGENTS.md`
- `docs/cluster_resource_policy.md`
- `outputs/long_run_policy_patch_report.md`
- `logs/ai_runs/2026-04-27_0525_long_run_policy_patch.md`
- `logs/ai_runs/INDEX.md`

## Commands Run
```bash
ls -l ~/.codex/AGENTS.md ./AGENTS.md docs/cluster_resource_policy.md Makefile 2>&1
rg -n "Long-Running Job|Safe Acceleration|claim_best_immediate|long-run-status|Cluster Resource|Slurm Resource|Login-node|Git Checkpoint|AI Run" ~/.codex/AGENTS.md ./AGENTS.md docs/cluster_resource_policy.md Makefile 2>/dev/null
test -f outputs/long_run_policy_patch_report.md && ls -l outputs/long_run_policy_patch_report.md || true
git status -sb
git rev-parse HEAD
sed -n '1,220p' ~/.codex/AGENTS.md
sed -n '1,190p' ./AGENTS.md
sed -n '1,220p' docs/cluster_resource_policy.md
ls -ld outputs logs/ai_runs
test -f .gitignore && sed -n '1,220p' .gitignore || true
git check-ignore -v outputs/long_run_policy_patch_report.md logs/ai_runs/INDEX.md docs/cluster_resource_policy.md AGENTS.md 2>/dev/null || true
date '+%Y-%m-%d %H:%M %Z'
```

Post-edit validation and checkpoint commands are recorded in the final response.

Additional post-edit commands:

```bash
rg -n "^## Long-Running Job Triage" ~/.codex/AGENTS.md ./AGENTS.md docs/cluster_resource_policy.md
sed -n '35,125p' ~/.codex/AGENTS.md
sed -n '35,76p' ./AGENTS.md
sed -n '1,55p' docs/cluster_resource_policy.md
git diff --check -- AGENTS.md docs/cluster_resource_policy.md logs/ai_runs/INDEX.md logs/ai_runs/2026-04-27_0525_long_run_policy_patch.md outputs/long_run_policy_patch_report.md
git diff --stat -- AGENTS.md docs/cluster_resource_policy.md logs/ai_runs/INDEX.md logs/ai_runs/2026-04-27_0525_long_run_policy_patch.md outputs/long_run_policy_patch_report.md
```

## Validation Results
- Found inserted policy sections in `/home/haizhe/.codex/AGENTS.md`, repo-local `AGENTS.md`, and `docs/cluster_resource_policy.md`.
- Verified the global AGENTS policy contains the full long-running triage and safe acceleration policy.
- Verified repo-local `AGENTS.md` contains the inherited project-specific triage section.
- Verified `docs/cluster_resource_policy.md` contains a short long-running job triage summary.
- `git diff --check` over the changed policy/report/log files produced no output.
- No Slurm job was submitted, no job was killed, and no raw data was modified.

## Final Codex Response Text
Pending until checkpoint completion. The final response reports the commit and push status; embedding the final commit hash in this log would change the hash again.

## Remaining Risks / Blockers
- The global policy file is outside the repository and cannot be included in a git commit.
- The repository has many pre-existing unrelated modified and untracked paths; they are not part of this policy patch.

## Git Commit After Task
Pending before commit. See final response for exact commit and push status.

# AGENTS.md - stnbeta Project Working Rules

## Repository Identity
- This repository is the STN-LFP beta-burst SNN / NSM detector project for eventual DYNAP-SE1 evaluation.
- STN-LFP is the primary signal. Do not add MEG to the primary detector pipeline.
- Brian2 / Brian2CUDA is the primary SNN simulation path.
- PyTorch may be used for training, diagnostics, SG-BPTT-style upper bounds, and feasibility checks, but PyTorch-only models are not accepted final SNN/NSM detectors unless replayed or represented in Brian2 / Brian2-equivalent simulation.
- Accepted SNN/NSM candidates must have a Brian2, Brian2CUDA, or Brian2-equivalent simulation/replay path.
- NumPy, Pandas, and SciPy analysis is allowed for audits, summaries, and deterministic probes, but NumPy-only branches are not accepted final SNN/NSM detector candidates.
- Do not replace the final detector with a generic black-box classifier.
- Hardware-aware candidates must report DYNAP-SE1 feasibility: binary synapses, 64 CAM slots per neuron, parallel connections for graded weights, shared biases per 256-neuron core, fan-in/core mapping, quantization, and mismatch.

## Scientific Boundaries
- Do not alter frozen labels, frozen splits, evaluation boundaries, phase gates, or benchmark definitions unless explicitly requested by the user. Any approved change must be recorded in `docs/decisions.md` or an ADR.
- Do not deploy to DYNAP-SE1 hardware unless explicitly requested.
- Do not modify raw or research data while probing, debugging, or running analyses.
- Do not turn `AGENTS.md` into a history file for old phases. Keep only active, project-wide guardrails here.

## Environment Boot Rule
- Start every compute-node session and every Slurm batch script with:
  `source /scratch/haizhe/stn/start_stn.sh`
- The boot script activates `/scratch/haizhe/stn/stn_env`, loads CUDA, sets CUDA/Brian2CUDA library paths, preserves Slurm GPU visibility, and sets JAX memory behavior.
- New GPU training, Brian2CUDA, JAX, PyTorch, CuPy, or Numba work should use `stn_env`.
- `.venv-phase4` is legacy / archival and should be used only when a specific old script requires that exact environment.
- Lightweight validation:
  `source /scratch/haizhe/stn/start_stn.sh && python scripts/validate_stn_env.py`

## Project-wide UZH Slurm Resource Policy
- This repo inherits the global UZH Slurm Cluster Resource Policy from `~/.codex/AGENTS.md`; do not duplicate the global resource ladder here.
- The user's time is more expensive than compute. For GPU work, heavy CPU work, memory-intensive jobs, large data processing, dense analysis, embedding generation, LLM inference, or model training, request the strongest immediately available valid resource configuration first.
- On this UZH Slurm cluster, V100 is the strongest available GPU type. Do not search for, wait for, or request unavailable A100, H100, or newer GPUs.
- For any GPU, heavy CPU, memory-intensive, large data processing, embedding, inference, or training task, use `~/bin/claim_best_immediate_resource.sh` unless the user explicitly requests a different resource strategy.
- Resource strength includes V100 GPU count, CPU cores, memory, wall time, and node count when relevant.
- Use immediate/no-wait allocation behavior for exploratory and production heavy jobs by default. Use `--immediate=120` when claiming resources through Slurm unless the user explicitly asks for a queued job.
- Never submit jobs that wait indefinitely. Never silently choose weaker resources. Log every failed stronger attempt and why a weaker configuration was selected.
- Never use CPU fallback for GPU tasks. If a task requires GPU, abort if PyTorch/CUDA sees zero GPUs or fewer GPUs than requested.
- If multiple GPUs are requested, verify that the code actually uses them; allocation alone is not enough.
- If the code is CPU-only, do not request GPUs just to make the allocation look stronger; instead request strong CPU/memory resources and verify CPU utilization.
- Do not under-request CPU or memory for V100 jobs. For memory-heavy jobs, increase memory before reducing CPU/GPU count when possible.
- Do not kill productive running jobs.
- Long runs must be checkpointed, resumable, and have manifests, config hashes, logs, and status files.

## Long-Running Job Triage and Safe Acceleration Policy
- This repo inherits the full long-running job triage policy from `~/.codex/AGENTS.md`; do not duplicate the global checklist or resource ladder here.
- Do not passively wait for long-running jobs that are slow, stalled, under-resourced, not producing expected outputs, or likely to miss useful turnaround. Analyze progress, estimate ETA, diagnose the bottleneck, and decide whether safe acceleration is possible.
- Never kill a productive running job unless the user explicitly authorizes it. Never modify raw/research data. Never corrupt, overwrite, race, or duplicate an active output path.
- Safe acceleration must leave the original job untouched, preserve the same input/output contract, write to a separate staging path such as `outputs/<task>/accelerated_<timestamp>/`, use stronger immediate Slurm resources when appropriate, log why it was launched, and validate outputs before treating them as final.
- Unsafe acceleration includes rerunning the same command against the same output path, overwriting active outputs, editing raw data, switching GPU work to CPU, silently reducing resources, or claiming success from an incomplete/different schema.
- When a long-running job needs triage, write or update `outputs/current_status.md` with job id, command, working directory, elapsed time, requested resources, observed usage, expected/current outputs, last output modification time, progress estimate, ETA, bottleneck hypothesis, whether the original job is left running, any acceleration recommendation, exact accelerated resource request, staging output path, validation plan, and assumptions/uncertainties.
- If a user says "do not kill" or "do not duplicate," interpret that as do not kill the productive job and do not duplicate the unsafe write path. Still analyze and, when safe, run an isolated accelerated equivalent in a staging path.
- For accelerated heavy jobs, use `~/bin/claim_best_immediate_resource.sh` unless the user explicitly requests a different strategy: `--mode gpu` for GPU tasks and `--mode cpu` for CPU/memory-heavy tasks. Use immediate allocations only; do not submit waiting jobs.
- If an accelerated attempt finishes first, validate schema, row counts, checksums or deterministic fields where applicable, compare against partial/expected outputs where possible, do not delete original outputs, and do not kill the original job without explicit approval.

## Slurm Submission Rules
- For ad hoc heavy commands and resource-sensitive work, prefer `~/bin/claim_best_immediate_resource.sh`.
- For production runs, prefer `sbatch` only after the resource request has been selected according to the strongest-immediate policy.
- Do not use ordinary queued `sbatch` by default unless the user explicitly requests a queued job or immediate allocation is unsuitable and the reason is logged.
- Before submitting any existing `sbatch` script for a heavy job, inspect its `#SBATCH` resource lines and compare them against the current strongest-immediate-resource policy.
- If the script uses a weaker or fixed allocation, either patch/wrap the command through `~/bin/claim_best_immediate_resource.sh`, or explicitly log why the script's allocation is appropriate.
- A successful `sbatch --test-only` or immediate-start prediction is not enough; Codex must also justify CPU/GPU/memory/wall-time choice.
- For production `sbatch` jobs, static scripts are acceptable only when they encode the selected strong resource request, include required preflight checks, and write logs/status outputs.

## Required Slurm Preflight and Post-run Checks
Before running any heavy command, log this preflight:

```bash
hostname
echo SLURM_JOB_ID=$SLURM_JOB_ID
echo SLURM_STEP_ID=$SLURM_STEP_ID
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE
echo SLURM_MEM_PER_CPU=$SLURM_MEM_PER_CPU
nproc
free -h
ulimit -a
nvidia-smi || true
python - <<'PY'
import os
print("python preflight")
print("cpu_count_os", os.cpu_count())
try:
    import psutil
    print("cpu_count_psutil", psutil.cpu_count())
    print("virtual_memory", psutil.virtual_memory())
except Exception as e:
    print("psutil_unavailable", e)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, torch.cuda.get_device_name(i), p.total_memory)
except Exception as e:
    print("torch_preflight_failed", e)
PY
```
- For GPU tasks, abort if CUDA/GPU visibility is wrong.
- For CPU-only tasks, verify CPU count and memory before running.
- After Slurm jobs, always run and record:
  `sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- Also run and record `seff <JOBID>` when available.
- Do not trust outputs until `sacct` confirms successful completion.
- Low CPU efficiency means the code may not be using allocated CPUs; do not blindly request more CPUs without inspecting parallelism/vectorization.

## Login-node Protection
- Do not run MNE, FIF loading, raw LFP processing, dense training, full real-data analysis, large matrix computations, embedding generation, LLM inference, GPU jobs, or large preprocessing on the login node.
- No heavy jobs, model training, dense data processing, or GPU allocations should be started unless the task explicitly calls for them.
- If `CPU time limit exceeded` appears at a login-node prompt, check for local/background processes before blaming Slurm allocation.
- Use `squeue`, `sacct`, `ps`, and `jobs -l` to distinguish Slurm jobs from login-node processes.
- Do not leave background compute terminals running on the login node.

## Output Preservation and Reproducibility
- Do not overwrite completed results.
- New runners should be resume-safe and checkpointed when runtime is nontrivial.
- Write manifests, config hashes, logs, and status JSONs for new outputs.
- Preserve baseline outputs before any new sweep or comparison.
- Record important scientific decisions in `docs/decisions.md` or in a dedicated ADR under `docs/adr/`.
- Decision records should state the context, decision, rationale, affected phase or dataset, and expected consequence for later phases.

## AI Run Provenance Logging Policy
- For every non-trivial Codex task, create one task-level Markdown log under `logs/ai_runs/YYYY-MM-DD_HHMM_<short_task_slug>.md`.
- Each task log must include: date and local time; task or phase name; Git commit hash before the task from `git rev-parse HEAD`; the complete original user prompt for the task; a concise operational plan; files inspected; files modified or created; commands run; validation, test, smoke-check, or audit results; the complete final Codex output for the task; remaining risks, blockers, or recommended next actions; and the Git commit hash after the task if a commit was created.
- Use one Markdown file per task or run. Do not collapse all runs into a single giant log file.
- Update `logs/ai_runs/INDEX.md` with the date, task, log path, commit hash, and status for each task log.
- Do not paste massive raw stdout, huge intermediate model outputs, raw data dumps, subject-identifiable data, private credentials, local environment contents, ignored files, or files excluded by `.gitignore` into AI run logs. Summarize long outputs and point to artifact paths instead.
- If the user explicitly constrains a task to a different file set, obey that narrower scope and do not create a run log.
- End-of-task AI run logging is mandatory for every non-trivial task unless the user explicitly narrows the task scope in a way that forbids log creation.
- Before the final response, ensure the task log contains the final response text that will be reported to the user, including commit and push status.

## Git Checkpoint / Commit / Push Policy
- This repository uses GitHub as a remote research checkpoint system, not only as a software release target.
- At the end of every completed non-trivial task, Codex should automatically create a local commit and push it to the configured GitHub remote unless the user explicitly says not to commit or push.
- This policy is task-boundary based. Push only after the task has reached a coherent checkpoint, not after every small intermediate edit.
- Before committing and pushing, run and record:
  - `git status --short`
  - `git diff --stat`
  - `git diff --check`
  - A hard large-file check equivalent to:
    ```bash
    git ls-files -s | awk '{print $4}' | while read -r f; do
      [ -f "$f" ] || continue
      size=$(stat -c%s "$f")
      if [ "$size" -ge 100000000 ]; then
        printf '%s\t%s\n' "$size" "$f"
      fi
    done
    ```
- The large-file check must produce no output before commit or push.
- Do not commit or push `raw/`, local environments such as `stn_env/`, `stn_env_corrupted/`, `.venv/`, `venv/`, or `env/`, cache directories, ignored files, files larger than or equal to 100 MB, subject-identifiable or non-shareable raw research data, failed temporary outputs unless intentionally preserved as diagnostics, or secrets or credentials.
- Before staging, prepare an explicit safe staging list split into policy/logging files; source/config/docs/tests/slurm files; files to ignore or leave untracked; and files needing user decision.
- Never use `git add .`, `git add -A`, broad shell globs, or other repository-wide staging shortcuts for checkpoint commits in this repo.
- Stage only reviewed, explicit paths with `git add -- <path> ...`. If unsafe untracked paths, subject-level files, generated outputs, local environments, or large files are present, show the proposed staging list to the user before staging.
- If checks pass and staging has been reviewed where required, Codex may run path-limited commands such as:
  ```bash
  git add -- <explicit-safe-path-1> <explicit-safe-path-2>
  git commit -m "<clear project/task commit message>"
  git push
  ```
- Commit messages must identify the project task and briefly summarize what changed. Avoid vague messages such as `update` or `fix`.
- If a task is exploratory and produces no file changes, Codex should still create or update the AI run log and push that log unless the user explicitly says not to commit or push.
- If the user gives a task-specific no-commit or no-push instruction, that explicit instruction overrides the automatic checkpoint policy for that task.
- The final Codex response must report whether the AI run log was created or updated, whether a commit was created, whether a push was attempted or completed, and if commit or push was skipped, the reason.

## Commit / Push Failure Behavior
- If commit or push fails, do not keep retrying blindly.
- Record the failure in the AI run log, show the exact failure summary to the user, and leave the repository in a safe, inspectable state.
- Do not rewrite Git history unless the user explicitly asks.

## AGENTS.md-Only Policy Updates
- When the user asks to update `AGENTS.md` only, modify only `AGENTS.md`.
- After editing `AGENTS.md`, show the diff for `AGENTS.md` only and verify that no other files were modified by the task.
- Do not commit or push an `AGENTS.md`-only policy update when the user explicitly asks only to prepare the modification or says not to commit or push.

## Lightweight Test Commands
- Environment validation: `source /scratch/haizhe/stn/start_stn.sh && python scripts/validate_stn_env.py`
- Quick full suite: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest tests/ -q -p no:cacheprovider`
- Diff hygiene: `git diff --check`

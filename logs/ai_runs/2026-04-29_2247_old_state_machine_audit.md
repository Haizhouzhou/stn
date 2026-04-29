# AI Run Log: Old Brian2 State-Machine Audit

Date and local time: 2026-04-29 22:47 CEST

Task / phase name: Legacy old state-machine audit

Git commit before task: `47e9bd64d679bde05d37f063fe2544b6a40d809a`

## Original User Prompt

```text
You are working in the legacy STN beta repository:

    /scratch/haizhe/stn

Goal:
Verify whether my memory is correct:

1. The old Brian2/Brian2CUDA status/state machine worked well on synthetic simulation tests.
2. The old Tinkhauser/Tinkerhauser-style real-data beta-burst onset-alarm task performed badly or failed the low-false-positive event-scoring gate.
3. Therefore, the old state-machine mechanism should be treated as reusable engineering substrate, but the old Tinkhauser onset-alarm task should remain abandoned/archived as negative evidence.

Important constraints:
- Do not run heavy jobs on the login node.
- Do not run full FIF extraction, full real-data sweeps, GPU sweeps, or Brian2CUDA jobs.
- Prefer read-only inspection, grep/ripgrep, lightweight pytest, and synthetic runtime tests only.
- If Brian2 or CUDA is unavailable, do not install anything; just report what could not be run.
- Do not change source code unless explicitly needed for a small report file.
- Keep a clear distinction between:
    a) state-machine mechanism validation
    b) real-data detector performance
    c) old Tinkhauser/onset-alarm target
    d) new adaptive beta-burden state target

Start with these commands:

    cd /scratch/haizhe/stn
    pwd
    git status --short
    source /scratch/haizhe/stn/start_stn.sh || true
    python scripts/validate_stn_env.py || true

Inspect the key files:

    rg -n "low-false|false positive|false-positive|Tinkhauser|Tinkerhauser|onset|alarm|negative|failed|Phase 5|synthetic|duration progression|short-event|readout|D2|D2\\+" docs src scripts tests configs results -S

    ls -la
    ls -la scripts tests configs docs src/stnbeta/snn_brian2 src/stnbeta/phase5 2>/dev/null || true

Open and summarize these files if present:

    docs/PHASE4_RUNBOOK.md
    docs/PHASE5_RUNBOOK.md
    scripts/04b_validate_state_machine_synthetic.py
    scripts/05a_validate_state_machine_synthetic.py
    scripts/05b_run_phase5_dev.py
    tests/test_state_machine_synthetic.py
    tests/test_phase5_state_machine.py
    src/stnbeta/snn_brian2/runner.py
    src/stnbeta/snn_brian2/neuron_model.py
    src/stnbeta/snn_brian2/synapse_model.py
    src/stnbeta/snn_brian2/topologies/nsm_monotonic_duration.py
    src/stnbeta/phase5/readout.py
    src/stnbeta/phase5/metrics.py
    configs/nsm_mono.yaml
    configs/nsm_mono_frozen.yaml

Run only lightweight tests if the environment is ready:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest \
      tests/test_state_machine_synthetic.py \
      tests/test_phase5_state_machine.py \
      -q -p no:cacheprovider

If pytest is successful or mostly successful, optionally run the lightweight synthetic Phase 5 topology test only:

    python scripts/05a_validate_state_machine_synthetic.py \
      --level topology \
      --backend runtime \
      --no-grid \
      --out results/phase5_synthetic/codex_quick_check_topology

Do not run end-to-end grids unless they are very small and already documented as safe.

Search for historical result tables:

    find results -maxdepth 5 -type f \( \
      -name "*.tsv" -o -name "*.csv" -o -name "*.json" -o -name "*.md" -o -name "*.txt" \
    \) | sort | sed -n '1,300p'

Then inspect likely result summaries:

    rg -n "precision|recall|F1|false|FP|min|event|onset|latency|Tinkhauser|Tinkerhauser|negative|failed|D2|stable|synthetic" results docs -S

If tables are found, use Python to summarize filenames, shapes, columns, and any performance fields:

    python - <<'PY'
    from pathlib import Path
    import pandas as pd

    paths = []
    for ext in ("*.tsv", "*.csv"):
        paths.extend(Path("results").rglob(ext))

    keywords = ["phase5", "real", "dev", "metric", "summary", "synthetic", "tink", "onset", "event"]
    selected = [p for p in paths if any(k in str(p).lower() for k in keywords)]

    print(f"Found {len(paths)} tables total, {len(selected)} selected")
    for p in selected[:80]:
        try:
            sep = "\t" if p.suffix == ".tsv" else ","
            df = pd.read_csv(p, sep=sep)
            print("\n==", p)
            print("shape:", df.shape)
            print("columns:", list(df.columns)[:30])
            interesting = [c for c in df.columns if any(k in c.lower() for k in [
                "precision","recall","f1","false","fp","latency","event","score","synthetic","stable"
            ])]
            if interesting:
                print(df[interesting].head(10).to_string(index=False))
        except Exception as e:
            print("Could not read", p, e)
    PY

Produce a final audit report with the following sections:

1. Verdict:
   - Was the old state/status machine successful on synthetic simulations?
   - Did the old Tinkhauser-style real-data/onset-alarm detector fail or underperform?
   - Should the old state-machine core be copied into the new project?

2. Evidence from files:
   - Exact file paths.
   - Relevant line numbers if possible.
   - Short quotes or paraphrases.

3. Synthetic validation:
   - Which tests/scripts exist?
   - Which cases are tested?
   - Which tests were run just now?
   - Pass/fail status.

4. Real-data / Tinkhauser-style task:
   - What was the target?
   - What scoring gate was used?
   - What results or documentation indicate failure/negative evidence?
   - Was the failure due to the state-machine dynamics, the front-end features, the event target, or the scoring definition?

5. Reusable architecture:
   - Identify the old state-machine populations and topology:
       encoder/input
       quiet/reset
       excitatory state clusters
       inhibitory clusters
       forward progression
       recurrent sustain
       lateral suppression
       readout
   - Identify what should be copied.
   - Identify what should not be copied blindly.

6. New-project recommendation:
   - Treat the old state machine as a reusable SNN substrate.
   - Do not revive the old Tinkhauser onset-alarm task.
   - Retune/revalidate against the new adaptive STN beta-burden state target.
   - Keep SNN input as spike streams, not raw analog LFP.

7. Uncertainty:
   - State clearly if any historical result tables were missing or could not be inspected.
   - State clearly if any test could not be run due to environment limitations.

Save the report to:

    results/old_state_machine_audit/codex_old_state_machine_check.md

Create the directory if needed.
```

## Operational Plan

1. Run the requested startup/environment commands without launching heavy work.
2. Inspect requested docs, scripts, tests, configs, source, and preserved results with `rg`, `ls`, `nl`, and small Python table summaries.
3. Avoid full FIF extraction, real-data sweeps, GPU sweeps, Brian2CUDA jobs, and heavy login-node computation.
4. Run only lightweight tests if safe; avoid the CUDA smoke because no Slurm GPU was allocated and no GPU was visible.
5. Write the requested report under `results/old_state_machine_audit/`.
6. Record this AI run and checkpoint only explicit safe files.

## Files Inspected

- `docs/PHASE4_RUNBOOK.md`
- `docs/PHASE5_RUNBOOK.md`
- `docs/PHASE5_FINAL_DECISION.md`
- `docs/decisions.md`
- `docs/PHASE7_REPO_SPLIT_AND_LEGACY_CLOSEOUT.md`
- `docs/ADR_PROJECT_SPLIT_TO_STNBETA_STATE_MONITOR.md`
- `docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE_REVISED.md`
- `docs/PHASE5_2C_PRE_ADR_BOUNDED_REMEDIATION_ANALYSIS.md`
- `docs/PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md`
- `scripts/04b_validate_state_machine_synthetic.py`
- `scripts/05a_validate_state_machine_synthetic.py`
- `scripts/05b_run_phase5_dev.py`
- `tests/test_state_machine_synthetic.py`
- `tests/test_phase5_state_machine.py`
- `src/stnbeta/snn_brian2/runner.py`
- `src/stnbeta/snn_brian2/neuron_model.py`
- `src/stnbeta/snn_brian2/synapse_model.py`
- `src/stnbeta/snn_brian2/topologies/nsm_monotonic_duration.py`
- `src/stnbeta/phase5/readout.py`
- `src/stnbeta/phase5/metrics.py`
- `configs/nsm_mono.yaml`
- `configs/nsm_mono_frozen.yaml`
- selected Phase 4/5 result summary TSV/CSV files under `results/`

## Files Modified or Created

- Created `results/old_state_machine_audit/codex_old_state_machine_check.md`.
- Created this run log: `logs/ai_runs/2026-04-29_2247_old_state_machine_audit.md`.
- Updated `logs/ai_runs/INDEX.md`.

No source code was modified.

## Commands Run

- `pwd`
- `git status --short`
- `source /scratch/haizhe/stn/start_stn.sh || true`
- `python scripts/validate_stn_env.py || true`
- Broad `rg` over `docs src scripts tests configs results`; stopped after it matched huge result tables and produced excessive output.
- `ls -la`
- `ls -la scripts tests configs docs src/stnbeta/snn_brian2 src/stnbeta/phase5 2>/dev/null || true`
- `wc -l` on requested key files.
- Targeted `rg` and `nl -ba ... | sed -n ...` inspections on docs, tests, runner, metrics, readout, configs, and final-decision files.
- `find results -maxdepth 5 -type f ... | sort | sed -n '1,300p'`
- Targeted result `rg`; stopped after matching dense/large result tables.
- Safety-filtered Python/pandas table summarizer over likely Phase 5 result summaries, excluding dense feature matrices and very large trace/prediction files.
- Lightweight pytest attempt excluding CUDA smoke:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /scratch/haizhe/stn/stn_env/bin/python -m pytest tests/test_state_machine_synthetic.py tests/test_phase5_state_machine.py -q -p no:cacheprovider -k 'not cuda_smoke'`

## Validation / Audit Results

- Environment validation completed. Brian2 and Brian2CUDA were importable in `stn_env`.
- CUDA was not usable in the login-node session: `CUDA_VISIBLE_DEVICES` was unset, `nvidia-smi` could not communicate with the driver, and PyTorch reported zero CUDA devices.
- The requested pytest file set contains a CUDA smoke test gated only by command availability. I excluded it to comply with the no-Brian2CUDA/no-GPU-job constraint.
- The remaining lightweight pytest command did not complete promptly and was stopped to preserve the lightweight boundary. No fresh pytest pass/fail claim is made.
- Optional `scripts/05a_validate_state_machine_synthetic.py --level topology ...` was not run because pytest did not complete successfully or mostly successfully.
- The audit report concludes:
  - the synthetic/topology state-machine mechanism worked and is reusable;
  - the real-data low-FP onset-alarm detector path failed or underperformed;
  - the old Tinkhauser-style onset-alarm task should remain archived as negative evidence;
  - the new target should retune/revalidate the substrate for adaptive STN beta-burden/state monitoring.

## Remaining Risks / Blockers / Recommended Next Actions

- Fresh runtime tests remain unresolved because the lightweight pytest attempt did not finish promptly.
- No fresh Brian2CUDA/GPU validation was possible without a Slurm GPU allocation, and this task explicitly forbade Brian2CUDA jobs.
- Not every historical table under `results/` was inspected; the audit focused on preserved summary tables and high-value closeout documents.
- A future compute-node task could run a bounded CPU synthetic topology smoke if the owner wants a fresh empirical check, but it is not needed for the historical verdict.

## Final Codex Output Draft

The audit report is saved at `results/old_state_machine_audit/codex_old_state_machine_check.md`.

Verdict: your memory is basically correct. The old Brian2 state-machine mechanism is reusable engineering substrate because the topology/synthetic evidence supports short-event rejection, duration progression, reset behavior, and negative-control specificity. The old Tinkhauser-style real-data onset-alarm path failed or underperformed the low-false-positive event-recovery gate, so it should remain archived as negative evidence rather than revived as the active target.

I did not run any heavy jobs, full FIF extraction, GPU sweeps, or Brian2CUDA. Brian2/Brian2CUDA imported, but no GPU was visible; the lightweight pytest attempt excluding the CUDA smoke did not finish promptly and was stopped, so I did not make a new test-pass claim. The report separates synthetic mechanism validation, real-data detector performance, the old onset-alarm target, and the new adaptive beta-burden/state recommendation.

Commit/push status will be reported in the final chat response after the explicit-path checkpoint is attempted.

## Git Commit After Task

Pending at log write time; exact checkpoint status is reported in the final response to avoid self-referential commit-hash churn.

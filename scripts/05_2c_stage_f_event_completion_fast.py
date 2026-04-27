#!/usr/bin/env python
"""Run Phase 5_2C Stage F event scoring with the optimized scorer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stnbeta.phase5_2c import stage_f_event_metrics as stage
from stnbeta.phase5_2c.io import append_command_log, load_config
from stnbeta.phase5_2c.stage_f_event_metrics_fast import compute_event_outputs_fast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase5_2c.yaml")
    parser.add_argument("--causal-config", default="configs/phase5_2c_causal_frontend.yaml")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--table-dir", default=None, help="Optional table output directory for staged validation runs.")
    parser.add_argument("--require-job-exited", default=None, help="Refuse to run while this Slurm job id is still visible in squeue.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    if args.require_job_exited and slurm_job_running(args.require_job_exited):
        print(f"Refusing to run: Slurm job {args.require_job_exited} is still running or pending.", file=sys.stderr)
        return 2
    config = load_config(args.config, out_root=args.out_root)
    if args.table_dir is not None:
        config.setdefault("paths", {})["table_dir"] = args.table_dir
    config.setdefault("execution", {})["resume"] = bool(args.resume)
    with Path(args.causal_config).open("r", encoding="utf-8") as f:
        causal = yaml.safe_load(f) or {}
    config.setdefault("inputs", {}).update({k: v for k, v in causal.get("inputs", {}).items() if k.startswith("causal_")})
    stage.compute_event_outputs = compute_event_outputs_fast
    append_command_log(config, sys.argv, status="started", message="Stage F event completion with optimized scorer")
    result = stage.run_stage_f_event_completion(config)
    append_command_log(config, sys.argv, status="completed", message=str(result))
    print(result)
    return 0


def slurm_job_running(job_id: str) -> bool:
    try:
        result = subprocess.run(["squeue", "-h", "-j", str(job_id)], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return False
    if result.returncode not in {0, 1}:
        return False
    return bool(result.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(main())

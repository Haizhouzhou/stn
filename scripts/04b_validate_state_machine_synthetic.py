"""Validate the monotonic duration-bucket state machine on synthetic traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stnbeta.encoding.lif_encoder import (
    currents_from_filtered_bands,
    load_lif_encoder_config,
)
from stnbeta.phase4.config import config_hash, load_yaml
from stnbeta.phase4.front_end import (
    apply_lif_overrides,
    apply_nsm_overrides,
    apply_rectify_overrides,
    nsm_with_lif_defaults,
    split_front_end_overrides,
)
from stnbeta.phase4.grid import expand_grid_points, filter_grid_points, split_rebuild_overrides
from stnbeta.phase4.manifests import collect_runtime_manifest, write_manifest
from stnbeta.phase4.metrics import synthetic_case_metrics
from stnbeta.preprocessing.filter_bank import apply_filter_bank, load_filter_bank_config
from stnbeta.preprocessing.rectify_amplify import load_rectify_amplify_config
from stnbeta.snn_brian2.runner import StandaloneStateMachineProject, derive_quiet_drive, run_state_machine
from stnbeta.snn_brian2.topologies.nsm_monotonic_duration import load_nsm_config
from stnbeta.synthetic.beta_burst_generator import SyntheticTrace, generate_trace_suite


def _parse_grid_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _score_metrics(df: pd.DataFrame) -> float:
    monotonic_score = df["monotonic_progression_correct"].sum() / max(df["n_ground_truth_bursts"].sum(), 1)
    reset_score = df["reset_success_count"].sum() / max(df["reset_total"].sum(), 1)
    readout_score = df["correct_readout_count"].sum() / max(df["expected_readout_total"].sum(), 1)
    rejection_score = df["too_short_rejection_success_count"].sum() / max(df["too_short_total"].sum(), 1)
    skipped_penalty = 0.1 * df["skipped_bucket_transitions"].sum()
    false_positive_penalty = df["false_positive_rate_hz"].mean()
    unexpected_penalty = 0.5 * df["unexpected_readout_count"].sum()
    missed_penalty = 0.25 * df["missed_readout_count"].sum()
    return float(
        monotonic_score
        + reset_score
        + readout_score
        + rejection_score
        - skipped_penalty
        - false_positive_penalty
        - unexpected_penalty
        - missed_penalty
    )


def _representative_figure(
    trace: SyntheticTrace,
    currents: np.ndarray,
    quiet_drive: np.ndarray,
    result,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(trace.time_s, trace.signal, color="black", lw=0.8)
    for row in trace.annotations.itertuples(index=False):
        axes[0].axvspan(row.onset_s, row.offset_s, color="#d95f02", alpha=0.2)
    axes[0].set_ylabel("Signal")
    axes[0].set_title(f"Synthetic trace: {trace.name}")

    axes[1].plot(trace.time_s, currents.mean(axis=0), color="#1b9e77", lw=1.0, label="mean band current")
    axes[1].plot(trace.time_s, quiet_drive, color="#7570b3", lw=0.9, label="quiet drive")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Current")

    axes[2].scatter(result.bucket_spike_times_s, result.bucket_spike_indices, s=8, color="#e7298a", label="bucket")
    if len(result.readout_spike_times_s) > 0:
        axes[2].vlines(
            result.readout_spike_times_s,
            ymin=-0.5,
            ymax=max(result.bucket_thresholds_ms) / 100.0,
            color="#66a61e",
            lw=1.0,
            label="readout",
        )
    axes[2].set_ylabel("Bucket")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-config", type=Path, default=Path("configs/synthetic_beta.yaml"))
    parser.add_argument("--filter-bank-config", type=Path, default=Path("configs/filter_bank.yaml"))
    parser.add_argument("--lif-config", type=Path, default=Path("configs/lif_encoder.yaml"))
    parser.add_argument("--nsm-config", type=Path, default=Path("configs/nsm_mono.yaml"))
    parser.add_argument("--grid-config", type=Path, default=Path("configs/gridsearch_phase4.yaml"))
    parser.add_argument("--grid-indices", default=None)
    parser.add_argument("--worker-label", default=None)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--max-grid-points", type=int, default=None)
    parser.add_argument("--backend", choices=["runtime", "cpp_standalone", "cuda_standalone"], default="runtime")
    parser.add_argument("--out", type=Path, default=Path("results/phase4_synthetic"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-capability", type=float, default=None)
    parser.add_argument("--cuda-runtime-version", type=float, default=None)
    args = parser.parse_args()

    worker_dir = args.out if args.worker_label is None else args.out / "workers" / args.worker_label
    worker_dir.mkdir(parents=True, exist_ok=True)

    traces = generate_trace_suite(args.synthetic_config)
    filter_bank_config = load_filter_bank_config(args.filter_bank_config)
    base_lif_config = load_lif_encoder_config(args.lif_config)
    base_rectify_config = load_rectify_amplify_config(args.lif_config)
    base_nsm_config = nsm_with_lif_defaults(load_nsm_config(args.nsm_config), base_lif_config)
    grid_cfg = {} if args.no_grid else load_yaml(args.grid_config)
    grid_points = expand_grid_points(grid_cfg, "synthetic")
    if args.max_grid_points is not None:
        grid_points = grid_points[: args.max_grid_points]
    indexed_points = filter_grid_points(grid_points, _parse_grid_indices(args.grid_indices))
    rebuild_axes = dict(grid_cfg.get("rebuild_axes", {}))

    preprocessed = []
    for trace in traces:
        filtered = apply_filter_bank(trace.signal, trace.sfreq_hz, filter_bank_config)
        band_names = [band.name for band in filter_bank_config.bands]
        band_roles = [band.role for band in filter_bank_config.bands]
        preprocessed.append((trace, band_names, band_roles, filtered))

    summary_rows: list[dict] = []
    best_result = None
    best_trace = None
    best_currents = None
    best_quiet = None
    best_score = -np.inf
    project_cache: dict[tuple[tuple[tuple[str, object], ...], int, int], StandaloneStateMachineProject] = {}

    for point_index, overrides in indexed_points:
        lif_overrides, rectify_overrides, nsm_overrides = split_front_end_overrides(overrides)
        lif_config = apply_lif_overrides(base_lif_config, lif_overrides)
        rectify_config = apply_rectify_overrides(base_rectify_config, rectify_overrides)
        nsm_with_encoder = nsm_with_lif_defaults(base_nsm_config, lif_config)
        build_overrides, run_overrides = split_rebuild_overrides(nsm_overrides, rebuild_axes)
        build_config = apply_nsm_overrides(nsm_with_encoder, build_overrides)
        run_config = apply_nsm_overrides(build_config, run_overrides)

        case_rows: list[dict] = []
        point_results: dict[str, tuple[SyntheticTrace, np.ndarray, np.ndarray, object]] = {}
        for trace, band_names, band_roles, filtered in preprocessed:
            currents = currents_from_filtered_bands(
                filtered,
                band_names,
                rectify_config=rectify_config,
                sfreq_hz=trace.sfreq_hz,
            )
            quiet_drive = derive_quiet_drive(currents, band_roles)
            if args.backend == "runtime":
                result = run_state_machine(
                    currents,
                    band_roles,
                    run_config,
                    backend="runtime",
                    quiet_drive=quiet_drive,
                    seed=args.seed,
                )
            else:
                key = (tuple(sorted(build_overrides.items())), currents.shape[0], currents.shape[-1])
                if key not in project_cache:
                    project_cache[key] = StandaloneStateMachineProject(
                        n_steps=currents.shape[-1],
                        n_inputs=currents.shape[0],
                        band_roles=band_roles,
                        config=build_config,
                        backend=args.backend,
                        build_dir=worker_dir / "standalone_build" / f"group_{len(project_cache):02d}",
                        record_voltage=False,
                        compute_capability=args.compute_capability,
                        cuda_runtime_version=args.cuda_runtime_version,
                    )
                result = project_cache[key].run(
                    currents,
                    quiet_drive=quiet_drive,
                    results_directory=worker_dir / "runs" / f"grid_{point_index}_{trace.name}",
                    overrides=run_overrides,
                )
            metrics = synthetic_case_metrics(trace, result)
            metrics.update({"grid_index": point_index, **overrides})
            case_rows.append(metrics)
            point_results[trace.name] = (trace, currents, quiet_drive, result)

        case_df = pd.DataFrame(case_rows)
        point_score = _score_metrics(case_df)
        summary_rows.append({"grid_index": point_index, "score": point_score, **overrides})
        case_df.to_csv(worker_dir / f"synthetic_case_metrics_grid_{point_index}.tsv", sep="\t", index=False)

        if point_score > best_score:
            best_score = point_score
            bundle = point_results.get("long_pathological") or next(iter(point_results.values()))
            best_trace, best_currents, best_quiet, best_result = bundle

    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    summary_df.to_csv(worker_dir / "grid_summary.tsv", sep="\t", index=False)
    if args.worker_label is None:
        Path("results/tables/04_phase4").mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(Path("results/tables/04_phase4/synthetic_grid_summary.tsv"), sep="\t", index=False)
        if best_trace is not None and best_result is not None and best_currents is not None and best_quiet is not None:
            _representative_figure(
                best_trace,
                best_currents,
                best_quiet,
                best_result,
                Path("results/figures/04_phase4/synthetic_state_machine_qc.png"),
            )

    manifest = collect_runtime_manifest(
        backend=args.backend,
        config_hash_value=config_hash(
            {
                "synthetic_config": load_yaml(args.synthetic_config),
                "filter_bank_config": load_yaml(args.filter_bank_config),
                "lif_config": load_yaml(args.lif_config),
                "nsm_config": load_yaml(args.nsm_config),
                "grid_config": grid_cfg,
            }
        ),
        seed=args.seed,
        extra={
            "best_score": best_score,
            "grid_indices": [index for index, _ in indexed_points],
            "worker_label": args.worker_label,
        },
    )
    write_manifest(worker_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

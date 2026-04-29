"""Validate the Phase 5 duration-bucket state machine on synthetic data."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stnbeta.encoding.lif_encoder import currents_from_filtered_bands, load_lif_encoder_config
from stnbeta.phase4.config import config_hash, load_yaml
from stnbeta.phase4.manifests import collect_runtime_manifest, write_manifest
from stnbeta.phase4.gpu import ensure_cuda_runtime_libraries
from stnbeta.phase5.grid import expand_grid_points, filter_grid_points, split_rebuild_overrides
from stnbeta.phase5.metrics import evaluate_synthetic_case, summarize_synthetic_metrics
from stnbeta.phase5.readout import build_readout_summary
from stnbeta.phase5.synthetic_suite import generate_end_to_end_suite, generate_topology_suite
from stnbeta.preprocessing.filter_bank import apply_filter_bank, load_filter_bank_config
from stnbeta.preprocessing.rectify_amplify import load_rectify_amplify_config
from stnbeta.snn_brian2.runner import (
    StandaloneDurationBucketProject,
    derive_quiet_drive,
    prepare_phase5_entry_currents,
    run_duration_bucket_state_machine,
)
from stnbeta.snn_brian2.topologies.nsm_monotonic_duration import (
    DurationBucketClusterConfig,
    load_duration_bucket_cluster_config,
)


def _parse_grid_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _phase5_with_lif_defaults(
    config: DurationBucketClusterConfig,
    *,
    lif_config,
) -> DurationBucketClusterConfig:
    return replace(
        config,
        dt_ms=lif_config.dt_ms,
        encoder_tau_ms=lif_config.tau_ms,
        encoder_threshold=lif_config.threshold,
        encoder_reset=lif_config.reset,
        encoder_refractory_ms=lif_config.refractory_ms,
        encoder_gain=lif_config.gain,
        encoder_bias=lif_config.bias,
    )


def _score_metrics(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    summary = summarize_synthetic_metrics(df)
    score = (
        3.0 * summary["short_burst_rejection_rate"]
        + 3.0 * summary["threshold_and_long_detection_rate"]
        + 2.0 * summary["reset_success_rate"]
        + 1.0 * summary["interrupt_behavior_rate"]
        - 0.5 * summary["skipped_bucket_transitions"]
        - 5.0 * summary["negative_false_positive_rate_hz"]
    )
    return float(score), summary


def _representative_bundle(case_rows: list[tuple[object, np.ndarray, object, object]]) -> tuple[object, np.ndarray, object, object]:
    preferred = {"threshold_crossing_120ms", "long_400ms_plus", "interrupted_burst_60_on_20_off_60_on"}
    for bundle in case_rows:
        if bundle[0].name in preferred:
            return bundle
    return case_rows[0]


def _plot_timeline(
    case,
    currents: np.ndarray,
    result,
    readout_summary,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    beta_indices = [index for index, role in enumerate(result.band_roles) if role == "beta" and index < currents.shape[0]]
    mean_beta = currents[beta_indices].mean(axis=0) if beta_indices else currents.mean(axis=0)
    axes[0].plot(case.time_s, mean_beta, color="#1b9e77", lw=1.0, label="beta evidence")
    for row in case.annotations.itertuples(index=False):
        axes[0].axvspan(row.onset_s, row.offset_s, color="#d95f02", alpha=0.18)
    axes[0].set_ylabel("Evidence")
    axes[0].legend(loc="upper right", fontsize=8)

    if case.signal is not None:
        axes[1].plot(case.time_s, case.signal, color="black", lw=0.7)
        axes[1].set_ylabel("LFP")
    else:
        axes[1].plot(case.time_s, currents[1:5].T, lw=0.5, alpha=0.8)
        axes[1].set_ylabel("Band Currents")

    for state_index, state_name in enumerate(result.state_names):
        axes[2].plot(case.time_s, result.occupancy[state_index] + 0.03 * state_index, lw=1.0, label=state_name)
    axes[2].legend(loc="upper right", fontsize=7, ncols=3)
    axes[2].set_ylabel("Occupancy")

    axes[3].plot(case.time_s, readout_summary.score, color="#e7298a", lw=1.0, label="readout score")
    axes[3].fill_between(
        case.time_s,
        0.0,
        np.where(readout_summary.stable_mask, readout_summary.score.max(initial=0.0), 0.0),
        color="#66a61e",
        alpha=0.18,
        label="stable readout",
    )
    axes[3].set_ylabel("Readout")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_pass_fail(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    passes = (
        (metrics_df["skipped_bucket_transitions"] == 0)
        & (metrics_df["reset_success_count"] == metrics_df["reset_total"])
        & (metrics_df["unexpected_readout_count"] == 0)
    )
    colors = np.where(passes, "#1b9e77", "#d95f02")
    ax.bar(metrics_df["trace_name"], metrics_df["correct_readout_count"] + metrics_df["too_short_rejection_success_count"], color=colors)
    ax.set_ylabel("Successful Outcomes")
    ax.set_title("Synthetic Phase 5 pass/fail panel")
    ax.tick_params(axis="x", labelrotation=35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_reset_panel(case, result, readout_summary, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    non_idle = result.occupancy[1:].sum(axis=0)
    axes[0].plot(case.time_s, non_idle, color="#7570b3", lw=1.0)
    for row in case.annotations.itertuples(index=False):
        axes[0].axvspan(row.onset_s, row.offset_s, color="#d95f02", alpha=0.18)
    axes[0].set_ylabel("Non-idle Occupancy")

    axes[1].plot(case.time_s, readout_summary.score, color="#e7298a", lw=1.0)
    axes[1].fill_between(case.time_s, 0.0, readout_summary.score, where=readout_summary.stable_mask, color="#66a61e", alpha=0.20)
    axes[1].set_ylabel("Readout")
    axes[1].set_xlabel("Time (s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["topology", "end_to_end"], default="topology")
    parser.add_argument("--filter-bank-config", type=Path, default=Path("configs/filter_bank.yaml"))
    parser.add_argument("--lif-config", type=Path, default=Path("configs/lif_encoder.yaml"))
    parser.add_argument("--nsm-config", type=Path, default=Path("configs/nsm_mono.yaml"))
    parser.add_argument("--grid-config", type=Path, default=Path("configs/gridsearch_phase5.yaml"))
    parser.add_argument("--grid-indices", default=None)
    parser.add_argument("--worker-label", default=None)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--backend", choices=["runtime", "cpp_standalone", "cuda_standalone"], default="runtime")
    parser.add_argument("--out", type=Path, default=Path("results/phase5_synthetic"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-capability", type=float, default=None)
    parser.add_argument("--cuda-runtime-version", type=float, default=None)
    args = parser.parse_args()

    if args.backend == "cuda_standalone":
        ensure_cuda_runtime_libraries()

    worker_dir = args.out if args.worker_label is None else args.out / "workers" / args.worker_label
    worker_dir.mkdir(parents=True, exist_ok=True)

    lif_config = load_lif_encoder_config(args.lif_config)
    base_config = _phase5_with_lif_defaults(
        load_duration_bucket_cluster_config(args.nsm_config),
        lif_config=lif_config,
    )
    grid_cfg = {} if args.no_grid else load_yaml(args.grid_config)
    section = "synthetic_topology" if args.level == "topology" else "synthetic_end_to_end"
    grid_points = expand_grid_points(grid_cfg, section)
    indexed_points = filter_grid_points(grid_points, _parse_grid_indices(args.grid_indices))
    rebuild_axes = dict(grid_cfg.get("rebuild_axes", {}))

    if args.level == "topology":
        cases = generate_topology_suite()
        preprocessed = [(case, np.asarray(case.direct_currents, dtype=np.float32)) for case in cases]
    else:
        filter_bank_config = load_filter_bank_config(args.filter_bank_config)
        rectify_config = load_rectify_amplify_config(args.lif_config)
        band_names = [band.name for band in filter_bank_config.bands]
        cases = generate_end_to_end_suite()
        preprocessed = []
        for case in cases:
            filtered = apply_filter_bank(np.asarray(case.signal, dtype=np.float32), case.sfreq_hz, filter_bank_config)
            currents = currents_from_filtered_bands(
                filtered,
                band_names,
                rectify_config=rectify_config,
                sfreq_hz=case.sfreq_hz,
            )
            preprocessed.append((case, currents))

    active_project_key: tuple[tuple[tuple[str, object], ...], int, int] | None = None
    active_project: StandaloneDurationBucketProject | None = None
    project_counter = 0
    summary_rows = []
    best_score = -np.inf
    best_df = None
    best_case_bundle = None

    for grid_index, overrides in indexed_points:
        build_overrides, run_overrides = split_rebuild_overrides(overrides, rebuild_axes)
        build_config = replace(base_config, **build_overrides)
        run_config = replace(build_config, **run_overrides)
        case_rows = []
        case_bundles = []
        for case, currents in preprocessed:
            model_currents, model_band_roles = prepare_phase5_entry_currents(
                currents,
                case.band_roles,
                mode=run_config.evidence_aggregation_mode,
            )
            quiet_drive = derive_quiet_drive(model_currents, model_band_roles)
            if args.backend == "runtime":
                result = run_duration_bucket_state_machine(
                    model_currents,
                    model_band_roles,
                    run_config,
                    backend="runtime",
                    quiet_drive=quiet_drive,
                    seed=args.seed,
                )
            else:
                key = (
                    tuple(sorted(build_overrides.items())),
                    model_currents.shape[0],
                    model_currents.shape[-1],
                )
                if key != active_project_key:
                    active_project = StandaloneDurationBucketProject(
                        n_steps=model_currents.shape[-1],
                        n_inputs=model_currents.shape[0],
                        band_roles=model_band_roles,
                        config=build_config,
                        backend=args.backend,
                        build_dir=worker_dir / "standalone_build" / f"group_{project_counter:02d}",
                        seed=args.seed,
                        compute_capability=args.compute_capability,
                        cuda_runtime_version=args.cuda_runtime_version,
                    )
                    active_project_key = key
                    project_counter += 1
                if active_project is None:
                    raise RuntimeError("Standalone project was not initialized")
                result = active_project.run(
                    model_currents,
                    quiet_drive=quiet_drive,
                    results_directory=worker_dir / "runs" / f"{args.level}_grid_{grid_index}_{case.name}",
                    overrides=run_overrides,
                )
            readout_summary = build_readout_summary(result, run_config)
            metrics = evaluate_synthetic_case(case, result, run_config, readout_summary)
            metrics.update({"grid_index": grid_index, **overrides})
            case_rows.append(metrics)
            case_bundles.append((case, model_currents, result, readout_summary))

        case_df = pd.DataFrame(case_rows)
        case_df.to_csv(worker_dir / f"{args.level}_case_metrics_grid_{grid_index}.tsv", sep="\t", index=False)
        score, extras = _score_metrics(case_df)
        summary_rows.append({"grid_index": grid_index, "score": score, **extras, **overrides})
        if score > best_score:
            best_score = score
            best_df = case_df
            best_case_bundle = _representative_bundle(case_bundles)

    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    summary_df.to_csv(worker_dir / "grid_summary.tsv", sep="\t", index=False)

    if args.worker_label is None and best_df is not None and best_case_bundle is not None:
        case, currents, result, readout_summary = best_case_bundle
        figures_dir = Path("results/figures/05_phase5")
        tables_dir = Path("results/tables/05_phase5")
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_name = "synthetic_suite_summary.tsv" if args.level == "topology" else "synthetic_end_to_end_summary.tsv"
        grid_name = "synthetic_sweep_summary.tsv" if args.level == "topology" else "synthetic_end_to_end_grid_summary.tsv"
        best_df.to_csv(tables_dir / summary_name, sep="\t", index=False)
        summary_df.to_csv(tables_dir / grid_name, sep="\t", index=False)
        _plot_timeline(case, currents, result, readout_summary, figures_dir / f"{args.level}_state_timeline.png")
        if args.level == "topology":
            _plot_pass_fail(best_df, figures_dir / "synthetic_pass_fail_panel.png")
            reset_case = next((bundle for bundle in preprocessed if bundle[0].name == "two_bursts_with_quiet_gap"), None)
            if reset_case is not None:
                reset_bundle = next(
                    (bundle for bundle in case_bundles if bundle[0].name == "two_bursts_with_quiet_gap"),
                    best_case_bundle,
                )
                _plot_reset_panel(reset_bundle[0], reset_bundle[2], reset_bundle[3], figures_dir / "synthetic_reset_panel.png")
        else:
            _plot_pass_fail(best_df, figures_dir / "synthetic_front_end_pass_fail_panel.png")

    manifest = collect_runtime_manifest(
        backend=args.backend,
        config_hash_value=config_hash(
            {
                "lif_config": load_yaml(args.lif_config),
                "nsm_config": load_yaml(args.nsm_config),
                "grid_config": grid_cfg,
                "level": args.level,
            }
        ),
        seed=args.seed,
        extra={
            "level": args.level,
            "best_score": best_score,
            "grid_indices": [index for index, _ in indexed_points],
            "worker_label": args.worker_label,
        },
    )
    write_manifest(worker_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

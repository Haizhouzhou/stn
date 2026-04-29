"""Run the Phase 5 duration-bucket detector on the primary real-data dev case."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from stnbeta.encoding.lif_encoder import currents_from_filtered_bands, load_lif_encoder_config
from stnbeta.ground_truth.pipeline import get_epoch_mask
from stnbeta.phase4.config import config_hash, load_yaml
from stnbeta.phase4.gpu import ensure_cuda_runtime_libraries
from stnbeta.phase4.manifests import collect_runtime_manifest, write_manifest
from stnbeta.phase4.real_data import load_real_condition_case
from stnbeta.phase5.grid import expand_grid_points, filter_grid_points, split_rebuild_overrides
from stnbeta.phase5.metrics import (
    band_contribution_table,
    evaluate_real_case,
    evaluate_readout_against_reference,
    latency_decomposition_table,
    match_readout_events_to_bursts,
    merge_event_tables,
    state_occupancy_table,
    summarize_latency_decomposition,
    summarize_real_metrics,
)
from stnbeta.phase5.readout import build_readout_summary, events_from_mask
from stnbeta.preprocessing.filter_bank import apply_filter_bank, load_filter_bank_config
from stnbeta.preprocessing.rectify_amplify import load_rectify_amplify_config
from stnbeta.snn_brian2.runner import (
    StandaloneDurationBucketProject,
    aggregate_beta_evidence,
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


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
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


def _condition_suffix(description: str) -> str | None:
    lowered = description.lower()
    if lowered == "rest":
        return "Rest"
    if lowered.startswith("hold"):
        return "Hold"
    if lowered.startswith("move"):
        return "Move"
    return None


def _rest_mask_from_fif(fif_path: Path) -> np.ndarray:
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    mask = np.zeros(len(raw.times), dtype=bool)
    for description in sorted({ann["description"] for ann in raw.annotations}):
        if _condition_suffix(description) == "Rest":
            mask |= get_epoch_mask(raw, description)
    return mask


def _available_channels(bursts_root: Path, subject_id: str, band_mode: str, condition: str) -> list[str]:
    band_dir = bursts_root / subject_id / band_mode
    prefix = f"{condition}_"
    return sorted(path.stem.replace(prefix, "", 1) for path in band_dir.glob(f"{condition}_*.parquet"))


def _load_cases(
    *,
    subject_id: str,
    conditions: list[str],
    band_mode: str,
    extracted_root: Path,
    bursts_root: Path,
    channels: list[str] | None = None,
) -> list[tuple[object, np.ndarray]]:
    rows = []
    channel_filter = None if not channels else set(channels)
    for condition in conditions:
        available = _available_channels(bursts_root, subject_id, band_mode, condition)
        selected = [channel for channel in available if channel_filter is None or channel in channel_filter]
        for channel in selected:
            try:
                case = load_real_condition_case(
                    subject_id=subject_id,
                    condition=condition,
                    channel=channel,
                    extracted_root=extracted_root,
                    bursts_root=bursts_root,
                    band_mode=band_mode,
                )
            except ValueError:
                continue
            rest_mask = _rest_mask_from_fif(case.fif_path)
            rows.append((case, rest_mask))
    return rows


def _score_metrics(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    summary = summarize_real_metrics(df)
    score = (
        4.0 * summary["auc_mean"]
        - 0.015 * abs(summary["median_latency_ms"])
        - 0.20 * summary["false_positive_per_min_mean"]
        + 0.50 * summary["readout_separation_mean"]
    )
    return float(score), summary


def _hemi_from_channel(channel: str) -> str:
    lowered = channel.lower()
    if "left" in lowered:
        return "left"
    if "right" in lowered:
        return "right"
    return "unknown"


def _dilate_mask(mask: np.ndarray, half_window_samples: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if half_window_samples <= 0:
        return values
    window = np.ones(2 * half_window_samples + 1, dtype=np.int32)
    hits = np.convolve(values.astype(np.int32), window, mode="same")
    return hits > 0


def _entry_band_names(
    band_names: list[str],
    band_roles: list[str],
    mode: str,
) -> list[str]:
    mode_normalized = mode.lower().strip()
    if mode_normalized in {"raw", "none"}:
        return list(band_names)
    if not any(role == "beta" for role in band_roles):
        return list(band_names)
    non_beta_names = [name for name, role in zip(band_names, band_roles, strict=False) if role != "beta"]
    return [f"pooled_beta_{mode_normalized}"] + non_beta_names


def _plot_overlay(
    case,
    currents: np.ndarray,
    result,
    readout_summary,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
    mean_beta = currents[[index for index, role in enumerate(result.band_roles) if role == "beta"]].mean(axis=0)
    axes[0].plot(np.arange(len(case.signal)) / case.sfreq_hz, case.signal, color="black", lw=0.6)
    axes[0].set_ylabel("LFP")
    axes[0].set_title(f"{case.subject_id} {case.condition} {case.channel}")

    axes[1].plot(np.arange(len(mean_beta)) / case.sfreq_hz, mean_beta, color="#1b9e77", lw=0.9)
    axes[1].set_ylabel("Beta Curr.")

    axes[2].scatter(result.encoder_spike_times_s, result.encoder_spike_indices, s=2, color="#7570b3")
    axes[2].set_ylabel("Enc.")

    for state_index, state_name in enumerate(result.state_names):
        axes[3].plot(np.arange(result.occupancy.shape[1]) / case.sfreq_hz, result.occupancy[state_index] + 0.025 * state_index, lw=0.9, label=state_name)
    axes[3].legend(loc="upper right", fontsize=7, ncols=3)
    axes[3].set_ylabel("States")

    time_s = np.arange(len(readout_summary.score)) / case.sfreq_hz
    axes[4].plot(time_s, readout_summary.score, color="#e7298a", lw=1.0, label="readout score")
    axes[4].fill_between(time_s, 0.0, readout_summary.score, where=readout_summary.stable_mask, color="#66a61e", alpha=0.18, label="stable readout")
    axes[4].fill_between(time_s, 0.0, np.full_like(time_s, readout_summary.score.max(initial=0.0)), where=case.burst_mask.astype(bool), color="#d95f02", alpha=0.10, label="Phase 3 burst")
    axes[4].set_ylabel("Readout")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_latency_distribution(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = metrics_df["median_latency_ms"].dropna().to_numpy(dtype=float)
    if len(values):
        ax.hist(values, bins=min(12, len(values)), color="#1b9e77", alpha=0.85)
    ax.axvline(0.0, color="black", lw=1.0, linestyle="--")
    ax.set_xlabel("Median onset lead/lag (ms)")
    ax.set_ylabel("Channel count")
    ax.set_title("Phase 5 onset lead/lag distribution")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _event_window(
    time_s: float,
    *,
    total_duration_s: float,
    pre_s: float = 0.30,
    post_s: float = 0.50,
) -> tuple[float, float]:
    start_s = max(0.0, time_s - pre_s)
    stop_s = min(total_duration_s, time_s + post_s)
    return start_s, stop_s


def _plot_timing_alignment(
    case,
    currents: np.ndarray,
    causal_currents: np.ndarray,
    result,
    readout_summary,
    latency_df: pd.DataFrame,
    *,
    aggregation_mode: str,
    out_path: Path,
) -> None:
    if latency_df.empty:
        return
    preferred = latency_df[latency_df["kind"] == "true_positive"]
    event_row = preferred.iloc[0] if not preferred.empty else latency_df.iloc[0]
    anchor_s = (
        float(event_row["phase3_onset_s"])
        if pd.notna(event_row["phase3_onset_s"])
        else float(event_row["stable_readout_onset_s"])
    )
    start_s, stop_s = _event_window(anchor_s, total_duration_s=len(case.signal) / case.sfreq_hz)
    start = int(round(start_s * case.sfreq_hz))
    stop = int(round(stop_s * case.sfreq_hz))
    time_s = np.arange(start, stop) / case.sfreq_hz

    pooled = aggregate_beta_evidence(currents, result.band_roles, mode=aggregation_mode)
    causal_pooled = aggregate_beta_evidence(causal_currents, result.band_roles, mode=aggregation_mode)

    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(time_s, case.signal[start:stop], color="black", lw=0.7)
    axes[0].set_ylabel("LFP")
    axes[0].set_title(f"{case.subject_id} {case.condition} {case.channel} timing alignment")

    axes[1].plot(time_s, pooled[start:stop], color="#1b9e77", lw=1.0, label="frozen pooled evidence")
    axes[1].plot(time_s, causal_pooled[start:stop], color="#1f78b4", lw=1.0, alpha=0.9, label="causalized proxy")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Evidence")

    beta_mask = np.array([role == "beta" for role in result.band_roles], dtype=bool)
    beta_spike_mask = beta_mask[np.asarray(result.encoder_spike_indices, dtype=int)]
    axes[2].scatter(
        result.encoder_spike_times_s[beta_spike_mask],
        result.encoder_spike_indices[beta_spike_mask],
        s=4,
        color="#7570b3",
    )
    axes[2].set_ylabel("Enc.")

    for state_name in ("D0", "D1", "D2", "D3", "D4"):
        state_index = result.state_names.index(state_name)
        axes[3].plot(time_s, result.occupancy[state_index, start:stop], lw=0.9, label=state_name)
    axes[3].legend(loc="upper right", fontsize=7, ncols=3)
    axes[3].set_ylabel("States")

    axes[4].plot(time_s, readout_summary.score[start:stop], color="#e7298a", lw=1.0, label="readout score")
    axes[4].fill_between(
        time_s,
        0.0,
        readout_summary.score[start:stop],
        where=readout_summary.stable_mask[start:stop],
        color="#66a61e",
        alpha=0.18,
        label="stable readout",
    )
    axes[4].fill_between(
        time_s,
        0.0,
        np.full(stop - start, readout_summary.score[start:stop].max(initial=0.0)),
        where=case.burst_mask[start:stop].astype(bool),
        color="#d95f02",
        alpha=0.10,
        label="Phase 3 burst",
    )
    for marker_name, color in [
        ("phase3_onset_s", "#d95f02"),
        ("causalized_aux_onset_s", "#1f78b4"),
        ("phase4_encoder_onset_s", "#7570b3"),
        ("D0_onset_s", "#66a61e"),
        ("D2_onset_s", "#e6ab02"),
        ("stable_readout_onset_s", "#e7298a"),
    ]:
        value = event_row.get(marker_name)
        if pd.notna(value):
            axes[4].axvline(float(value), color=color, lw=1.0, linestyle="--")
    axes[4].set_ylabel("Readout")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_example_panels(
    case,
    currents: np.ndarray,
    causal_currents: np.ndarray,
    result,
    readout_summary,
    latency_df: pd.DataFrame,
    *,
    aggregation_mode: str,
    out_path: Path,
) -> None:
    if latency_df.empty:
        return
    pooled = aggregate_beta_evidence(currents, result.band_roles, mode=aggregation_mode)
    causal_pooled = aggregate_beta_evidence(causal_currents, result.band_roles, mode=aggregation_mode)
    rows = []
    for kind in ("true_positive", "miss", "false_positive"):
        frame = latency_df[latency_df["kind"] == kind]
        if not frame.empty:
            rows.append((kind, frame.iloc[0]))
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.2 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = np.asarray([axes])
    total_duration_s = len(case.signal) / case.sfreq_hz

    for row_index, (kind, event_row) in enumerate(rows):
        anchor = (
            float(event_row["phase3_onset_s"])
            if pd.notna(event_row["phase3_onset_s"])
            else float(event_row["stable_readout_onset_s"])
        )
        start_s, stop_s = _event_window(anchor, total_duration_s=total_duration_s)
        start = int(round(start_s * case.sfreq_hz))
        stop = int(round(stop_s * case.sfreq_hz))
        time_s = np.arange(start, stop) / case.sfreq_hz

        axes[row_index, 0].plot(time_s, case.signal[start:stop], color="black", lw=0.7)
        axes[row_index, 0].set_ylabel(kind.replace("_", " "))

        axes[row_index, 1].plot(time_s, pooled[start:stop], color="#1b9e77", lw=1.0)
        axes[row_index, 1].plot(time_s, causal_pooled[start:stop], color="#1f78b4", lw=1.0, alpha=0.9)

        for state_name in ("D0", "D1", "D2"):
            state_index = result.state_names.index(state_name)
            axes[row_index, 2].plot(time_s, result.occupancy[state_index, start:stop], lw=1.0, label=state_name)
        axes[row_index, 2].plot(time_s, readout_summary.score[start:stop], color="#e7298a", lw=1.0, alpha=0.9)
        axes[row_index, 2].fill_between(
            time_s,
            0.0,
            readout_summary.score[start:stop],
            where=readout_summary.stable_mask[start:stop],
            color="#66a61e",
            alpha=0.15,
        )

        for col in range(3):
            if pd.notna(event_row["phase3_onset_s"]):
                axes[row_index, col].axvline(float(event_row["phase3_onset_s"]), color="#d95f02", linestyle="--", lw=1.0)
            if pd.notna(event_row["causalized_aux_onset_s"]):
                axes[row_index, col].axvline(float(event_row["causalized_aux_onset_s"]), color="#1f78b4", linestyle="--", lw=1.0)
            if pd.notna(event_row["stable_readout_onset_s"]):
                axes[row_index, col].axvline(float(event_row["stable_readout_onset_s"]), color="#e7298a", linestyle="--", lw=1.0)

    axes[0, 0].set_title("LFP")
    axes[0, 1].set_title("Frozen vs causalized evidence")
    axes[0, 2].set_title("States + readout")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    axes[0, 2].legend(loc="upper right", fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted", type=Path, default=Path("extracted"))
    parser.add_argument("--bursts", type=Path, default=Path("results/bursts"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--subject", default="sub-0cGdk9")
    parser.add_argument("--conditions", default="MedOff_Hold")
    parser.add_argument("--band-mode", default="fixed_13_30")
    parser.add_argument("--filter-bank-config", type=Path, default=Path("configs/filter_bank.yaml"))
    parser.add_argument("--lif-config", type=Path, default=Path("configs/lif_encoder.yaml"))
    parser.add_argument("--nsm-config", type=Path, default=Path("configs/nsm_mono.yaml"))
    parser.add_argument("--grid-config", type=Path, default=Path("configs/gridsearch_phase5.yaml"))
    parser.add_argument("--grid-section", default="real_dev")
    parser.add_argument("--grid-indices", default=None)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--architectures", default="per_channel")
    parser.add_argument("--consensus-k", default="1,2")
    parser.add_argument("--consensus-window-ms", type=float, default=25.0)
    parser.add_argument("--worker-label", default=None)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--backend", choices=["runtime", "cpp_standalone", "cuda_standalone"], default="runtime")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-capability", type=float, default=None)
    parser.add_argument("--cuda-runtime-version", type=float, default=None)
    args = parser.parse_args(argv)

    if args.backend == "cuda_standalone":
        ensure_cuda_runtime_libraries()

    out_dir = args.out / "phase5_real_dev"
    worker_dir = out_dir if args.worker_label is None else out_dir / "workers" / args.worker_label
    worker_dir.mkdir(parents=True, exist_ok=True)

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    cases = _load_cases(
        subject_id=args.subject,
        conditions=conditions,
        band_mode=args.band_mode,
        extracted_root=args.extracted,
        bursts_root=args.bursts,
        channels=_parse_csv(args.channels),
    )
    if not cases:
        raise RuntimeError("No single-file Phase 5 dev cases were found for the requested subject/conditions")

    filter_bank_config = load_filter_bank_config(args.filter_bank_config)
    band_names = [band.name for band in filter_bank_config.bands]
    band_roles = [band.role for band in filter_bank_config.bands]
    rectify_config = load_rectify_amplify_config(args.lif_config)
    lif_config = load_lif_encoder_config(args.lif_config)
    base_config = _phase5_with_lif_defaults(
        load_duration_bucket_cluster_config(args.nsm_config),
        lif_config=lif_config,
    )
    grid_cfg = {} if args.no_grid else load_yaml(args.grid_config)
    grid_points = expand_grid_points(grid_cfg, args.grid_section)
    indexed_points = filter_grid_points(grid_points, _parse_grid_indices(args.grid_indices))
    rebuild_axes = dict(grid_cfg.get("rebuild_axes", {}))
    architectures = set(_parse_csv(args.architectures) or ["per_channel"])
    consensus_ks = _parse_int_csv(args.consensus_k) or [1, 2]

    preprocessed = []
    for case, rest_mask in cases:
        filtered = apply_filter_bank(case.signal, case.sfreq_hz, filter_bank_config)
        currents = currents_from_filtered_bands(
            filtered,
            band_names,
            rectify_config=rectify_config,
            sfreq_hz=case.sfreq_hz,
        )
        causal_filtered = apply_filter_bank(case.signal, case.sfreq_hz, filter_bank_config, causal=True)
        causal_currents = currents_from_filtered_bands(
            causal_filtered,
            band_names,
            rectify_config=rectify_config,
            sfreq_hz=case.sfreq_hz,
            causal=True,
        )
        preprocessed.append(
            {
                "case": case,
                "rest_mask": rest_mask,
                "currents": currents,
                "causal_currents": causal_currents,
                "hemi": _hemi_from_channel(case.channel),
            }
        )

    active_project_key: tuple[tuple[tuple[str, object], ...], int, int] | None = None
    active_project: StandaloneDurationBucketProject | None = None
    project_counter = 0
    summary_rows = []
    best_score = -np.inf
    best_metrics_df = None
    best_occupancy_df = None
    best_band_df = None
    best_event_df = None
    best_latency_df = None
    best_latency_summary_df = None
    best_architecture_df = None
    best_run_config = None
    best_bundle = None

    for grid_index, overrides in indexed_points:
        build_overrides, run_overrides = split_rebuild_overrides(overrides, rebuild_axes)
        build_config = replace(base_config, **build_overrides)
        run_config = replace(build_config, **run_overrides)
        print(
            f"[phase5-dev] grid_index={grid_index} subject={args.subject} conditions={','.join(conditions)} "
            f"channels={len(preprocessed)} backend={args.backend} overrides={dict(overrides)}",
            flush=True,
        )

        metrics_rows = []
        occupancy_frames = []
        band_frames = []
        event_frames = []
        latency_frames = []
        case_bundles = []
        progress_dir = worker_dir / "progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        for item in preprocessed:
            case = item["case"]
            rest_mask = item["rest_mask"]
            currents = item["currents"]
            causal_currents = item["causal_currents"]
            model_currents, model_band_roles = prepare_phase5_entry_currents(
                currents,
                band_roles,
                mode=run_config.evidence_aggregation_mode,
            )
            causal_model_currents, _ = prepare_phase5_entry_currents(
                causal_currents,
                band_roles,
                mode=run_config.evidence_aggregation_mode,
            )
            model_band_names = _entry_band_names(
                band_names,
                band_roles,
                run_config.evidence_aggregation_mode,
            )
            print(
                f"[phase5-dev] case_start grid_index={grid_index} "
                f"condition={case.condition} channel={case.channel} n_samples={len(case.signal)}",
                flush=True,
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
                    results_directory=worker_dir / "runs" / f"grid_{grid_index}_{case.condition}_{case.channel}",
                    overrides=run_overrides,
                )
            readout_summary = build_readout_summary(result, run_config)
            metrics_row = evaluate_real_case(case, result, run_config, readout_summary, rest_mask=rest_mask)
            metrics_row["architecture"] = "per_channel"
            metrics_row["hemi"] = item["hemi"]
            metrics_row["n_channels"] = 1
            metrics_rows.append(metrics_row)
            occupancy_frames.append(state_occupancy_table(case, result, rest_mask=rest_mask))
            band_frames.append(
                band_contribution_table(
                    case,
                    result,
                    model_band_names,
                    list(model_band_roles),
                    readout_summary,
                    currents=model_currents,
                )
            )
            readout_events = events_from_mask(readout_summary.stable_mask & case.task_mask.astype(bool), dt_ms=run_config.dt_ms)
            event_df = match_readout_events_to_bursts(case.events.loc[:, ["onset_s", "offset_s"]], readout_events)
            event_df["subject_id"] = case.subject_id
            event_df["condition"] = case.condition
            event_df["channel"] = case.channel
            event_df["architecture"] = "per_channel"
            event_frames.append(event_df)
            latency_df = latency_decomposition_table(
                case,
                result,
                run_config,
                readout_summary,
                evidence_trace=aggregate_beta_evidence(currents, band_roles, mode=run_config.evidence_aggregation_mode),
                causal_evidence_trace=aggregate_beta_evidence(
                    causal_currents,
                    band_roles,
                    mode=run_config.evidence_aggregation_mode,
                ),
            )
            latency_df["architecture"] = "per_channel"
            latency_frames.append(latency_df)
            case_bundles.append((case, model_currents, causal_model_currents, result, readout_summary, latency_df))

            pd.DataFrame(metrics_rows).to_csv(
                progress_dir / f"dev_case_metrics_grid_{grid_index}.partial.tsv",
                sep="\t",
                index=False,
            )
            pd.concat(occupancy_frames, ignore_index=True).to_csv(
                progress_dir / f"state_occupancy_grid_{grid_index}.partial.tsv",
                sep="\t",
                index=False,
            )
            pd.concat(band_frames, ignore_index=True).to_csv(
                progress_dir / f"band_contributions_grid_{grid_index}.partial.tsv",
                sep="\t",
                index=False,
            )
            pd.concat(event_frames, ignore_index=True).to_csv(
                progress_dir / f"event_examples_grid_{grid_index}.partial.tsv",
                sep="\t",
                index=False,
            )
            pd.concat(latency_frames, ignore_index=True).to_csv(
                progress_dir / f"latency_decomposition_grid_{grid_index}.partial.tsv",
                sep="\t",
                index=False,
            )
            print(
                f"[phase5-dev] case_done grid_index={grid_index} condition={case.condition} "
                f"channel={case.channel} partial_cases={len(metrics_rows)}",
                flush=True,
            )

        metrics_df = pd.DataFrame(metrics_rows)
        occupancy_df = pd.concat(occupancy_frames, ignore_index=True)
        band_df = pd.concat(band_frames, ignore_index=True)
        event_df = pd.concat(event_frames, ignore_index=True)
        latency_df = pd.concat(latency_frames, ignore_index=True)
        latency_summary_df = summarize_latency_decomposition(latency_df)
        architecture_rows = []

        grouped_runs: dict[tuple[str, str], list[dict[str, object]]] = {}
        for item, bundle in zip(preprocessed, case_bundles, strict=False):
            grouped_runs.setdefault((item["case"].condition, item["hemi"]), []).append(
                {
                    "case": item["case"],
                    "rest_mask": item["rest_mask"],
                    "currents": item["currents"],
                    "causal_currents": item["causal_currents"],
                    "result": bundle[3],
                    "readout_summary": bundle[4],
                }
            )

        if len(preprocessed) > 1 and (architectures & {"consensus", "pooled_entry"}):
            consensus_window = max(0, int(round(args.consensus_window_ms / run_config.dt_ms)))
            for (condition, hemi), group_runs in grouped_runs.items():
                if not group_runs:
                    continue
                union_events = merge_event_tables(
                    [item["case"].events.loc[:, ["onset_s", "offset_s"]] for item in group_runs]
                )
                union_burst_mask = np.vstack([item["case"].burst_mask for item in group_runs]).any(axis=0)
                union_task_mask = np.vstack([item["case"].task_mask for item in group_runs]).any(axis=0)
                rest_mask = np.asarray(group_runs[0]["rest_mask"], dtype=bool)
                channel_label = f"{hemi}_all_channels"

                if "consensus" in architectures:
                    dilated_masks = [
                        _dilate_mask(item["readout_summary"].stable_mask, consensus_window)
                        for item in group_runs
                    ]
                    count_trace = np.vstack(dilated_masks).sum(axis=0).astype(np.float32)
                    for k in consensus_ks:
                        if k <= 0 or k > len(group_runs):
                            continue
                        architecture_rows.append(
                            {
                                **evaluate_readout_against_reference(
                                    subject_id=args.subject,
                                    condition=condition,
                                    channel=f"{channel_label}_consensus_k{k}",
                                    band_mode=args.band_mode,
                                    burst_mask=union_burst_mask,
                                    task_mask=union_task_mask,
                                    burst_events=union_events,
                                    score=count_trace / len(group_runs),
                                    stable_mask=count_trace >= k,
                                    sfreq_hz=group_runs[0]["case"].sfreq_hz,
                                    dt_ms=run_config.dt_ms,
                                    rest_mask=rest_mask,
                                ),
                                "architecture": f"consensus_k{k}",
                                "hemi": hemi,
                                "n_channels": len(group_runs),
                            }
                        )

                if "pooled_entry" in architectures:
                    pooled_raw_currents = np.concatenate([item["currents"] for item in group_runs], axis=0)
                    pooled_raw_band_roles = tuple(band_roles) * len(group_runs)
                    pooled_currents, pooled_band_roles = prepare_phase5_entry_currents(
                        pooled_raw_currents,
                        pooled_raw_band_roles,
                        mode=run_config.evidence_aggregation_mode,
                    )
                    quiet_drive = derive_quiet_drive(pooled_currents, pooled_band_roles)
                    if args.backend == "runtime":
                        pooled_result = run_duration_bucket_state_machine(
                            pooled_currents,
                            pooled_band_roles,
                            run_config,
                            backend="runtime",
                            quiet_drive=quiet_drive,
                            seed=args.seed,
                        )
                    else:
                        pooled_key = (
                            tuple(sorted(build_overrides.items())),
                            pooled_currents.shape[0],
                            pooled_currents.shape[-1],
                        )
                        if pooled_key != active_project_key:
                            active_project = StandaloneDurationBucketProject(
                                n_steps=pooled_currents.shape[-1],
                                n_inputs=pooled_currents.shape[0],
                                band_roles=pooled_band_roles,
                                config=build_config,
                                backend=args.backend,
                                build_dir=worker_dir / "standalone_build" / f"group_{project_counter:02d}",
                                seed=args.seed,
                                compute_capability=args.compute_capability,
                                cuda_runtime_version=args.cuda_runtime_version,
                            )
                            active_project_key = pooled_key
                            project_counter += 1
                        if active_project is None:
                            raise RuntimeError("Standalone project was not initialized")
                        pooled_result = active_project.run(
                            pooled_currents,
                            quiet_drive=quiet_drive,
                            results_directory=worker_dir / "runs" / f"grid_{grid_index}_{condition}_{hemi}_pooled_entry",
                            overrides=run_overrides,
                        )
                    pooled_readout = build_readout_summary(pooled_result, run_config)
                    architecture_rows.append(
                        {
                            **evaluate_readout_against_reference(
                                subject_id=args.subject,
                                condition=condition,
                                channel=f"{channel_label}_pooled_entry",
                                band_mode=args.band_mode,
                                burst_mask=union_burst_mask,
                                task_mask=union_task_mask,
                                burst_events=union_events,
                                score=pooled_readout.score,
                                stable_mask=pooled_readout.stable_mask,
                                sfreq_hz=group_runs[0]["case"].sfreq_hz,
                                dt_ms=run_config.dt_ms,
                                rest_mask=rest_mask,
                            ),
                            "architecture": "pooled_entry",
                            "hemi": hemi,
                            "n_channels": len(group_runs),
                        }
                    )

        architecture_df = pd.DataFrame(architecture_rows)
        metrics_df.to_csv(worker_dir / f"dev_case_metrics_grid_{grid_index}.tsv", sep="\t", index=False)
        occupancy_df.to_csv(worker_dir / f"state_occupancy_grid_{grid_index}.tsv", sep="\t", index=False)
        band_df.to_csv(worker_dir / f"band_contributions_grid_{grid_index}.tsv", sep="\t", index=False)
        event_df.to_csv(worker_dir / f"event_examples_grid_{grid_index}.tsv", sep="\t", index=False)
        latency_df.to_csv(worker_dir / f"latency_decomposition_grid_{grid_index}.tsv", sep="\t", index=False)
        latency_summary_df.to_csv(worker_dir / f"latency_decomposition_summary_grid_{grid_index}.tsv", sep="\t", index=False)
        if not architecture_df.empty:
            architecture_df.to_csv(worker_dir / f"architecture_metrics_grid_{grid_index}.tsv", sep="\t", index=False)

        score, extras = _score_metrics(metrics_df)
        summary_row = {"grid_index": grid_index, "score": score, **extras, **overrides}
        if not architecture_df.empty:
            best_arch = architecture_df.sort_values(["auc", "false_positive_per_min"], ascending=[False, True]).iloc[0]
            summary_row.update(
                {
                    "best_architecture": str(best_arch["architecture"]),
                    "best_architecture_channel": str(best_arch["channel"]),
                    "best_architecture_auc": float(best_arch["auc"]),
                    "best_architecture_false_positive_per_min": float(best_arch["false_positive_per_min"]),
                }
            )
        summary_rows.append(summary_row)
        if score > best_score:
            best_score = score
            best_metrics_df = metrics_df
            best_occupancy_df = occupancy_df
            best_band_df = band_df
            best_event_df = event_df
            best_latency_df = latency_df
            best_latency_summary_df = latency_summary_df
            best_architecture_df = architecture_df
            best_run_config = run_config
            best_bundle = max(
                case_bundles,
                key=lambda bundle: float(
                    metrics_df.loc[
                        metrics_df["channel"] == bundle[0].channel,
                        "auc",
                    ].max()
                ),
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    summary_df.to_csv(worker_dir / "grid_summary.tsv", sep="\t", index=False)
    print(
        f"[phase5-dev] complete subject={args.subject} backend={args.backend} "
        f"grid_points={len(summary_df)} best_score={best_score:.4f}",
        flush=True,
    )

    if args.worker_label is None and best_metrics_df is not None and best_bundle is not None:
        figures_dir = Path("results/figures/05_phase5")
        tables_dir = Path("results/tables/05_phase5")
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(tables_dir / "real_dev_metrics_summary.tsv", sep="\t", index=False)
        best_metrics_df.to_csv(out_dir / "best_case_metrics.tsv", sep="\t", index=False)
        best_occupancy_df.to_csv(out_dir / "state_occupancy_summary.tsv", sep="\t", index=False)
        best_band_df.to_csv(out_dir / "band_contributions.tsv", sep="\t", index=False)
        best_event_df.to_csv(out_dir / "event_examples.tsv", sep="\t", index=False)
        if best_latency_df is not None:
            best_latency_df.to_csv(out_dir / "latency_decomposition.tsv", sep="\t", index=False)
            best_latency_summary_df.to_csv(out_dir / "latency_decomposition_summary.tsv", sep="\t", index=False)
            best_latency_summary_df.to_csv(tables_dir / "latency_decomposition_summary.tsv", sep="\t", index=False)
        if best_architecture_df is not None and not best_architecture_df.empty:
            best_architecture_df.to_csv(out_dir / "architecture_metrics.tsv", sep="\t", index=False)
            best_architecture_df.to_csv(tables_dir / "real_dev_architecture_metrics.tsv", sep="\t", index=False)
        _plot_overlay(best_bundle[0], best_bundle[1], best_bundle[3], best_bundle[4], figures_dir / "real_dev_overlay.png")
        _plot_latency_distribution(best_metrics_df, figures_dir / "latency_distribution.png")
        if best_latency_df is not None:
            _plot_timing_alignment(
                best_bundle[0],
                best_bundle[1],
                best_bundle[2],
                best_bundle[3],
                best_bundle[4],
                best_bundle[5],
                aggregation_mode=best_run_config.evidence_aggregation_mode,
                out_path=figures_dir / "timing_alignment.png",
            )
            _plot_example_panels(
                best_bundle[0],
                best_bundle[1],
                best_bundle[2],
                best_bundle[3],
                best_bundle[4],
                best_bundle[5],
                aggregation_mode=best_run_config.evidence_aggregation_mode,
                out_path=figures_dir / "dev_event_examples.png",
            )

    manifest = collect_runtime_manifest(
        backend=args.backend,
        config_hash_value=config_hash(
            {
                "filter_bank_config": load_yaml(args.filter_bank_config),
                "lif_config": load_yaml(args.lif_config),
                "nsm_config": load_yaml(args.nsm_config),
                "grid_config": grid_cfg,
                "subject": args.subject,
                "conditions": conditions,
                "channels": _parse_csv(args.channels),
                "band_mode": args.band_mode,
            }
        ),
        seed=args.seed,
        extra={
            "subject": args.subject,
            "conditions": conditions,
            "band_mode": args.band_mode,
            "channels": _parse_csv(args.channels),
            "architectures": sorted(architectures),
            "consensus_k": consensus_ks,
            "consensus_window_ms": args.consensus_window_ms,
            "best_score": best_score,
            "grid_indices": [index for index, _ in indexed_points],
            "worker_label": args.worker_label,
        },
    )
    write_manifest(worker_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

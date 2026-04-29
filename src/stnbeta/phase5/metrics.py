"""Metrics and summaries for Phase 5 synthetic and real-data validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .readout import events_from_mask
from stnbeta.snn_brian2.runner import aggregate_beta_evidence


def _first_true_time(mask: np.ndarray, *, start_s: float, stop_s: float, dt_ms: float) -> float | None:
    start = max(0, int(round(start_s * 1000.0 / dt_ms)))
    stop = min(len(mask), int(round(stop_s * 1000.0 / dt_ms)))
    if stop <= start:
        return None
    hits = np.flatnonzero(mask[start:stop])
    if len(hits) == 0:
        return None
    return (start + int(hits[0])) * dt_ms / 1000.0


def _any_true(mask: np.ndarray, *, start_s: float, stop_s: float, dt_ms: float) -> bool:
    start = max(0, int(round(start_s * 1000.0 / dt_ms)))
    stop = min(len(mask), int(round(stop_s * 1000.0 / dt_ms)))
    if stop <= start:
        return False
    return bool(np.asarray(mask[start:stop], dtype=bool).any())


def _first_false_time(mask: np.ndarray, *, start_s: float, stop_s: float, dt_ms: float) -> float | None:
    start = max(0, int(round(start_s * 1000.0 / dt_ms)))
    stop = min(len(mask), int(round(stop_s * 1000.0 / dt_ms)))
    if stop <= start:
        return None
    hits = np.flatnonzero(~np.asarray(mask[start:stop], dtype=bool))
    if len(hits) == 0:
        return None
    return (start + int(hits[0])) * dt_ms / 1000.0


def _first_spike_time(spike_times_s: np.ndarray, *, start_s: float, stop_s: float) -> float | None:
    values = np.asarray(spike_times_s, dtype=float)
    mask = (values >= start_s) & (values <= stop_s)
    if not mask.any():
        return None
    return float(values[mask][0])


def _ms_diff(lhs: float | None, rhs: float | None) -> float:
    if lhs is None or rhs is None:
        return np.nan
    return float((lhs - rhs) * 1000.0)


def _run_bounds(mask: np.ndarray, *, anchor_s: float, dt_ms: float) -> tuple[float | None, float | None, float]:
    if anchor_s is None:
        return None, None, 0.0
    values = np.asarray(mask, dtype=bool)
    if not len(values):
        return None, None, 0.0
    anchor = int(np.clip(round(anchor_s * 1000.0 / dt_ms), 0, len(values) - 1))
    if not values[anchor]:
        return None, None, 0.0
    start = anchor
    stop = anchor + 1
    while start > 0 and values[start - 1]:
        start -= 1
    while stop < len(values) and values[stop]:
        stop += 1
    onset_s = start * dt_ms / 1000.0
    offset_s = stop * dt_ms / 1000.0
    return onset_s, offset_s, float((stop - start) * dt_ms)


def _first_true_or_spike_time(
    mask: np.ndarray,
    *,
    start_s: float,
    stop_s: float,
    dt_ms: float,
) -> float | None:
    return _first_true_time(mask, start_s=start_s, stop_s=stop_s, dt_ms=dt_ms)


def _half_height_onset(
    trace: np.ndarray | None,
    *,
    start_s: float,
    stop_s: float,
    dt_ms: float,
    floor: float = 0.15,
) -> float | None:
    if trace is None:
        return None
    values = np.asarray(trace, dtype=np.float32)
    start = max(0, int(round(start_s * 1000.0 / dt_ms)))
    stop = min(len(values), int(round(stop_s * 1000.0 / dt_ms)))
    if stop <= start:
        return None
    window = values[start:stop]
    peak = float(window.max(initial=0.0))
    if peak <= 0.0:
        return None
    threshold = max(floor, 0.5 * peak)
    hits = np.flatnonzero(window >= threshold)
    if len(hits) == 0:
        return None
    return (start + int(hits[0])) * dt_ms / 1000.0


def evaluate_synthetic_case(
    case: Any,
    result: Any,
    config: Any,
    readout_summary: Any,
    *,
    reset_margin_s: float = 0.05,
) -> dict[str, float | int | str]:
    """Evaluate one Phase 5 synthetic case."""
    annotations = case.annotations
    bucket_names = tuple(result.state_names[1:])
    state_masks = readout_summary.state_masks
    bucket_masks = [state_masks[name] for name in bucket_names]
    non_idle_mask = np.vstack(bucket_masks).any(axis=0) if bucket_masks else np.zeros_like(readout_summary.stable_mask)
    readout_events = pd.DataFrame(
        {
            "onset_s": readout_summary.event_onsets_s,
            "offset_s": readout_summary.event_offsets_s,
        }
    )

    if annotations.empty:
        false_positive_rate_hz = float(len(readout_events) / max(result.duration_s, 1e-9))
        return {
            "trace_name": case.name,
            "level": case.level,
            "n_ground_truth_bursts": 0,
            "monotonic_progression_correct": 1,
            "skipped_bucket_transitions": 0,
            "reset_success_count": 1,
            "reset_total": 1,
            "expected_readout_total": 0,
            "correct_readout_count": 0,
            "missed_readout_count": 0,
            "too_short_total": 0,
            "too_short_rejection_success_count": 0,
            "unexpected_readout_count": int(len(readout_events)),
            "mean_detection_latency_ms": np.nan,
            "false_positive_rate_hz": false_positive_rate_hz,
            "max_bucket_reached": -1,
            "interrupt_behavior_ok": 1,
        }

    monotonic = 0
    skipped = 0
    reset_success = 0
    expected_readout_total = 0
    correct_readout = 0
    missed_readout = 0
    too_short_total = 0
    too_short_rejection_success = 0
    unexpected_readout = 0
    detection_latencies_ms: list[float] = []
    max_bucket_reached = -1

    for burst_index, row in annotations.iterrows():
        onset_s = float(row["onset_s"])
        offset_s = float(row["offset_s"])
        expected_bucket = int(row["expected_bucket_index"])
        next_onset = (
            float(annotations.iloc[burst_index + 1]["onset_s"])
            if burst_index + 1 < len(annotations)
            else result.duration_s
        )

        state_first_times = [
            _first_true_time(mask, start_s=onset_s, stop_s=offset_s + 0.25, dt_ms=config.dt_ms)
            for mask in bucket_masks
        ]
        reached = [index for index, value in enumerate(state_first_times) if value is not None]
        if reached:
            ordered = [state_first_times[index] for index in reached]
            monotonic += int(all(a <= b for a, b in zip(ordered, ordered[1:])))
            highest = max(reached)
            skipped += max(0, highest + 1 - len(reached))
            max_bucket_reached = max(max_bucket_reached, highest)
        else:
            skipped += 0 if not bool(row["expect_readout"]) else expected_bucket + 1

        detected_time = _first_true_time(
            readout_summary.stable_mask,
            start_s=onset_s - 0.1,
            stop_s=offset_s + 0.35,
            dt_ms=config.dt_ms,
        )
        if bool(row["expect_readout"]):
            expected_readout_total += 1
            correct_readout += int(detected_time is not None)
            missed_readout += int(detected_time is None)
            if detected_time is not None:
                detection_latencies_ms.append((detected_time - onset_s) * 1000.0)
        else:
            too_short_total += 1
            too_short_rejection_success += int(detected_time is None)
            unexpected_readout += int(detected_time is not None)

        reset_success += int(
            not _any_true(
                non_idle_mask,
                start_s=offset_s + reset_margin_s,
                stop_s=next_onset,
                dt_ms=config.dt_ms,
            )
        )

    false_positive_events = 0
    for event in readout_events.itertuples(index=False):
        overlaps = ((annotations["onset_s"] <= event.offset_s) & (annotations["offset_s"] >= event.onset_s)).any()
        if not overlaps:
            false_positive_events += 1

    interrupt_behavior_ok = 1
    if "interrupted_burst_60_on_20_off_60_on" in case.name:
        gap_start_s = 0.61
        gap_stop_s = 0.63
        interrupt_behavior_ok = int(
            not _any_true(non_idle_mask, start_s=gap_start_s, stop_s=gap_stop_s, dt_ms=config.dt_ms)
        )

    return {
        "trace_name": case.name,
        "level": case.level,
        "n_ground_truth_bursts": int(len(annotations)),
        "monotonic_progression_correct": monotonic,
        "skipped_bucket_transitions": skipped,
        "reset_success_count": reset_success,
        "reset_total": int(len(annotations)),
        "expected_readout_total": expected_readout_total,
        "correct_readout_count": correct_readout,
        "missed_readout_count": missed_readout,
        "too_short_total": too_short_total,
        "too_short_rejection_success_count": too_short_rejection_success,
        "unexpected_readout_count": unexpected_readout,
        "mean_detection_latency_ms": float(np.nanmean(detection_latencies_ms)) if detection_latencies_ms else np.nan,
        "false_positive_rate_hz": false_positive_events / max(result.duration_s, 1e-9),
        "max_bucket_reached": max_bucket_reached,
        "interrupt_behavior_ok": interrupt_behavior_ok,
    }


def summarize_synthetic_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate core synthetic acceptance metrics across the suite."""
    negative = metrics_df[metrics_df["n_ground_truth_bursts"] == 0]
    positives = metrics_df[metrics_df["n_ground_truth_bursts"] > 0]
    return {
        "short_burst_rejection_rate": float(
            positives["too_short_rejection_success_count"].sum() / max(positives["too_short_total"].sum(), 1)
        ),
        "threshold_and_long_detection_rate": float(
            positives["correct_readout_count"].sum() / max(positives["expected_readout_total"].sum(), 1)
        ),
        "reset_success_rate": float(
            positives["reset_success_count"].sum() / max(positives["reset_total"].sum(), 1)
        ),
        "negative_false_positive_rate_hz": float(negative["false_positive_rate_hz"].mean()) if len(negative) else 0.0,
        "skipped_bucket_transitions": float(metrics_df["skipped_bucket_transitions"].sum()),
        "interrupt_behavior_rate": float(metrics_df["interrupt_behavior_ok"].mean()),
    }


def _auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    pos_ranks = ranks[labels]
    return float((pos_ranks.sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def match_readout_events_to_bursts(
    burst_events: pd.DataFrame,
    readout_events: pd.DataFrame,
) -> pd.DataFrame:
    """Label readout events as TP/FP and unmatched bursts as misses."""
    rows: list[dict[str, float | str | int]] = []
    matched_readout: set[int] = set()
    for burst_index, burst in burst_events.reset_index(drop=True).iterrows():
        overlap_index = None
        for readout_index, readout in readout_events.reset_index(drop=True).iterrows():
            overlaps = float(readout["onset_s"]) <= float(burst["offset_s"]) and float(readout["offset_s"]) >= float(burst["onset_s"])
            if overlaps:
                overlap_index = readout_index
                break
        if overlap_index is None:
            rows.append(
                {
                    "kind": "miss",
                    "burst_index": burst_index,
                    "burst_onset_s": float(burst["onset_s"]),
                    "burst_offset_s": float(burst["offset_s"]),
                    "readout_onset_s": np.nan,
                    "readout_offset_s": np.nan,
                }
            )
            continue
        matched_readout.add(overlap_index)
        readout = readout_events.iloc[overlap_index]
        rows.append(
            {
                "kind": "true_positive",
                "burst_index": burst_index,
                "burst_onset_s": float(burst["onset_s"]),
                "burst_offset_s": float(burst["offset_s"]),
                "readout_onset_s": float(readout["onset_s"]),
                "readout_offset_s": float(readout["offset_s"]),
            }
        )

    for readout_index, readout in readout_events.reset_index(drop=True).iterrows():
        if readout_index in matched_readout:
            continue
        rows.append(
            {
                "kind": "false_positive",
                "burst_index": -1,
                "burst_onset_s": np.nan,
                "burst_offset_s": np.nan,
                "readout_onset_s": float(readout["onset_s"]),
                "readout_offset_s": float(readout["offset_s"]),
            }
        )
    return pd.DataFrame(rows)


def state_occupancy_table(
    case: Any,
    result: Any,
    *,
    rest_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize mean occupancy by state for burst and non-burst windows."""
    rows = []
    burst_mask = np.asarray(case.burst_mask, dtype=bool)
    task_mask = np.asarray(case.task_mask, dtype=bool)
    nonburst_mask = task_mask & ~burst_mask
    rest_values = None if rest_mask is None else np.asarray(rest_mask, dtype=bool)

    for state_index, state_name in enumerate(result.state_names):
        trace = np.asarray(result.occupancy[state_index], dtype=np.float32)
        rows.append(
            {
                "subject_id": case.subject_id,
                "condition": case.condition,
                "channel": case.channel,
                "state_name": state_name,
                "burst_mean_occupancy": float(trace[task_mask & burst_mask].mean()) if (task_mask & burst_mask).any() else np.nan,
                "nonburst_mean_occupancy": float(trace[nonburst_mask].mean()) if nonburst_mask.any() else np.nan,
                "rest_mean_occupancy": float(trace[rest_values].mean()) if rest_values is not None and rest_values.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def band_contribution_table(
    case: Any,
    result: Any,
    band_names: list[str],
    band_roles: list[str],
    readout_summary: Any,
    *,
    currents: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize per-band evidence contributions for one real case."""
    burst_mask = np.asarray(case.burst_mask, dtype=bool)
    task_mask = np.asarray(case.task_mask, dtype=bool)
    nonburst_mask = task_mask & ~burst_mask
    current_values = np.asarray(result.encoder_currents if currents is None else currents, dtype=np.float32)
    rows = []
    for band_index, band_name in enumerate(band_names):
        if band_index >= current_values.shape[0]:
            continue
        trace = np.asarray(current_values[band_index], dtype=np.float32)
        spike_times = np.asarray(result.encoder_spike_times_s[result.encoder_spike_indices == band_index], dtype=float)
        if len(spike_times):
            spike_samples = np.clip(np.round(spike_times * case.sfreq_hz).astype(int), 0, len(task_mask) - 1)
            burst_spike_rate = float((task_mask & burst_mask)[spike_samples].sum() / max((task_mask & burst_mask).sum() / case.sfreq_hz, 1e-9))
            nonburst_spike_rate = float(nonburst_mask[spike_samples].sum() / max(nonburst_mask.sum() / case.sfreq_hz, 1e-9))
        else:
            burst_spike_rate = 0.0
            nonburst_spike_rate = 0.0
        rows.append(
            {
                "subject_id": case.subject_id,
                "condition": case.condition,
                "channel": case.channel,
                "band_name": band_name,
                "band_role": band_roles[band_index],
                "burst_current_mean": float(trace[task_mask & burst_mask].mean()) if (task_mask & burst_mask).any() else np.nan,
                "nonburst_current_mean": float(trace[nonburst_mask].mean()) if nonburst_mask.any() else np.nan,
                "burst_spike_rate_hz": burst_spike_rate,
                "nonburst_spike_rate_hz": nonburst_spike_rate,
                "readout_score_correlation": float(np.corrcoef(trace[task_mask], readout_summary.score[task_mask])[0, 1]) if task_mask.sum() > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_readout_against_reference(
    *,
    subject_id: str,
    condition: str,
    channel: str,
    band_mode: str,
    burst_mask: np.ndarray,
    task_mask: np.ndarray,
    burst_events: pd.DataFrame,
    score: np.ndarray,
    stable_mask: np.ndarray,
    sfreq_hz: float,
    dt_ms: float,
    rest_mask: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Evaluate a readout trace against a frozen burst reference."""
    burst_mask = np.asarray(burst_mask, dtype=bool)
    task_mask = np.asarray(task_mask, dtype=bool)
    rest_values = None if rest_mask is None else np.asarray(rest_mask, dtype=bool)
    readout_mask = np.asarray(stable_mask, dtype=bool)
    score = np.asarray(score, dtype=np.float32)

    eval_mask = task_mask
    auc = _auc_score(burst_mask[eval_mask], score[eval_mask])
    readout_events = events_from_mask(readout_mask & eval_mask, dt_ms=dt_ms)
    burst_events = burst_events.loc[:, ["onset_s", "offset_s"]].copy()
    matched = match_readout_events_to_bursts(burst_events, readout_events)

    latencies_ms = []
    for row in matched.itertuples(index=False):
        if row.kind != "true_positive":
            continue
        latencies_ms.append((float(row.readout_onset_s) - float(row.burst_onset_s)) * 1000.0)

    nonburst_mask = eval_mask & ~burst_mask
    false_positive_events = int((matched["kind"] == "false_positive").sum()) if not matched.empty else 0
    false_positive_per_min = false_positive_events / max(nonburst_mask.sum() / sfreq_hz / 60.0, 1e-9)
    rest_false_positive_per_min = np.nan
    if rest_values is not None and rest_values.any():
        rest_events = events_from_mask(readout_mask & rest_values, dt_ms=dt_ms)
        rest_false_positive_per_min = float(len(rest_events) / max(rest_values.sum() / sfreq_hz / 60.0, 1e-9))

    return {
        "subject_id": subject_id,
        "condition": condition,
        "channel": channel,
        "band_mode": band_mode,
        "auc": auc,
        "median_latency_ms": float(np.median(latencies_ms)) if latencies_ms else np.nan,
        "false_positive_per_min": false_positive_per_min,
        "rest_false_positive_per_min": rest_false_positive_per_min,
        "readout_burst_mean": float(score[eval_mask & burst_mask].mean()) if (eval_mask & burst_mask).any() else np.nan,
        "readout_nonburst_mean": float(score[nonburst_mask].mean()) if nonburst_mask.any() else np.nan,
        "readout_separation": float(score[eval_mask & burst_mask].mean() - score[nonburst_mask].mean()) if (eval_mask & burst_mask).any() and nonburst_mask.any() else np.nan,
        "n_bursts": int(len(burst_events)),
        "n_readout_events": int(len(readout_events)),
        "true_positive_count": int((matched["kind"] == "true_positive").sum()) if not matched.empty else 0,
        "miss_count": int((matched["kind"] == "miss").sum()) if not matched.empty else int(len(burst_events)),
        "false_positive_count": false_positive_events,
    }


def evaluate_real_case(
    case: Any,
    result: Any,
    config: Any,
    readout_summary: Any,
    *,
    rest_mask: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Evaluate one real Phase 5 dev/QC case against frozen Phase 3 labels."""
    return evaluate_readout_against_reference(
        subject_id=case.subject_id,
        condition=case.condition,
        channel=case.channel,
        band_mode=case.band_mode,
        burst_mask=case.burst_mask,
        task_mask=case.task_mask,
        burst_events=case.events,
        score=readout_summary.score,
        stable_mask=readout_summary.stable_mask,
        sfreq_hz=case.sfreq_hz,
        dt_ms=config.dt_ms,
        rest_mask=rest_mask,
    )


def summarize_real_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate key Phase 5 real-data metrics across cases."""
    return {
        "auc_mean": float(metrics_df["auc"].mean()),
        "auc_median": float(metrics_df["auc"].median()),
        "median_latency_ms": float(metrics_df["median_latency_ms"].median()),
        "false_positive_per_min_mean": float(metrics_df["false_positive_per_min"].mean()),
        "readout_separation_mean": float(metrics_df["readout_separation"].mean()),
    }


def merge_event_tables(event_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge overlapping onset/offset rows into one interval table."""
    if not event_frames:
        return pd.DataFrame(columns=["onset_s", "offset_s"])
    merged = pd.concat(event_frames, ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=["onset_s", "offset_s"])
    merged = merged.sort_values(["onset_s", "offset_s"]).reset_index(drop=True)
    rows: list[dict[str, float]] = []
    current_onset = float(merged.iloc[0]["onset_s"])
    current_offset = float(merged.iloc[0]["offset_s"])
    for row in merged.iloc[1:].itertuples(index=False):
        onset_s = float(row.onset_s)
        offset_s = float(row.offset_s)
        if onset_s <= current_offset:
            current_offset = max(current_offset, offset_s)
            continue
        rows.append({"onset_s": current_onset, "offset_s": current_offset})
        current_onset = onset_s
        current_offset = offset_s
    rows.append({"onset_s": current_onset, "offset_s": current_offset})
    return pd.DataFrame(rows)


def latency_decomposition_table(
    case: Any,
    result: Any,
    config: Any,
    readout_summary: Any,
    *,
    evidence_trace: np.ndarray | None = None,
    causal_evidence_trace: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return event-level timing decomposition for burst, miss, and FP analysis."""
    readout_events = events_from_mask(np.asarray(readout_summary.stable_mask, dtype=bool), dt_ms=config.dt_ms)
    matched = match_readout_events_to_bursts(case.events.loc[:, ["onset_s", "offset_s"]], readout_events)
    beta_indices = np.array([index for index, role in enumerate(result.band_roles) if role == "beta"], dtype=int)
    beta_encoder_times = np.asarray(
        [
            time_s
            for time_s, index in zip(result.encoder_spike_times_s, result.encoder_spike_indices, strict=False)
            if int(index) in beta_indices
        ],
        dtype=float,
    )
    quiet_spike_times = np.asarray(result.quiet_spike_times_s, dtype=float)
    readout_spike_times = np.asarray(result.readout_spike_times_s, dtype=float)
    state_masks = {key: np.asarray(value, dtype=bool) for key, value in readout_summary.state_masks.items()}
    d2_plus_mask = np.vstack(
        [state_masks[name] for name in result.state_names[config.readout_bucket_index + 1 :]]
    ).any(axis=0)
    non_idle_mask = np.vstack([state_masks[name] for name in result.state_names[1:]]).any(axis=0)
    pooled_evidence = (
        np.asarray(evidence_trace, dtype=np.float32)
        if evidence_trace is not None
        else aggregate_beta_evidence(result.encoder_currents, result.band_roles, mode="mean")
    )
    duration_s = len(readout_summary.score) * config.dt_ms / 1000.0
    event_rows: list[dict[str, float | int | str]] = []

    for event_index, row in enumerate(matched.itertuples(index=False)):
        if row.kind == "false_positive":
            anchor_onset_s = float(row.readout_onset_s)
            anchor_offset_s = float(row.readout_offset_s)
            burst_index = -1
            phase3_onset_s = np.nan
            phase3_offset_s = np.nan
        else:
            anchor_onset_s = float(row.burst_onset_s)
            anchor_offset_s = float(row.burst_offset_s)
            burst_index = int(row.burst_index)
            phase3_onset_s = anchor_onset_s
            phase3_offset_s = anchor_offset_s

        search_start_s = max(0.0, anchor_onset_s - 0.20)
        search_stop_s = min(duration_s, anchor_offset_s + 0.35)
        reset_stop_s = min(duration_s, anchor_offset_s + 0.60)

        d0_onset_s = _first_true_or_spike_time(state_masks["D0"], start_s=search_start_s, stop_s=search_stop_s, dt_ms=config.dt_ms)
        d1_onset_s = _first_true_or_spike_time(state_masks["D1"], start_s=search_start_s, stop_s=search_stop_s, dt_ms=config.dt_ms)
        d2_onset_s = _first_true_or_spike_time(state_masks["D2"], start_s=search_start_s, stop_s=search_stop_s, dt_ms=config.dt_ms)
        stable_readout_onset_s = (
            None if pd.isna(row.readout_onset_s) else float(row.readout_onset_s)
        )
        readout_spike_time_s = _first_spike_time(readout_spike_times, start_s=search_start_s, stop_s=search_stop_s)
        quiet_activation_s = _first_spike_time(quiet_spike_times, start_s=anchor_offset_s, stop_s=reset_stop_s)
        reset_onset_s = _first_false_time(non_idle_mask, start_s=anchor_offset_s, stop_s=reset_stop_s, dt_ms=config.dt_ms)
        phase4_encoder_onset_s = _first_spike_time(beta_encoder_times, start_s=search_start_s, stop_s=search_stop_s)
        phase4_evidence_onset_s = _half_height_onset(
            pooled_evidence,
            start_s=search_start_s,
            stop_s=search_stop_s,
            dt_ms=config.dt_ms,
            floor=0.15,
        )
        causal_proxy_onset_s = _half_height_onset(
            causal_evidence_trace,
            start_s=search_start_s,
            stop_s=search_stop_s,
            dt_ms=config.dt_ms,
            floor=0.15,
        )
        d2_anchor_s = d2_onset_s if d2_onset_s is not None else (stable_readout_onset_s if stable_readout_onset_s is not None else anchor_onset_s)
        d2_anchor_source = "D2_onset_s" if d2_onset_s is not None else ("stable_readout_onset_s" if stable_readout_onset_s is not None else "event_anchor_s")
        d2_run_onset_s, d2_run_offset_s, d2_run_duration_ms = _run_bounds(d2_plus_mask, anchor_s=d2_anchor_s, dt_ms=config.dt_ms)

        event_rows.append(
            {
                "event_index": event_index,
                "kind": row.kind,
                "burst_index": burst_index,
                "subject_id": case.subject_id,
                "condition": case.condition,
                "channel": case.channel,
                "phase3_onset_s": phase3_onset_s,
                "phase3_offset_s": phase3_offset_s,
                "causalized_aux_onset_s": causal_proxy_onset_s,
                "phase4_encoder_onset_s": phase4_encoder_onset_s,
                "phase4_evidence_onset_s": phase4_evidence_onset_s,
                "D0_onset_s": d0_onset_s,
                "D1_onset_s": d1_onset_s,
                "D2_onset_s": d2_onset_s,
                "stable_readout_onset_s": stable_readout_onset_s,
                "readout_spike_time_s": readout_spike_time_s,
                "quiet_detector_activation_s": quiet_activation_s,
                "reset_onset_s": reset_onset_s,
                "D2_plus_anchor_s": d2_anchor_s,
                "D2_plus_anchor_source": d2_anchor_source,
                "D2_plus_run_onset_s": d2_run_onset_s,
                "D2_plus_run_offset_s": d2_run_offset_s,
                "D2_plus_occupancy_ms": d2_run_duration_ms,
                "stable_readout_latency_vs_phase3_ms": _ms_diff(stable_readout_onset_s, None if np.isnan(phase3_onset_s) else phase3_onset_s),
                "stable_readout_latency_vs_causalized_ms": _ms_diff(stable_readout_onset_s, causal_proxy_onset_s),
                "phase4_encoder_latency_vs_phase3_ms": _ms_diff(phase4_encoder_onset_s, None if np.isnan(phase3_onset_s) else phase3_onset_s),
                "D0_latency_vs_phase3_ms": _ms_diff(d0_onset_s, None if np.isnan(phase3_onset_s) else phase3_onset_s),
                "D1_latency_vs_phase3_ms": _ms_diff(d1_onset_s, None if np.isnan(phase3_onset_s) else phase3_onset_s),
                "D2_latency_vs_phase3_ms": _ms_diff(d2_onset_s, None if np.isnan(phase3_onset_s) else phase3_onset_s),
            }
        )
    return pd.DataFrame(event_rows)


def summarize_latency_decomposition(latency_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize event-level timing decomposition by event kind."""
    if latency_df.empty:
        return pd.DataFrame(
            columns=[
                "kind",
                "n_events",
                "median_phase4_encoder_latency_ms",
                "median_D0_latency_ms",
                "median_D1_latency_ms",
                "median_D2_latency_ms",
                "median_readout_latency_vs_phase3_ms",
                "median_readout_latency_vs_causalized_ms",
                "median_D2_plus_occupancy_ms",
            ]
        )
    rows = []
    for kind, frame in latency_df.groupby("kind", dropna=False):
        rows.append(
            {
                "kind": kind,
                "n_events": int(len(frame)),
                "median_phase4_encoder_latency_ms": float(frame["phase4_encoder_latency_vs_phase3_ms"].median()),
                "median_D0_latency_ms": float(frame["D0_latency_vs_phase3_ms"].median()),
                "median_D1_latency_ms": float(frame["D1_latency_vs_phase3_ms"].median()),
                "median_D2_latency_ms": float(frame["D2_latency_vs_phase3_ms"].median()),
                "median_readout_latency_vs_phase3_ms": float(frame["stable_readout_latency_vs_phase3_ms"].median()),
                "median_readout_latency_vs_causalized_ms": float(frame["stable_readout_latency_vs_causalized_ms"].median()),
                "median_D2_plus_occupancy_ms": float(frame["D2_plus_occupancy_ms"].median()),
            }
        )
    return pd.DataFrame(rows)

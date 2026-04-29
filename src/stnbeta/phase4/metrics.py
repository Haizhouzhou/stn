"""Metrics for synthetic and real-data Phase 4 validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stnbeta.phase4.real_data import RealConditionCase
from stnbeta.snn_brian2.runner import StateMachineResult
from stnbeta.synthetic.beta_burst_generator import SyntheticTrace


def _first_spike_in_window(spike_times_s: np.ndarray, start_s: float, stop_s: float) -> float | None:
    hits = spike_times_s[(spike_times_s >= start_s) & (spike_times_s <= stop_s)]
    if len(hits) == 0:
        return None
    return float(hits[0])


def synthetic_case_metrics(
    trace: SyntheticTrace,
    result: StateMachineResult,
    *,
    reset_margin_s: float = 0.05,
) -> dict[str, float | int | str]:
    """Aggregate synthetic benchmark metrics for one trace."""
    rows = trace.annotations
    negative_duration_s = float(len(trace.signal) / trace.sfreq_hz)
    detection_times = result.bucket_spike_times_s[
        result.bucket_spike_indices >= result.readout_bucket_index
    ]
    if rows.empty:
        false_positive_rate = len(detection_times) / max(negative_duration_s, 1e-9)
        return {
            "trace_name": trace.name,
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
            "unexpected_readout_count": 0,
            "mean_detection_latency_ms": np.nan,
            "false_positive_rate_hz": false_positive_rate,
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
    bucket_times_by_index = [
        result.bucket_spike_times_s[result.bucket_spike_indices == bucket_index]
        for bucket_index in range(len(result.bucket_thresholds_ms))
    ]

    for burst_index, row in rows.iterrows():
        onset_s = float(row["onset_s"])
        offset_s = float(row["offset_s"])
        expected_bucket = int(row["expected_bucket_index"])
        expects_readout = expected_bucket >= result.readout_bucket_index
        bucket_first_times = [
            _first_spike_in_window(times, onset_s, offset_s + 0.2)
            for times in bucket_times_by_index
        ]
        reached = [index for index, value in enumerate(bucket_first_times) if value is not None]
        if reached:
            ordered_times = [bucket_first_times[index] for index in reached]
            monotonic += int(
                all(earlier <= later for earlier, later in zip(ordered_times, ordered_times[1:]))
            )
            highest = max(reached)
            skipped += max(0, highest - len(reached) + 1)
        else:
            skipped += max(expected_bucket + 1, 0)

        detected = _first_spike_in_window(detection_times, onset_s, offset_s + 0.2)
        if expects_readout:
            expected_readout_total += 1
            correct_readout += int(detected is not None)
            missed_readout += int(detected is None)
        else:
            too_short_total += 1
            too_short_rejection_success += int(detected is None)
            unexpected_readout += int(detected is not None)
        if expects_readout and detected is not None:
            detection_latencies_ms.append((detected - onset_s) * 1000.0)

        next_onset = float(rows.iloc[burst_index + 1]["onset_s"]) if burst_index + 1 < len(rows) else result.duration_s
        post_reset_spikes = result.bucket_spike_times_s[
            (result.bucket_spike_times_s >= offset_s + reset_margin_s)
            & (result.bucket_spike_times_s < next_onset)
        ]
        reset_success += int(len(post_reset_spikes) == 0)

    false_positives = 0
    for spike in detection_times:
        in_burst = ((rows["onset_s"] <= spike) & (rows["offset_s"] >= spike)).any()
        if not in_burst:
            false_positives += 1

    return {
        "trace_name": trace.name,
        "n_ground_truth_bursts": int(len(rows)),
        "monotonic_progression_correct": monotonic,
        "skipped_bucket_transitions": skipped,
        "reset_success_count": reset_success,
        "reset_total": int(len(rows)),
        "expected_readout_total": expected_readout_total,
        "correct_readout_count": correct_readout,
        "missed_readout_count": missed_readout,
        "too_short_total": too_short_total,
        "too_short_rejection_success_count": too_short_rejection_success,
        "unexpected_readout_count": unexpected_readout,
        "mean_detection_latency_ms": float(np.nanmean(detection_latencies_ms)) if detection_latencies_ms else np.nan,
        "false_positive_rate_hz": false_positives / max(result.duration_s, 1e-9),
    }


def spike_rate_in_mask(spike_times_s: np.ndarray, sample_mask: np.ndarray, sfreq_hz: float) -> float:
    """Compute spikes/sec restricted to *sample_mask*."""
    duration_s = float(sample_mask.sum()) / sfreq_hz
    if duration_s <= 0.0:
        return np.nan
    spike_samples = np.clip(np.round(spike_times_s * sfreq_hz).astype(int), 0, len(sample_mask) - 1)
    return float(sample_mask[spike_samples].sum() / duration_s)


def real_case_band_metrics(
    case: RealConditionCase,
    band_names: list[str],
    band_roles: list[str],
    encoder_spike_times_s: np.ndarray,
    encoder_spike_indices: np.ndarray,
) -> pd.DataFrame:
    """Summarize burst-vs-non-burst firing rates per band."""
    rows: list[dict[str, float | str]] = []
    task_nonburst_mask = case.task_mask & ~case.burst_mask

    for band_index, band_name in enumerate(band_names):
        spike_times = encoder_spike_times_s[encoder_spike_indices == band_index]
        burst_rate = spike_rate_in_mask(spike_times, case.task_mask & case.burst_mask, case.sfreq_hz)
        nonburst_rate = spike_rate_in_mask(spike_times, task_nonburst_mask, case.sfreq_hz)
        lead_lags = []
        for event in case.events.itertuples(index=False):
            first_spike = _first_spike_in_window(spike_times, float(event.onset_s) - 0.2, float(event.onset_s) + 0.2)
            if first_spike is not None:
                lead_lags.append((first_spike - float(event.onset_s)) * 1000.0)
        rows.append(
            {
                "subject_id": case.subject_id,
                "condition": case.condition,
                "channel": case.channel,
                "band_name": band_name,
                "band_role": band_roles[band_index],
                "burst_rate_hz": burst_rate,
                "nonburst_rate_hz": nonburst_rate,
                "burst_to_nonburst_ratio": burst_rate / nonburst_rate if nonburst_rate and not np.isnan(nonburst_rate) else np.nan,
                "median_onset_lead_lag_ms": float(np.nanmedian(lead_lags)) if lead_lags else np.nan,
            }
        )
    return pd.DataFrame(rows)

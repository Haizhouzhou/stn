"""Faster Stage F event scoring for Phase 5_2C.

This module preserves the public Stage F event-output contract while avoiding
the repeated pandas sort/groupby work in the original scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import stage_f_event_metrics as stage


@dataclass(frozen=True)
class AlarmResult:
    by_group: dict[int, np.ndarray]
    n_alarms: int


@dataclass
class EventCache:
    order: np.ndarray
    group_sorted: np.ndarray
    subject_sorted: np.ndarray
    times_sorted: np.ndarray
    group_codes: np.ndarray
    subject_codes: np.ndarray
    group_labels: np.ndarray
    subject_labels: np.ndarray
    group_slices: list[tuple[int, int, int]]
    group_subject: dict[int, int]
    events_by_group: dict[int, tuple[np.ndarray, np.ndarray]]
    events_by_subject_group: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]
    minutes_total: float
    minutes_by_subject: dict[int, float]
    n_events: int


def compute_event_outputs_fast(config: dict[str, Any], frame: pd.DataFrame, bundles: list[stage.ScoreBundle], target: dict[str, Any]) -> dict[str, pd.DataFrame]:
    threshold_rows = []
    subject_rows = []
    tier_rows = {"causal_tier1_event_metrics.tsv": [], "causal_tier2_event_metrics.tsv": [], "causal_tier3_event_metrics.tsv": []}
    cache = prepare_event_cache(frame)
    y_window = frame["is_true_event"].to_numpy(dtype=bool)
    timing = stage.event_timing_policy(config)
    for bundle in bundles:
        metrics_by_grid = []
        score_diagnostics = stage.score_diagnostic_metrics(y_window, bundle.score)
        thresholds = stage.threshold_candidates(bundle.score)
        evaluated = []
        alarm_cache: dict[float, AlarmResult] = {}
        for threshold in thresholds:
            alarms = build_alarm_result(cache, bundle.score, float(threshold), timing=timing)
            alarm_cache[float(threshold)] = alarms
            metrics = evaluate_alarm_result(alarms, cache.events_by_group, cache.n_events, cache.minutes_total)
            metrics["threshold"] = float(threshold)
            evaluated.append(metrics)
        for fp_cap in stage.FP_GRID:
            best = None
            feasible = [metrics for metrics in evaluated if metrics["fp_per_min_achieved"] <= fp_cap]
            if feasible:
                best = max(
                    feasible,
                    key=lambda metrics: (
                        stage.safe_float(metrics.get("recall")),
                        stage.safe_float(metrics.get("F1")),
                        stage.safe_float(metrics.get("precision")),
                        -stage.safe_float(metrics.get("fp_per_min_achieved")),
                    ),
                ).copy()
                best["target_fp_min"] = fp_cap
            if best is None:
                best = stage.empty_event_metrics(fp_cap, thresholds[-1] if len(thresholds) else np.nan, "no threshold achieved requested FP/min cap")
            best.update(score_diagnostics)
            best.update(stage.target_event_fields(target))
            best.update(stage.tier_info(bundle))
            metrics_by_grid.append(best)
            threshold_rows.append(best.copy())
            if len(thresholds):
                alarms = alarm_cache[float(best["threshold"])]
                subject_rows.extend(subject_event_rows_fast(cache, bundle, float(best["threshold"]), alarms))
        table_name = {
            "tier1_continuous": "causal_tier1_event_metrics.tsv",
            "tier2_quantized": "causal_tier2_event_metrics.tsv",
            "tier3_quantized_mismatched": "causal_tier3_event_metrics.tsv",
        }[bundle.tier]
        tier_rows[table_name].extend(metrics_by_grid)
    outputs = {}
    for name, rows in tier_rows.items():
        outputs[name] = stage.universal_event_frame(config, pd.DataFrame(rows), name)
    outputs["causal_event_threshold_grid.tsv"] = stage.universal_event_frame(config, pd.DataFrame(threshold_rows), "causal_event_threshold_grid.tsv")
    outputs["causal_event_alarm_trace_summary.tsv"] = stage.universal_event_frame(config, pd.DataFrame(subject_rows), "causal_event_alarm_trace_summary.tsv")
    outputs["causal_three_tier_event_summary.tsv"] = stage.three_tier_event_summary(config, outputs, target)
    return outputs


def prepare_event_cache(frame: pd.DataFrame) -> EventCache:
    group_key = frame["fif_path"].astype(str) + "\0" + frame["channel"].astype(str)
    group_codes, group_labels = pd.factorize(group_key, sort=False)
    subject_codes, subject_labels = pd.factorize(frame["subject_id"].astype(str), sort=True)
    times = pd.to_numeric(frame["window_start_s"], errors="coerce").to_numpy(dtype=float) + 0.150
    order = np.lexsort((times, group_codes))
    group_sorted = group_codes[order].astype(np.int64, copy=False)
    subject_sorted = subject_codes[order].astype(np.int64, copy=False)
    times_sorted = times[order]
    start_idx = np.r_[0, np.flatnonzero(np.diff(group_sorted)) + 1]
    stop_idx = np.r_[start_idx[1:], len(order)]
    group_slices = [(int(group_sorted[start]), int(start), int(stop)) for start, stop in zip(start_idx, stop_idx, strict=False)]
    group_subject = {group: int(subject_sorted[start]) for group, start, _ in group_slices}
    minutes_total, minutes_by_subject = recording_minutes_from_slices(frame, order, group_slices, group_subject)
    events_by_group, events_by_subject_group, n_events = event_arrays(frame, group_labels, subject_labels)
    return EventCache(
        order=order,
        group_sorted=group_sorted,
        subject_sorted=subject_sorted,
        times_sorted=times_sorted,
        group_codes=group_codes.astype(np.int64, copy=False),
        subject_codes=subject_codes.astype(np.int64, copy=False),
        group_labels=np.asarray(group_labels, dtype=object),
        subject_labels=np.asarray(subject_labels, dtype=object),
        group_slices=group_slices,
        group_subject=group_subject,
        events_by_group=events_by_group,
        events_by_subject_group=events_by_subject_group,
        minutes_total=minutes_total,
        minutes_by_subject=minutes_by_subject,
        n_events=n_events,
    )


def recording_minutes_from_slices(frame: pd.DataFrame, order: np.ndarray, group_slices: list[tuple[int, int, int]], group_subject: dict[int, int]) -> tuple[float, dict[int, float]]:
    starts = pd.to_numeric(frame["window_start_s"], errors="coerce").to_numpy(dtype=float)[order]
    stops = pd.to_numeric(frame["window_stop_s"], errors="coerce").to_numpy(dtype=float)[order]
    total = 0.0
    by_subject: dict[int, float] = {}
    for group, start, stop in group_slices:
        group_start = np.nanmin(starts[start:stop])
        group_stop = np.nanmax(stops[start:stop])
        if np.isfinite(group_start) and np.isfinite(group_stop) and group_stop > group_start:
            minutes = float(group_stop - group_start) / 60.0
            total += minutes
            subject = group_subject[group]
            by_subject[subject] = by_subject.get(subject, 0.0) + minutes
    return total, by_subject


def event_arrays(frame: pd.DataFrame, group_labels: np.ndarray, subject_labels: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, dict[int, tuple[np.ndarray, np.ndarray]]], int]:
    events = stage.truth_events(frame)
    group_to_code = {str(label): idx for idx, label in enumerate(group_labels)}
    subject_to_code = {str(label): idx for idx, label in enumerate(subject_labels)}
    by_group: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    by_subject_group: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    events = events.copy()
    events["_group_key"] = events["fif_path"].astype(str) + "\0" + events["channel"].astype(str)
    for (subject, group_key), ev in events.groupby(["subject_id", "_group_key"], sort=False):
        group_code = group_to_code.get(str(group_key))
        subject_code = subject_to_code.get(str(subject))
        if group_code is None or subject_code is None:
            continue
        onsets = pd.to_numeric(ev["anchor_onset_s"], errors="coerce").to_numpy(dtype=float)
        offsets = pd.to_numeric(ev["anchor_offset_s"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(onsets) & np.isfinite(offsets)
        if not valid.any():
            continue
        order = np.argsort(onsets[valid], kind="mergesort")
        pair = (onsets[valid][order], offsets[valid][order])
        by_group[group_code] = pair
        by_subject_group.setdefault(subject_code, {})[group_code] = pair
    return by_group, by_subject_group, int(len(events))


def build_alarm_result(cache: EventCache, score: np.ndarray, threshold: float, *, timing: dict[str, float]) -> AlarmResult:
    score_sorted = np.asarray(score, dtype=float)[cache.order]
    finite = np.isfinite(score_sorted)
    by_group: dict[int, np.ndarray] = {}
    n_alarms = 0
    for group, start, stop in cache.group_slices:
        local = finite[start:stop] & (score_sorted[start:stop] >= threshold)
        if not local.any():
            continue
        idx = np.flatnonzero(local)
        times = cache.times_sorted[start:stop][idx]
        keep = stage.refractory_keep_mask(times, refractory_s=timing["refractory_s"], merge_window_s=timing["merge_window_s"])
        if keep.any():
            kept = times[keep]
            by_group[group] = kept
            n_alarms += int(kept.size)
    return AlarmResult(by_group=by_group, n_alarms=n_alarms)


def evaluate_alarm_result(alarms: AlarmResult, events_by_group: dict[int, tuple[np.ndarray, np.ndarray]], n_events: int, total_minutes: float) -> dict[str, Any]:
    if alarms.n_alarms == 0:
        return stage.event_metric_row(0, 0, n_events, 0, total_minutes, [], [], "no alarms at threshold")
    latencies: list[float] = []
    early_latencies: list[float] = []
    one_alarm_events = 0
    success_alarm_count = 0
    matched_event_count = 0
    for group, (onsets, offsets) in events_by_group.items():
        alarm_times = alarms.by_group.get(group)
        if alarm_times is None or alarm_times.size == 0 or onsets.size == 0:
            continue
        start = np.searchsorted(alarm_times, onsets, side="left")
        stop = np.searchsorted(alarm_times, offsets, side="right")
        counts = stop - start
        matched = counts > 0
        matched_event_count += int(matched.sum())
        one_alarm_events += int((counts == 1).sum())
        success_alarm_count += int(counts[matched].sum())
        if matched.any():
            latencies.extend(((alarm_times[start[matched]] - onsets[matched]) * 1000.0).tolist())
        early_start = np.searchsorted(alarm_times, onsets - 0.500, side="left")
        early_stop = np.searchsorted(alarm_times, onsets, side="left")
        early_matched = early_stop > early_start
        if early_matched.any():
            early_latencies.extend(((alarm_times[early_start[early_matched]] - onsets[early_matched]) * 1000.0).tolist())
    tp_alarms = min(success_alarm_count, alarms.n_alarms)
    fp = int(max(alarms.n_alarms - tp_alarms, 0))
    return stage.event_metric_row(
        matched_event_count,
        tp_alarms,
        n_events,
        fp,
        total_minutes,
        latencies,
        early_latencies,
        "",
        one_alarm_events=one_alarm_events,
        success_alarm_count=success_alarm_count,
    )


def subject_event_rows_fast(cache: EventCache, bundle: stage.ScoreBundle, threshold: float, alarms: AlarmResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject_code, events_by_group in sorted(cache.events_by_subject_group.items(), key=lambda item: str(cache.subject_labels[item[0]])):
        subject_alarm_groups = {group: alarm_times for group, alarm_times in alarms.by_group.items() if cache.group_subject.get(group) == subject_code}
        subject_alarms = AlarmResult(subject_alarm_groups, sum(int(times.size) for times in subject_alarm_groups.values()))
        n_events = sum(int(onsets.size) for onsets, _ in events_by_group.values())
        minutes = cache.minutes_by_subject.get(subject_code, 0.0)
        metrics = evaluate_alarm_result(subject_alarms, events_by_group, n_events, minutes)
        metrics.update(stage.tier_info(bundle))
        metrics["subject_id"] = str(cache.subject_labels[subject_code])
        metrics["threshold"] = threshold
        rows.append(metrics)
    return rows

"""Synthetic robustness-family audit for the Phase 5_2C duration state machine.

This module is a deterministic topology surrogate and output-contract harness for
the legacy Brian2 state-machine path. It does not claim Brian2 or DYNAP-SE1
execution. Its purpose is to lock the robustness protocol, metrics, perturbation
battery, and table schemas before the same audit is run against the full
``stnbeta.snn_brian2`` implementation in the teammate/cluster checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DT_MS = 1.0
TOTAL_MS = 1400
POSITIVE_CASES = {"threshold_crossing_120ms", "sustained_200ms", "long_400ms_plus", "two_bursts_with_quiet_gap"}
SHORT_NEGATIVE_CASES = {"short_40ms", "near_threshold_90ms"}
NEGATIVE_CONTROL_CASES = {"no_burst", "power_shift_no_burst", "sustained_only"}


@dataclass(frozen=True)
class StateMachineConfig:
    config_id: str
    population_exc: int
    population_inh: int
    duration_thresholds_ms: tuple[int, int, int, int]
    sustain_weight: float
    forward_weight: float
    reset_strength: float
    quiet_gain: float


@dataclass(frozen=True)
class NoiseCondition:
    condition_id: str
    family: str
    level: float
    seed_count: int
    description: str


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    expected_role: str
    events_ms: tuple[tuple[int, int], ...]
    baseline_level: float
    beta_background: float
    interpretation: str


def synthetic_cases() -> list[SyntheticCase]:
    return [
        SyntheticCase("no_burst", "negative_control", (), 0.05, 0.00, "No beta event should produce no readout."),
        SyntheticCase("short_40ms", "short_negative", ((300, 340),), 0.05, 0.00, "Below-duration beta event should be rejected."),
        SyntheticCase("near_threshold_90ms", "short_negative", ((300, 390),), 0.05, 0.00, "Near-threshold event should remain below stable D2+ readout."),
        SyntheticCase("threshold_crossing_120ms", "positive", ((300, 420),), 0.05, 0.00, "Minimal positive event crossing D2 lower bound."),
        SyntheticCase("sustained_200ms", "positive", ((300, 500),), 0.05, 0.00, "Sustained event should produce stable readout."),
        SyntheticCase("long_400ms_plus", "positive", ((300, 760),), 0.05, 0.00, "Long event should progress into later duration buckets."),
        SyntheticCase("two_bursts_with_quiet_gap", "positive", ((250, 430), (760, 980)), 0.05, 0.00, "Quiet gap should reset between two valid bursts."),
        SyntheticCase("interrupted_burst", "interrupted", ((300, 360), (385, 445),), 0.05, 0.00, "Short interruption tests quiet holdoff and gap bridging."),
        SyntheticCase("power_shift_no_burst", "negative_control", (), 0.38, 0.00, "Sustained subthreshold power shift must not become a transient burst."),
        SyntheticCase("sustained_only", "negative_control", ((250, 950),), 0.05, 0.00, "Long sustained beta state is a state-tracking control, not transient-burst evidence."),
    ]


def literature_noise_battery(seed_count: int) -> list[NoiseCondition]:
    rows = [
        NoiseCondition("clean", "clean", 0.0, 1, "No perturbation; establishes baseline operating region."),
        NoiseCondition("input_gaussian_0p02", "input_current_additive_gaussian", 0.02, seed_count, "Additive current noise at 2% of current std."),
        NoiseCondition("input_gaussian_0p05", "input_current_additive_gaussian", 0.05, seed_count, "Additive current noise at 5% of current std."),
        NoiseCondition("input_gaussian_0p10", "input_current_additive_gaussian", 0.10, seed_count, "Additive current noise at 10% of current std."),
        NoiseCondition("gain_noise_0p05", "input_current_multiplicative_gain", 0.05, seed_count, "Per-run multiplicative encoder gain noise."),
        NoiseCondition("slow_drift_0p10", "input_current_slow_drift", 0.10, seed_count, "Linear gain drift across the synthetic trial."),
        NoiseCondition("colored_1f_0p05", "physiological_colored_noise", 0.05, seed_count, "1/f-like current perturbation for end-to-end LFP risk."),
        NoiseCondition("line_50hz_0p05", "physiological_line_noise", 0.05, seed_count, "50 Hz sinusoidal current contamination."),
        NoiseCondition("event_jitter_5ms", "event_timing_jitter", 5.0, seed_count, "Event boundary jitter of +/-5 ms."),
        NoiseCondition("event_dropout_0p10", "event_dropout", 0.10, seed_count, "Random dropout of active beta evidence samples."),
        NoiseCondition("false_spikes_0p02", "false_background_spikes", 0.02, seed_count, "Sparse false beta-evidence samples in negative intervals."),
        NoiseCondition("impulse_artifact_0p25", "artifact_impulse", 0.25, seed_count, "Short high-amplitude impulse artifacts."),
        NoiseCondition("clipping_0p75", "artifact_clipping", 0.75, seed_count, "Amplitude clipping that distorts strong beta evidence."),
        NoiseCondition("mismatch_0p05", "hardware_mismatch", 0.05, seed_count, "Gaussian mismatch on thresholds, weights, reset, and quiet gain."),
        NoiseCondition("mismatch_0p10", "hardware_mismatch", 0.10, seed_count, "Gaussian mismatch at literature-grounded 10% level."),
        NoiseCondition("mismatch_0p20", "hardware_mismatch", 0.20, seed_count, "Stress-test mismatch at 20% level."),
        NoiseCondition("quant_8bit", "quantization", 8.0, seed_count, "8-bit current/readout quantization."),
        NoiseCondition("quant_6bit", "quantization", 6.0, seed_count, "6-bit current/readout quantization."),
        NoiseCondition("quant_4bit", "quantization", 4.0, seed_count, "4-bit DYNAP-SE-style quantization stress test."),
    ]
    return rows


def config_grid(max_configs: int | None = 384) -> list[StateMachineConfig]:
    populations = [(8, 2), (16, 4), (32, 8), (64, 16)]
    thresholds = [(40, 80, 160, 320), (50, 100, 200, 400), (75, 150, 300, 600), (100, 200, 350, 500)]
    sustain = [0.5, 0.75, 1.0, 1.25, 1.5]
    forward = [0.5, 0.75, 1.0, 1.25, 1.5]
    reset = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    quiet = [0.5, 0.75, 1.0, 1.25, 1.5]
    configs: list[StateMachineConfig] = []
    for pop_exc, pop_inh in populations:
        for th in thresholds:
            for sw in sustain:
                for fw in forward:
                    for rw in reset:
                        for qg in quiet:
                            raw = f"{pop_exc}-{pop_inh}-{th}-{sw}-{fw}-{rw}-{qg}"
                            cid = "cfg_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
                            configs.append(StateMachineConfig(cid, pop_exc, pop_inh, th, sw, fw, rw, qg))
    if max_configs is not None and len(configs) > max_configs:
        indices = np.linspace(0, len(configs) - 1, max_configs, dtype=int)
        configs = [configs[int(i)] for i in indices]
    return configs


def evidence_for_case(case: SyntheticCase, rng: np.random.Generator | None = None, noise: NoiseCondition | None = None) -> np.ndarray:
    evidence = np.full(TOTAL_MS, case.baseline_level + case.beta_background, dtype=float)
    events = list(case.events_ms)
    if noise and noise.family == "event_timing_jitter" and rng is not None:
        jitter = int(noise.level)
        events = [(max(0, s + int(rng.integers(-jitter, jitter + 1))), min(TOTAL_MS, e + int(rng.integers(-jitter, jitter + 1)))) for s, e in events]
    for start, stop in events:
        if stop > start:
            evidence[start:stop] = 1.0
    if noise is None or noise.family == "clean":
        return evidence
    if rng is None:
        rng = np.random.default_rng(0)
    std = max(float(np.std(evidence)), 0.05)
    if noise.family == "input_current_additive_gaussian":
        evidence = evidence + rng.normal(0.0, noise.level * std, size=evidence.shape)
    elif noise.family == "input_current_multiplicative_gain":
        evidence = evidence * max(0.0, rng.normal(1.0, noise.level))
    elif noise.family == "input_current_slow_drift":
        evidence = evidence * np.linspace(1.0 - noise.level, 1.0 + noise.level, evidence.size)
    elif noise.family == "physiological_colored_noise":
        evidence = evidence + colored_noise(evidence.size, rng) * noise.level
    elif noise.family == "physiological_line_noise":
        t = np.arange(evidence.size) / 1000.0
        evidence = evidence + noise.level * np.sin(2.0 * np.pi * 50.0 * t)
    elif noise.family == "event_dropout":
        active = evidence > 0.8
        drop = rng.random(evidence.size) < noise.level
        evidence[active & drop] = case.baseline_level
    elif noise.family == "false_background_spikes":
        false = rng.random(evidence.size) < noise.level
        evidence[false] = np.maximum(evidence[false], 1.0)
    elif noise.family == "artifact_impulse":
        for _ in range(3):
            start = int(rng.integers(0, evidence.size - 8))
            evidence[start : start + 8] = np.maximum(evidence[start : start + 8], 1.0 + noise.level)
    elif noise.family == "artifact_clipping":
        evidence = np.clip(evidence, 0.0, noise.level)
    elif noise.family == "quantization":
        evidence = quantize(evidence, int(noise.level), low=0.0, high=1.25)
    return np.clip(evidence, 0.0, 1.5)


def colored_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    amp = 1.0 / np.sqrt(freqs)
    amp[0] = 0.0
    phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.size)
    x = np.fft.irfft(amp * np.exp(1j * phases), n=n)
    return x / (np.std(x) + 1e-12)


def quantize(x: np.ndarray, bits: int, *, low: float, high: float) -> np.ndarray:
    levels = max(2, 2**bits)
    clipped = np.clip(x, low, high)
    return np.round((clipped - low) / (high - low) * (levels - 1)) / (levels - 1) * (high - low) + low


def perturbed_config(config: StateMachineConfig, noise: NoiseCondition | None, rng: np.random.Generator | None) -> StateMachineConfig:
    if noise is None or noise.family != "hardware_mismatch" or rng is None:
        return config
    scale = noise.level
    th = tuple(max(10, int(round(v * max(0.25, rng.normal(1.0, scale))))) for v in config.duration_thresholds_ms)
    return StateMachineConfig(
        config.config_id,
        config.population_exc,
        config.population_inh,
        th,
        max(0.05, config.sustain_weight * max(0.25, rng.normal(1.0, scale))),
        max(0.05, config.forward_weight * max(0.25, rng.normal(1.0, scale))),
        max(0.05, config.reset_strength * max(0.25, rng.normal(1.0, scale))),
        max(0.05, config.quiet_gain * max(0.25, rng.normal(1.0, scale))),
    )


def simulate_case(config: StateMachineConfig, case: SyntheticCase, *, noise: NoiseCondition | None = None, seed: int = 0) -> dict[str, float | str | bool | int]:
    rng = np.random.default_rng(seed)
    cfg = perturbed_config(config, noise, rng)
    evidence = evidence_for_case(case, rng=rng, noise=noise)
    quiet_drive = np.clip(1.0 - evidence, 0.0, 1.5) * cfg.quiet_gain
    readout_duration = cfg.duration_thresholds_ms[1]
    state_thresholds = cfg.duration_thresholds_ms
    entry_threshold = effective_entry_threshold(cfg)
    active = evidence >= entry_threshold

    duration = 0.0
    readout_mask = np.zeros_like(active, dtype=bool)
    state_index = np.zeros_like(active, dtype=int)
    latencies = []
    event_hits = 0

    event_windows = list(case.events_ms)
    for i, is_active in enumerate(active):
        if is_active:
            duration += DT_MS * max(0.35, min(1.75, cfg.forward_weight))
        else:
            decay = DT_MS * max(0.0, quiet_drive[i]) * cfg.reset_strength * 5.0
            duration = max(0.0, duration - decay)
        bucket = int(np.searchsorted(np.asarray(state_thresholds), duration, side="right"))
        state_index[i] = bucket
        readout_mask[i] = duration >= readout_duration
    # The topology surrogate tests D2+ readout plumbing. The full Brian2 runner may
    # apply a longer post-hoc dwell filter; that should be audited separately.
    readout_mask = apply_dwell(readout_mask, dwell_ms=20)
    predicted_events = mask_to_events(readout_mask)
    true_events = event_windows if case.case_id != "sustained_only" else []
    true_state_events = event_windows if case.case_id == "sustained_only" else true_events
    for start, stop in true_state_events:
        hit_starts = [p_start for p_start, p_stop in predicted_events if p_stop >= start and p_start <= stop + 150]
        if hit_starts:
            event_hits += 1
            latencies.append(float(min(hit_starts) - start))
    false_alarms = count_false_alarms(predicted_events, true_state_events)
    n_true = len(true_state_events)
    recall = event_hits / n_true if n_true else (0.0 if predicted_events else np.nan)
    precision = event_hits / len(predicted_events) if predicted_events else (np.nan if n_true else 1.0)
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else np.nan
    progression_violations = int(np.sum(np.diff(state_index) > 1))
    if cfg.forward_weight > 1.35 and cfg.sustain_weight > 1.25:
        progression_violations += int(np.max(state_index) >= 3)
    minutes = TOTAL_MS / 60000.0
    return {
        "config_id": config.config_id,
        "case_id": case.case_id,
        "expected_role": case.expected_role,
        "noise_condition": noise.condition_id if noise else "clean",
        "seed": seed,
        "n_true_events": n_true,
        "n_readout_events": len(predicted_events),
        "event_hits": event_hits,
        "false_alarms": false_alarms,
        "recall": recall,
        "precision": precision,
        "F1": f1,
        "false_alarm_per_min": false_alarms / minutes,
        "latency_median_ms": float(np.median(latencies)) if latencies else np.nan,
        "latency_iqr_ms": float(np.percentile(latencies, 75) - np.percentile(latencies, 25)) if len(latencies) > 1 else 0.0 if latencies else np.nan,
        "progression_violations": progression_violations,
        "reset_failure_rate": reset_failure_rate(state_index, event_windows),
        "state_dwell_time_error_ms": dwell_time_error(state_index, case),
        "max_state_index": int(np.max(state_index)),
        "short_event_false_readout": bool(case.case_id in SHORT_NEGATIVE_CASES and len(predicted_events) > 0),
        "transient_claim_for_sustained_only": bool(case.case_id == "sustained_only" and len(predicted_events) > 0),
    }


def effective_entry_threshold(config: StateMachineConfig) -> float:
    population_factor = np.sqrt(32.0 / max(config.population_exc, 1))
    weight_factor = np.sqrt(max(config.sustain_weight * config.forward_weight, 0.05))
    inhibition_factor = 1.0 + 0.02 * max(config.population_inh - config.population_exc / 4.0, 0.0)
    return float(np.clip(0.62 * population_factor * inhibition_factor / weight_factor, 0.25, 1.15))


def apply_dwell(mask: np.ndarray, dwell_ms: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for start, stop in mask_to_events(mask):
        if stop - start >= dwell_ms:
            out[start:stop] = True
    return out


def mask_to_events(mask: np.ndarray) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    in_event = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_event:
            start = i
            in_event = True
        elif not val and in_event:
            events.append((start, i))
            in_event = False
    if in_event:
        events.append((start, len(mask)))
    return events


def count_false_alarms(predicted: list[tuple[int, int]], truth: list[tuple[int, int]]) -> int:
    count = 0
    for p_start, p_stop in predicted:
        if not any(p_stop >= t_start and p_start <= t_stop + 150 for t_start, t_stop in truth):
            count += 1
    return count


def reset_failure_rate(state_index: np.ndarray, event_windows: list[tuple[int, int]]) -> float:
    if not event_windows:
        return 0.0
    failures = 0
    for idx, (_start, stop) in enumerate(event_windows):
        next_start = event_windows[idx + 1][0] if idx + 1 < len(event_windows) else len(state_index)
        check_start = min(len(state_index), stop + 150)
        if next_start <= check_start:
            continue
        if np.any(state_index[check_start:next_start] > 0):
            failures += 1
    return failures / max(1, len(event_windows))


def dwell_time_error(state_index: np.ndarray, case: SyntheticCase) -> float:
    if not case.events_ms:
        return float(np.sum(state_index > 0))
    expected = sum(stop - start for start, stop in case.events_ms)
    observed = int(np.sum(state_index > 0))
    return float(abs(observed - expected))


def hard_gate_pass(row: pd.Series) -> bool:
    case = str(row["case_id"])
    if int(row["progression_violations"]) != 0:
        return False
    if float(row["reset_failure_rate"]) > 0.05:
        return False
    if case == "no_burst" and float(row["false_alarm_per_min"]) > 0.0:
        return False
    if case in SHORT_NEGATIVE_CASES and bool(row["short_event_false_readout"]):
        return False
    if case in POSITIVE_CASES and not (float(row["recall"]) >= 0.95):
        return False
    if case == "power_shift_no_burst" and float(row["false_alarm_per_min"]) > 0.0:
        return False
    return True


def run_clean_sweep(configs: list[StateMachineConfig]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [simulate_case(cfg, case) for cfg in configs for case in synthetic_cases() if case.case_id != "sustained_only"]
    frame = pd.DataFrame(rows)
    frame["hard_gate_pass"] = frame.apply(hard_gate_pass, axis=1)
    summary = summarize_configs(frame, configs)
    return frame, summary


def summarize_configs(frame: pd.DataFrame, configs: list[StateMachineConfig]) -> pd.DataFrame:
    config_rows = []
    cfg_by_id = {c.config_id: c for c in configs}
    for cid, group in frame.groupby("config_id", sort=False):
        positive = group[group["case_id"].isin(POSITIVE_CASES)]
        negative = group[group["case_id"].isin(SHORT_NEGATIVE_CASES | {"no_burst", "power_shift_no_burst"})]
        cfg = cfg_by_id[cid]
        config_rows.append({
            **asdict(cfg),
            "duration_thresholds_ms": ",".join(str(v) for v in cfg.duration_thresholds_ms),
            "clean_hard_gate_pass": bool(group["hard_gate_pass"].all()),
            "case_pass_rate": float(group["hard_gate_pass"].mean()),
            "positive_recall_mean": finite_mean(positive["recall"].to_numpy(dtype=float), default=0.0),
            "negative_false_alarm_per_min_sum": float(negative["false_alarm_per_min"].sum()),
            "latency_median_ms": finite_median(positive["latency_median_ms"].to_numpy(dtype=float), default=1000.0),
            "progression_violations_total": int(group["progression_violations"].sum()),
            "reset_failure_rate_max": float(group["reset_failure_rate"].max()),
            "pareto_score": pareto_score(positive, negative, group),
        })
    return pd.DataFrame(config_rows).sort_values(["clean_hard_gate_pass", "pareto_score"], ascending=[False, False])


def pareto_score(positive: pd.DataFrame, negative: pd.DataFrame, group: pd.DataFrame) -> float:
    recall = finite_mean(positive["recall"].to_numpy(dtype=float), default=0.0) if len(positive) else 0.0
    latency = finite_median(positive["latency_median_ms"].to_numpy(dtype=float), default=1000.0) if len(positive) else 1000.0
    fp = float(negative["false_alarm_per_min"].sum()) if len(negative) else 100.0
    violations = float(group["progression_violations"].sum())
    reset = float(group["reset_failure_rate"].max())
    return recall - 0.001 * max(latency, 0.0) - 0.01 * fp - 0.05 * violations - reset


def finite_mean(values: np.ndarray, *, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else default


def finite_median(values: np.ndarray, *, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else default


def select_family(summary: pd.DataFrame, family_size: int) -> pd.DataFrame:
    passed = summary[summary["clean_hard_gate_pass"]].copy()
    if passed.empty:
        return summary.head(min(family_size, len(summary))).copy()
    return passed.head(min(family_size, len(passed))).copy()


def run_noise_family(family: pd.DataFrame, configs: list[StateMachineConfig], seed_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg_by_id = {cfg.config_id: cfg for cfg in configs}
    family_ids = set(family["config_id"].tolist())
    selected = [cfg_by_id[cid] for cid in family_ids if cid in cfg_by_id]
    rows = []
    battery = literature_noise_battery(seed_count)
    eval_cases = synthetic_cases()
    for noise in battery:
        seeds = [0] if noise.family == "clean" else list(range(seed_count))
        for cfg in selected:
            for seed in seeds:
                for case in eval_cases:
                    rows.append(simulate_case(cfg, case, noise=noise, seed=seed))
    frame = pd.DataFrame(rows)
    frame["hard_gate_pass"] = frame.apply(hard_gate_pass, axis=1)
    summary = summarize_noise(frame)
    negative = negative_control_summary(frame)
    return frame, summary, negative


def summarize_noise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in frame.groupby("noise_condition", sort=False):
        core = group[~group["case_id"].eq("sustained_only")]
        config_pass = core.groupby(["config_id", "seed"], sort=False)["hard_gate_pass"].all().astype(float)
        positive = group[group["case_id"].isin(POSITIVE_CASES)]
        rows.append({
            "noise_condition": condition,
            "family_pass_rate": float(config_pass.mean()),
            "robustness_volume": float(core["hard_gate_pass"].mean()),
            "positive_recall_mean": float(positive["recall"].mean()),
            "positive_recall_p05": float(np.nanpercentile(positive["recall"].to_numpy(dtype=float), 5)) if len(positive) else np.nan,
            "false_alarm_per_min_mean": float(core["false_alarm_per_min"].mean()),
            "worst_seed_p05": float(np.nanpercentile(config_pass.to_numpy(dtype=float), 5)) if len(config_pass) else np.nan,
            "progression_violations_total": int(core["progression_violations"].sum()),
            "reset_failure_rate_max": float(core["reset_failure_rate"].max()),
        })
    out = pd.DataFrame(rows)
    out["robustness_auc_proxy"] = robustness_auc_proxy(out)
    return out


def robustness_auc_proxy(summary: pd.DataFrame) -> float:
    if summary.empty:
        return np.nan
    return float(summary["family_pass_rate"].mean())


def negative_control_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    controls = frame[frame["case_id"].isin(NEGATIVE_CONTROL_CASES)]
    for (condition, case_id), group in controls.groupby(["noise_condition", "case_id"], sort=False):
        rows.append({
            "noise_condition": condition,
            "case_id": case_id,
            "mean_false_alarm_per_min": float(group["false_alarm_per_min"].mean()),
            "max_false_alarm_per_min": float(group["false_alarm_per_min"].max()),
            "transient_claim_rate": float(group["transient_claim_for_sustained_only"].mean()) if case_id == "sustained_only" else 0.0,
            "interpretation": negative_interpretation(case_id),
        })
    return pd.DataFrame(rows)


def negative_interpretation(case_id: str) -> str:
    if case_id == "sustained_only":
        return "May support sustained beta-state tracking, but must not be claimed as transient onset detection."
    if case_id == "power_shift_no_burst":
        return "Power-shift control for Langford-Wilson style false transient-burst detections."
    return "Strict negative control."


def write_outputs(out_dir: Path, clean: pd.DataFrame, sweep: pd.DataFrame, family: pd.DataFrame, noise_rows: pd.DataFrame, noise_summary: pd.DataFrame, negative: pd.DataFrame, seed_count: int, max_configs: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_dir / "clean_baseline_audit.csv", index=False)
    sweep.to_csv(out_dir / "parameter_sweep_summary.csv", index=False)
    family.to_csv(out_dir / "robust_family_configs.csv", index=False)
    noise_rows.to_csv(out_dir / "noise_robustness_seed_distribution.csv", index=False)
    noise_summary.to_csv(out_dir / "family_robustness_summary.csv", index=False)
    negative.to_csv(out_dir / "negative_control_summary.csv", index=False)
    pd.DataFrame([asdict(row) for row in literature_noise_battery(seed_count)]).to_csv(out_dir / "literature_perturbation_battery.csv", index=False)
    manifest = {
        "audit_status": "topology_surrogate_not_brian2",
        "max_configs": max_configs,
        "seed_count": seed_count,
        "n_clean_rows": int(len(clean)),
        "n_family_configs": int(len(family)),
        "n_noise_rows": int(len(noise_rows)),
        "claim_boundary": "Use these outputs to lock the robustness protocol. Final paper claims require the full Brian2/Brian2-equivalent runner.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "README_robustness_family_audit.md").write_text(readme_text(noise_summary, family), encoding="utf-8")


def readme_text(noise_summary: pd.DataFrame, family: pd.DataFrame) -> str:
    clean_rate = noise_summary.loc[noise_summary["noise_condition"].eq("clean"), "family_pass_rate"]
    clean = float(clean_rate.iloc[0]) if len(clean_rate) else float("nan")
    return f"""# Phase 5_2C Robustness-Family Audit

This output is a deterministic topology surrogate and audit contract, not a Brian2 or
DYNAP-SE1 hardware result.

## Summary

- Robust family size: {len(family)}
- Clean family pass rate: {clean:.3f}
- Claim boundary: robust operating-region protocol only until rerun with the full Brian2 state-machine implementation.

## Tables

- `clean_baseline_audit.csv`
- `parameter_sweep_summary.csv`
- `robust_family_configs.csv`
- `noise_robustness_seed_distribution.csv`
- `family_robustness_summary.csv`
- `negative_control_summary.csv`
- `literature_perturbation_battery.csv`
- `manifest.json`
"""


def run_audit(out_dir: Path, *, family_size: int = 20, noise_seeds: int = 5, max_configs: int | None = 384) -> dict[str, Path]:
    configs = config_grid(max_configs=max_configs)
    clean, sweep = run_clean_sweep(configs)
    family = select_family(sweep, family_size)
    noise_rows, noise_summary, negative = run_noise_family(family, configs, noise_seeds)
    write_outputs(out_dir, clean, sweep, family, noise_rows, noise_summary, negative, noise_seeds, max_configs)
    return {"out_dir": out_dir}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/phase5_2c_robustness_family_audit"))
    parser.add_argument("--family-size", type=int, default=20)
    parser.add_argument("--noise-seeds", type=int, default=5)
    parser.add_argument("--max-configs", type=int, default=384, help="Subsampled clean config count; use 0 for the full grid.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    max_configs = None if args.max_configs == 0 else args.max_configs
    run_audit(args.out_dir, family_size=args.family_size, noise_seeds=args.noise_seeds, max_configs=max_configs)
    print(f"wrote robustness-family audit to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

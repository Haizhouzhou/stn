"""Synthetic STN-like beta-burst traces for Phase 4 state-machine validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import signal

from stnbeta.phase4.config import load_yaml

DEFAULT_SFREQ_HZ = 1000.0
DURATION_BUCKETS_S = (0.10, 0.20, 0.35, 0.50)


@dataclass(frozen=True)
class BurstSpec:
    """One embedded synthetic beta burst."""

    onset_s: float
    duration_s: float
    amplitude: float
    center_hz: float
    freq_drift_hz: float = 0.0
    ramp_s: float = 0.02
    interruption_offset_s: float | None = None
    interruption_s: float = 0.0
    label: str = "beta_burst"


@dataclass(frozen=True)
class SyntheticTraceConfig:
    """Configuration for a single synthetic trace."""

    name: str
    seed: int
    duration_s: float = 6.0
    sfreq_hz: float = DEFAULT_SFREQ_HZ
    noise_std: float = 0.18
    drift_std: float = 0.05
    pink_noise_std: float = 0.10
    bursts: tuple[BurstSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SyntheticTrace:
    """Generated signal plus ground-truth annotations."""

    name: str
    seed: int
    sfreq_hz: float
    time_s: np.ndarray
    signal: np.ndarray
    annotations: pd.DataFrame
    components: dict[str, np.ndarray]


def duration_bucket_index(duration_s: float) -> int:
    """Return the highest duration bucket reached by *duration_s*."""
    bucket = -1
    for index, threshold_s in enumerate(DURATION_BUCKETS_S):
        if duration_s >= threshold_s:
            bucket = index
    return bucket


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / DEFAULT_SFREQ_HZ)
    freqs[0] = 1.0
    amplitude = 1.0 / np.sqrt(freqs)
    amplitude[0] = 0.0
    phases = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
    spectrum = amplitude * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n_samples)
    return noise / (noise.std() + 1e-12)


def _low_frequency_drift(n_samples: int, sfreq_hz: float, rng: np.random.Generator) -> np.ndarray:
    sos = signal.butter(2, 1.5, btype="lowpass", fs=sfreq_hz, output="sos")
    drift = signal.sosfiltfilt(sos, rng.standard_normal(n_samples))
    return drift / (drift.std() + 1e-12)


def _burst_waveform(
    spec: BurstSpec,
    time_s: np.ndarray,
    sfreq_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    burst = np.zeros_like(time_s)
    envelope = np.zeros_like(time_s)
    start = int(round(spec.onset_s * sfreq_hz))
    stop = int(round((spec.onset_s + spec.duration_s) * sfreq_hz))
    stop = min(stop, len(time_s))
    if stop <= start:
        return burst, envelope

    local_t = np.arange(stop - start) / sfreq_hz
    freqs = np.linspace(
        spec.center_hz - spec.freq_drift_hz / 2.0,
        spec.center_hz + spec.freq_drift_hz / 2.0,
        stop - start,
    )
    phase = 2.0 * np.pi * np.cumsum(freqs) / sfreq_hz
    ramp_n = max(1, int(round(spec.ramp_s * sfreq_hz)))
    window = np.ones(stop - start, dtype=float)
    if 2 * ramp_n < len(window):
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp_n))
        window[:ramp_n] = ramp
        window[-ramp_n:] = ramp[::-1]

    if spec.interruption_s > 0.0:
        interruption_offset_s = spec.interruption_offset_s
        if interruption_offset_s is None:
            interruption_offset_s = spec.duration_s / 2.0 - spec.interruption_s / 2.0
        gap_start = int(round(interruption_offset_s * sfreq_hz))
        gap_stop = int(round((interruption_offset_s + spec.interruption_s) * sfreq_hz))
        gap_start = max(0, gap_start)
        gap_stop = min(len(window), gap_stop)
        window[gap_start:gap_stop] = 0.0

    envelope[start:stop] = spec.amplitude * window
    burst[start:stop] = envelope[start:stop] * np.sin(phase)
    return burst, envelope


def generate_trace(config: SyntheticTraceConfig) -> SyntheticTrace:
    """Generate one deterministic synthetic trace from *config*."""
    rng = np.random.default_rng(config.seed)
    n_samples = int(round(config.duration_s * config.sfreq_hz))
    time_s = np.arange(n_samples, dtype=float) / config.sfreq_hz

    white = rng.standard_normal(n_samples)
    pink = _pink_noise(n_samples, rng)
    drift = _low_frequency_drift(n_samples, config.sfreq_hz, rng)
    noise = (
        config.noise_std * white
        + config.pink_noise_std * pink
        + config.drift_std * drift
    )

    burst_signal = np.zeros(n_samples, dtype=float)
    burst_envelope = np.zeros(n_samples, dtype=float)
    rows: list[dict[str, Any]] = []
    for burst_index, burst_spec in enumerate(config.bursts):
        waveform, envelope = _burst_waveform(burst_spec, time_s, config.sfreq_hz)
        burst_signal += waveform
        burst_envelope += envelope
        rows.append(
            {
                "trace_name": config.name,
                "burst_index": burst_index,
                "label": burst_spec.label,
                "onset_s": burst_spec.onset_s,
                "offset_s": burst_spec.onset_s + burst_spec.duration_s,
                "duration_s": burst_spec.duration_s,
                "duration_ms": burst_spec.duration_s * 1000.0,
                "amplitude": burst_spec.amplitude,
                "center_hz": burst_spec.center_hz,
                "freq_drift_hz": burst_spec.freq_drift_hz,
                "interruption_s": burst_spec.interruption_s,
                "expected_bucket_index": duration_bucket_index(burst_spec.duration_s),
            }
        )

    annotations = pd.DataFrame(rows)
    if annotations.empty:
        annotations = pd.DataFrame(
            columns=[
                "trace_name",
                "burst_index",
                "label",
                "onset_s",
                "offset_s",
                "duration_s",
                "duration_ms",
                "amplitude",
                "center_hz",
                "freq_drift_hz",
                "interruption_s",
                "expected_bucket_index",
            ]
        )

    signal_out = (noise + burst_signal).astype(np.float32)
    return SyntheticTrace(
        name=config.name,
        seed=config.seed,
        sfreq_hz=config.sfreq_hz,
        time_s=time_s.astype(np.float32),
        signal=signal_out,
        annotations=annotations,
        components={
            "noise": noise.astype(np.float32),
            "burst_signal": burst_signal.astype(np.float32),
            "burst_envelope": burst_envelope.astype(np.float32),
        },
    )


def _coerce_bursts(items: Iterable[Mapping[str, Any]]) -> tuple[BurstSpec, ...]:
    return tuple(BurstSpec(**dict(item)) for item in items)


def _trace_config_from_mapping(base: Mapping[str, Any], item: Mapping[str, Any]) -> SyntheticTraceConfig:
    merged = dict(base)
    merged.update(dict(item))
    bursts = _coerce_bursts(merged.pop("bursts", []))
    return SyntheticTraceConfig(bursts=bursts, **merged)


def generate_trace_suite(config: Mapping[str, Any] | str | Path) -> list[SyntheticTrace]:
    """Generate a deterministic benchmark suite from a mapping or YAML path."""
    if isinstance(config, (str, Path)):
        config = load_yaml(config)
    base = dict(config.get("base", {}))
    traces = [
        generate_trace(_trace_config_from_mapping(base, item))
        for item in config.get("cases", [])
    ]
    return traces

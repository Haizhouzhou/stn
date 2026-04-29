"""Deterministic Phase 5 synthetic validation suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SFREQ_HZ = 1000.0
DEFAULT_BAND_NAMES = (
    "boundary_low",
    "beta_1",
    "beta_2",
    "beta_3",
    "beta_4",
    "boundary_high",
)
DEFAULT_BAND_ROLES = (
    "boundary",
    "beta",
    "beta",
    "beta",
    "beta",
    "boundary",
)
STATE_NAMES = ("D0", "D1", "D2", "D3", "D4")
STATE_LOWER_BOUNDS_MS = (0.0, 50.0, 100.0, 200.0, 400.0)


@dataclass(frozen=True)
class SyntheticDurationCase:
    """One deterministic synthetic case for Phase 5 validation."""

    name: str
    level: str
    sfreq_hz: float
    time_s: np.ndarray
    annotations: pd.DataFrame
    band_names: tuple[str, ...]
    band_roles: tuple[str, ...]
    direct_currents: np.ndarray | None
    signal: np.ndarray | None
    interruption_policy: str = "bridge_short_gaps"
    note: str = ""


def expected_bucket_for_duration_ms(duration_ms: float) -> int:
    """Return the highest duration bucket reached by *duration_ms*."""
    bucket = 0
    for index, lower_bound_ms in enumerate(STATE_LOWER_BOUNDS_MS):
        if duration_ms >= lower_bound_ms:
            bucket = index
    return bucket


def _base_case_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "no_burst",
            "duration_s": 2.0,
            "segments": [],
            "events": [],
            "note": "Negative control.",
        },
        {
            "name": "short_40ms",
            "duration_s": 2.0,
            "segments": [(0.55, 0.04, 1.35, 1.35)],
            "events": [(0.55, 0.04, "short_40ms")],
        },
        {
            "name": "near_threshold_90ms",
            "duration_s": 2.0,
            "segments": [(0.55, 0.09, 0.90, 0.90)],
            "events": [(0.55, 0.09, "near_threshold_90ms")],
        },
        {
            "name": "threshold_crossing_120ms",
            "duration_s": 2.0,
            "segments": [(0.55, 0.12, 1.35, 1.35)],
            "events": [(0.55, 0.12, "threshold_crossing_120ms")],
        },
        {
            "name": "sustained_200ms",
            "duration_s": 2.0,
            "segments": [(0.55, 0.20, 1.42, 1.34)],
            "events": [(0.55, 0.20, "sustained_200ms")],
        },
        {
            "name": "long_400ms_plus",
            "duration_s": 2.9,
            "segments": [(0.55, 0.90, 2.25, 2.00)],
            "events": [(0.55, 0.90, "long_400ms_plus")],
        },
        {
            "name": "two_bursts_with_quiet_gap",
            "duration_s": 2.4,
            "segments": [(0.45, 0.12, 1.32, 1.32), (1.25, 0.24, 1.35, 1.25)],
            "events": [(0.45, 0.12, "burst_a"), (1.25, 0.24, "burst_b")],
        },
        {
            "name": "interrupted_burst_60_on_20_off_60_on",
            "duration_s": 2.2,
            "segments": [(0.55, 0.06, 1.35, 1.35), (0.63, 0.06, 1.30, 1.20)],
            "events": [(0.55, 0.14, "bridged_interrupt")],
            "note": "Frozen design choice: a 20 ms quiet gap is treated as a reset-sized interruption.",
        },
        {
            "name": "decaying_burst",
            "duration_s": 2.2,
            "segments": [(0.55, 0.22, 1.55, 0.80)],
            "events": [(0.55, 0.22, "decaying_burst")],
        },
        {
            "name": "noisy_jittered_burst",
            "duration_s": 2.2,
            "segments": [(0.55, 0.16, 1.30, 1.10)],
            "events": [(0.55, 0.16, "noisy_jittered_burst")],
            "note": "Adds within-burst amplitude jitter and stochastic dips.",
        },
    ]


def _event_table(events: list[tuple[float, float, str]]) -> pd.DataFrame:
    rows = []
    for burst_index, (onset_s, duration_s, label) in enumerate(events):
        duration_ms = duration_s * 1000.0
        expected_bucket = expected_bucket_for_duration_ms(duration_ms)
        rows.append(
            {
                "burst_index": burst_index,
                "label": label,
                "onset_s": onset_s,
                "offset_s": onset_s + duration_s,
                "duration_s": duration_s,
                "duration_ms": duration_ms,
                "expected_bucket_index": expected_bucket,
                "expected_state": STATE_NAMES[expected_bucket],
                "expect_readout": expected_bucket >= 2,
            }
        )
    return pd.DataFrame(rows)


def _piecewise_envelope(
    n_samples: int,
    sfreq_hz: float,
    segments: list[tuple[float, float, float, float]],
) -> np.ndarray:
    envelope = np.zeros(n_samples, dtype=np.float32)
    for onset_s, duration_s, amp_start, amp_end in segments:
        start = max(0, int(round(onset_s * sfreq_hz)))
        stop = min(n_samples, int(round((onset_s + duration_s) * sfreq_hz)))
        if stop <= start:
            continue
        values = np.linspace(amp_start, amp_end, stop - start, dtype=np.float32)
        ramp = max(1, int(round(0.01 * sfreq_hz)))
        if 2 * ramp < len(values):
            window = np.ones(len(values), dtype=np.float32)
            taper = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp, dtype=np.float32))
            window[:ramp] = taper
            window[-ramp:] = taper[::-1]
            values *= window
        envelope[start:stop] += values
    return envelope


def _direct_currents_from_envelope(
    envelope: np.ndarray,
    *,
    seed: int,
    case_name: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = len(envelope)
    noise = rng.normal(0.0, 0.015, size=(len(DEFAULT_BAND_NAMES), n_samples)).astype(np.float32)
    beta_scales = np.asarray([0.0, 1.00, 0.92, 1.08, 0.84, 0.0], dtype=np.float32)[:, None]
    direct = 0.04 + noise
    direct += beta_scales * envelope[None, :]

    if case_name == "noisy_jittered_burst":
        jitter = rng.normal(1.0, 0.18, size=(4, n_samples)).astype(np.float32)
        gap_mask = rng.random(n_samples) < 0.08
        jitter[:, gap_mask] *= 0.45
        direct[1:5] *= jitter
    elif case_name == "decaying_burst":
        direct[1:5] *= np.linspace(1.0, 0.85, n_samples, dtype=np.float32)[None, :]

    direct[0] += 0.08 * np.maximum(0.0, 0.15 - envelope)
    direct[-1] += 0.06 * np.maximum(0.0, 0.12 - envelope)
    return np.clip(direct, 0.0, None).astype(np.float32)


def _lfp_signal_from_envelope(
    envelope: np.ndarray,
    *,
    sfreq_hz: float,
    seed: int,
    case_name: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time_s = np.arange(len(envelope), dtype=np.float32) / sfreq_hz
    freq = 20.0 + 2.5 * np.sin(2.0 * np.pi * 0.35 * time_s)
    phase = 2.0 * np.pi * np.cumsum(freq) / sfreq_hz
    carrier = np.sin(phase) + 0.35 * np.sin(phase * 0.83 + 0.5)

    white = rng.normal(0.0, 0.18, len(envelope)).astype(np.float32)
    low = rng.normal(0.0, 1.0, len(envelope)).astype(np.float32)
    low = np.convolve(low, np.ones(31, dtype=np.float32) / 31.0, mode="same")
    low = 0.06 * (low / (np.std(low) + 1e-9))
    burst = envelope * carrier.astype(np.float32)

    if case_name == "noisy_jittered_burst":
        burst *= rng.normal(1.0, 0.22, len(envelope)).astype(np.float32)
    elif case_name == "decaying_burst":
        burst *= np.linspace(1.0, 0.8, len(envelope), dtype=np.float32)

    return (white + low + burst).astype(np.float32)


def _build_cases(level: str) -> list[SyntheticDurationCase]:
    cases: list[SyntheticDurationCase] = []
    for seed_offset, spec in enumerate(_base_case_specs(), start=1):
        sfreq_hz = DEFAULT_SFREQ_HZ
        n_samples = int(round(spec["duration_s"] * sfreq_hz))
        time_s = np.arange(n_samples, dtype=np.float32) / sfreq_hz
        envelope = _piecewise_envelope(n_samples, sfreq_hz, spec["segments"])
        annotations = _event_table(spec["events"])
        direct_currents = _direct_currents_from_envelope(
            envelope,
            seed=3100 + seed_offset,
            case_name=spec["name"],
        )
        signal = _lfp_signal_from_envelope(
            envelope,
            sfreq_hz=sfreq_hz,
            seed=9100 + seed_offset,
            case_name=spec["name"],
        )
        cases.append(
            SyntheticDurationCase(
                name=spec["name"],
                level=level,
                sfreq_hz=sfreq_hz,
                time_s=time_s,
                annotations=annotations,
                band_names=DEFAULT_BAND_NAMES,
                band_roles=DEFAULT_BAND_ROLES,
                direct_currents=direct_currents if level == "topology" else None,
                signal=signal if level == "end_to_end" else None,
                interruption_policy="bridge_short_gaps",
                note=spec.get("note", ""),
            )
        )
    return cases


def generate_topology_suite() -> list[SyntheticDurationCase]:
    """Return the Phase 5 topology-only synthetic suite."""
    return _build_cases("topology")


def generate_end_to_end_suite() -> list[SyntheticDurationCase]:
    """Return the Phase 5 end-to-end synthetic suite through the frozen front end."""
    return _build_cases("end_to_end")

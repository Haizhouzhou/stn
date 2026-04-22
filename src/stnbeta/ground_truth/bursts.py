"""Core Tinkhauser-style beta-burst algorithms."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal

logger = logging.getLogger(__name__)


def bandpass(raw, fmin: float, fmax: float, picks: list[str]) -> np.ndarray:
    """Zero-phase Butterworth order 4, applied to the named picks. Returns ndarray (n_ch, n_samples)."""
    data = raw.get_data(picks=picks)
    sfreq = raw.info["sfreq"]
    nyq = sfreq / 2.0
    b, a = scipy.signal.butter(4, [fmin / nyq, fmax / nyq], btype="bandpass")
    return scipy.signal.filtfilt(b, a, data, axis=1)


def hilbert_envelope(x: np.ndarray, sfreq: float, lowpass_hz: float = 5.0) -> np.ndarray:
    """Analytic amplitude, smoothed with an order-2 Butterworth at lowpass_hz. Per-channel."""
    analytic = scipy.signal.hilbert(x, axis=1)
    env = np.abs(analytic)
    nyq = sfreq / 2.0
    b, a = scipy.signal.butter(2, lowpass_hz / nyq, btype="low")
    return scipy.signal.filtfilt(b, a, env, axis=1)


def burst_threshold(
    env: np.ndarray,
    method: str = "percentile_75",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """One threshold scalar per channel, computed on mask-selected samples (e.g., Rest MedOff).
    method='percentile_75' is Tinkhauser 2017; also support 'percentile_80', 'fixed_sigma_1.5'.
    """
    data = env[:, mask] if mask is not None else env
    if method == "percentile_75":
        return np.percentile(data, 75, axis=1)
    elif method == "percentile_80":
        return np.percentile(data, 80, axis=1)
    elif method == "fixed_sigma_1.5":
        return np.mean(data, axis=1) + 1.5 * np.std(data, axis=1)
    else:
        raise ValueError(f"Unknown threshold method: {method!r}")


def label_bursts(
    env: np.ndarray,
    threshold: np.ndarray,
    sfreq: float,
    min_duration_ms: float = 100,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (is_burst per sample, events dataframe with columns
    [channel, onset_s, offset_s, duration_ms, peak_amp, mean_amp]).
    """
    min_samples = max(1, int(np.round(min_duration_ms / 1000.0 * sfreq)))
    n_ch, n_samples = env.shape
    is_burst = np.zeros((n_ch, n_samples), dtype=bool)
    rows: list[dict] = []

    for ch_idx in range(n_ch):
        above = env[ch_idx] > threshold[ch_idx]
        labeled, n_features = scipy.ndimage.label(above)
        for burst_id in range(1, n_features + 1):
            indices = np.where(labeled == burst_id)[0]
            if len(indices) >= min_samples:
                is_burst[ch_idx, indices] = True
                rows.append(
                    {
                        "channel": ch_idx,
                        "onset_s": float(indices[0] / sfreq),
                        "offset_s": float((indices[-1] + 1) / sfreq),
                        "duration_ms": float(len(indices) / sfreq * 1000.0),
                        "peak_amp": float(env[ch_idx, indices].max()),
                        "mean_amp": float(env[ch_idx, indices].mean()),
                    }
                )

    if rows:
        events = pd.DataFrame(rows)
    else:
        events = pd.DataFrame(
            columns=["channel", "onset_s", "offset_s", "duration_ms", "peak_amp", "mean_amp"]
        )
    return is_burst, events


def burst_stats(events: pd.DataFrame, total_duration_s: float) -> dict:
    """Rate (bursts/min), mean_duration_ms, p25/p50/p75/p90/p99 of duration, mean_peak_amp."""
    if len(events) == 0:
        return {
            "n_bursts": 0,
            "rate_per_min": 0.0,
            "mean_duration_ms": 0.0,
            "p25_duration_ms": 0.0,
            "p50_duration_ms": 0.0,
            "p75_duration_ms": 0.0,
            "p90_duration_ms": 0.0,
            "p99_duration_ms": 0.0,
            "mean_peak_amp": 0.0,
        }
    dur = events["duration_ms"].values
    return {
        "n_bursts": int(len(events)),
        "rate_per_min": float(len(events) / max(total_duration_s, 1e-6) * 60.0),
        "mean_duration_ms": float(dur.mean()),
        "p25_duration_ms": float(np.percentile(dur, 25)),
        "p50_duration_ms": float(np.percentile(dur, 50)),
        "p75_duration_ms": float(np.percentile(dur, 75)),
        "p90_duration_ms": float(np.percentile(dur, 90)),
        "p99_duration_ms": float(np.percentile(dur, 99)),
        "mean_peak_amp": float(events["peak_amp"].mean()),
    }

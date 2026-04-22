"""Unit tests for is_beta_active_channel() in contact_selection.py.

All tests use synthetic signals — no file I/O, safe on login node.
"""

import mne
import numpy as np
import pytest

from stnbeta.ground_truth.contact_selection import is_beta_active_channel


def _make_raw(signal: np.ndarray, sfreq: float, ch_name: str = "LFP-left-01") -> mne.io.RawArray:
    info = mne.create_info([ch_name], sfreq, ch_types=["eeg"])
    return mne.io.RawArray(signal[np.newaxis, :], info, verbose=False)


def _pink_noise(n_samples: int, sfreq: float, rng: np.random.Generator) -> np.ndarray:
    """1/f pink noise via inverse FFT with 1/f amplitude spectrum."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    freqs[0] = 1.0  # avoid division by zero at DC
    amplitude = 1.0 / np.sqrt(freqs)
    amplitude[0] = 0.0  # zero DC
    phases = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
    spectrum = amplitude * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n_samples)
    return noise / (noise.std() + 1e-12) * 1e-6


def test_synthetic_pure_beta_is_active():
    """120 s of 20 Hz sinusoid + 1/f noise → active=True, peak near 20 Hz."""
    sfreq = 1000.0
    duration = 120.0
    n = int(duration * sfreq)
    t = np.arange(n) / sfreq
    rng = np.random.default_rng(42)

    signal = np.sin(2.0 * np.pi * 20.0 * t) * 5e-6 + _pink_noise(n, sfreq, rng)
    raw = _make_raw(signal, sfreq)
    mask = np.ones(n, dtype=bool)

    result = is_beta_active_channel(raw, "LFP-left-01", mask)

    assert result["active"] is True, f"Expected active=True, got reason={result['reason']}"
    assert result["peak_freq_hz"] is not None
    assert 13.0 <= result["peak_freq_hz"] <= 35.0, (
        f"Peak freq {result['peak_freq_hz']} Hz not in beta range"
    )
    assert result["peak_power_db"] is not None
    assert result["peak_power_db"] >= 3.0


def test_pink_noise_only_is_inactive():
    """Pure 1/f noise with no oscillatory component → active=False."""
    sfreq = 1000.0
    duration = 120.0
    n = int(duration * sfreq)
    rng = np.random.default_rng(0)

    signal = _pink_noise(n, sfreq, rng)
    raw = _make_raw(signal, sfreq)
    mask = np.ones(n, dtype=bool)

    result = is_beta_active_channel(raw, "LFP-left-01", mask)

    assert result["active"] is False, (
        f"Expected active=False for pure pink noise, got peak at {result['peak_freq_hz']} Hz "
        f"with {result['peak_power_db']} dB"
    )


def test_insufficient_duration_returns_specific_reason():
    """Rest mask covering only 30 s → reason='insufficient_rest_duration'."""
    sfreq = 1000.0
    duration = 120.0
    n = int(duration * sfreq)
    t = np.arange(n) / sfreq

    # Signal that would pass if there were enough data
    signal = np.sin(2.0 * np.pi * 20.0 * t) * 5e-6
    raw = _make_raw(signal, sfreq)

    # Only 30 seconds of rest mask — below min_rest_duration_s=60
    mask = np.zeros(n, dtype=bool)
    mask[: int(30 * sfreq)] = True

    result = is_beta_active_channel(raw, "LFP-left-01", mask)

    assert result["active"] is False
    assert result["reason"] == "insufficient_rest_duration", (
        f"Expected 'insufficient_rest_duration', got '{result['reason']}'"
    )

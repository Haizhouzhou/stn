"""Functional contact selection: does a bipolar channel exhibit a detectable beta peak?

Reference: Tinkhauser 2017, Neumann 2016, Lofredi 2019 — require each bipolar channel
to show a beta peak above 1/f aperiodic background on MedOff Rest before including it
in burst analysis. Rassoulou 2024 usage notes recommend functional criteria because
anatomical contact positions within STN are not provided.
"""

from __future__ import annotations

import logging
import warnings
from typing import TypedDict

import mne
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

try:
    from specparam import SpectralModel as _FOOOFModel
except ImportError:
    from fooof import FOOOF as _FOOOFModel  # type: ignore[no-redef]


class BetaActiveResult(TypedDict):
    active: bool
    peak_freq_hz: float | None
    peak_power_db: float | None
    aperiodic_offset: float | None
    aperiodic_exponent: float | None
    reason: str


def _get_peaks(fm) -> np.ndarray | None:
    """Extract (n_peaks, 3) [CF, PW, BW] array from fitted model; None if no peaks."""
    # specparam v2: fm.results.get_results().peak_fit
    if hasattr(fm, "results") and hasattr(fm.results, "get_results"):
        try:
            fit_res = fm.results.get_results()
            peaks = fit_res.peak_fit
            if peaks is not None and np.asarray(peaks).size > 0:
                return np.atleast_2d(peaks)
            return None
        except Exception:
            pass
    # fooof v1 / specparam v1 via get_params
    try:
        peaks = fm.get_params("peak_params")
        if peaks is not None and hasattr(peaks, "__len__") and len(peaks) > 0:
            return np.atleast_2d(peaks)
    except Exception:
        pass
    # fooof v1 direct attribute
    if hasattr(fm, "peak_params_"):
        peaks = fm.peak_params_
        if peaks is not None and np.asarray(peaks).size > 0:
            return np.atleast_2d(peaks)
    return None


def _get_aperiodic(fm) -> tuple[float | None, float | None]:
    """Return (offset, exponent) from a fitted model, or (None, None) on failure."""
    # specparam v2
    if hasattr(fm, "results") and hasattr(fm.results, "get_results"):
        try:
            ap = fm.results.get_results().aperiodic_fit
            if ap is not None and len(ap) >= 2:
                return float(ap[0]), float(ap[1])
        except Exception:
            pass
    # fooof v1 via get_params
    try:
        ap = fm.get_params("aperiodic_params")
        if ap is not None and hasattr(ap, "__len__") and len(ap) >= 2:
            return float(ap[0]), float(ap[1])
    except Exception:
        pass
    # fooof v1 direct attribute
    if hasattr(fm, "aperiodic_params_"):
        ap = fm.aperiodic_params_
        if ap is not None and len(ap) >= 2:
            return float(ap[0]), float(ap[1])
    return None, None


def is_beta_active_channel(
    raw: mne.io.BaseRaw,
    channel: str,
    rest_mask: np.ndarray,
    min_peak_power_db: float = 3.0,
    beta_range: tuple[float, float] = (13.0, 35.0),
    fit_range: tuple[float, float] = (2.0, 45.0),
    min_rest_duration_s: float = 60.0,
) -> BetaActiveResult:
    """Return whether *channel* has a detectable beta peak on MedOff Rest.

    rest_mask: boolean mask over raw.times, True where MedOff Rest is active
               and no BAD_* annotation overlaps (get_epoch_mask output).
    min_peak_power_db: peak height above the aperiodic baseline in dB
                       (= log10-power × 10 ≥ this value → PW ≥ 0.3 for default 3.0 dB).
    """
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data(picks=[channel])[0, rest_mask]

    rest_s = len(data) / sfreq
    if rest_s < min_rest_duration_s:
        return BetaActiveResult(
            active=False,
            peak_freq_hz=None,
            peak_power_db=None,
            aperiodic_offset=None,
            aperiodic_exponent=None,
            reason="insufficient_rest_duration",
        )

    nperseg = min(len(data), int(4 * sfreq))
    noverlap = int(2 * sfreq)
    freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

    try:
        fm = _FOOOFModel(
            peak_width_limits=(2, 12),
            max_n_peaks=6,
            min_peak_height=0.0,
            aperiodic_mode="fixed",
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fm.fit(freqs, psd, freq_range=list(fit_range))
    except Exception as err:
        return BetaActiveResult(
            active=False,
            peak_freq_hz=None,
            peak_power_db=None,
            aperiodic_offset=None,
            aperiodic_exponent=None,
            reason=f"fooof_failed:{err}",
        )

    ap_offset, ap_exponent = _get_aperiodic(fm)
    peaks_arr = _get_peaks(fm)

    if peaks_arr is None:
        return BetaActiveResult(
            active=False,
            peak_freq_hz=None,
            peak_power_db=None,
            aperiodic_offset=ap_offset,
            aperiodic_exponent=ap_exponent,
            reason="no_beta_peak_above_threshold",
        )

    # peaks_arr: (n_peaks, 3) — [CF, PW, BW]
    # PW is log10-power above the aperiodic fit; PW * 10 converts to dB
    for peak in peaks_arr:
        cf, pw = float(peak[0]), float(peak[1])
        if beta_range[0] <= cf <= beta_range[1] and pw * 10.0 >= min_peak_power_db:
            return BetaActiveResult(
                active=True,
                peak_freq_hz=round(cf, 2),
                peak_power_db=round(pw * 10.0, 2),
                aperiodic_offset=ap_offset,
                aperiodic_exponent=ap_exponent,
                reason="beta_active",
            )

    return BetaActiveResult(
        active=False,
        peak_freq_hz=None,
        peak_power_db=None,
        aperiodic_offset=ap_offset,
        aperiodic_exponent=ap_exponent,
        reason="no_beta_peak_above_threshold",
    )

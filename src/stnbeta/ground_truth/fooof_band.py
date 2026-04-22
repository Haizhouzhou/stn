"""Individualized beta-band fitting using specparam (formerly fooof).

specparam v2 API: fm.results.get_results().peak_fit → (n_peaks, 3) array [CF, PW, BW]
fooof v1 API:     fm.get_params('peak_params')       → same shape
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


def _get_peaks(fm) -> np.ndarray | None:
    """Extract (n_peaks, 3) [CF, PW, BW] array from a fitted model regardless of version."""
    # specparam v2+
    if hasattr(fm, "results") and hasattr(fm.results, "get_results"):
        try:
            fit_res = fm.results.get_results()
            peaks = fit_res.peak_fit
            if peaks is not None and len(peaks) > 0:
                return np.atleast_2d(peaks)
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
        if peaks is not None and len(peaks) > 0:
            return np.atleast_2d(peaks)

    return None


def fit_individual_beta(
    raw,
    channel: str,
    mask: np.ndarray | None,
    default: tuple[float, float] = (13.0, 30.0),
) -> tuple[float, float]:
    """Fit FOOOF (or specparam) on the PSD from masked samples of one channel.
    Return subject-specific beta peak center ± 3 Hz, clamped to [12, 35].
    If fit fails or peak power < 0.05 above aperiodic, return default and log a warning.
    """
    try:
        import specparam as sp

        ModelClass = sp.SpectralModel
    except ImportError:
        try:
            import fooof as sp

            ModelClass = sp.FOOOF
        except ImportError:
            logger.warning("Neither specparam nor fooof installed; using default %s", default)
            return default

    data = raw.get_data(picks=[channel])[0]
    sfreq = raw.info["sfreq"]

    if mask is not None and mask.any():
        data = data[mask]

    if len(data) < int(sfreq * 2):
        logger.warning(
            "Too little data for %s (%d samples), using default %s", channel, len(data), default
        )
        return default

    freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=int(sfreq * 4))

    try:
        fm = ModelClass(
            peak_width_limits=[1.0, 8.0],
            min_peak_height=0.05,
            aperiodic_mode="fixed",
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fm.fit(freqs, psd, freq_range=[3.0, 40.0])
    except Exception as exc:
        logger.warning("Spectral fit failed for %s: %s — using default %s", channel, exc, default)
        return default

    peaks_arr = _get_peaks(fm)
    if peaks_arr is None:
        logger.warning("No peaks found for %s, using default %s", channel, default)
        return default

    # peaks_arr shape: (n_peaks, 3) — [CF, PW, BW]
    beta_mask = (peaks_arr[:, 0] >= 12.0) & (peaks_arr[:, 0] <= 35.0)
    if not beta_mask.any():
        logger.warning("No beta-range peak for %s, using default %s", channel, default)
        return default

    beta_peaks = peaks_arr[beta_mask]
    strongest = beta_peaks[np.argmax(beta_peaks[:, 1])]
    cf, pw = float(strongest[0]), float(strongest[1])

    if pw < 0.05:
        logger.warning(
            "Beta peak power %.4f < 0.05 for %s, using default %s", pw, channel, default
        )
        return default

    lo = max(12.0, cf - 3.0)
    hi = min(35.0, cf + 3.0)
    logger.debug("Individual beta for %s: CF=%.1f Hz → band [%.1f, %.1f]", channel, cf, lo, hi)
    return (lo, hi)

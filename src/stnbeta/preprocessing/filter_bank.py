"""Configurable beta-band filter bank for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mne
import numpy as np
from scipy import signal

from stnbeta.phase4.config import load_yaml


@dataclass(frozen=True)
class BandSpec:
    """One band-pass filter definition."""

    name: str
    fmin_hz: float
    fmax_hz: float
    role: str = "beta"


@dataclass(frozen=True)
class FilterBankConfig:
    """Full filter bank definition."""

    bands: tuple[BandSpec, ...]
    order: int = 4


def _config_from_mapping(config: Mapping[str, Any]) -> FilterBankConfig:
    bands = tuple(BandSpec(**dict(item)) for item in config["bands"])
    return FilterBankConfig(bands=bands, order=int(config.get("order", 4)))


def load_filter_bank_config(path: str | Path) -> FilterBankConfig:
    """Load a filter bank config from YAML."""
    return _config_from_mapping(load_yaml(path))


def default_filter_bank_config() -> FilterBankConfig:
    """Return the repo default filter bank."""
    return load_filter_bank_config(Path(__file__).resolve().parents[3] / "configs" / "filter_bank.yaml")


def apply_filter_bank(
    data: np.ndarray,
    sfreq_hz: float,
    config: FilterBankConfig,
    *,
    causal: bool = False,
) -> dict[str, np.ndarray]:
    """Apply the configured band-pass bank to 1-D or 2-D data."""
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D input, got shape {array.shape}")

    outputs: dict[str, np.ndarray] = {}
    for band in config.bands:
        sos = signal.butter(
            config.order,
            [band.fmin_hz, band.fmax_hz],
            btype="bandpass",
            fs=sfreq_hz,
            output="sos",
        )
        filtered = signal.sosfilt(sos, array, axis=-1) if causal else signal.sosfiltfilt(sos, array, axis=-1)
        outputs[band.name] = filtered.astype(np.float32)
    return outputs


def apply_filter_bank_to_raw(
    raw: mne.io.BaseRaw,
    picks: list[str],
    config: FilterBankConfig,
    *,
    causal: bool = False,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Apply the filter bank to selected channels from an MNE Raw."""
    data = raw.get_data(picks=picks)
    return picks, apply_filter_bank(data, float(raw.info["sfreq"]), config, causal=causal)

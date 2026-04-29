"""LI&F encoder utilities for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from stnbeta.phase4.config import load_yaml
from stnbeta.preprocessing.filter_bank import (
    FilterBankConfig,
    apply_filter_bank,
    default_filter_bank_config,
    load_filter_bank_config,
)
from stnbeta.preprocessing.rectify_amplify import (
    RectifyAmplifyConfig,
    rectify_and_amplify,
)


def _lazy_brian(import_cuda: bool = False):
    import brian2 as b2

    if import_cuda:
        import brian2cuda  # noqa: F401

    return b2


@dataclass(frozen=True)
class LIFEncoderConfig:
    """Dimensionless LI&F encoder parameters."""

    dt_ms: float = 1.0
    tau_ms: float = 15.0
    refractory_ms: float = 4.0
    threshold: float = 1.0
    reset: float = 0.0
    bias: float = 0.0
    gain: float = 1.0


@dataclass(frozen=True)
class LIFEncodingResult:
    """Encoder spikes and optional voltage traces."""

    duration_s: float
    spike_times_s: np.ndarray
    spike_indices: np.ndarray
    voltages: np.ndarray | None


def load_lif_encoder_config(path: str | Path | Mapping[str, Any]) -> LIFEncoderConfig:
    """Load encoder config from YAML or a mapping."""
    if isinstance(path, Mapping):
        mapping = dict(path)
    else:
        mapping = load_yaml(path)
    if "lif" in mapping:
        mapping = dict(mapping["lif"])

    allowed = {field.name for field in fields(LIFEncoderConfig)}
    filtered = {key: value for key, value in mapping.items() if key in allowed}
    return LIFEncoderConfig(**filtered)


def band_currents_from_signal(
    signal_1d: np.ndarray,
    sfreq_hz: float,
    *,
    filter_bank_config: FilterBankConfig | None = None,
    rectify_config: RectifyAmplifyConfig | None = None,
    causal: bool = False,
) -> tuple[list[str], list[str], np.ndarray]:
    """Return rectified band currents for one 1-D signal."""
    if filter_bank_config is None:
        filter_bank_config = default_filter_bank_config()
    if rectify_config is None:
        rectify_config = RectifyAmplifyConfig()

    filtered = apply_filter_bank(signal_1d, sfreq_hz, filter_bank_config, causal=causal)
    band_names = [band.name for band in filter_bank_config.bands]
    band_roles = [band.role for band in filter_bank_config.bands]
    currents = currents_from_filtered_bands(
        filtered,
        band_names,
        rectify_config=rectify_config,
        sfreq_hz=sfreq_hz,
        causal=causal,
    )
    return band_names, band_roles, currents


def currents_from_filtered_bands(
    filtered: Mapping[str, np.ndarray],
    band_names: list[str],
    *,
    rectify_config: RectifyAmplifyConfig,
    sfreq_hz: float,
    causal: bool = False,
) -> np.ndarray:
    """Convert already filtered band signals to rectified encoder currents."""
    rows = []
    for name in band_names:
        values = np.asarray(filtered[name], dtype=float)
        if values.ndim == 2:
            if values.shape[0] != 1:
                raise ValueError(f"Expected one channel per filtered band, got shape {values.shape} for {name}")
            values = values[0]
        rows.append(values)
    stacked = np.vstack(rows)
    return rectify_and_amplify(stacked, rectify_config, sfreq_hz=sfreq_hz, causal=causal)


def run_lif_encoder(
    currents: np.ndarray,
    config: LIFEncoderConfig,
    *,
    backend: str = "runtime",
    seed: int = 0,
    record_voltage: bool = False,
) -> LIFEncodingResult:
    """Run a small LI&F encoder smoke simulation."""
    if backend != "runtime":
        raise ValueError("run_lif_encoder currently supports the runtime backend only")

    b2 = _lazy_brian()
    b2.start_scope()
    b2.seed(seed)
    b2.defaultclock.dt = config.dt_ms * b2.ms

    currents = np.asarray(currents, dtype=np.float32)
    if currents.ndim == 1:
        currents = currents[np.newaxis, :]

    n_neurons, n_steps = currents.shape
    drive = b2.TimedArray(currents.T, dt=config.dt_ms * b2.ms, name="encoder_drive")
    neurons = b2.NeuronGroup(
        n_neurons,
        """
        dv/dt = (-v + gain * encoder_drive(t, i) + bias) / tau : 1 (unless refractory)
        gain : 1 (shared, constant)
        bias : 1 (shared, constant)
        tau : second (shared, constant)
        threshold_param : 1 (shared, constant)
        reset_level : 1 (shared, constant)
        refractory_period : second (shared, constant)
        """,
        threshold="v > threshold_param",
        reset="v = reset_level",
        refractory="refractory_period",
        method="euler",
        name="lif_encoder",
        namespace={"encoder_drive": drive},
    )
    neurons.gain = config.gain
    neurons.bias = config.bias
    neurons.tau = config.tau_ms * b2.ms
    neurons.threshold_param = config.threshold
    neurons.reset_level = config.reset
    neurons.refractory_period = config.refractory_ms * b2.ms

    spike_mon = b2.SpikeMonitor(neurons)
    state_mon = b2.StateMonitor(neurons, "v", record=True) if record_voltage else None
    b2.run(n_steps * config.dt_ms * b2.ms, namespace={})

    voltages = None if state_mon is None else np.asarray(state_mon.v, dtype=np.float32)
    return LIFEncodingResult(
        duration_s=n_steps * config.dt_ms / 1000.0,
        spike_times_s=np.asarray(spike_mon.t / b2.second, dtype=float),
        spike_indices=np.asarray(spike_mon.i, dtype=int),
        voltages=voltages,
    )

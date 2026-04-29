"""Full-wave rectify and amplify stage for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import signal

from stnbeta.phase4.config import load_yaml


@dataclass(frozen=True)
class RectifyAmplifyConfig:
    """Rectify/amplify parameters."""

    gain: float = 1.0
    offset: float = 0.0
    power: float = 1.0
    normalize_percentile: float | None = 95.0
    clip_max: float | None = None
    smooth_hz: float | None = None
    smooth_order: int = 2
    epsilon: float = 1e-9


def load_rectify_amplify_config(path: str | Path | Mapping[str, Any]) -> RectifyAmplifyConfig:
    """Load rectify/amplify config from YAML or a mapping."""
    if isinstance(path, Mapping):
        mapping = dict(path)
    else:
        mapping = load_yaml(path)
    if "rectify_amplify" in mapping:
        mapping = dict(mapping["rectify_amplify"])
    elif "rectify" in mapping:
        mapping = dict(mapping["rectify"])

    allowed = {field.name for field in fields(RectifyAmplifyConfig)}
    filtered = {key: value for key, value in mapping.items() if key in allowed}
    return RectifyAmplifyConfig(**filtered)


def rectify_and_amplify(
    data: np.ndarray,
    config: RectifyAmplifyConfig,
    *,
    reference: np.ndarray | None = None,
    sfreq_hz: float | None = None,
    causal: bool = False,
) -> np.ndarray:
    """Full-wave rectify, optionally normalize, then amplify."""
    array = np.abs(np.asarray(data, dtype=float))
    ref = array if reference is None else np.abs(np.asarray(reference, dtype=float))

    if config.smooth_hz is not None:
        if sfreq_hz is None:
            raise ValueError("sfreq_hz is required when RectifyAmplifyConfig.smooth_hz is set")
        sos = signal.butter(
            int(config.smooth_order),
            float(config.smooth_hz),
            btype="lowpass",
            fs=float(sfreq_hz),
            output="sos",
        )
        filter_fn = signal.sosfilt if causal else signal.sosfiltfilt
        array = filter_fn(sos, array, axis=-1)
        ref = filter_fn(sos, ref, axis=-1)
        array = np.clip(array, 0.0, None)
        ref = np.clip(ref, 0.0, None)

    if config.normalize_percentile is not None:
        denom = np.percentile(ref, config.normalize_percentile, axis=-1, keepdims=True)
        array = array / np.maximum(denom, config.epsilon)
    if config.power != 1.0:
        array = np.power(array, config.power)
    array = config.gain * array + config.offset
    if config.clip_max is not None:
        array = np.clip(array, 0.0, config.clip_max)
    return array.astype(np.float32)

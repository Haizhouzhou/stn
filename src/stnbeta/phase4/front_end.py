"""Helpers for aligning Phase 4 front-end configs across scripts."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from stnbeta.encoding.lif_encoder import LIFEncoderConfig
from stnbeta.preprocessing.rectify_amplify import RectifyAmplifyConfig
from stnbeta.snn_brian2.topologies.nsm_monotonic_duration import MonotonicStateMachineConfig


_ENCODER_OVERRIDE_MAP = {
    "encoder_tau_ms": "tau_ms",
    "encoder_threshold": "threshold",
    "encoder_reset": "reset",
    "encoder_refractory_ms": "refractory_ms",
    "encoder_gain": "gain",
    "encoder_bias": "bias",
}


def nsm_with_lif_defaults(
    nsm_config: MonotonicStateMachineConfig,
    lif_config: LIFEncoderConfig,
) -> MonotonicStateMachineConfig:
    """Copy base LI&F settings into the state-machine encoder block."""
    return replace(
        nsm_config,
        dt_ms=lif_config.dt_ms,
        encoder_tau_ms=lif_config.tau_ms,
        encoder_threshold=lif_config.threshold,
        encoder_reset=lif_config.reset,
        encoder_refractory_ms=lif_config.refractory_ms,
        encoder_gain=lif_config.gain,
        encoder_bias=lif_config.bias,
    )


def split_front_end_overrides(
    overrides: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Split overrides into LI&F, rectify, and state-machine subsets."""
    lif_overrides: dict[str, object] = {}
    rectify_overrides: dict[str, object] = {}
    nsm_overrides: dict[str, object] = {}

    for key, value in overrides.items():
        if key == "dt_ms":
            lif_overrides["dt_ms"] = value
            nsm_overrides[key] = value
            continue
        if key in _ENCODER_OVERRIDE_MAP:
            lif_overrides[_ENCODER_OVERRIDE_MAP[key]] = value
            nsm_overrides[key] = value
            continue
        if key.startswith("rectify_"):
            rectify_overrides[key.removeprefix("rectify_")] = value
            continue
        nsm_overrides[key] = value

    return lif_overrides, rectify_overrides, nsm_overrides


def apply_lif_overrides(
    base: LIFEncoderConfig,
    overrides: Mapping[str, object],
) -> LIFEncoderConfig:
    """Return a copy of *base* with LI&F overrides applied."""
    return base if not overrides else replace(base, **dict(overrides))


def apply_rectify_overrides(
    base: RectifyAmplifyConfig,
    overrides: Mapping[str, object],
) -> RectifyAmplifyConfig:
    """Return a copy of *base* with rectify/amplify overrides applied."""
    return base if not overrides else replace(base, **dict(overrides))


def apply_nsm_overrides(
    base: MonotonicStateMachineConfig,
    overrides: Mapping[str, object],
) -> MonotonicStateMachineConfig:
    """Return a copy of *base* with state-machine overrides applied."""
    return base if not overrides else replace(base, **dict(overrides))

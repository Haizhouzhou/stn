"""Monotonic duration-bucket state-machine topology definitions.

This module keeps the lightweight Phase 4 config loader intact and adds the
Phase 5 clustered duration-bucket config used by the Brian2/Brian2CUDA SNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from stnbeta.phase4.config import load_yaml


@dataclass(frozen=True)
class MonotonicStateMachineConfig:
    """Config for the synthetic duration-bucket state machine."""

    bucket_thresholds_ms: tuple[float, ...] = (100.0, 200.0, 350.0, 500.0)
    dt_ms: float = 1.0
    encoder_tau_ms: float = 15.0
    encoder_threshold: float = 1.0
    encoder_reset: float = 0.0
    encoder_refractory_ms: float = 4.0
    encoder_gain: float = 1.0
    encoder_bias: float = 0.0
    quiet_tau_ms: float = 25.0
    quiet_threshold: float = 0.70
    quiet_gain: float = 1.0
    quiet_refractory_ms: float = 8.0
    bucket_tau_ms: float = 45.0
    bucket_refractory_ms: float = 10.0
    bucket_threshold_scale: float = 0.01
    input_weight: float = 0.40
    forward_weight: float = 0.90
    sustain_weight: float = 0.15
    boundary_inhibition_weight: float = 0.10
    readout_bucket_index: int = 2
    readout_weight: float = 1.2
    readout_tau_ms: float = 35.0
    readout_threshold: float = 1.0
    readout_refractory_ms: float = 10.0
    synaptic_delay_ms: float = 0.0

    @property
    def bucket_thresholds(self) -> tuple[float, ...]:
        return tuple(ms * self.bucket_threshold_scale for ms in self.bucket_thresholds_ms)


@dataclass(frozen=True)
class DurationBucketClusterConfig:
    """Phase 5 clustered monotonic duration-bucket state-machine config."""

    state_names: tuple[str, ...] = ("D_idle", "D0", "D1", "D2", "D3", "D4")
    state_lower_bounds_ms: tuple[float, ...] = (0.0, 0.0, 50.0, 100.0, 200.0, 400.0)
    state_upper_bounds_ms: tuple[float | None, ...] = (None, 50.0, 100.0, 200.0, 400.0, None)
    state_threshold_scales: tuple[float, ...] = (0.80, 1.00, 1.18, 1.55, 2.00, 2.60)
    cluster_exc_size: int = 32
    cluster_inh_size: int = 8
    dt_ms: float = 1.0
    encoder_tau_ms: float = 15.0
    encoder_threshold: float = 1.0
    encoder_reset: float = 0.0
    encoder_refractory_ms: float = 4.0
    encoder_gain: float = 1.0
    encoder_bias: float = 0.0
    quiet_tau_ms: float = 25.0
    quiet_threshold: float = 0.60
    quiet_gain: float = 1.0
    quiet_refractory_ms: float = 12.0
    quiet_holdoff_ms: float = 25.0
    neuron_tau_ms: float = 12.0
    inhibitory_tau_ms: float = 8.0
    refractory_ms: float = 5.0
    threshold_base: float = 1.0
    reset_level: float = 0.0
    input_tau_syn_ms: float = 20.0
    feedforward_tau_syn_ms: float = 20.0
    recurrent_tau_syn_ms: float = 20.0
    inhibitory_tau_syn_ms: float = 15.0
    reset_tau_syn_ms: float = 50.0
    readout_tau_syn_ms: float = 20.0
    input_weight: float = 0.70
    recurrent_weight: float = 0.050
    idle_recurrent_weight: float = 0.020
    feedforward_weight: float = 0.120
    local_ei_weight: float = 0.080
    local_ie_weight: float = 0.060
    lateral_inhibition_weight: float = 0.060
    reset_weight: float = 0.220
    readout_weight: float = 0.060
    idle_bias: float = 0.020
    mismatch_cov_pct: float = 0.0
    readout_threshold_state: str = "D2"
    readout_dwell_ms: float = 10.0
    readout_threshold: float = 0.35
    readout_refractory_ms: float = 10.0
    occupancy_tau_ms: float = 20.0
    occupancy_active_threshold: float = 0.020
    evidence_aggregation_mode: str = "mean"

    @property
    def bucket_state_names(self) -> tuple[str, ...]:
        return self.state_names[1:]

    @property
    def bucket_thresholds_ms(self) -> tuple[float, ...]:
        return tuple(self.state_lower_bounds_ms[1:])

    @property
    def readout_bucket_index(self) -> int:
        try:
            index = self.state_names.index(self.readout_threshold_state)
        except ValueError as exc:
            raise ValueError(
                f"Unknown readout threshold state {self.readout_threshold_state!r}; "
                f"expected one of {self.state_names}"
            ) from exc
        if index == 0:
            raise ValueError("readout_threshold_state must be one of D0-D4, not D_idle")
        return index - 1


def _coerce_phase5_mapping(path: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(path, Mapping):
        raw = dict(path)
    else:
        raw = load_yaml(path)
    if "phase5" in raw and isinstance(raw["phase5"], Mapping):
        return dict(raw["phase5"])
    return raw


def load_duration_bucket_cluster_config(
    path: str | Path | Mapping[str, Any],
) -> DurationBucketClusterConfig:
    """Load the Phase 5 clustered duration-bucket config from YAML or a mapping."""
    raw = _coerce_phase5_mapping(path)
    states = dict(raw.get("states", {}))
    cluster = dict(raw.get("cluster", {}))
    encoder = dict(raw.get("encoder", {}))
    quiet = dict(raw.get("quiet", {}))
    topology = dict(raw.get("topology", {}))
    readout = dict(raw.get("readout", {}))
    analysis = dict(raw.get("analysis", {}))

    state_names = tuple(states.get("names", ("D_idle", "D0", "D1", "D2", "D3", "D4")))
    lower_bounds = tuple(states.get("lower_bounds_ms", (0.0, 0.0, 50.0, 100.0, 200.0, 400.0)))
    upper_bounds_raw = states.get("upper_bounds_ms", (None, 50.0, 100.0, 200.0, 400.0, None))
    upper_bounds = tuple(item if item is None else float(item) for item in upper_bounds_raw)
    threshold_scales = tuple(states.get("threshold_scales", (0.80, 1.00, 1.18, 1.55, 2.00, 2.60)))

    cluster_exc_size = int(cluster.get("excitatory_size", 32))
    cluster_inh_size = int(
        cluster.get(
            "inhibitory_size",
            max(1, int(round(cluster_exc_size * float(cluster.get("inhibitory_ratio", 0.25))))),
        )
    )

    return DurationBucketClusterConfig(
        state_names=state_names,
        state_lower_bounds_ms=tuple(float(item) for item in lower_bounds),
        state_upper_bounds_ms=upper_bounds,
        state_threshold_scales=tuple(float(item) for item in threshold_scales),
        cluster_exc_size=cluster_exc_size,
        cluster_inh_size=cluster_inh_size,
        dt_ms=float(topology.get("dt_ms", 1.0)),
        encoder_tau_ms=float(encoder.get("tau_ms", 15.0)),
        encoder_threshold=float(encoder.get("threshold", 1.0)),
        encoder_reset=float(encoder.get("reset", 0.0)),
        encoder_refractory_ms=float(encoder.get("refractory_ms", 4.0)),
        encoder_gain=float(encoder.get("gain", 1.0)),
        encoder_bias=float(encoder.get("bias", 0.0)),
        quiet_tau_ms=float(quiet.get("tau_ms", 25.0)),
        quiet_threshold=float(quiet.get("threshold", 0.60)),
        quiet_gain=float(quiet.get("gain", 1.0)),
        quiet_refractory_ms=float(quiet.get("refractory_ms", 12.0)),
        quiet_holdoff_ms=float(quiet.get("holdoff_ms", 25.0)),
        neuron_tau_ms=float(topology.get("neuron_tau_ms", 12.0)),
        inhibitory_tau_ms=float(topology.get("inhibitory_tau_ms", 8.0)),
        refractory_ms=float(topology.get("refractory_ms", 5.0)),
        threshold_base=float(topology.get("threshold_base", 1.0)),
        reset_level=float(topology.get("reset_level", 0.0)),
        input_tau_syn_ms=float(topology.get("input_tau_syn_ms", topology.get("feedforward_tau_syn_ms", 20.0))),
        feedforward_tau_syn_ms=float(topology.get("feedforward_tau_syn_ms", 20.0)),
        recurrent_tau_syn_ms=float(topology.get("recurrent_tau_syn_ms", 20.0)),
        inhibitory_tau_syn_ms=float(topology.get("inhibitory_tau_syn_ms", 15.0)),
        reset_tau_syn_ms=float(topology.get("reset_tau_syn_ms", 50.0)),
        readout_tau_syn_ms=float(readout.get("tau_syn_ms", 20.0)),
        input_weight=float(topology.get("input_weight", 0.70)),
        recurrent_weight=float(topology.get("recurrent_weight", 0.050)),
        idle_recurrent_weight=float(topology.get("idle_recurrent_weight", 0.020)),
        feedforward_weight=float(topology.get("feedforward_weight", 0.120)),
        local_ei_weight=float(topology.get("local_ei_weight", 0.080)),
        local_ie_weight=float(topology.get("local_ie_weight", 0.060)),
        lateral_inhibition_weight=float(topology.get("lateral_inhibition_weight", 0.060)),
        reset_weight=float(topology.get("reset_weight", 0.220)),
        readout_weight=float(readout.get("weight", 0.060)),
        idle_bias=float(topology.get("idle_bias", 0.020)),
        mismatch_cov_pct=float(topology.get("mismatch_cov_pct", 0.0)),
        readout_threshold_state=str(readout.get("threshold_state", "D2")),
        readout_dwell_ms=float(readout.get("dwell_ms", 10.0)),
        readout_threshold=float(readout.get("threshold", 0.35)),
        readout_refractory_ms=float(readout.get("refractory_ms", 10.0)),
        occupancy_tau_ms=float(analysis.get("occupancy_tau_ms", 20.0)),
        occupancy_active_threshold=float(analysis.get("occupancy_active_threshold", 0.020)),
        evidence_aggregation_mode=str(analysis.get("evidence_aggregation_mode", "mean")),
    )


def load_nsm_config(path: str | Path | Mapping[str, Any]) -> MonotonicStateMachineConfig:
    """Load the state-machine config from YAML or a mapping."""
    if isinstance(path, Mapping):
        raw = dict(path)
    else:
        raw = load_yaml(path)

    bucket_thresholds_ms = tuple(raw.get("bucket_thresholds_ms", []))
    encoder = raw.get("encoder", {})
    quiet = raw.get("quiet", {})
    state_machine = raw.get("state_machine", {})
    return MonotonicStateMachineConfig(
        bucket_thresholds_ms=bucket_thresholds_ms or (100.0, 200.0, 350.0, 500.0),
        dt_ms=float(state_machine.get("dt_ms", 1.0)),
        encoder_tau_ms=float(encoder.get("tau_ms", 15.0)),
        encoder_threshold=float(encoder.get("threshold", 1.0)),
        encoder_reset=float(encoder.get("reset", 0.0)),
        encoder_refractory_ms=float(encoder.get("refractory_ms", 4.0)),
        encoder_gain=float(encoder.get("gain", 1.0)),
        encoder_bias=float(encoder.get("bias", 0.0)),
        quiet_tau_ms=float(quiet.get("tau_ms", 25.0)),
        quiet_threshold=float(quiet.get("threshold", 0.70)),
        quiet_gain=float(quiet.get("gain", 1.0)),
        quiet_refractory_ms=float(quiet.get("refractory_ms", 8.0)),
        bucket_tau_ms=float(state_machine.get("bucket_tau_ms", 45.0)),
        bucket_threshold_scale=float(state_machine.get("bucket_threshold_scale", 0.01)),
        input_weight=float(state_machine.get("input_weight", 0.40)),
        forward_weight=float(state_machine.get("forward_weight", 0.90)),
        sustain_weight=float(state_machine.get("sustain_weight", 0.15)),
        boundary_inhibition_weight=float(state_machine.get("boundary_inhibition_weight", 0.10)),
        readout_bucket_index=int(state_machine.get("readout_bucket_index", 2)),
        readout_weight=float(state_machine.get("readout_weight", 1.2)),
        synaptic_delay_ms=float(state_machine.get("synaptic_delay_ms", 0.0)),
    )

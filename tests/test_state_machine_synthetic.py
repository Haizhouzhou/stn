from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stnbeta.encoding.lif_encoder import band_currents_from_signal
from stnbeta.phase4.metrics import synthetic_case_metrics
from stnbeta.snn_brian2.runner import derive_quiet_drive, run_state_machine
from stnbeta.snn_brian2.topologies.nsm_monotonic_duration import load_nsm_config
from stnbeta.synthetic.beta_burst_generator import BurstSpec, SyntheticTraceConfig, generate_trace


brian2 = pytest.importorskip("brian2", reason="brian2 not installed")


def _fast_config():
    return replace(
        load_nsm_config("configs/nsm_mono.yaml"),
        bucket_threshold_scale=0.012,
        input_weight=0.4,
        forward_weight=0.9,
        boundary_inhibition_weight=0.1,
        quiet_threshold=0.7,
        encoder_gain=1.0,
    )


def test_state_machine_progresses_monotonically():
    trace = generate_trace(
        SyntheticTraceConfig(
            name="mono",
            seed=7,
            duration_s=2.0,
            bursts=(BurstSpec(onset_s=0.5, duration_s=0.35, amplitude=1.3, center_hz=20.0),),
        )
    )
    _, roles, currents = band_currents_from_signal(trace.signal, trace.sfreq_hz)
    result = run_state_machine(currents, roles, _fast_config(), backend="runtime", quiet_drive=derive_quiet_drive(currents, roles))

    first_times = []
    for bucket_index in range(len(result.bucket_thresholds_ms)):
        spikes = result.bucket_spike_times_s[result.bucket_spike_indices == bucket_index]
        if len(spikes):
            first_times.append(float(spikes[0]))
    assert first_times == sorted(first_times)


def test_state_machine_reset_and_negative_control():
    trace = generate_trace(
        SyntheticTraceConfig(
            name="negative",
            seed=11,
            duration_s=2.0,
            bursts=(),
        )
    )
    _, roles, currents = band_currents_from_signal(trace.signal, trace.sfreq_hz)
    result = run_state_machine(currents, roles, _fast_config(), backend="runtime", quiet_drive=derive_quiet_drive(currents, roles))
    metrics = synthetic_case_metrics(trace, result)

    assert metrics["false_positive_rate_hz"] == 0.0

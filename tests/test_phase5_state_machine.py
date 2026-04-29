from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import shutil

import numpy as np
import pytest

from stnbeta.phase5.metrics import latency_decomposition_table
from stnbeta.phase5.readout import build_readout_summary
from stnbeta.phase5.synthetic_suite import generate_topology_suite
from stnbeta.snn_brian2.runner import (
    aggregate_beta_evidence,
    phase5_entry_weight,
    prepare_phase5_entry_currents,
    run_duration_bucket_state_machine,
)
from stnbeta.snn_brian2.topologies.nsm_monotonic_duration import load_duration_bucket_cluster_config


brian2 = pytest.importorskip("brian2", reason="brian2 not installed")


def _runtime_config():
    return replace(
        load_duration_bucket_cluster_config("configs/nsm_mono.yaml"),
        input_weight=0.70,
        feedforward_weight=0.12,
        recurrent_weight=0.05,
        local_ei_weight=0.08,
        local_ie_weight=0.06,
        readout_weight=0.06,
        idle_recurrent_weight=0.02,
        idle_bias=0.02,
        readout_dwell_ms=8.0,
        occupancy_active_threshold=0.02,
        cluster_exc_size=16,
        cluster_inh_size=4,
        mismatch_cov_pct=0.0,
    )


def test_phase5_runtime_rejects_short_and_detects_threshold_crossing():
    config = _runtime_config()
    suite = {case.name: case for case in generate_topology_suite()}

    short_case = suite["short_40ms"]
    short_result = run_duration_bucket_state_machine(short_case.direct_currents, short_case.band_roles, config, backend="runtime")
    short_readout = build_readout_summary(short_result, config)
    assert short_readout.stable_mask.sum() == 0

    threshold_case = suite["threshold_crossing_120ms"]
    threshold_result = run_duration_bucket_state_machine(threshold_case.direct_currents, threshold_case.band_roles, config, backend="runtime")
    threshold_readout = build_readout_summary(threshold_result, config)
    assert threshold_readout.stable_mask.sum() > 0


def test_phase5_runtime_progresses_without_skipping():
    config = _runtime_config()
    case = next(item for item in generate_topology_suite() if item.name == "long_400ms_plus")
    result = run_duration_bucket_state_machine(case.direct_currents, case.band_roles, config, backend="runtime")
    masks = build_readout_summary(result, config).state_masks
    reached = [name for name in result.state_names[1:] if masks[name].any()]
    assert reached == list(result.state_names[1 : 1 + len(reached)])


def test_phase5_entry_weight_scales_with_aggregation_mode():
    config = replace(_runtime_config(), evidence_aggregation_mode="mean", input_weight=0.8)
    assert phase5_entry_weight(config, ("beta", "beta", "beta", "beta", "boundary", "boundary")) == pytest.approx(0.2)

    config = replace(config, evidence_aggregation_mode="top2_mean")
    assert phase5_entry_weight(config, ("beta", "beta", "beta", "beta")) == pytest.approx(0.4)


def test_phase5_entry_currents_explicitly_pool_beta_before_d0():
    currents = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 4.0, 6.0],
            [4.0, 2.0, 8.0],
            [8.0, 6.0, 2.0],
        ],
        dtype=np.float32,
    )
    roles = ("boundary", "beta", "beta", "beta")

    raw_currents, raw_roles = prepare_phase5_entry_currents(currents, roles, mode="raw")
    assert raw_roles == roles
    assert raw_currents.shape == currents.shape

    mean_currents, mean_roles = prepare_phase5_entry_currents(currents, roles, mode="mean")
    assert mean_roles == ("beta", "boundary")
    np.testing.assert_allclose(mean_currents[0], [14.0 / 3.0, 4.0, 16.0 / 3.0])
    np.testing.assert_allclose(mean_currents[1], currents[0])

    top2_currents, top2_roles = prepare_phase5_entry_currents(currents, roles, mode="top2_mean")
    assert top2_roles == ("beta", "boundary")
    np.testing.assert_allclose(top2_currents[0], [6.0, 5.0, 7.0])


def test_phase5_latency_decomposition_plumbing():
    config = _runtime_config()
    case = next(item for item in generate_topology_suite() if item.name == "threshold_crossing_120ms")
    result = run_duration_bucket_state_machine(case.direct_currents, case.band_roles, config, backend="runtime")
    readout = build_readout_summary(result, config)
    proxy_case = SimpleNamespace(
        subject_id="synthetic",
        condition="topology",
        channel=case.name,
        events=case.annotations.loc[:, ["onset_s", "offset_s"]].copy(),
    )
    latency_df = latency_decomposition_table(
        proxy_case,
        result,
        config,
        readout,
        evidence_trace=aggregate_beta_evidence(case.direct_currents, case.band_roles, mode="mean"),
    )

    assert not latency_df.empty
    assert {"phase4_encoder_onset_s", "D0_onset_s", "D2_onset_s", "stable_readout_onset_s"}.issubset(latency_df.columns)


@pytest.mark.skipif(shutil.which("nvidia-smi") is None or shutil.which("nvcc") is None, reason="CUDA toolchain or GPU not visible")
def test_phase5_cuda_smoke(tmp_path):
    pytest.importorskip("brian2cuda", reason="brian2cuda not installed")
    config = _runtime_config()
    case = next(item for item in generate_topology_suite() if item.name == "threshold_crossing_120ms")
    result = run_duration_bucket_state_machine(
        case.direct_currents,
        case.band_roles,
        config,
        backend="cuda_standalone",
        seed=0,
    )
    assert result.occupancy.shape[1] == case.direct_currents.shape[1]

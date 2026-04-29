from __future__ import annotations

from pathlib import Path

import numpy as np

from stnbeta.phase5_2c.robustness_family_audit import (
    NEGATIVE_CONTROL_CASES,
    StateMachineConfig,
    hard_gate_pass,
    literature_noise_battery,
    run_audit,
    run_clean_sweep,
    select_family,
    simulate_case,
    synthetic_cases,
)


def _default_config() -> StateMachineConfig:
    return StateMachineConfig(
        config_id="unit_default",
        population_exc=32,
        population_inh=8,
        duration_thresholds_ms=(50, 100, 200, 400),
        sustain_weight=1.0,
        forward_weight=1.0,
        reset_strength=1.0,
        quiet_gain=1.0,
    )


def test_clean_default_rejects_no_burst_and_short_events() -> None:
    cfg = _default_config()
    cases = {case.case_id: case for case in synthetic_cases()}

    no_burst = simulate_case(cfg, cases["no_burst"])
    short = simulate_case(cfg, cases["short_40ms"])
    positive = simulate_case(cfg, cases["threshold_crossing_120ms"])

    assert no_burst["false_alarm_per_min"] == 0.0
    assert short["short_event_false_readout"] is False
    assert positive["recall"] == 1.0
    assert hard_gate_pass(_series_like(positive))


def test_noise_battery_is_not_gaussian_only() -> None:
    families = {row.family for row in literature_noise_battery(seed_count=3)}
    assert "input_current_additive_gaussian" in families
    assert "hardware_mismatch" in families
    assert "quantization" in families
    assert "artifact_impulse" in families
    assert "event_dropout" in families


def test_clean_sweep_selects_config_family() -> None:
    configs = [
        _default_config(),
        StateMachineConfig("late_readout", 8, 2, (100, 200, 350, 500), 0.5, 0.5, 1.0, 1.0),
        StateMachineConfig("large_pop", 64, 16, (50, 100, 200, 400), 1.0, 1.0, 1.25, 1.25),
    ]
    _, summary = run_clean_sweep(configs)
    family = select_family(summary, family_size=2)
    assert len(family) >= 1
    assert family.iloc[0]["clean_hard_gate_pass"] in {True, np.bool_(True)}


def test_run_audit_writes_expected_tables(tmp_path: Path) -> None:
    out_dir = tmp_path / "audit"
    run_audit(out_dir, family_size=3, noise_seeds=2, max_configs=16)

    expected = {
        "clean_baseline_audit.csv",
        "parameter_sweep_summary.csv",
        "robust_family_configs.csv",
        "noise_robustness_seed_distribution.csv",
        "family_robustness_summary.csv",
        "negative_control_summary.csv",
        "literature_perturbation_battery.csv",
        "manifest.json",
    }
    assert expected.issubset({path.name for path in out_dir.iterdir()})


def test_negative_control_case_set_is_explicit() -> None:
    assert {"no_burst", "power_shift_no_burst", "sustained_only"}.issubset(NEGATIVE_CONTROL_CASES)


def _series_like(row: dict):
    import pandas as pd

    return pd.Series(row)

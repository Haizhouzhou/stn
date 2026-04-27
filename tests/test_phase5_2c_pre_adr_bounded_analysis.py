from __future__ import annotations

import numpy as np
import pandas as pd

from stnbeta.phase5_2c import stage_f_event_metrics_fast as fast
from stnbeta.phase5_2c.pre_adr_bounded_analysis import (
    ScoringWindow,
    aggregate_metric_rows,
    evaluate_alarm_result_with_policy,
    rolling_count_bool,
    rolling_mean_by_group,
)


def test_rolling_count_bool_is_causal() -> None:
    mask = np.array([True, False, True, True, False])
    assert rolling_count_bool(mask, 3).tolist() == [1, 1, 2, 2, 2]


def test_evaluate_alarm_result_with_onset_tolerance() -> None:
    alarms = fast.AlarmResult(by_group={0: np.array([0.95, 2.50])}, n_alarms=2)
    events = {0: (np.array([1.00]), np.array([1.20]))}
    scoring = ScoringWindow("test", "onset", -0.10, "onset", 0.10, False, "unit")
    row = evaluate_alarm_result_with_policy(alarms, events, 1, 1.0, scoring)
    assert row["true_positive_events"] == 1
    assert row["false_positive_alarms"] == 1
    assert row["recall"] == 1.0


def test_rolling_mean_by_group_uses_past_window_only() -> None:
    frame = pd.DataFrame(
            {
                "subject_id": ["s1", "s1", "s1"],
                "condition": ["cnd"] * 3,
                "fif_path": ["f"] * 3,
                "channel": ["c"] * 3,
            "window_start_s": [0.0, 1.0, 2.0],
            "window_stop_s": [0.3, 1.3, 2.3],
            "window_type": ["negative", "true_full_burst", "negative"],
            "anchor_onset_s": [np.nan, 1.0, np.nan],
            "anchor_offset_s": [np.nan, 1.2, np.nan],
        }
    )
    frame["is_true_event"] = frame["window_type"].eq("true_full_burst")
    frame["event_key"] = ["neg0", "event1", "neg2"]
    cache = fast.prepare_event_cache(frame)
    out = rolling_mean_by_group(cache, np.array([0.0, 1.0, 0.0]), 1.0)
    assert out.tolist() == [0.0, 0.5, 0.5]


def test_aggregate_metric_rows_sums_counts() -> None:
    rows = [
        {"n_true_events": 10, "n_alarms": 3, "true_positive_events": 2, "true_positive_alarms": 2, "false_positive_alarms": 1, "minutes": 1.0, "one_alarm_per_burst_fraction": 0.2},
        {"n_true_events": 5, "n_alarms": 1, "true_positive_events": 1, "true_positive_alarms": 1, "false_positive_alarms": 0, "minutes": 1.0, "one_alarm_per_burst_fraction": 0.2},
    ]
    agg = aggregate_metric_rows(rows)
    assert agg["n_true_events"] == 15
    assert agg["true_positive_events"] == 3
    assert agg["recall"] == 0.2
    assert agg["fp_per_min_achieved"] == 0.5

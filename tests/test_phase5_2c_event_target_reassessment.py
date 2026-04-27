from __future__ import annotations

import numpy as np
import pandas as pd

from stnbeta.phase5_2c.event_target_reassessment import assign_fp_grid_to_subject_summary, hash_subject, score_threshold_stats


def test_hash_subject_does_not_expose_raw_id() -> None:
    raw = "sub-secret"
    key = hash_subject(raw)
    assert key.startswith("subject_")
    assert raw not in key
    assert key == hash_subject(raw)


def test_assign_fp_grid_to_subject_summary_uses_group_order() -> None:
    rows = []
    for fp_idx in range(4):
        for subject in ["s1", "s2"]:
            rows.append({"tier": "tier1_continuous", "mismatch_seed": np.nan, "subject_id": subject, "recall": fp_idx})
    out = assign_fp_grid_to_subject_summary(pd.DataFrame(rows))
    assert out["fp_per_min"].tolist() == [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 5.0, 5.0]


def test_score_threshold_stats_counts_quantized_ties() -> None:
    score = np.array([0.0, 1.0, 1.0, 0.5, np.nan])
    stats = score_threshold_stats(score, 1.0)
    assert stats["tie_count_at_threshold"] == 2
    assert stats["n_scores_at_or_above_threshold"] == 2
    assert stats["threshold_rank_or_quantile"] == 1.0

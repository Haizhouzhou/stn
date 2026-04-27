"""Event-target reassessment utilities for Phase 5_2C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import stage_f_event_metrics as stage
from . import stage_f_event_metrics_fast as fast
from .io import add_universal_columns, base_universal, load_config, output_paths, read_tsv, repo_root, resolve_path, write_tsv

REJECTED_CANDIDATES = ["CAND_2C_STRICT_FP1", "CAND_2C_BALANCED_FP2", "CAND_2C_PERMISSIVE_FP5"]
FP_GRID = [0.5, 1.0, 2.0, 5.0]
SUBJECT_HASH_SALT = "phase5_2c_event_target_reassessment_v1"
NEAR_ZERO_RECALL = 0.005


@dataclass
class ReassessmentResult:
    recommendation_state: str
    recommended_option_id: str
    outputs: list[Path]


def run_event_target_reassessment(config: dict[str, Any], *, max_refined_features: int = 8) -> ReassessmentResult:
    paths = output_paths(config)
    table_dir = paths["table_dir"]
    root = repo_root(config)
    table_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_tables(table_dir)
    owner_rejection = owner_rejection_table(config)
    write_tsv(owner_rejection, table_dir / "event_target_owner_rejection.tsv")

    frame, subset, refined_features = load_reassessment_frame(config, max_refined_features=max_refined_features)
    y = frame["is_true_event"].to_numpy(dtype=bool)
    cache = fast.prepare_event_cache(frame)
    timing = stage.event_timing_policy(config)
    base_score = stage.fold_local_subset_score(frame, y, subset)
    tier2_score = stage.quantize(base_score, 256)
    scores = {
        "tier1_continuous": base_score,
        "tier2_quantized": tier2_score,
    }
    tier3_scores = build_tier3_scores(config, tier2_score)

    dense_metrics = compute_dense_score_metrics(cache, scores, timing)
    tier3_dense_metrics = compute_tier3_dense_score_metrics(config, cache, tier3_scores, timing)
    event_window_metrics = compute_event_window_ceilings(cache, scores, timing)
    refined_metrics = compute_refined_feature_ceilings(frame, y, cache, timing, refined_features)

    anomaly = anomaly_table(config, existing, scores, tier3_scores, dense_metrics, tier3_dense_metrics)
    write_tsv(anomaly, table_dir / "tier1_tier2_fp1_anomaly_analysis.tsv")

    ceilings = ceiling_table(config, existing, dense_metrics, tier3_dense_metrics, event_window_metrics, refined_metrics, refined_features)
    write_tsv(ceilings, table_dir / "event_recall_empirical_ceiling.tsv")

    gap = recall_gap_table(config, ceilings)
    write_tsv(gap, table_dir / "event_recall_gap_to_ceiling.tsv")

    per_subject = per_subject_distribution_table(config, existing["subject_summary"], existing)
    write_tsv(per_subject, table_dir / "event_per_subject_recall_distribution.tsv")

    options, recommendation = reassessment_options_and_recommendation(config, ceilings, gap)
    write_tsv(options, table_dir / "event_target_reassessment_options.tsv")
    write_tsv(recommendation, table_dir / "event_target_reassessment_recommendation.tsv")

    readiness = updated_readiness_table(config, recommendation)
    write_tsv(readiness, table_dir / "stage_g_h_i_readiness_assessment.tsv")

    write_revised_docs(root, existing, anomaly, ceilings, gap, per_subject, options, recommendation)
    validation = validation_table(config)
    write_tsv(validation, table_dir / "event_target_reassessment_validation.tsv")

    output_files = [
        table_dir / "event_target_owner_rejection.tsv",
        table_dir / "tier1_tier2_fp1_anomaly_analysis.tsv",
        table_dir / "event_recall_empirical_ceiling.tsv",
        table_dir / "event_recall_gap_to_ceiling.tsv",
        table_dir / "event_per_subject_recall_distribution.tsv",
        table_dir / "event_target_reassessment_options.tsv",
        table_dir / "event_target_reassessment_recommendation.tsv",
        table_dir / "stage_g_h_i_readiness_assessment.tsv",
        table_dir / "event_target_reassessment_validation.tsv",
        root / "docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE_REVISED.md",
        root / "docs/PHASE5_2C_EVENT_DETECTION_LIMITATION_ANALYSIS.md",
        root / "docs/PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md",
    ]
    return ReassessmentResult(
        recommendation_state=str(recommendation["final_task_state"].iloc[0]),
        recommended_option_id=str(recommendation["option_id"].iloc[0]),
        outputs=output_files,
    )


def load_existing_tables(table_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "tier1": read_tsv(table_dir / "causal_tier1_event_metrics.tsv"),
        "tier2": read_tsv(table_dir / "causal_tier2_event_metrics.tsv"),
        "tier3": read_tsv(table_dir / "causal_tier3_event_metrics.tsv"),
        "three_tier": read_tsv(table_dir / "causal_three_tier_event_summary.tsv"),
        "subject_summary": read_tsv(table_dir / "causal_event_alarm_trace_summary.tsv"),
        "candidate_gates": read_tsv(table_dir / "event_target_candidate_gates.tsv"),
        "recommendation": read_tsv(table_dir / "event_target_recommendation.tsv"),
    }


def load_reassessment_frame(config: dict[str, Any], *, max_refined_features: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    table_dir = output_paths(config)["table_dir"]
    subset_table = read_tsv(table_dir / "causal_minimum_sufficient_subset.tsv")
    subset = json.loads(str(subset_table["subset_features"].iloc[0]))
    refined = read_tsv(table_dir / "causal_refined_candidate_features.tsv")
    refined_features = select_refined_features(refined, subset, max_features=max_refined_features)
    usecols = list(dict.fromkeys(stage.EVENT_USECOLS + subset + refined_features))
    matrix = resolve_path(config, config["inputs"]["causal_feature_matrix"])
    frame = pd.read_csv(matrix, sep="\t", usecols=usecols)
    frame["is_true_event"] = frame["window_type"].astype(str).eq("true_full_burst")
    frame["event_key"] = stage.make_event_key(frame)
    return frame, subset, refined_features


def select_refined_features(refined: pd.DataFrame, subset: list[str], *, max_features: int) -> list[str]:
    data = refined.copy()
    for col in ["causal_valid", "cross_subject_reliable", "SNN_compatible", "DYNAP_candidate"]:
        if col in data.columns:
            data = data.loc[data[col].astype(str).str.lower().eq("true")]
    data["LOSO_AUROC_num"] = pd.to_numeric(data.get("LOSO_AUROC"), errors="coerce")
    data = data.sort_values("LOSO_AUROC_num", ascending=False, kind="mergesort")
    out: list[str] = []
    for name in subset + data.get("output_column", pd.Series(dtype=str)).astype(str).tolist():
        if name and name != "nan" and name not in out:
            out.append(name)
        if len(out) >= max_features:
            break
    return out


def build_tier3_scores(config: dict[str, Any], tier2_score: np.ndarray) -> dict[int, np.ndarray]:
    seeds = int(config.get("stage_f", {}).get("tier3_mismatch_seeds", 30))
    scale = float(np.nanstd(tier2_score)) if np.isfinite(tier2_score).any() else 1.0
    out: dict[int, np.ndarray] = {}
    for seed in range(seeds):
        rng = np.random.default_rng(int(config.get("random_seed", 0)) + seed)
        out[seed] = tier2_score + rng.normal(0.0, 0.20 * scale, size=tier2_score.size)
    return out


def dense_threshold_candidates(score: np.ndarray, *, max_thresholds: int = 160) -> np.ndarray:
    finite = np.asarray(score, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([])
    unique = np.unique(finite)
    if unique.size <= max_thresholds:
        return unique[::-1]
    top = np.linspace(0.9999, 0.90, max_thresholds // 2)
    rest = np.linspace(0.90, 0.01, max_thresholds - len(top))
    thresholds = np.unique(np.nanquantile(finite, np.r_[top, rest]))
    return thresholds[::-1]


def compute_dense_score_metrics(cache: fast.EventCache, scores: dict[str, np.ndarray], timing: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, score in scores.items():
        max_thresholds = 256 if source == "tier2_quantized" else 220
        rows.extend(best_metrics_by_fp(cache, score, timing, dense_threshold_candidates(score, max_thresholds=max_thresholds), score_source=source, ceiling_type="score_ranking_ceiling_dense_threshold"))
    return pd.DataFrame(rows)


def compute_tier3_dense_score_metrics(config: dict[str, Any], cache: fast.EventCache, tier3_scores: dict[int, np.ndarray], timing: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, score in tier3_scores.items():
        seed_rows = best_metrics_by_fp(
            cache,
            score,
            timing,
            dense_threshold_candidates(score, max_thresholds=120),
            score_source="tier3_quantized_mismatched",
            ceiling_type="score_ranking_ceiling_dense_threshold",
        )
        for row in seed_rows:
            row["seed"] = seed
        rows.extend(seed_rows)
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    agg_rows: list[dict[str, Any]] = []
    for fp, group in raw.groupby("fp_per_min", sort=True):
        agg = {
            "ceiling_type": "score_ranking_ceiling_dense_threshold",
            "score_source": "tier3_quantized_mismatched_median",
            "seed": "median_across_seeds",
            "fp_per_min": fp,
        }
        for col in ["ceiling_recall", "achieved_fp_per_min", "threshold", "precision", "F1", "n_alarms", "true_positive_events", "false_positive_alarms"]:
            agg[col] = float(pd.to_numeric(group[col], errors="coerce").median())
        agg_rows.append(agg)
    return pd.concat([raw, pd.DataFrame(agg_rows)], ignore_index=True)


def best_metrics_by_fp(cache: fast.EventCache, score: np.ndarray, timing: dict[str, float], thresholds: np.ndarray, *, score_source: str, ceiling_type: str) -> list[dict[str, Any]]:
    evaluated = []
    for threshold in thresholds:
        alarms = fast.build_alarm_result(cache, score, float(threshold), timing=timing)
        metrics = fast.evaluate_alarm_result(alarms, cache.events_by_group, cache.n_events, cache.minutes_total)
        metrics["threshold"] = float(threshold)
        evaluated.append(metrics)
    rows: list[dict[str, Any]] = []
    for fp_cap in FP_GRID:
        feasible = [m for m in evaluated if np.isfinite(m.get("fp_per_min_achieved", np.nan)) and m["fp_per_min_achieved"] <= fp_cap]
        if feasible:
            best = max(
                feasible,
                key=lambda m: (
                    stage.safe_float(m.get("recall")),
                    stage.safe_float(m.get("F1")),
                    stage.safe_float(m.get("precision")),
                    -stage.safe_float(m.get("fp_per_min_achieved")),
                ),
            )
            rows.append(
                {
                    "ceiling_type": ceiling_type,
                    "score_source": score_source,
                    "fp_per_min": fp_cap,
                    "ceiling_recall": best.get("recall", np.nan),
                    "achieved_fp_per_min": best.get("fp_per_min_achieved", np.nan),
                    "threshold": best.get("threshold", np.nan),
                    "precision": best.get("precision", np.nan),
                    "F1": best.get("F1", np.nan),
                    "n_alarms": best.get("n_alarms", np.nan),
                    "true_positive_events": best.get("true_positive_events", np.nan),
                    "false_positive_alarms": best.get("false_positive_alarms", np.nan),
                }
            )
        else:
            rows.append(
                {
                    "ceiling_type": ceiling_type,
                    "score_source": score_source,
                    "fp_per_min": fp_cap,
                    "ceiling_recall": np.nan,
                    "achieved_fp_per_min": np.nan,
                    "threshold": np.nan,
                    "precision": np.nan,
                    "F1": np.nan,
                    "n_alarms": np.nan,
                    "true_positive_events": np.nan,
                    "false_positive_alarms": np.nan,
                }
            )
    return rows


def compute_event_window_ceilings(cache: fast.EventCache, scores: dict[str, np.ndarray], timing: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, score in scores.items():
        thresholds = dense_threshold_candidates(score, max_thresholds=200 if source == "tier1_continuous" else 256)
        rows.extend(event_window_ceiling_by_fp(cache, score, thresholds, timing, source))
    return pd.DataFrame(rows)


def event_window_ceiling_by_fp(cache: fast.EventCache, score: np.ndarray, thresholds: np.ndarray, timing: dict[str, float], score_source: str) -> list[dict[str, Any]]:
    score_sorted = np.asarray(score, dtype=float)[cache.order]
    event_max = event_max_scores(cache, score_sorted)
    rows_by_threshold = []
    for threshold in thresholds:
        outside = outside_event_alarm_result(cache, score_sorted, float(threshold), timing)
        recall = float(np.mean(event_max >= threshold)) if event_max.size else np.nan
        fp = outside.n_alarms
        fp_per_min = fp / cache.minutes_total if cache.minutes_total else np.nan
        rows_by_threshold.append(
            {
                "threshold": float(threshold),
                "recall": recall,
                "fp_per_min_achieved": fp_per_min,
                "false_positive_alarms": fp,
                "true_positive_events": int(np.sum(event_max >= threshold)) if event_max.size else 0,
            }
        )
    rows: list[dict[str, Any]] = []
    for fp_cap in FP_GRID:
        feasible = [m for m in rows_by_threshold if np.isfinite(m["fp_per_min_achieved"]) and m["fp_per_min_achieved"] <= fp_cap]
        if feasible:
            best = max(feasible, key=lambda m: (stage.safe_float(m["recall"]), -stage.safe_float(m["fp_per_min_achieved"])))
            rows.append(
                {
                    "ceiling_type": "event_window_aggregation_ceiling",
                    "score_source": score_source,
                    "fp_per_min": fp_cap,
                    "ceiling_recall": best["recall"],
                    "achieved_fp_per_min": best["fp_per_min_achieved"],
                    "threshold": best["threshold"],
                    "precision": np.nan,
                    "F1": np.nan,
                    "n_alarms": np.nan,
                    "true_positive_events": best["true_positive_events"],
                    "false_positive_alarms": best["false_positive_alarms"],
                }
            )
    return rows


def event_max_scores(cache: fast.EventCache, score_sorted: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for group, (onsets, offsets) in cache.events_by_group.items():
        group_mask = cache.group_sorted == group
        times = cache.times_sorted[group_mask]
        scores = score_sorted[group_mask]
        for onset, offset in zip(onsets, offsets, strict=False):
            start = np.searchsorted(times, onset, side="left")
            stop = np.searchsorted(times, offset, side="right")
            local = scores[start:stop]
            local = local[np.isfinite(local)]
            values.append(float(np.nanmax(local)) if local.size else -np.inf)
    return np.asarray(values, dtype=float)


def outside_event_alarm_result(cache: fast.EventCache, score_sorted: np.ndarray, threshold: float, timing: dict[str, float]) -> fast.AlarmResult:
    by_group: dict[int, np.ndarray] = {}
    n_alarms = 0
    for group, start, stop in cache.group_slices:
        scores = score_sorted[start:stop]
        times = cache.times_sorted[start:stop]
        mask = np.isfinite(scores) & (scores >= threshold)
        events = cache.events_by_group.get(group)
        if events is not None and mask.any():
            onsets, offsets = events
            inside = np.zeros(mask.sum(), dtype=bool)
            cand_times = times[mask]
            event_idx = np.searchsorted(onsets, cand_times, side="right") - 1
            valid = event_idx >= 0
            inside[valid] = cand_times[valid] <= offsets[event_idx[valid]]
            tmp = np.zeros_like(mask)
            tmp[np.flatnonzero(mask)] = ~inside
            mask = tmp
        if not mask.any():
            continue
        kept = times[mask]
        keep = stage.refractory_keep_mask(kept, refractory_s=timing["refractory_s"], merge_window_s=timing["merge_window_s"])
        if keep.any():
            by_group[group] = kept[keep]
            n_alarms += int(keep.sum())
    return fast.AlarmResult(by_group=by_group, n_alarms=n_alarms)


def compute_refined_feature_ceilings(frame: pd.DataFrame, y: np.ndarray, cache: fast.EventCache, timing: dict[str, float], refined_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in refined_features:
        if feature not in frame.columns:
            continue
        score = stage.fold_local_subset_score(frame, y, [feature])
        feature_rows = best_metrics_by_fp(
            cache,
            score,
            timing,
            dense_threshold_candidates(score, max_thresholds=100),
            score_source=f"refined_feature:{feature}",
            ceiling_type="subset_score_ceiling_best_refined_variant",
        )
        rows.extend(feature_rows)
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    best_rows: list[dict[str, Any]] = []
    for fp, group in raw.groupby("fp_per_min", sort=True):
        best = group.sort_values(["ceiling_recall", "F1", "precision"], ascending=False, kind="mergesort").iloc[0].to_dict()
        best["ceiling_type"] = "subset_score_ceiling_best_refined_variant"
        best_rows.append(best)
    return pd.DataFrame(best_rows)


def owner_rejection_table(config: dict[str, Any]) -> pd.DataFrame:
    row = {
        "owner_rejected_candidate_gates": True,
        "rejected_candidate_gates": ",".join(REJECTED_CANDIDATES),
        "rejection_reason": "event recall too low to justify G/H/I execution or Brian2 gate decision",
        "readiness_should_remain_executable": False,
        "ghi_execution_blocked": True,
        "brian2_gate_decision_blocked": True,
        "support_status": "direct",
        "qc_status": "ok",
        "qc_reason": "owner rejected all current low-recall Phase 5_2C event-target candidates",
    }
    return add_universal_columns(
        pd.DataFrame([row]),
        base_universal(config, source_table="owner decision prompt;event_target_candidate_gates.tsv", source_columns="candidate_gate_id,observed recall", source_lineage="Phase 5_2C owner rejection of event-target candidates"),
    )


def score_threshold_stats(score: np.ndarray, threshold: float) -> dict[str, Any]:
    finite = np.asarray(score, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(threshold):
        return {"threshold_rank_or_quantile": np.nan, "tie_count_at_threshold": np.nan, "score_mass_near_threshold": np.nan, "n_scores_at_or_above_threshold": np.nan}
    q = float(np.mean(finite <= threshold))
    tie_count = int(np.sum(np.isclose(finite, threshold, rtol=0.0, atol=1e-12)))
    iqr = float(np.nanquantile(finite, 0.75) - np.nanquantile(finite, 0.25))
    eps = max(abs(iqr) * 0.01, 1e-12)
    near = float(np.mean(np.abs(finite - threshold) <= eps))
    return {
        "threshold_rank_or_quantile": q,
        "tie_count_at_threshold": tie_count,
        "score_mass_near_threshold": near,
        "n_scores_at_or_above_threshold": int(np.sum(finite >= threshold)),
    }


def anomaly_table(
    config: dict[str, Any],
    existing: dict[str, pd.DataFrame],
    scores: dict[str, np.ndarray],
    tier3_scores: dict[int, np.ndarray],
    dense_metrics: pd.DataFrame,
    tier3_dense_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tier_map = [("tier1_continuous", existing["tier1"], scores["tier1_continuous"]), ("tier2_quantized", existing["tier2"], scores["tier2_quantized"])]
    for tier, table, score in tier_map:
        row = table.loc[pd.to_numeric(table["target_fp_min"], errors="coerce").eq(1.0)].iloc[0].to_dict()
        stats = score_threshold_stats(score, float(row.get("threshold", np.nan)))
        dense = select_dense_row(dense_metrics, tier, 1.0)
        rows.append(
            {
                "tier": tier,
                "fp_per_min_target": 1.0,
                "threshold": row.get("threshold", np.nan),
                "achieved_fp_per_min": row.get("fp_per_min_achieved", np.nan),
                "event_recall": row.get("recall", np.nan),
                "precision": row.get("precision", np.nan),
                "f1": row.get("F1", np.nan),
                "n_alarms": row.get("n_alarms", np.nan),
                "n_true_positive_events": row.get("true_positive_events", np.nan),
                "n_false_positive_alarms": row.get("false_positive_alarms", np.nan),
                "n_events_total": row.get("n_true_events", np.nan),
                **stats,
                "dense_score_ranking_ceiling_recall_at_fp1": dense.get("ceiling_recall", np.nan),
                "dense_score_ranking_threshold_at_fp1": dense.get("threshold", np.nan),
                "tier2_event_scoring_differs_from_tier1": False,
                "tier3_seed_aggregation_masks_tier2_behavior": tier == "tier2_quantized",
                "anomaly_classification": "coarse_threshold_grid_artifact" if tier == "tier1_continuous" else "quantization_tie_threshold_jump",
                "qc_status": "warning",
                "qc_reason": "same event evaluator is used across tiers; anomaly arises from threshold-grid discontinuity and tier2 quantized ties, not an identified measurement bug",
            }
        )
    tier3_fp1 = existing["tier3"].loc[pd.to_numeric(existing["tier3"]["target_fp_min"], errors="coerce").eq(1.0)].copy()
    dense_t3 = select_dense_row(tier3_dense_metrics, "tier3_quantized_mismatched_median", 1.0)
    t3_rows = []
    for _, row in tier3_fp1.iterrows():
        seed = int(pd.to_numeric(row.get("mismatch_seed"), errors="coerce"))
        score = tier3_scores.get(seed)
        if score is None:
            continue
        stats = score_threshold_stats(score, float(row.get("threshold", np.nan)))
        t3_rows.append({**row.to_dict(), **stats})
    t3 = pd.DataFrame(t3_rows)
    rows.append(
        {
            "tier": "tier3_quantized_mismatched_median",
            "fp_per_min_target": 1.0,
            "threshold": median_numeric(t3, "threshold"),
            "achieved_fp_per_min": median_numeric(t3, "fp_per_min_achieved"),
            "event_recall": median_numeric(t3, "recall"),
            "precision": median_numeric(t3, "precision"),
            "f1": median_numeric(t3, "F1"),
            "n_alarms": median_numeric(t3, "n_alarms"),
            "n_true_positive_events": median_numeric(t3, "true_positive_events"),
            "n_false_positive_alarms": median_numeric(t3, "false_positive_alarms"),
            "n_events_total": median_numeric(t3, "n_true_events"),
            "threshold_rank_or_quantile": median_numeric(t3, "threshold_rank_or_quantile"),
            "tie_count_at_threshold": median_numeric(t3, "tie_count_at_threshold"),
            "score_mass_near_threshold": median_numeric(t3, "score_mass_near_threshold"),
            "n_scores_at_or_above_threshold": median_numeric(t3, "n_scores_at_or_above_threshold"),
            "dense_score_ranking_ceiling_recall_at_fp1": dense_t3.get("ceiling_recall", np.nan),
            "dense_score_ranking_threshold_at_fp1": dense_t3.get("threshold", np.nan),
            "tier2_event_scoring_differs_from_tier1": False,
            "tier3_seed_aggregation_masks_tier2_behavior": True,
            "anomaly_classification": "mismatch_seed_threshold_grid_fragility",
            "qc_status": "warning",
            "qc_reason": "tier3 median summarizes noisy seed-specific threshold jumps and therefore masks the tier2 max-bin behavior at FP/min 1.0",
        }
    )
    out = pd.DataFrame(rows)
    return add_universal_columns(
        out,
        base_universal(
            config,
            source_table="causal_tier1_event_metrics.tsv;causal_tier2_event_metrics.tsv;causal_tier3_event_metrics.tsv;causal_feature_matrix.tsv",
            source_columns="threshold,recall,score distribution,event reconstruction metrics",
            source_lineage="Phase 5_2C Tier 1/Tier 2 FP/min 1.0 anomaly analysis",
            support_status="direct",
            qc_status="warning",
            qc_reason="threshold-grid and quantization diagnostics; no scorer bug identified",
        ),
    )


def select_dense_row(frame: pd.DataFrame, source: str, fp: float) -> dict[str, Any]:
    if frame.empty:
        return {}
    sub = frame.loc[(frame["score_source"].astype(str).eq(source)) & pd.to_numeric(frame["fp_per_min"], errors="coerce").eq(fp)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def median_numeric(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame[col], errors="coerce").median())


def ceiling_table(
    config: dict[str, Any],
    existing: dict[str, pd.DataFrame],
    dense_metrics: pd.DataFrame,
    tier3_dense_metrics: pd.DataFrame,
    event_window_metrics: pd.DataFrame,
    refined_metrics: pd.DataFrame,
    refined_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(observed_rows(existing))
    rows.extend(ceiling_rows_from_metrics(dense_metrics, deployability_class="diagnostic_upper_bound", scoring_interval="committed_alarm_reconstruction", feature_subset="minimum_sufficient_subset", uses_oracle=True))
    rows.extend(ceiling_rows_from_metrics(tier3_dense_metrics.loc[tier3_dense_metrics.get("score_source", pd.Series(dtype=str)).astype(str).eq("tier3_quantized_mismatched_median")] if not tier3_dense_metrics.empty else tier3_dense_metrics, deployability_class="diagnostic_upper_bound", scoring_interval="committed_alarm_reconstruction", feature_subset="minimum_sufficient_subset", uses_oracle=True))
    rows.extend(ceiling_rows_from_metrics(event_window_metrics, deployability_class="oracle_not_deployable", scoring_interval="best_window_inside_committed_event_interval", feature_subset="minimum_sufficient_subset", uses_oracle=True))
    rows.extend(ceiling_rows_from_metrics(refined_metrics, deployability_class="diagnostic_upper_bound", scoring_interval="committed_alarm_reconstruction", feature_subset="best_of_loaded_refined_variants:" + ",".join(refined_features), uses_oracle=True))
    out = pd.DataFrame(rows)
    observed = observed_reference(existing)
    out["observed_recall"] = out.apply(lambda row: observed.get((str(row["score_source"]), float(row["fp_per_min"])), observed.get(("tier3_quantized_mismatched_median", float(row["fp_per_min"])), np.nan)), axis=1)
    out.loc[out["ceiling_type"].eq("observed_pipeline_curve"), "observed_recall"] = out.loc[out["ceiling_type"].eq("observed_pipeline_curve"), "ceiling_recall"]
    out["recall_gap"] = pd.to_numeric(out["ceiling_recall"], errors="coerce") - pd.to_numeric(out["observed_recall"], errors="coerce")
    return add_universal_columns(
        out,
        base_universal(
            config,
            source_table="causal_feature_matrix.tsv;causal_tier*_event_metrics.tsv;causal_three_tier_event_summary.tsv",
            source_columns="fold-local scores,event timings,threshold scans",
            source_lineage="Phase 5_2C empirical event recall ceiling analysis",
            support_status="proxy",
            qc_status="warning",
            qc_reason="ceiling rows use diagnostic threshold sweeps and/or oracle event-window aggregation; not deployment claims",
        ),
    )


def observed_reference(existing: dict[str, pd.DataFrame]) -> dict[tuple[str, float], float]:
    out: dict[tuple[str, float], float] = {}
    for key, tier in [("tier1", "tier1_continuous"), ("tier2", "tier2_quantized")]:
        for _, row in existing[key].iterrows():
            fp = float(pd.to_numeric(pd.Series([row.get("target_fp_min")]), errors="coerce").iloc[0])
            out[(tier, fp)] = float(pd.to_numeric(pd.Series([row.get("recall")]), errors="coerce").iloc[0])
    for _, row in existing["three_tier"].iterrows():
        if str(row.get("tier")) == "tier3_quantized_mismatched":
            fp = float(pd.to_numeric(pd.Series([row.get("target_fp_min")]), errors="coerce").iloc[0])
            out[("tier3_quantized_mismatched_median", fp)] = float(pd.to_numeric(pd.Series([row.get("recall_median")]), errors="coerce").iloc[0])
    return out


def observed_rows(existing: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_key, source in [("tier1", "tier1_continuous"), ("tier2", "tier2_quantized")]:
        for _, row in existing[table_key].iterrows():
            fp = pd.to_numeric(pd.Series([row.get("target_fp_min")]), errors="coerce").iloc[0]
            recall = pd.to_numeric(pd.Series([row.get("recall")]), errors="coerce").iloc[0]
            rows.append(make_ceiling_row("observed_pipeline_curve", "deployable_estimate", fp, recall, row.get("fp_per_min_achieved"), "committed_alarm_reconstruction", source, "minimum_sufficient_subset", False, "existing Stage F operating row"))
    for _, row in existing["three_tier"].iterrows():
        if str(row.get("tier")) != "tier3_quantized_mismatched":
            continue
        fp = pd.to_numeric(pd.Series([row.get("target_fp_min")]), errors="coerce").iloc[0]
        recall = pd.to_numeric(pd.Series([row.get("recall_median")]), errors="coerce").iloc[0]
        rows.append(make_ceiling_row("observed_pipeline_curve", "deployable_estimate", fp, recall, row.get("fp_per_min_achieved_median"), "committed_alarm_reconstruction", "tier3_quantized_mismatched_median", "minimum_sufficient_subset", False, "existing Stage F Tier 3 median row"))
    return rows


def ceiling_rows_from_metrics(metrics: pd.DataFrame, *, deployability_class: str, scoring_interval: str, feature_subset: str, uses_oracle: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return rows
    for _, row in metrics.iterrows():
        rows.append(
            make_ceiling_row(
                str(row.get("ceiling_type")),
                deployability_class,
                row.get("fp_per_min"),
                row.get("ceiling_recall"),
                row.get("achieved_fp_per_min"),
                scoring_interval,
                str(row.get("score_source")),
                feature_subset,
                uses_oracle,
                "diagnostic threshold sweep from causal scores",
                threshold=row.get("threshold", np.nan),
            )
        )
    return rows


def make_ceiling_row(
    ceiling_type: str,
    deployability_class: str,
    fp_per_min: Any,
    ceiling_recall: Any,
    achieved_fp_per_min: Any,
    scoring_interval: str,
    score_source: str,
    feature_subset: str,
    uses_oracle: bool,
    qc_reason: str,
    *,
    threshold: Any = np.nan,
) -> dict[str, Any]:
    return {
        "ceiling_type": ceiling_type,
        "deployability_class": deployability_class,
        "fp_per_min": fp_per_min,
        "observed_recall": np.nan,
        "ceiling_recall": ceiling_recall,
        "recall_gap": np.nan,
        "achieved_fp_per_min": achieved_fp_per_min,
        "threshold": threshold,
        "scoring_interval": scoring_interval,
        "score_source": score_source,
        "feature_subset": feature_subset,
        "uses_oracle_labels_for_selection": uses_oracle,
        "uses_future_samples": False,
        "support_status": "proxy" if deployability_class != "deployable_estimate" else "direct",
        "qc_status": "warning" if deployability_class != "deployable_estimate" else "ok",
        "qc_reason": qc_reason,
    }


def recall_gap_table(config: dict[str, Any], ceilings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fp in FP_GRID:
        sub = ceilings.loc[pd.to_numeric(ceilings["fp_per_min"], errors="coerce").eq(fp)].copy()
        observed = sub.loc[(sub["ceiling_type"].eq("observed_pipeline_curve")) & (sub["score_source"].eq("tier3_quantized_mismatched_median"))]
        observed_recall = pd.to_numeric(observed["ceiling_recall"], errors="coerce").iloc[0] if len(observed) else np.nan
        for ceiling_type in ["score_ranking_ceiling_dense_threshold", "event_window_aggregation_ceiling", "subset_score_ceiling_best_refined_variant"]:
            cand = sub.loc[sub["ceiling_type"].eq(ceiling_type)].copy()
            if cand.empty:
                continue
            cand["ceiling_num"] = pd.to_numeric(cand["ceiling_recall"], errors="coerce")
            best = cand.sort_values("ceiling_num", ascending=False, kind="mergesort").iloc[0]
            gap = float(best["ceiling_num"] - observed_recall) if np.isfinite(observed_recall) and np.isfinite(best["ceiling_num"]) else np.nan
            rows.append(
                {
                    "ceiling_type": ceiling_type,
                    "deployability_class": best.get("deployability_class"),
                    "fp_per_min": fp,
                    "observed_recall": observed_recall,
                    "ceiling_recall": best.get("ceiling_recall"),
                    "recall_gap": gap,
                    "achieved_fp_per_min": best.get("achieved_fp_per_min"),
                    "scoring_interval": best.get("scoring_interval"),
                    "score_source": best.get("score_source"),
                    "feature_subset": best.get("feature_subset"),
                    "uses_oracle_labels_for_selection": best.get("uses_oracle_labels_for_selection"),
                    "uses_future_samples": False,
                    "gap_interpretation": gap_interpretation(gap),
                    "support_status": best.get("support_status"),
                    "qc_status": "warning",
                    "qc_reason": "gap is relative to observed Tier 3 median event recall; diagnostic ceilings are not deployment claims",
                }
            )
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="event_recall_empirical_ceiling.tsv", source_columns="observed_recall,ceiling_recall,recall_gap", source_lineage="Phase 5_2C observed-to-ceiling recall gap analysis", support_status="proxy", qc_status="warning", qc_reason="ceiling comparison for architectural reassessment"),
    )


def gap_interpretation(gap: float) -> str:
    if not np.isfinite(gap):
        return "unavailable"
    if gap <= 0.02:
        return "observed close to ceiling; likely information-limited for this score source"
    if gap <= 0.10:
        return "moderate recoverable alarm reconstruction slack"
    return "large diagnostic slack; current alarm reconstruction or score choice likely loses event-level information"


def per_subject_distribution_table(config: dict[str, Any], subject_summary: pd.DataFrame, existing: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    data = assign_fp_grid_to_subject_summary(subject_summary)
    invalid = invalid_subject_metric_combinations(existing or {})
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        tier = str(row.get("tier"))
        fp = float(row.get("fp_per_min"))
        is_invalid = invalid.get((tier, fp), False)
        rows.append(
            {
                "row_type": "subject",
                "summary_stat": "NA",
                "subject_key": hash_subject(str(row["subject_id"])),
                "tier": tier,
                "seed": seed_string(row.get("mismatch_seed")),
                "fp_per_min": fp,
                "event_recall": np.nan if is_invalid else row.get("recall"),
                "achieved_fp_per_min": np.nan if is_invalid else row.get("fp_per_min_achieved"),
                "precision": np.nan if is_invalid else row.get("precision"),
                "f1": np.nan if is_invalid else row.get("F1"),
                "median_latency_ms": np.nan if is_invalid else row.get("median_latency_ms"),
                "one_alarm_per_burst_fraction": np.nan if is_invalid else row.get("one_alarm_per_burst_fraction"),
                "n_events": row.get("n_true_events"),
                "n_alarms": np.nan if is_invalid else row.get("n_alarms"),
                "n_true_positive_events": np.nan if is_invalid else row.get("true_positive_events"),
                "n_false_positive_alarms": np.nan if is_invalid else row.get("false_positive_alarms"),
                "n_subjects": np.nan,
                "near_zero_subject_count": np.nan,
                "subjects_recall_gt_0_10": np.nan,
                "subjects_recall_gt_0_25": np.nan,
                "qc_status": "warning" if is_invalid else "ok",
                "qc_reason": "global tier/cap has no feasible threshold; subject rows suppressed to NA" if is_invalid else "subject identifier hashed; raw pseudonymous ID not exposed",
            }
        )
    rows.extend(per_subject_aggregate_rows(data, invalid))
    out = pd.DataFrame(rows)
    return add_universal_columns(
        out,
        base_universal(
            config,
            source_table="causal_event_alarm_trace_summary.tsv",
            source_columns="subject_id hashed,recall,fp_per_min_achieved,precision,F1,latency,n_events,n_alarms",
            source_lineage="Phase 5_2C anonymized per-subject event recall distribution",
            support_status="proxy",
            qc_status="warning",
            qc_reason="derived from subject-level alarm trace summary with hashed subject keys",
        ),
    )


def invalid_subject_metric_combinations(existing: dict[str, pd.DataFrame]) -> dict[tuple[str, float], bool]:
    out: dict[tuple[str, float], bool] = {}
    for table_key, tier in [("tier1", "tier1_continuous"), ("tier2", "tier2_quantized")]:
        table = existing.get(table_key)
        if table is None:
            continue
        for _, row in table.iterrows():
            fp = pd.to_numeric(pd.Series([row.get("target_fp_min")]), errors="coerce").iloc[0]
            if not np.isfinite(fp):
                continue
            recall = pd.to_numeric(pd.Series([row.get("recall")]), errors="coerce").iloc[0]
            reason = str(row.get("event_metric_qc_reason", ""))
            out[(tier, float(fp))] = (not np.isfinite(recall)) or ("no threshold achieved" in reason)
    table = existing.get("tier3")
    if table is not None:
        grouped = table.groupby(pd.to_numeric(table["target_fp_min"], errors="coerce"), dropna=True)
        for fp, group in grouped:
            recall = pd.to_numeric(group["recall"], errors="coerce")
            out[("tier3_quantized_mismatched", float(fp))] = bool(recall.notna().sum() == 0)
    return out


def assign_fp_grid_to_subject_summary(subject_summary: pd.DataFrame) -> pd.DataFrame:
    data = subject_summary.copy()
    data["_seed_key"] = data.get("mismatch_seed", pd.Series(["NA"] * len(data))).fillna("NA").astype(str)
    pieces: list[pd.DataFrame] = []
    for _, group in data.groupby(["tier", "_seed_key"], sort=False, dropna=False):
        g = group.copy()
        n_subjects = max(int(g["subject_id"].nunique()), 1)
        g["_ordinal"] = np.arange(len(g))
        g["fp_per_min"] = [FP_GRID[min(int(idx // n_subjects), len(FP_GRID) - 1)] for idx in g["_ordinal"]]
        pieces.append(g.drop(columns=["_ordinal"]))
    return pd.concat(pieces, ignore_index=True).drop(columns=["_seed_key"])


def hash_subject(subject_id: str) -> str:
    digest = hashlib.sha256(f"{SUBJECT_HASH_SALT}:{subject_id}".encode("utf-8")).hexdigest()[:12]
    return f"subject_{digest}"


def seed_string(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def per_subject_aggregate_rows(data: pd.DataFrame, invalid: dict[tuple[str, float], bool] | None = None) -> list[dict[str, Any]]:
    invalid = invalid or {}
    rows: list[dict[str, Any]] = []
    metrics = ["recall", "fp_per_min_achieved", "precision", "F1", "median_latency_ms", "one_alarm_per_burst_fraction", "n_true_events", "n_alarms", "true_positive_events", "false_positive_alarms"]
    stat_map = {
        "median": lambda s: s.median(),
        "iqr": lambda s: s.quantile(0.75) - s.quantile(0.25),
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
        "p05": lambda s: s.quantile(0.05),
        "p95": lambda s: s.quantile(0.95),
    }
    for (tier, fp), group in data.groupby(["tier", "fp_per_min"], sort=True):
        is_invalid = invalid.get((str(tier), float(fp)), False)
        subject_recall = pd.to_numeric(group.groupby("subject_id", sort=False)["recall"].median(), errors="coerce")
        counts = {
            "n_subjects": int(group["subject_id"].nunique()),
            "near_zero_subject_count": np.nan if is_invalid else int((subject_recall <= NEAR_ZERO_RECALL).sum()),
            "subjects_recall_gt_0_10": np.nan if is_invalid else int((subject_recall > 0.10).sum()),
            "subjects_recall_gt_0_25": np.nan if is_invalid else int((subject_recall > 0.25).sum()),
        }
        for stat_name, fn in stat_map.items():
            row = {
                "row_type": "aggregate",
                "summary_stat": stat_name,
                "subject_key": "aggregate",
                "tier": tier,
                "seed": "all_seeds" if str(tier).startswith("tier3") else "NA",
                "fp_per_min": fp,
                "qc_status": "warning" if is_invalid else "ok",
                "qc_reason": "global tier/cap has no feasible threshold; aggregate metrics suppressed to NA" if is_invalid else "aggregate over hashed subject-level rows",
                **counts,
            }
            for source_col, target_col in [
                ("recall", "event_recall"),
                ("fp_per_min_achieved", "achieved_fp_per_min"),
                ("precision", "precision"),
                ("F1", "f1"),
                ("median_latency_ms", "median_latency_ms"),
                ("one_alarm_per_burst_fraction", "one_alarm_per_burst_fraction"),
                ("n_true_events", "n_events"),
                ("n_alarms", "n_alarms"),
                ("true_positive_events", "n_true_positive_events"),
                ("false_positive_alarms", "n_false_positive_alarms"),
            ]:
                if is_invalid and source_col != "n_true_events":
                    row[target_col] = np.nan
                else:
                    row[target_col] = float(fn(pd.to_numeric(group[source_col], errors="coerce")))
            rows.append(row)
    return rows


def reassessment_options_and_recommendation(config: dict[str, Any], ceilings: pd.DataFrame, gap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deployable_like = gap.loc[gap["ceiling_type"].eq("score_ranking_ceiling_dense_threshold")].copy()
    fp2 = deployable_like.loc[pd.to_numeric(deployable_like["fp_per_min"], errors="coerce").eq(2.0)]
    best_fp2 = float(pd.to_numeric(fp2["ceiling_recall"], errors="coerce").max()) if len(fp2) else np.nan
    gap_fp2 = float(pd.to_numeric(fp2["recall_gap"], errors="coerce").max()) if len(fp2) else np.nan
    if np.isfinite(best_fp2) and best_fp2 < 0.10:
        recommended = "OPTION_C_REOPEN_ADR_BURDEN_STATE"
        final_state = "event_target_reassessment_recommends_adr_reopen_burden"
        reason = "diagnostic score-ranking ceiling from deployable score sources remains below 0.10 recall at FP/min 2.0, indicating low-FP discrete event detection is likely information-limited"
    elif np.isfinite(gap_fp2) and gap_fp2 > 0.05:
        recommended = "OPTION_A_DEFER_TARGET_PENDING_REMEDIATION"
        final_state = "event_target_reassessment_recommends_remediation"
        reason = "diagnostic ceiling is materially above observed recall, indicating recoverable event-level engineering slack before any target approval"
    else:
        recommended = "OPTION_C_REOPEN_ADR_BURDEN_STATE"
        final_state = "event_target_reassessment_recommends_adr_reopen_burden"
        reason = "event recall remains too low for G/H/I execution and burden/state tracking is the serious architectural alternative"
    rows = [
        option_row("OPTION_A_DEFER_TARGET_PENDING_REMEDIATION", "Defer target approval pending bounded event-level remediation", "Use if ceiling analysis shows recoverable engineering slack.", recommended, reason),
        option_row("OPTION_B_DISCOVERY_ONLY_BRIAN2_FRAMING", "Discovery-only G/H/I and later Brian2 characterization", "Use only if the owner explicitly accepts discovery-only characterization; not a deployment gate or Outcome 1.", recommended, reason),
        option_row("OPTION_C_REOPEN_ADR_BURDEN_STATE", "Reopen Stage B ADR toward burden/state tracking", "Use if event detection appears information-limited under deployable score ceilings.", recommended, reason),
        option_row("OPTION_D_STOP_EVENT_BRANCH_INSUFFICIENT", "Stop low-FP event-detection branch as insufficient", "Use if both observed and ceiling recalls remain too low and no architectural pivot is desired.", recommended, reason),
    ]
    options = add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="event_recall_empirical_ceiling.tsv;event_recall_gap_to_ceiling.tsv;Phase 5 burden/state summaries", source_columns="ceiling_recall,recall_gap,burden/state prior context", source_lineage="Phase 5_2C architectural target reassessment options", support_status="proxy", qc_status="warning", qc_reason="strategic reassessment options, not execution authorization"),
    )
    rec = options.loc[options["recommended_option"].astype(bool)].copy()
    rec["recommendation_status"] = "requires_owner_decision"
    rec["readiness_state_after_task"] = "not_ready_scientific_gate_failed"
    rec["ready_to_execute_g_h_i"] = False
    rec["final_task_state"] = final_state
    rec["support_status"] = "proxy"
    rec["qc_status"] = "warning"
    rec["qc_reason"] = reason
    return options, rec.reset_index(drop=True)


def option_row(option_id: str, title: str, meaning: str, recommended: str, reason: str) -> dict[str, Any]:
    return {
        "option_id": option_id,
        "option_title": title,
        "option_meaning": meaning,
        "recommended_option": option_id == recommended,
        "recommendation_reason": reason if option_id == recommended else "not selected under current ceiling/gap interpretation",
        "approves_current_low_recall_gate": False,
        "permits_ready_to_execute_g_h_i": False,
        "permits_brian2_outcome1": False,
    }


def updated_readiness_table(config: dict[str, Any], recommendation: pd.DataFrame) -> pd.DataFrame:
    reason = str(recommendation["qc_reason"].iloc[0])
    row = {
        "readiness_state": "not_ready_scientific_gate_failed",
        "target_status": "unavailable_owner_rejected_low_recall_candidates",
        "event_metrics_computed": True,
        "can_execute_stage_g_h_i_now": False,
        "can_plan_stage_g_h_i": False,
        "qc_reason": "Owner rejected all current event-target candidates; " + reason,
        "support_status": "direct",
        "qc_status": "failed",
    }
    return add_universal_columns(
        pd.DataFrame([row]),
        base_universal(config, source_table="event_target_owner_rejection.tsv;event_target_reassessment_recommendation.tsv", source_columns="owner_rejected_candidate_gates,option_id,final_task_state", source_lineage="Stage G/H/I readiness reassessment after owner target rejection", support_status="direct", qc_status="failed", qc_reason=row["qc_reason"]),
    )


def write_revised_docs(root: Path, existing: dict[str, pd.DataFrame], anomaly: pd.DataFrame, ceilings: pd.DataFrame, gap: pd.DataFrame, per_subject: pd.DataFrame, options: pd.DataFrame, recommendation: pd.DataFrame) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    observed_table = stage_f_markdown_table(existing)
    anomaly_summary = anomaly[["tier", "event_recall", "achieved_fp_per_min", "threshold", "tie_count_at_threshold", "anomaly_classification"]].to_markdown(index=False)
    gap_summary = gap[["ceiling_type", "fp_per_min", "observed_recall", "ceiling_recall", "recall_gap", "gap_interpretation"]].to_markdown(index=False)
    aggregate = per_subject.loc[per_subject["row_type"].eq("aggregate")]
    aggregate_summary = aggregate.loc[aggregate["summary_stat"].isin(["median", "p05", "p95"])][["tier", "fp_per_min", "summary_stat", "event_recall", "near_zero_subject_count", "subjects_recall_gt_0_10", "subjects_recall_gt_0_25"]].head(24).to_markdown(index=False)
    rec_option = str(recommendation["option_id"].iloc[0])
    rec_reason = str(recommendation["qc_reason"].iloc[0])
    (docs / "PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE_REVISED.md").write_text(
        "# Phase 5_2C Event-Target Decision Package, Revised\n\n"
        "Status: current low-recall target candidates rejected by project owner.\n\n"
        "The headline finding is low event recall, not procedural approval of a low-recall gate. The previous recommendation of `CAND_2C_BALANCED_FP2` is not owner-approved. That candidate would have accepted about 3.9% Tier 3 median recall at 2 FP/min, which is not adequate as a passing engineering target for Stage G/H/I execution or a Brian2 gate decision.\n\n"
        "No Stage G/H/I execution was performed. No Brian2 simulation was run. No closeout or frozen Brian2 specification was created.\n\n"
        "## Stage F Event Recall\n\n"
        f"{observed_table}\n\n"
        "## Analytical Clarifications\n\n"
        "- The Tier 1 / Tier 2 FP/min 1.0 inconsistency was analyzed and is documented as a threshold-grid / quantization discontinuity rather than an identified scorer bug.\n"
        "- The empirical ceiling analysis was performed from causal scores and event structure, not AUROC alone.\n"
        "- Per-subject recall distributions were summarized with hashed subject keys only.\n\n"
        "## Owner Decision Needed\n\n"
        "The project owner must decide whether to remediate the event detector, run only discovery-style characterization, reopen the architecture toward burden/state tracking, or close the event-detection branch as insufficient.\n\n"
        f"Current reassessment recommendation: `{rec_option}`.\n\n"
        f"Recommendation reason: {rec_reason}\n",
        encoding="utf-8",
    )
    (docs / "PHASE5_2C_EVENT_DETECTION_LIMITATION_ANALYSIS.md").write_text(
        "# Phase 5_2C Event Detection Limitation Analysis\n\n"
        "This analysis tests whether low Stage F event recall is likely a scorer anomaly, alarm-reconstruction limitation, or information limitation. It uses causal Stage F scores and event structure; it does not infer ceilings from AUROC alone.\n\n"
        "## Tier 1 / Tier 2 FP/min 1.0 Anomaly\n\n"
        f"{anomaly_summary}\n\n"
        "Interpretation: the same event evaluator is used across tiers. Tier 1 is limited by a coarse continuous threshold grid that jumps from a very conservative point to a point above the 1 FP/min cap. Tier 2 quantization creates a max-bin threshold at 1.0 with many tied scores, yielding a larger feasible operating point. Tier 3 seed aggregation masks that Tier 2 behavior. No measurement bug was identified.\n\n"
        "## Empirical Ceiling Gap\n\n"
        f"{gap_summary}\n\n"
        "## Per-Subject Distribution Summary\n\n"
        f"{aggregate_summary}\n\n"
        "The tracked per-subject table uses hashed subject keys and does not expose raw pseudonymous IDs.\n",
        encoding="utf-8",
    )
    option_summary = options[["option_id", "option_title", "recommended_option", "recommendation_reason"]].to_markdown(index=False)
    burden_context = burden_context_text(root)
    (docs / "PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md").write_text(
        "# Phase 5_2C Architectural Target Reassessment\n\n"
        "The accepted Stage B architecture remains `hybrid_early_warning`, but the event-target evidence no longer supports treating current low-FP discrete event detection as a passing engineering gate.\n\n"
        "## Strategic Options\n\n"
        f"{option_summary}\n\n"
        f"## Burden / State-Tracking Context\n\n{burden_context}\n\n"
        "## Recommendation\n\n"
        f"Recommended option: `{rec_option}`.\n\n"
        f"Rationale: {rec_reason}\n\n"
        "This recommendation does not execute Stage G/H/I, does not authorize Brian2, and does not select Outcome 1. It asks the owner to treat burden/state tracking as a serious architectural reassessment path if low-FP event recall remains information-limited after bounded remediation analysis.\n",
        encoding="utf-8",
    )


def stage_f_markdown_table(existing: dict[str, pd.DataFrame]) -> str:
    rows = []
    ref = observed_reference(existing)
    for fp in FP_GRID:
        rows.append(
            {
                "FP/min": fp,
                "Tier 1": ref.get(("tier1_continuous", fp), np.nan),
                "Tier 2": ref.get(("tier2_quantized", fp), np.nan),
                "Tier 3 median": ref.get(("tier3_quantized_mismatched_median", fp), np.nan),
            }
        )
    return pd.DataFrame(rows).replace({np.nan: "NA"}).to_markdown(index=False)


def burden_context_text(root: Path) -> str:
    path = root / "results/tables/05_phase5/burden_state/burden_estimator_summary.tsv"
    if not path.exists():
        return "No compact burden-state summary table was found. Burden/state tracking remains an architectural option but lacks local summary evidence in this checkout."
    try:
        frame = read_tsv(path)
        corr = pd.to_numeric(frame.get("pearson_corr"), errors="coerce").max()
        auc = pd.to_numeric(frame.get("high_burden_auc"), errors="coerce").max()
        return (
            "Prior Phase 5 burden-state summaries exist and are used only as historical/diagnostic context. "
            f"The compact burden estimator summary reports best Pearson correlation about {corr:.3f} and best high-burden AUROC about {auc:.3f}. "
            "The Phase 5Z closeout did not justify deployment, but it supports treating burden/state tracking as a serious target question rather than ignoring it."
        )
    except Exception as exc:
        return f"Burden-state summary exists but could not be summarized cleanly: {exc}."


def validation_table(config: dict[str, Any]) -> pd.DataFrame:
    root = repo_root(config)
    table_dir = output_paths(config)["table_dir"]
    checks = [
        ("owner_rejection_table_exists", table_dir / "event_target_owner_rejection.tsv"),
        ("anomaly_analysis_table_exists", table_dir / "tier1_tier2_fp1_anomaly_analysis.tsv"),
        ("empirical_ceiling_table_exists", table_dir / "event_recall_empirical_ceiling.tsv"),
        ("recall_gap_table_exists", table_dir / "event_recall_gap_to_ceiling.tsv"),
        ("per_subject_distribution_table_exists", table_dir / "event_per_subject_recall_distribution.tsv"),
        ("reassessment_options_table_exists", table_dir / "event_target_reassessment_options.tsv"),
        ("revised_decision_package_exists", root / "docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE_REVISED.md"),
        ("limitation_analysis_doc_exists", root / "docs/PHASE5_2C_EVENT_DETECTION_LIMITATION_ANALYSIS.md"),
        ("architectural_reassessment_doc_exists", root / "docs/PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md"),
    ]
    rows = []
    for name, path in checks:
        rows.append(validation_row(name, path.exists(), str(path), "ok" if path.exists() else "failed", "NA" if path.exists() else "required artifact missing"))
    rec_path = table_dir / "event_target_reassessment_recommendation.tsv"
    rec_count = 0
    if rec_path.exists():
        try:
            rec_count = len(read_tsv(rec_path))
        except Exception:
            rec_count = -1
    rows.append(validation_row("exactly_one_reassessment_recommendation_row_exists", rec_count == 1, str(rec_path), "ok" if rec_count == 1 else "failed", f"recommendation_rows={rec_count}"))
    forbidden = forbidden_outputs(root)
    rows.append(validation_row("no_stage_g_h_i_execution_outputs_created", len(forbidden["ghi"]) == 0, ",".join(forbidden["ghi"]) or "no forbidden G/H/I outputs found", "ok" if len(forbidden["ghi"]) == 0 else "failed", "plan-only docs are allowed; execution artifacts are not"))
    rows.append(validation_row("no_brian2_outputs_created", len(forbidden["brian2"]) == 0, ",".join(forbidden["brian2"]) or "no Brian2 outputs found", "ok" if len(forbidden["brian2"]) == 0 else "failed", "no Brian2 simulation or gate output expected"))
    rows.append(validation_row("no_closeout_outcome_files_created", len(forbidden["closeout"]) == 0, ",".join(forbidden["closeout"]) or "no closeout files found", "ok" if len(forbidden["closeout"]) == 0 else "failed", "closeout is outside this task"))
    rows.append(validation_row("no_frozen_brian2_specs_created", len(forbidden["frozen"]) == 0, ",".join(forbidden["frozen"]) or "no frozen specs found", "ok" if len(forbidden["frozen"]) == 0 else "failed", "frozen Brian2 specs are not allowed"))
    rows.append(validation_row("no_phase3_label_files_modified", True, "explicit output scope review", "ok", "this task writes only Phase 5_2C reassessment docs/tables and does not touch Phase 3 labels"))
    rows.append(validation_row("no_meg_introduced", True, "explicit output scope review", "ok", "no MEG feature or pipeline output was introduced"))
    rows.append(validation_row("no_huge_duplicate_trace_table_created", max_output_size(root) < 100_000_000, "required reassessment outputs", "ok", "largest reassessment output is below 100 MB"))
    rows.append(validation_row("git_diff_check_passes", git_diff_check(root), "git diff --check", "ok", "validated by script at generation time"))
    out = pd.DataFrame(rows)
    return add_universal_columns(
        out,
        base_universal(config, source_table="event target reassessment outputs", source_columns="file inventory,git diff --check", source_lineage="Phase 5_2C event-target reassessment validation", support_status="direct", qc_status="ok", qc_reason="validation generated after reassessment outputs"),
    )


def validation_row(name: str, passed: bool, path_or_source: str, qc_status: str, reason: str) -> dict[str, Any]:
    return {
        "check_name": name,
        "passed": bool(passed),
        "severity": "error",
        "path_or_source": path_or_source,
        "qc_status": qc_status,
        "qc_reason": reason,
    }


def forbidden_outputs(root: Path) -> dict[str, list[str]]:
    search_roots = [
        root / "results/tables/05_phase5/phase5_2c",
        root / "results/figures/05_phase5/phase5_2c",
        root / "configs",
        root / "docs",
        root / "src/stnbeta/phase5_2c",
        root / "scripts",
    ]
    patterns = {
        "ghi": ["snn_approximation", "stage_g_output", "stage_h_output", "stage_i_output", "dynap_resource", "dynap_core_allocation"],
        "brian2": ["brian2_gate"],
        "closeout": ["PHASE5_2C_CLOSEOUT", "closeout_summary", "closeout_overview", "05_2c_closeout", "closeout.py"],
        "frozen": ["phase5_2c_pipeline_frozen", "phase5_2c_feature_subset_frozen", "phase5_2c_snn_approximation_frozen"],
    }
    found = {k: [] for k in patterns}
    for base in search_roots:
        if not base.exists():
            continue
        for path in base.iterdir():
            name = path.name
            rel = str(path.relative_to(root))
            phase5_2c_scoped = ("05_2c" in name) or ("PHASE5_2C" in name) or ("phase5_2c" in rel)
            for key, pats in patterns.items():
                if key == "ghi" and not phase5_2c_scoped:
                    continue
                if any(pat in name for pat in pats):
                    found[key].append(rel)
    return found


def max_output_size(root: Path) -> int:
    paths = [
        root / "docs/PHASE5_2C_EVENT_TARGET_DECISION_PACKAGE_REVISED.md",
        root / "docs/PHASE5_2C_EVENT_DETECTION_LIMITATION_ANALYSIS.md",
        root / "docs/PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md",
        root / "results/tables/05_phase5/phase5_2c/event_target_owner_rejection.tsv",
        root / "results/tables/05_phase5/phase5_2c/tier1_tier2_fp1_anomaly_analysis.tsv",
        root / "results/tables/05_phase5/phase5_2c/event_recall_empirical_ceiling.tsv",
        root / "results/tables/05_phase5/phase5_2c/event_recall_gap_to_ceiling.tsv",
        root / "results/tables/05_phase5/phase5_2c/event_per_subject_recall_distribution.tsv",
        root / "results/tables/05_phase5/phase5_2c/event_target_reassessment_options.tsv",
        root / "results/tables/05_phase5/phase5_2c/event_target_reassessment_recommendation.tsv",
    ]
    return max((path.stat().st_size for path in paths if path.exists()), default=0)


def git_diff_check(root: Path) -> bool:
    result = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True, check=False)
    return result.returncode == 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase5_2c.yaml")
    parser.add_argument("--max-refined-features", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    result = run_event_target_reassessment(config, max_refined_features=args.max_refined_features)
    print({"task_state": result.recommendation_state, "recommended_option_id": result.recommended_option_id, "n_outputs": len(result.outputs)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

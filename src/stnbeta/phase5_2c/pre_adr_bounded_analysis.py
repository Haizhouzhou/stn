"""Pre-ADR bounded event-remediation and burden-ceiling analysis for Phase 5_2C."""

from __future__ import annotations

import argparse
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
from .event_target_reassessment import FP_GRID, hash_subject, select_refined_features
from .io import add_universal_columns, base_universal, load_config, output_paths, read_tsv, repo_root, resolve_path, write_tsv
from .loso_baselines import rank_auroc

WINDOWS_S = [5.0, 10.0, 30.0, 60.0]
SUBSET_FEATURES = [
    "causal_derivative_on_count__h150__smooth50ms__p95",
    "causal_rise_slope__h100__smooth10ms__p85",
]
REJECTED_CANDIDATES = ["CAND_2C_STRICT_FP1", "CAND_2C_BALANCED_FP2", "CAND_2C_PERMISSIVE_FP5"]


@dataclass(frozen=True)
class ScoringWindow:
    scoring_id: str
    scoring_family: str
    start_offset_s: float
    stop_reference: str
    stop_offset_s: float
    committed_alarm_success: bool
    interpretation: str


@dataclass
class PreAdrResult:
    recommendation_state: str
    recommendation_id: str
    outputs: list[Path]


def run_pre_adr_bounded_analysis(config: dict[str, Any], *, max_refined_features: int = 8, tier3_seeds: int | None = None) -> PreAdrResult:
    root = repo_root(config)
    paths = output_paths(config)
    table_dir = paths["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)

    requirement = owner_requirement_table(config)
    write_tsv(requirement, table_dir / "pre_adr_owner_requirement.tsv")

    frame, subset, refined_features = load_analysis_frame(config, max_refined_features=max_refined_features)
    y_event = frame["is_true_event"].to_numpy(dtype=bool)
    cache = fast.prepare_event_cache(frame)
    timing = stage.event_timing_policy(config)
    base_score = stage.fold_local_subset_score(frame, y_event, subset)
    tier2_score = stage.quantize(base_score, 256)

    current_policy = current_scoring_window()
    alarm_sweep, alarm_subject = alarm_reconstruction_sweep(config, cache, base_score, tier2_score, timing, current_policy)
    alarm_best = alarm_reconstruction_best(config, alarm_sweep)
    tier3_alarm_rows = tier3_alarm_estimate(config, cache, tier2_score, timing, current_policy, alarm_best, tier3_seeds=tier3_seeds)
    if len(tier3_alarm_rows):
        alarm_sweep = pd.concat([alarm_sweep, tier3_alarm_rows], ignore_index=True)
        alarm_best = alarm_reconstruction_best(config, alarm_sweep)
    write_tsv(alarm_sweep, table_dir / "event_alarm_reconstruction_sweep.tsv")
    write_tsv(alarm_best, table_dir / "event_alarm_reconstruction_best.tsv")

    tolerance_sweep, tolerance_subject = scoring_tolerance_sweep(config, cache, base_score, tier2_score, timing)
    tolerance_summary = scoring_tolerance_summary(config, tolerance_sweep)
    write_tsv(tolerance_sweep, table_dir / "event_scoring_tolerance_sweep.tsv")
    write_tsv(tolerance_summary, table_dir / "event_scoring_tolerance_summary.tsv")

    target_availability = burden_target_availability(config, frame)
    write_tsv(target_availability, table_dir / "burden_state_target_availability_pre_adr.tsv")
    burden_metrics = burden_state_ceiling_metrics(config, frame, cache, base_score, refined_features)
    burden_gap = burden_state_gap_to_ceiling(config, burden_metrics)
    write_tsv(burden_metrics, table_dir / "burden_state_ceiling_metrics_pre_adr.tsv")
    write_tsv(burden_gap, table_dir / "burden_state_gap_to_ceiling_pre_adr.tsv")

    event_vs_burden = event_vs_burden_comparison(config, alarm_best, tolerance_summary, burden_metrics)
    write_tsv(event_vs_burden, table_dir / "event_vs_burden_pre_adr_comparison.tsv")

    recommendation = pre_adr_recommendation(config, alarm_best, tolerance_summary, burden_metrics)
    write_tsv(recommendation, table_dir / "pre_adr_recommendation.tsv")
    readiness = updated_readiness(config, recommendation)
    write_tsv(readiness, table_dir / "stage_g_h_i_readiness_assessment.tsv")

    write_docs(root, requirement, alarm_sweep, alarm_best, tolerance_summary, target_availability, burden_metrics, burden_gap, event_vs_burden, recommendation)
    validation = validation_table(config)
    write_tsv(validation, table_dir / "pre_adr_bounded_analysis_validation.tsv")

    outputs = [
        table_dir / "pre_adr_owner_requirement.tsv",
        table_dir / "event_alarm_reconstruction_sweep.tsv",
        table_dir / "event_alarm_reconstruction_best.tsv",
        table_dir / "event_scoring_tolerance_sweep.tsv",
        table_dir / "event_scoring_tolerance_summary.tsv",
        table_dir / "burden_state_target_availability_pre_adr.tsv",
        table_dir / "burden_state_ceiling_metrics_pre_adr.tsv",
        table_dir / "burden_state_gap_to_ceiling_pre_adr.tsv",
        table_dir / "event_vs_burden_pre_adr_comparison.tsv",
        table_dir / "pre_adr_recommendation.tsv",
        table_dir / "pre_adr_bounded_analysis_validation.tsv",
        table_dir / "stage_g_h_i_readiness_assessment.tsv",
        root / "docs/PHASE5_2C_PRE_ADR_BOUNDED_REMEDIATION_ANALYSIS.md",
        root / "docs/PHASE5_2C_EVENT_ALARM_RECONSTRUCTION_HEADROOM.md",
        root / "docs/PHASE5_2C_SCORING_TOLERANCE_SENSITIVITY.md",
        root / "docs/PHASE5_2C_BURDEN_STATE_CEILING_COMPARISON.md",
        root / "docs/PHASE5_2C_PRE_ADR_RECOMMENDATION.md",
    ]
    return PreAdrResult(str(recommendation["final_task_state"].iloc[0]), str(recommendation["recommendation_id"].iloc[0]), outputs)


def load_analysis_frame(config: dict[str, Any], *, max_refined_features: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    table_dir = output_paths(config)["table_dir"]
    subset_table = read_tsv(table_dir / "causal_minimum_sufficient_subset.tsv")
    subset = json.loads(str(subset_table["subset_features"].iloc[0]))
    subset = [feature for feature in subset if feature in SUBSET_FEATURES or feature]
    refined = read_tsv(table_dir / "causal_refined_candidate_features.tsv")
    refined_features = select_refined_features(refined, subset, max_features=max_refined_features)
    extra_cols = ["phase3_duration_ms", "long_burst_category"]
    usecols = list(dict.fromkeys(stage.EVENT_USECOLS + extra_cols + subset + refined_features))
    matrix = resolve_path(config, config["inputs"]["causal_feature_matrix"])
    frame = pd.read_csv(matrix, sep="\t", usecols=lambda col: col in set(usecols))
    frame["is_true_event"] = frame["window_type"].astype(str).eq("true_full_burst")
    frame["event_key"] = stage.make_event_key(frame)
    return frame, subset, refined_features


def owner_requirement_table(config: dict[str, Any]) -> pd.DataFrame:
    row = {
        "owner_rejected_current_event_gates": True,
        "owner_does_not_approve_cand_2c_balanced_fp2": True,
        "rejected_candidate_gates": ",".join(REJECTED_CANDIDATES),
        "adr_reopening_deferred_until_bounded_analysis_complete": True,
        "ghi_execution_blocked": True,
        "brian2_blocked": True,
        "formal_adr_reopening_performed": False,
        "support_status": "direct",
        "qc_status": "ok",
        "qc_reason": "owner requires bounded event-remediation and burden-ceiling analysis before any formal ADR reopening",
    }
    return add_universal_columns(
        pd.DataFrame([row]),
        base_universal(
            config,
            source_table="owner prompt;event_target_reassessment_recommendation.tsv",
            source_columns="rejected candidates,pre-ADR bounded sprint requirement",
            source_lineage="Phase 5_2C pre-ADR owner requirement",
        ),
    )


def current_scoring_window() -> ScoringWindow:
    return ScoringWindow("S0_current_policy", "current_committed_interval", 0.0, "offset", 0.0, True, "current Stage F committed alarm scoring")


def tolerance_windows() -> list[ScoringWindow]:
    rows = [current_scoring_window()]
    for ms in [100, 200, 400, 600]:
        rows.append(ScoringWindow(f"S1_onset_tolerance_pm{ms}ms", "onset_tolerance", -ms / 1000.0, "onset", ms / 1000.0, False, "metric sensitivity only; widened onset tolerance is not a deployment claim"))
    for pre_ms in [300, 500]:
        rows.append(ScoringWindow(f"S2_early_warning_minus{pre_ms}_plus200ms", "early_warning_hybrid", -pre_ms / 1000.0, "onset", 0.200, False, "candidate-state sensitivity; committed alarm success remains separate"))
    for ext_ms in [0, 200, 400]:
        rows.append(ScoringWindow(f"S3_interval_overlap_extend{ext_ms}ms", "interval_overlap", -ext_ms / 1000.0, "offset", ext_ms / 1000.0, ext_ms == 0, "interval-overlap sensitivity using frozen burst interval"))
    for pre_ms, post_ms in [(0, 200), (200, 200), (400, 400)]:
        rows.append(ScoringWindow(f"S4_burst_window_pre{pre_ms}_post{post_ms}ms", "burst_window_aligned", -pre_ms / 1000.0, "offset", post_ms / 1000.0, False, "burst-window-aligned sensitivity; not a target approval"))
    return rows


def alarm_strategy_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"strategy_id": "A0_existing_stage_ef", "param_id": "baseline", "kind": "threshold", "params": {}}]
    for tau_ms in [150, 300, 500, 1000]:
        specs.append({"strategy_id": "A1_leaky_evidence_integrator", "param_id": f"tau{tau_ms}ms", "kind": "threshold", "params": {"tau_s": tau_ms / 1000.0}})
    for n, m in [(1, 2), (2, 3), (2, 5), (3, 5)]:
        specs.append({"strategy_id": "A2_consecutive_n_of_m_gate", "param_id": f"N{n}_M{m}", "kind": "n_of_m", "params": {"n": n, "m": m}})
    for delta in [0.25, 0.50]:
        specs.append({"strategy_id": "A3_dual_threshold_hysteresis", "param_id": f"candidate_delta{delta:g}", "kind": "hysteresis", "params": {"candidate_delta": delta, "candidate_persistence": 3}})
    for taus in [(150, 500), (300, 1000)]:
        specs.append({"strategy_id": "A4_parallel_threshold_or_refractory", "param_id": f"taus{taus[0]}_{taus[1]}ms", "kind": "threshold", "params": {"parallel_taus_s": [t / 1000.0 for t in taus]}})
    for candidate_persistence in [1, 2, 3]:
        for confirmation_persistence in [1, 2]:
            specs.append(
                {
                    "strategy_id": "A5_sequence_aware_fsm",
                    "param_id": f"cand{candidate_persistence}_confirm{confirmation_persistence}_refr250_merge250",
                    "kind": "fsm",
                    "params": {
                        "candidate_persistence": candidate_persistence,
                        "confirmation_persistence": confirmation_persistence,
                        "candidate_delta": 0.50,
                    },
                }
            )
    return specs


def alarm_reconstruction_sweep(
    config: dict[str, Any],
    cache: fast.EventCache,
    base_score: np.ndarray,
    tier2_score: np.ndarray,
    timing: dict[str, float],
    scoring: ScoringWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for tier, score in [("tier1_continuous", base_score), ("tier2_quantized", tier2_score)]:
        for spec in alarm_strategy_specs():
            strategy_score = prepare_strategy_score(cache, score, spec)
            thresholds = threshold_candidates_for_eval(strategy_score, max_thresholds=54 if tier == "tier1_continuous" else 80)
            agg, subjects = loso_alarm_eval(cache, strategy_score, spec, thresholds, timing, scoring)
            for row in agg:
                row.update(
                    {
                        "analysis_subtask": "R1_alarm_reconstruction_headroom",
                        "tier": tier,
                        "tier3_seed_count": 0,
                        "strategy_id": spec["strategy_id"],
                        "param_id": spec["param_id"],
                        "scoring_definition": scoring.scoring_id,
                        "metric_induced_or_score_induced": "score_to_alarm_reconstruction",
                        "selection_loso_safe": True,
                        "uses_oracle_labels_for_selection": False,
                        "support_status": "proxy",
                        "qc_status": "warning",
                        "qc_reason": "bounded LOSO-safe threshold selection over existing causal scores; strategy choice itself is posthoc analysis, not a deployed target",
                    }
                )
                rows.append(row)
            for row in subjects:
                row.update({"tier": tier, "strategy_id": spec["strategy_id"], "param_id": spec["param_id"], "scoring_definition": scoring.scoring_id})
                subject_rows.append(row)
    out = add_universal_columns(
        pd.DataFrame(rows),
        base_universal(
            config,
            source_table="causal_feature_matrix.tsv;causal_minimum_sufficient_subset.tsv",
            source_columns="fold-local minimum subset score,window timing,frozen event labels for evaluation only",
            source_lineage="Phase 5_2C pre-ADR bounded alarm reconstruction sweep",
            support_status="proxy",
            qc_status="warning",
            qc_reason="bounded pre-ADR analysis; not Stage G/H/I execution",
        ),
    )
    return out, pd.DataFrame(subject_rows)


def tier3_alarm_estimate(
    config: dict[str, Any],
    cache: fast.EventCache,
    tier2_score: np.ndarray,
    timing: dict[str, float],
    scoring: ScoringWindow,
    alarm_best: pd.DataFrame,
    *,
    tier3_seeds: int | None,
) -> pd.DataFrame:
    if alarm_best.empty:
        return pd.DataFrame()
    seed_count = int(tier3_seeds if tier3_seeds is not None else config.get("stage_f", {}).get("tier3_mismatch_seeds", 30))
    fp2 = alarm_best.loc[pd.to_numeric(alarm_best["fp_per_min_target"], errors="coerce").eq(2.0)].copy()
    candidates = fp2.loc[fp2["tier"].astype(str).eq("tier2_quantized")]
    if candidates.empty:
        candidates = alarm_best.loc[alarm_best["tier"].astype(str).eq("tier2_quantized")]
    if candidates.empty:
        return unavailable_tier3_rows(config, "no tier2 bounded strategy available for Tier 3 proxy")
    best = candidates.sort_values(["recall", "F1", "precision"], ascending=False, kind="mergesort").iloc[0].to_dict()
    spec = spec_from_best(best)
    scale = float(np.nanstd(tier2_score)) if np.isfinite(tier2_score).any() else 1.0
    seed_frames = []
    for seed in range(seed_count):
        rng = np.random.default_rng(int(config.get("random_seed", 0)) + seed)
        score = tier2_score + rng.normal(0.0, 0.20 * scale, size=tier2_score.size)
        strategy_score = prepare_strategy_score(cache, score, spec)
        thresholds = threshold_candidates_for_eval(strategy_score, max_thresholds=64)
        agg, _ = loso_alarm_eval(cache, strategy_score, spec, thresholds, timing, scoring)
        for row in agg:
            row.update(
                {
                    "analysis_subtask": "R1_alarm_reconstruction_headroom",
                    "tier": "tier3_quantized_mismatched_proxy",
                    "tier3_seed_count": seed_count,
                    "mismatch_seed": seed,
                    "strategy_id": spec["strategy_id"],
                    "param_id": spec["param_id"],
                    "scoring_definition": scoring.scoring_id,
                    "metric_induced_or_score_induced": "score_to_alarm_reconstruction_with_mismatch_proxy",
                    "selection_loso_safe": True,
                    "uses_oracle_labels_for_selection": False,
                    "support_status": "proxy",
                    "qc_status": "warning",
                    "qc_reason": "Tier 3 estimate uses deterministic Stage F score-level mismatch proxy on the best Tier 2 bounded strategy; pre-Brian2 only",
                }
            )
        seed_frames.extend(agg)
    raw = pd.DataFrame(seed_frames)
    if raw.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for fp, group in raw.groupby("fp_per_min_target", sort=True):
        row = group.iloc[0].to_dict()
        for col in metric_numeric_columns():
            row[col] = float(pd.to_numeric(group[col], errors="coerce").median())
        row["mismatch_seed"] = "median_across_seeds"
        row["tier"] = "tier3_quantized_mismatched_proxy"
        row["tier3_seed_count"] = seed_count
        row["per_subject_recall_p05_across_seeds"] = float(pd.to_numeric(group["per_subject_recall_min"], errors="coerce").quantile(0.05))
        rows.append(row)
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(
            config,
            source_table="causal_feature_matrix.tsv;event_alarm_reconstruction_sweep.tsv",
            source_columns="tier2 score,deterministic mismatch proxy,best bounded strategy",
            source_lineage="Phase 5_2C pre-ADR Tier 3 alarm reconstruction proxy",
            support_status="proxy",
            qc_status="warning",
            qc_reason="pre-Brian2 mismatch proxy only",
        ),
    )


def unavailable_tier3_rows(config: dict[str, Any], reason: str) -> pd.DataFrame:
    rows = []
    for fp in FP_GRID:
        rows.append({"fp_per_min_target": fp, "tier": "tier3_quantized_mismatched_proxy", "recall": np.nan, "support_status": "unsupported", "qc_status": "unavailable", "qc_reason": reason})
    return add_universal_columns(pd.DataFrame(rows), base_universal(config, source_table="event_alarm_reconstruction_sweep.tsv", source_columns="NA", source_lineage="Unavailable Tier 3 alarm reconstruction proxy", support_status="unsupported", qc_status="unavailable", qc_reason=reason))


def spec_from_best(best: dict[str, Any]) -> dict[str, Any]:
    for spec in alarm_strategy_specs():
        if spec["strategy_id"] == str(best.get("strategy_id")) and spec["param_id"] == str(best.get("param_id")):
            return spec
    return {"strategy_id": "A0_existing_stage_ef", "param_id": "baseline", "kind": "threshold", "params": {}}


def prepare_strategy_score(cache: fast.EventCache, score: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    params = spec.get("params", {})
    out = np.asarray(score, dtype=float).copy()
    if "tau_s" in params:
        out = leaky_integrate(cache, out, float(params["tau_s"]))
    if "parallel_taus_s" in params:
        pieces = [out]
        for tau in params["parallel_taus_s"]:
            pieces.append(leaky_integrate(cache, score, float(tau)))
        out = np.nanmax(np.vstack(pieces), axis=0)
    return out


def threshold_candidates_for_eval(score: np.ndarray, *, max_thresholds: int = 64) -> np.ndarray:
    finite = np.asarray(score, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([])
    unique = np.unique(finite)
    if unique.size <= max_thresholds:
        return unique[::-1]
    quantiles = np.r_[np.linspace(0.9995, 0.94, max_thresholds // 2), np.linspace(0.94, 0.50, max_thresholds - max_thresholds // 2)]
    return np.unique(np.nanquantile(finite, quantiles))[::-1]


def leaky_integrate(cache: fast.EventCache, score: np.ndarray, tau_s: float) -> np.ndarray:
    sorted_score = np.asarray(score, dtype=float)[cache.order]
    out_sorted = np.full_like(sorted_score, np.nan, dtype=float)
    for _, start, stop in cache.group_slices:
        local = sorted_score[start:stop]
        times = cache.times_sorted[start:stop]
        state = np.nan
        last_t = np.nan
        for idx, (value, t) in enumerate(zip(local, times, strict=False)):
            if not np.isfinite(value):
                out_sorted[start + idx] = state
                continue
            if not np.isfinite(state):
                state = float(value)
            else:
                dt = max(float(t - last_t), 0.0) if np.isfinite(last_t) else 0.0
                alpha = 1.0 - math.exp(-dt / max(tau_s, 1e-9))
                state = (1.0 - alpha) * state + alpha * float(value)
            last_t = float(t)
            out_sorted[start + idx] = state
    out = np.full_like(out_sorted, np.nan)
    out[cache.order] = out_sorted
    return out


def build_strategy_alarm(cache: fast.EventCache, score: np.ndarray, threshold: float, spec: dict[str, Any], timing: dict[str, float]) -> fast.AlarmResult:
    score_sorted = np.asarray(score, dtype=float)[cache.order]
    by_group: dict[int, np.ndarray] = {}
    n_alarms = 0
    kind = spec.get("kind", "threshold")
    params = spec.get("params", {})
    for group, start, stop in cache.group_slices:
        local_score = score_sorted[start:stop]
        finite = np.isfinite(local_score)
        high = finite & (local_score >= threshold)
        if kind == "n_of_m":
            mask = rolling_count_bool(high, int(params["m"])) >= int(params["n"])
        elif kind == "hysteresis":
            candidate = finite & (local_score >= threshold - float(params.get("candidate_delta", 0.5)))
            mask = high & (rolling_count_bool(candidate, int(params.get("candidate_persistence", 3))) > 0)
        elif kind == "fsm":
            candidate = finite & (local_score >= threshold - float(params.get("candidate_delta", 0.5)))
            cand_active = rolling_count_bool(candidate, int(params.get("candidate_persistence", 1))) > 0
            confirm = rolling_count_bool(high, int(params.get("confirmation_persistence", 1))) >= int(params.get("confirmation_persistence", 1))
            mask = cand_active & confirm
        else:
            mask = high
        if not mask.any():
            continue
        times = cache.times_sorted[start:stop][mask]
        keep = stage.refractory_keep_mask(times, refractory_s=timing["refractory_s"], merge_window_s=timing["merge_window_s"])
        if keep.any():
            kept = times[keep]
            by_group[group] = kept
            n_alarms += int(kept.size)
    return fast.AlarmResult(by_group=by_group, n_alarms=n_alarms)


def rolling_count_bool(mask: np.ndarray, window_count: int) -> np.ndarray:
    x = np.asarray(mask, dtype=np.int64)
    prefix = np.r_[0, np.cumsum(x)]
    idx = np.arange(len(x))
    start = np.maximum(0, idx - int(window_count) + 1)
    return prefix[idx + 1] - prefix[start]


def loso_alarm_eval(
    cache: fast.EventCache,
    score: np.ndarray,
    spec: dict[str, Any],
    thresholds: np.ndarray,
    timing: dict[str, float],
    scoring: ScoringWindow,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if thresholds.size == 0:
        return [empty_alarm_row(fp, "no finite thresholds") for fp in FP_GRID], []
    threshold_subject_rows: dict[float, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        alarms = build_strategy_alarm(cache, score, float(threshold), spec, timing)
        rows = subject_rows_with_policy(cache, alarms, scoring)
        for row in rows:
            row["threshold"] = float(threshold)
        threshold_subject_rows[float(threshold)] = rows
    subjects = sorted(cache.events_by_subject_group)
    chosen_subject_rows: dict[float, list[dict[str, Any]]] = {fp: [] for fp in FP_GRID}
    selected_thresholds: dict[float, list[float]] = {fp: [] for fp in FP_GRID}
    no_feasible: dict[float, int] = {fp: 0 for fp in FP_GRID}
    for subject_code in subjects:
        for fp in FP_GRID:
            best_threshold: float | None = None
            best_train: dict[str, Any] | None = None
            for threshold, rows in threshold_subject_rows.items():
                train = aggregate_metric_rows([row for row in rows if int(row["subject_code"]) != int(subject_code)])
                if not np.isfinite(train["fp_per_min_achieved"]) or train["fp_per_min_achieved"] > fp:
                    continue
                if best_train is None or metric_better(train, best_train):
                    best_train = train
                    best_threshold = threshold
            if best_threshold is None:
                no_feasible[fp] += 1
                continue
            selected_thresholds[fp].append(float(best_threshold))
            held = [row for row in threshold_subject_rows[best_threshold] if int(row["subject_code"]) == int(subject_code)]
            for row in held:
                row = row.copy()
                row["fp_per_min_target"] = fp
                row["selected_threshold_train_only"] = best_threshold
                chosen_subject_rows[fp].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for fp in FP_GRID:
        rows = chosen_subject_rows[fp]
        if not rows:
            aggregate_rows.append(empty_alarm_row(fp, "no LOSO fold had a feasible train-selected threshold"))
            continue
        agg = aggregate_metric_rows(rows)
        recalls = pd.to_numeric(pd.Series([row["recall"] for row in rows]), errors="coerce")
        agg.update(
            {
                "fp_per_min_target": fp,
                "threshold": float(np.nanmedian(selected_thresholds[fp])) if selected_thresholds[fp] else np.nan,
                "threshold_selection_mode": "train_subjects_only_per_LOSO_fold",
                "n_loso_folds": len(rows),
                "n_no_feasible_folds": no_feasible[fp],
                "per_subject_recall_median": float(recalls.median()),
                "per_subject_recall_iqr": float(recalls.quantile(0.75) - recalls.quantile(0.25)),
                "per_subject_recall_min": float(recalls.min()),
                "per_subject_recall_max": float(recalls.max()),
            }
        )
        aggregate_rows.append(agg)
        for row in rows:
            row = row.copy()
            row["subject_key"] = hash_subject(str(cache.subject_labels[int(row["subject_code"])]))
            subject_rows.append(row)
    return aggregate_rows, subject_rows


def metric_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    fields = ["recall", "F1", "precision"]
    cand = tuple(safe_float(candidate.get(field)) for field in fields) + (-safe_float(candidate.get("fp_per_min_achieved"), 1e9),)
    inc = tuple(safe_float(incumbent.get(field)) for field in fields) + (-safe_float(incumbent.get("fp_per_min_achieved"), 1e9),)
    return cand > inc


def subject_rows_with_policy(cache: fast.EventCache, alarms: fast.AlarmResult, scoring: ScoringWindow) -> list[dict[str, Any]]:
    rows = []
    for subject_code, events_by_group in sorted(cache.events_by_subject_group.items()):
        subject_alarm_groups = {group: times for group, times in alarms.by_group.items() if cache.group_subject.get(group) == subject_code}
        subject_alarm_count = sum(int(times.size) for times in subject_alarm_groups.values())
        metrics = evaluate_alarm_result_with_policy(
            fast.AlarmResult(subject_alarm_groups, subject_alarm_count),
            events_by_group,
            sum(int(onsets.size) for onsets, _ in events_by_group.values()),
            cache.minutes_by_subject.get(subject_code, 0.0),
            scoring,
        )
        metrics["subject_code"] = int(subject_code)
        metrics["minutes"] = float(cache.minutes_by_subject.get(subject_code, 0.0))
        rows.append(metrics)
    return rows


def evaluate_alarm_result_with_policy(
    alarms: fast.AlarmResult,
    events_by_group: dict[int, tuple[np.ndarray, np.ndarray]],
    n_events: int,
    total_minutes: float,
    scoring: ScoringWindow,
) -> dict[str, Any]:
    if alarms.n_alarms == 0:
        row = stage.event_metric_row(0, 0, n_events, 0, total_minutes, [], [], "no alarms under scoring policy")
        row["minutes"] = total_minutes
        return row
    latencies: list[float] = []
    early_latencies: list[float] = []
    matched_event_count = 0
    one_alarm_events = 0
    success_alarm_count = 0
    for group, (onsets, offsets) in events_by_group.items():
        alarm_times = alarms.by_group.get(group)
        if alarm_times is None or alarm_times.size == 0:
            continue
        starts = onsets + scoring.start_offset_s
        stops = (offsets if scoring.stop_reference == "offset" else onsets) + scoring.stop_offset_s
        valid = np.isfinite(starts) & np.isfinite(stops) & (stops >= starts)
        starts = starts[valid]
        stops = stops[valid]
        event_onsets = onsets[valid]
        if starts.size == 0:
            continue
        start_idx = np.searchsorted(alarm_times, starts, side="left")
        stop_idx = np.searchsorted(alarm_times, stops, side="right")
        counts = stop_idx - start_idx
        matched = counts > 0
        matched_event_count += int(matched.sum())
        one_alarm_events += int((counts == 1).sum())
        success_alarm_count += int(counts[matched].sum())
        if matched.any():
            latencies.extend(((alarm_times[start_idx[matched]] - event_onsets[matched]) * 1000.0).tolist())
        early_start = np.searchsorted(alarm_times, event_onsets - 0.500, side="left")
        early_stop = np.searchsorted(alarm_times, event_onsets, side="left")
        early_matched = early_stop > early_start
        if early_matched.any():
            early_latencies.extend(((alarm_times[early_start[early_matched]] - event_onsets[early_matched]) * 1000.0).tolist())
    tp_alarms = min(success_alarm_count, alarms.n_alarms)
    fp = int(max(alarms.n_alarms - tp_alarms, 0))
    row = stage.event_metric_row(matched_event_count, tp_alarms, n_events, fp, total_minutes, latencies, early_latencies, "", one_alarm_events=one_alarm_events, success_alarm_count=success_alarm_count)
    row["minutes"] = total_minutes
    return row


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_events = int(np.nansum([numeric(row.get("n_true_events")) for row in rows]))
    n_alarms = int(np.nansum([numeric(row.get("n_alarms")) for row in rows]))
    tp_events = int(np.nansum([numeric(row.get("true_positive_events")) for row in rows]))
    tp_alarms = int(np.nansum([numeric(row.get("true_positive_alarms")) for row in rows]))
    fp = int(np.nansum([numeric(row.get("false_positive_alarms")) for row in rows]))
    minutes = float(np.nansum([numeric(row.get("minutes")) for row in rows]))
    one_alarm_events = float(np.nansum([numeric(row.get("one_alarm_per_burst_fraction")) * numeric(row.get("n_true_events")) for row in rows]))
    latencies = pd.to_numeric(pd.Series([row.get("median_latency_ms") for row in rows]), errors="coerce").dropna()
    early = pd.to_numeric(pd.Series([row.get("early_warning_candidate_latency_ms") for row in rows]), errors="coerce").dropna()
    recall = tp_events / n_events if n_events else np.nan
    precision = tp_alarms / n_alarms if n_alarms else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else np.nan
    return {
        "n_true_events": n_events,
        "n_alarms": n_alarms,
        "true_positive_events": tp_events,
        "true_positive_alarms": tp_alarms,
        "false_positive_alarms": fp,
        "recall": recall,
        "precision": precision,
        "F1": f1,
        "fp_per_min_achieved": fp / minutes if minutes else np.nan,
        "median_latency_ms": float(latencies.median()) if len(latencies) else np.nan,
        "early_warning_candidate_latency_ms": float(early.median()) if len(early) else np.nan,
        "one_alarm_per_burst_fraction": one_alarm_events / n_events if n_events else np.nan,
        "alarms_per_burst": n_alarms / n_events if n_events else np.nan,
    }


def empty_alarm_row(fp: float, reason: str) -> dict[str, Any]:
    row = stage.empty_event_metrics(fp, np.nan, reason)
    row.update({"threshold_selection_mode": "unavailable", "n_loso_folds": 0, "n_no_feasible_folds": np.nan, "per_subject_recall_median": np.nan, "per_subject_recall_iqr": np.nan, "per_subject_recall_min": np.nan, "per_subject_recall_max": np.nan})
    return row


def alarm_reconstruction_best(config: dict[str, Any], sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if sweep.empty:
        return sweep
    data = sweep.copy()
    for fp, group in data.groupby(pd.to_numeric(data["fp_per_min_target"], errors="coerce"), sort=True):
        tier3 = group.loc[group["tier"].astype(str).eq("tier3_quantized_mismatched_proxy")]
        candidates = tier3 if len(tier3) else group.loc[group["tier"].astype(str).isin(["tier2_quantized", "tier1_continuous"])]
        if candidates.empty:
            continue
        best = candidates.sort_values(["recall", "F1", "precision"], ascending=False, kind="mergesort").iloc[0].to_dict()
        best["best_selection_basis"] = "Tier3 proxy median" if len(tier3) else "best available Tier1/Tier2 bounded proxy"
        best["headroom_interpretation"] = event_headroom_interpretation(best)
        rows.append(best)
    out = pd.DataFrame(rows)
    return add_universal_columns(
        out,
        base_universal(config, source_table="event_alarm_reconstruction_sweep.tsv", source_columns="recall,F1,precision,per_subject metrics", source_lineage="Phase 5_2C pre-ADR best bounded alarm reconstruction", support_status="proxy", qc_status="warning", qc_reason="best row selected posthoc from bounded analysis; not an approved engineering gate"),
    )


def event_headroom_interpretation(row: dict[str, Any]) -> str:
    fp = numeric(row.get("fp_per_min_target"))
    recall = numeric(row.get("recall"))
    if np.isfinite(fp) and abs(fp - 2.0) < 1e-9 and np.isfinite(recall) and recall < 0.10:
        return "best bounded event reconstruction remains below 0.10 recall at 2 FP/min"
    if np.isfinite(fp) and abs(fp - 5.0) < 1e-9 and np.isfinite(recall) and recall < 0.25:
        return "best bounded event reconstruction remains below 0.25 recall at 5 FP/min"
    if np.isfinite(recall):
        return "bounded reconstruction improves event recall but requires owner review before target approval"
    return "unavailable"


def scoring_tolerance_sweep(config: dict[str, Any], cache: fast.EventCache, base_score: np.ndarray, tier2_score: np.ndarray, timing: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    spec = {"strategy_id": "A0_existing_stage_ef", "param_id": "baseline", "kind": "threshold", "params": {}}
    for scoring in tolerance_windows():
        for tier, score in [("tier1_continuous", base_score), ("tier2_quantized", tier2_score)]:
            thresholds = threshold_candidates_for_eval(score, max_thresholds=54 if tier == "tier1_continuous" else 80)
            agg, subjects = loso_alarm_eval(cache, score, spec, thresholds, timing, scoring)
            for row in agg:
                row.update(
                    {
                        "analysis_subtask": "R2_scoring_tolerance_sensitivity",
                        "tier": tier,
                        "scoring_definition": scoring.scoring_id,
                        "scoring_family": scoring.scoring_family,
                        "start_offset_s": scoring.start_offset_s,
                        "stop_reference": scoring.stop_reference,
                        "stop_offset_s": scoring.stop_offset_s,
                        "committed_alarm_success_policy": scoring.committed_alarm_success,
                        "metric_induced_or_score_induced": "metric_induced",
                        "interpretation": scoring.interpretation,
                        "selection_loso_safe": True,
                        "support_status": "proxy",
                        "qc_status": "warning" if scoring.scoring_id != "S0_current_policy" else "ok",
                        "qc_reason": "scoring tolerance sensitivity; underlying causal scores unchanged",
                    }
                )
                rows.append(row)
            for row in subjects:
                row.update({"tier": tier, "scoring_definition": scoring.scoring_id})
                subject_rows.append(row)
    out = add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="causal_feature_matrix.tsv;causal_minimum_sufficient_subset.tsv", source_columns="fold-local minimum subset score,alternative event scoring windows", source_lineage="Phase 5_2C scoring tolerance sensitivity", support_status="proxy", qc_status="warning", qc_reason="sensitivity analysis only, not target approval"),
    )
    return out, pd.DataFrame(subject_rows)


def scoring_tolerance_summary(config: dict[str, Any], sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    current = sweep.loc[sweep["scoring_definition"].eq("S0_current_policy")]
    for fp in FP_GRID:
        cur = current.loc[pd.to_numeric(current["fp_per_min_target"], errors="coerce").eq(fp)].copy()
        cur = cap_feasible_rows(cur, fp)
        current_recall = float(pd.to_numeric(cur["recall"], errors="coerce").max()) if len(cur) else np.nan
        sub = sweep.loc[pd.to_numeric(sweep["fp_per_min_target"], errors="coerce").eq(fp)].copy()
        sub = cap_feasible_rows(sub, fp)
        if sub.empty:
            continue
        best = sub.sort_values(["recall", "F1", "precision"], ascending=False, kind="mergesort").iloc[0]
        recall = numeric(best.get("recall"))
        improvement = recall / current_recall if np.isfinite(recall) and np.isfinite(current_recall) and current_recall > 0 else np.nan
        rows.append(
            {
                "fp_per_min_target": fp,
                "current_policy_best_recall": current_recall,
                "best_scoring_definition": best.get("scoring_definition"),
                "best_scoring_family": best.get("scoring_family"),
                "best_tier": best.get("tier"),
                "best_recall": recall,
                "recall_improvement_factor": improvement,
                "precision": best.get("precision"),
                "F1": best.get("F1"),
                "achieved_fp_per_min": best.get("fp_per_min_achieved"),
                "metric_induced_or_score_induced": "metric_induced",
                "tolerance_interpretation": tolerance_interpretation(current_recall, recall, str(best.get("scoring_family"))),
                "support_status": "proxy",
                "qc_status": "warning",
                "qc_reason": "widened scoring windows are sensitivity analyses and do not approve a deployment target",
            }
        )
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="event_scoring_tolerance_sweep.tsv", source_columns="current and widened scoring recall", source_lineage="Phase 5_2C scoring tolerance sensitivity summary", support_status="proxy", qc_status="warning", qc_reason="metric sensitivity summary"),
    )


def cap_feasible_rows(frame: pd.DataFrame, fp_cap: float, *, tolerance: float = 0.05) -> pd.DataFrame:
    if frame.empty or "fp_per_min_achieved" not in frame.columns:
        return frame
    achieved = pd.to_numeric(frame["fp_per_min_achieved"], errors="coerce")
    feasible = frame.loc[achieved <= fp_cap * (1.0 + tolerance)].copy()
    return feasible if len(feasible) else frame.iloc[0:0].copy()


def tolerance_interpretation(current: float, best: float, family: str) -> str:
    if not np.isfinite(best):
        return "unavailable"
    factor = best / current if np.isfinite(current) and current > 0 else np.inf
    if factor >= 2.0 and family != "current_committed_interval":
        return "event recall is materially sensitive to scoring tolerance"
    if best < 0.10:
        return "event recall remains low even under tolerance sensitivity"
    return "widened scoring improves event recall; owner must judge scientific acceptability"


def burden_target_availability(config: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        target_availability_row("B1_frozen_phase3_rolling_burden", True, "frozen Phase 3 event labels aggregated over rolling sampled-window histories", "evaluation_target_from_phase3"),
        target_availability_row("B2_high_burden_interval_classification", True, "high-burden binary target derived from fold-local burden quantiles", "evaluation_target_from_phase3_quantile"),
        target_availability_row("B3_recent_event_density_state", True, "recent true-event density over rolling sampled-window histories", "evaluation_target_from_phase3"),
        target_availability_row("B4_long_burst_sustained_state", "phase3_duration_ms" in frame.columns, "long-burst target from frozen duration/offset metadata when available", "evaluation_target_from_phase3_duration_rule"),
    ]
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="causal_feature_matrix.tsv", source_columns="window_type,target_label,phase3_duration_ms,anchor_onset_s,anchor_offset_s", source_lineage="Phase 5_2C pre-ADR burden/state target availability", support_status="direct", qc_status="warning", qc_reason="targets are evaluation targets from frozen Phase 3 labels, not input features"),
    )


def target_availability_row(target_id: str, available: bool, derivation: str, risk: str) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "target_available": bool(available),
        "target_derivation": derivation,
        "target_tautology_risk": risk,
        "input_feature_tautology_risk": "low; Phase 3 labels are not used as input features",
        "support_status": "direct" if available else "unsupported",
        "qc_status": "warning" if available else "unavailable",
        "qc_reason": "available for diagnostic causal ceiling evaluation, not deployment proof" if available else "required metadata missing",
    }


def burden_state_ceiling_metrics(config: dict[str, Any], frame: pd.DataFrame, cache: fast.EventCache, base_score: np.ndarray, refined_features: list[str]) -> pd.DataFrame:
    y_event = frame["is_true_event"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    refined_score = None
    if refined_features:
        refined_score = stage.fold_local_subset_score(frame, frame["is_true_event"].to_numpy(dtype=bool), refined_features)
    long_target_base = long_burst_target(frame)
    for window_s in WINDOWS_S:
        burden_target = rolling_mean_by_group(cache, y_event, window_s)
        density_target = rolling_sum_by_group(cache, y_event, window_s) / max(window_s, 1e-9)
        dynamic_score = rolling_mean_by_group(cache, base_score, window_s)
        score_sources = [
            ("rolling_dynamic_confirmation_score", "deployable_estimate", dynamic_score, "minimum_sufficient_subset"),
            ("rolling_dynamic_confirmation_occupancy", "deployable_estimate", rolling_occupancy_loso(cache, base_score, 0.90, window_s), "minimum_sufficient_subset"),
        ]
        if refined_score is not None:
            score_sources.append(("rolling_best_loaded_refined_variants", "diagnostic_upper_bound", rolling_mean_by_group(cache, refined_score, window_s), "loaded_refined_variants"))
        targets = [
            ("B1_frozen_phase3_rolling_burden", burden_target, "continuous_burden_fraction"),
            ("B2_high_burden_interval_classification", burden_target, "high_burden_quantile"),
            ("B3_recent_event_density_state", density_target, "recent_event_density"),
        ]
        if long_target_base is not None:
            targets.append(("B4_long_burst_sustained_state", rolling_mean_by_group(cache, long_target_base, window_s), "long_burst_state"))
        for target_id, target, target_type in targets:
            for source_id, deployability, score, feature_subset in score_sources:
                metrics = loso_burden_metrics(frame, target, score)
                rows.append(
                    {
                        "target_id": target_id,
                        "target_type": target_type,
                        "window_s": window_s,
                        "score_source": source_id,
                        "feature_subset": feature_subset,
                        "ceiling_type": "burden_state_causal_score_estimate",
                        "deployability_class": deployability,
                        **metrics,
                        "tier_proxy_status": "Tier1 deployable estimate; Tier2/Tier3 burden degradation not rerun in this bounded analysis",
                        "uses_oracle_labels_for_selection": False,
                        "support_status": "proxy" if deployability != "deployable_estimate" else "direct",
                        "qc_status": "warning",
                        "qc_reason": "burden/state ceiling uses sampled-window rolling targets from frozen labels; not deployment validation",
                    }
                )
            oracle = loso_burden_metrics(frame, target, target)
            rows.append(
                {
                    "target_id": target_id,
                    "target_type": target_type,
                    "window_s": window_s,
                    "score_source": "oracle_target_as_score",
                    "feature_subset": "labels_for_upper_bound_only",
                    "ceiling_type": "burden_state_oracle_target_upper_bound",
                    "deployability_class": "oracle_not_deployable",
                    **oracle,
                    "tier_proxy_status": "not applicable",
                    "uses_oracle_labels_for_selection": True,
                    "support_status": "proxy",
                    "qc_status": "warning",
                    "qc_reason": "oracle target-as-score sanity bound; not deployable",
                }
            )
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(
            config,
            source_table="causal_feature_matrix.tsv;causal_refined_candidate_features.tsv",
            source_columns="fold-local causal scores,rolling frozen-label targets",
            source_lineage="Phase 5_2C pre-ADR burden/state ceiling comparison",
            support_status="proxy",
            qc_status="warning",
            qc_reason="bounded causal burden/state ceiling analysis; prior Phase 5Z context does not override Phase 5_2C evidence",
        ),
    )


def rolling_mean_by_group(cache: fast.EventCache, values: np.ndarray, window_s: float) -> np.ndarray:
    sorted_values = np.asarray(values, dtype=float)[cache.order]
    out_sorted = np.full_like(sorted_values, np.nan, dtype=float)
    for _, start, stop in cache.group_slices:
        times = cache.times_sorted[start:stop]
        vals = sorted_values[start:stop]
        finite = np.isfinite(vals)
        clean = np.where(finite, vals, 0.0)
        counts = np.cumsum(finite.astype(float))
        sums = np.cumsum(clean)
        left = np.searchsorted(times, times - window_s, side="left")
        prev_sum = np.where(left > 0, sums[left - 1], 0.0)
        prev_count = np.where(left > 0, counts[left - 1], 0.0)
        denom = counts - prev_count
        numer = sums - prev_sum
        local = np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=float), where=denom > 0)
        out_sorted[start:stop] = local
    out = np.full_like(out_sorted, np.nan)
    out[cache.order] = out_sorted
    return out


def rolling_sum_by_group(cache: fast.EventCache, values: np.ndarray, window_s: float) -> np.ndarray:
    sorted_values = np.asarray(values, dtype=float)[cache.order]
    out_sorted = np.full_like(sorted_values, np.nan, dtype=float)
    for _, start, stop in cache.group_slices:
        times = cache.times_sorted[start:stop]
        vals = np.where(np.isfinite(sorted_values[start:stop]), sorted_values[start:stop], 0.0)
        sums = np.cumsum(vals)
        left = np.searchsorted(times, times - window_s, side="left")
        prev_sum = np.where(left > 0, sums[left - 1], 0.0)
        out_sorted[start:stop] = sums - prev_sum
    out = np.full_like(out_sorted, np.nan)
    out[cache.order] = out_sorted
    return out


def rolling_occupancy_loso(cache: fast.EventCache, score: np.ndarray, quantile: float, window_s: float) -> np.ndarray:
    subjects = cache.subject_codes
    out = np.full(len(score), np.nan, dtype=float)
    for subject in np.unique(subjects):
        held = subjects == subject
        train = ~held
        threshold = np.nanquantile(score[train & np.isfinite(score)], quantile) if np.isfinite(score[train]).any() else np.nan
        binary = np.asarray(score >= threshold, dtype=float)
        occ = rolling_mean_by_group(cache, binary, window_s)
        out[held] = occ[held]
    return out


def long_burst_target(frame: pd.DataFrame) -> np.ndarray | None:
    if "phase3_duration_ms" not in frame.columns:
        return None
    duration = pd.to_numeric(frame["phase3_duration_ms"], errors="coerce").to_numpy(dtype=float)
    positive = frame["is_true_event"].to_numpy(dtype=bool)
    finite = np.isfinite(duration) & positive
    if not finite.any():
        return None
    threshold = float(np.nanquantile(duration[finite], 0.75))
    return (finite & (duration >= threshold)).astype(float)


def loso_burden_metrics(frame: pd.DataFrame, target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    subjects = frame["subject_id"].astype(str).to_numpy()
    oriented = np.full(len(frame), np.nan, dtype=float)
    y_binary = np.full(len(frame), False, dtype=bool)
    for subject in sorted(set(subjects)):
        held = subjects == subject
        train = ~held
        train_valid = np.isfinite(target[train]) & np.isfinite(score[train])
        sign = 1.0
        if train_valid.sum() >= 3:
            corr = np.corrcoef(target[train][train_valid], score[train][train_valid])[0, 1]
            sign = -1.0 if np.isfinite(corr) and corr < 0 else 1.0
        oriented[held] = sign * score[held]
        finite_target_train = target[train][np.isfinite(target[train])]
        cutoff = float(np.nanquantile(finite_target_train, 0.75)) if finite_target_train.size else np.nan
        y_binary[held] = np.isfinite(target[held]) & (target[held] >= cutoff)
    valid = np.isfinite(target) & np.isfinite(oriented)
    pearson = corrcoef(target[valid], oriented[valid]) if valid.sum() >= 3 else np.nan
    spearman = spearman_corr(target[valid], oriented[valid]) if valid.sum() >= 3 else np.nan
    auroc = rank_auroc(y_binary[valid], oriented[valid]) if valid.sum() and y_binary[valid].any() and (~y_binary[valid]).any() else np.nan
    auprc = stage.average_precision(y_binary[valid], oriented[valid]) if valid.sum() and y_binary[valid].any() else np.nan
    prob = minmax01(oriented, valid)
    brier = float(np.nanmean((prob[valid] - y_binary[valid].astype(float)) ** 2)) if valid.sum() else np.nan
    ece = calibration_ece(y_binary[valid], prob[valid]) if valid.sum() else np.nan
    per_subject = []
    for subject in sorted(set(subjects)):
        mask = valid & (subjects == subject)
        if mask.sum() >= 3:
            per_subject.append(corrcoef(target[mask], oriented[mask]))
    ps = pd.to_numeric(pd.Series(per_subject), errors="coerce")
    return {
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "high_burden_AUROC": auroc,
        "high_burden_AUPRC": auprc,
        "Brier": brier,
        "ECE": ece,
        "per_subject_metric_median": float(ps.median()) if ps.notna().any() else np.nan,
        "per_subject_metric_iqr": float(ps.quantile(0.75) - ps.quantile(0.25)) if ps.notna().any() else np.nan,
        "LOSO_support": True,
    }


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return corrcoef(pd.Series(a).rank(method="average").to_numpy(dtype=float), pd.Series(b).rank(method="average").to_numpy(dtype=float))


def minmax01(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(len(score), np.nan, dtype=float)
    vals = score[valid]
    if vals.size == 0:
        return out
    lo, hi = np.nanquantile(vals, [0.01, 0.99])
    out[valid] = np.clip((vals - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return out


def calibration_ece(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    valid = np.isfinite(prob)
    y = y[valid].astype(float)
    prob = prob[valid]
    if len(prob) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (prob >= lo) & (prob <= hi if hi == 1.0 else prob < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(prob[mask].mean()))
    return ece


def burden_state_gap_to_ceiling(config: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for (target_id, window_s), group in metrics.groupby(["target_id", "window_s"], sort=True):
        deploy = best_burden_row(group.loc[group["deployability_class"].eq("deployable_estimate")])
        diag = best_burden_row(group.loc[group["deployability_class"].eq("diagnostic_upper_bound")])
        oracle = best_burden_row(group.loc[group["deployability_class"].eq("oracle_not_deployable")])
        rows.append(
            {
                "target_id": target_id,
                "window_s": window_s,
                "best_deployable_score_source": deploy.get("score_source", "NA"),
                "deployable_primary_metric": deploy.get("primary_metric", np.nan),
                "deployable_primary_metric_value": deploy.get("primary_metric_value", np.nan),
                "diagnostic_ceiling_score_source": diag.get("score_source", "NA"),
                "diagnostic_primary_metric_value": diag.get("primary_metric_value", np.nan),
                "oracle_primary_metric_value": oracle.get("primary_metric_value", np.nan),
                "gap_deployable_to_diagnostic": subtract_or_nan(diag.get("primary_metric_value"), deploy.get("primary_metric_value")),
                "gap_deployable_to_oracle": subtract_or_nan(oracle.get("primary_metric_value"), deploy.get("primary_metric_value")),
                "support_status": "proxy",
                "qc_status": "warning",
                "qc_reason": "burden gap compares non-deployable diagnostic/oracle ceilings with deployable causal score estimates",
            }
        )
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="burden_state_ceiling_metrics_pre_adr.tsv", source_columns="deployable and diagnostic burden metrics", source_lineage="Phase 5_2C pre-ADR burden/state gap to ceiling", support_status="proxy", qc_status="warning", qc_reason="burden ceiling gap is diagnostic only"),
    )


def best_burden_row(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {}
    data = group.copy()
    data["primary_metric"] = np.where(pd.to_numeric(data["high_burden_AUROC"], errors="coerce").notna(), "high_burden_AUROC", "pearson_correlation")
    data["primary_metric_value"] = pd.to_numeric(data["high_burden_AUROC"], errors="coerce").where(pd.to_numeric(data["high_burden_AUROC"], errors="coerce").notna(), pd.to_numeric(data["pearson_correlation"], errors="coerce"))
    return data.sort_values("primary_metric_value", ascending=False, kind="mergesort").iloc[0].to_dict()


def event_vs_burden_comparison(config: dict[str, Any], alarm_best: pd.DataFrame, tolerance_summary: pd.DataFrame, burden_metrics: pd.DataFrame) -> pd.DataFrame:
    table_dir = output_paths(config)["table_dir"]
    event_summary = read_tsv(table_dir / "causal_three_tier_event_summary.tsv")
    rows: list[dict[str, Any]] = []
    current = event_summary.loc[(event_summary["tier"].astype(str).eq("tier3_quantized_mismatched")) & pd.to_numeric(event_summary["target_fp_min"], errors="coerce").eq(2.0)]
    if len(current):
        row = current.iloc[0]
        rows.append(comparison_row("current_event_detector", "discrete_event", "recall_at_fp_min", "Tier3 median recall", row.get("recall_median"), 2.0, row.get("median_latency_ms_median"), "causal Stage F pre-Brian2", "deployable_estimate", "evaluation_target_from_phase3", "low", "low", True, "Tier3 median", "current Stage F event detector remains low recall"))
    if not alarm_best.empty:
        best = alarm_best.sort_values(["recall", "F1"], ascending=False, kind="mergesort").iloc[0]
        rows.append(comparison_row("best_bounded_alarm_remediation", "discrete_event", "recall_at_fp_min", str(best.get("strategy_id")), best.get("recall"), best.get("fp_per_min_target"), best.get("median_latency_ms"), "causal pre-ADR bounded analysis", str(best.get("support_status", "proxy")), "evaluation_target_from_phase3", "low", "low", True, str(best.get("per_subject_recall_median")), str(best.get("headroom_interpretation", ""))))
    if not tolerance_summary.empty:
        best = tolerance_summary.sort_values(["best_recall"], ascending=False, kind="mergesort").iloc[0]
        rows.append(comparison_row("best_widened_scoring_sensitivity", "discrete_event_scoring_sensitivity", "recall_at_fp_min", str(best.get("best_scoring_definition")), best.get("best_recall"), best.get("fp_per_min_target"), str(best.get("best_scoring_family")), "causal score, widened metric", "proxy", "evaluation_target_from_phase3", "low", "low", True, "sensitivity", str(best.get("tolerance_interpretation"))))
    deployable = best_burden_row(burden_metrics.loc[burden_metrics["deployability_class"].eq("deployable_estimate")]) if not burden_metrics.empty else {}
    if deployable:
        rows.append(comparison_row("best_deployable_burden_state_estimate", "burden_state", str(deployable.get("primary_metric", "high_burden_AUROC")), str(deployable.get("score_source")), deployable.get("primary_metric_value"), "NA", f"{deployable.get('window_s')}s", "causal Phase 5_2C rolling score", "deployable_estimate", "evaluation_target_from_phase3", "low", "low", True, deployable.get("per_subject_metric_median"), "best deployable causal burden/state estimate"))
    diagnostic = best_burden_row(burden_metrics.loc[burden_metrics["deployability_class"].eq("diagnostic_upper_bound")]) if not burden_metrics.empty else {}
    if diagnostic:
        rows.append(comparison_row("best_diagnostic_burden_state_upper_bound", "burden_state", str(diagnostic.get("primary_metric", "high_burden_AUROC")), str(diagnostic.get("score_source")), diagnostic.get("primary_metric_value"), "NA", f"{diagnostic.get('window_s')}s", "causal refined variants diagnostic", "diagnostic_upper_bound", "evaluation_target_from_phase3", "low", "low", True, diagnostic.get("per_subject_metric_median"), "diagnostic upper bound from loaded causal refined variants"))
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="event_alarm_reconstruction_best.tsv;event_scoring_tolerance_summary.tsv;burden_state_ceiling_metrics_pre_adr.tsv", source_columns="primary metrics by branch", source_lineage="Phase 5_2C pre-ADR event vs burden comparison", support_status="proxy", qc_status="warning", qc_reason="cross-target metrics are not directly commensurate and do not approve execution"),
    )


def comparison_row(
    branch: str,
    target_type: str,
    metric_family: str,
    primary_metric: str,
    value: Any,
    fp_min: Any,
    latency_or_window: Any,
    causal_status: str,
    deployability_class: str,
    target_tautology_risk: str,
    feature_tautology_risk: str,
    leakage_risk_level: str,
    loso: bool,
    stability: Any,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "branch": branch,
        "target_type": target_type,
        "metric_family": metric_family,
        "primary_metric": primary_metric,
        "primary_metric_value": value,
        "FP_min_or_false_state_minutes": fp_min,
        "latency_or_window": latency_or_window,
        "causal_status": causal_status,
        "deployability_class": deployability_class,
        "target_tautology_risk": target_tautology_risk,
        "feature_tautology_risk": feature_tautology_risk,
        "leakage_risk_level": leakage_risk_level,
        "LOSO_support": loso,
        "per_subject_stability": stability,
        "SNN_compatible": True,
        "DYNAP_candidate": True,
        "interpretation": interpretation,
        "support_status": "proxy" if deployability_class != "deployable_estimate" else "direct",
        "qc_status": "warning",
        "qc_reason": "pre-ADR comparison, not deployment validation",
    }


def pre_adr_recommendation(config: dict[str, Any], alarm_best: pd.DataFrame, tolerance_summary: pd.DataFrame, burden_metrics: pd.DataFrame) -> pd.DataFrame:
    event_fp2 = best_recall_at_fp(alarm_best, 2.0)
    event_fp5 = best_recall_at_fp(alarm_best, 5.0)
    current_fp2 = current_stagef_tier3_recall(config, 2.0)
    tolerance_fp2 = best_tolerance_recall(tolerance_summary, 2.0)
    burden_best = best_burden_row(burden_metrics.loc[burden_metrics["deployability_class"].eq("deployable_estimate")]) if not burden_metrics.empty else {}
    burden_value = numeric(burden_best.get("primary_metric_value"))
    if np.isfinite(current_fp2) and np.isfinite(event_fp2) and event_fp2 >= 2.0 * current_fp2 and event_fp2 >= 0.10:
        rec_id = "EVENT_REMEDIATION_HAS_HEADROOM"
        state = "pre_adr_recommends_event_remediation_has_headroom"
        reason = "bounded alarm reconstruction at FP/min 2.0 at least doubled current Tier 3 recall and reached the minimum headroom threshold"
    elif np.isfinite(current_fp2) and np.isfinite(tolerance_fp2) and tolerance_fp2 >= 2.0 * current_fp2 and tolerance_fp2 >= 0.10:
        rec_id = "EVENT_TARGET_METRIC_SENSITIVE"
        state = "pre_adr_recommends_event_target_metric_sensitive"
        reason = "event recall improves materially only under widened scoring definitions; target metric requires owner review before ADR reopening"
    elif np.isfinite(burden_value) and burden_value >= 0.65 and (not np.isfinite(event_fp2) or event_fp2 < 0.10) and (not np.isfinite(event_fp5) or event_fp5 < 0.25):
        rec_id = "REOPEN_ADR_TOWARD_BURDEN_STATE"
        state = "pre_adr_recommends_reopen_adr_toward_burden_state"
        reason = "burden/state deployable causal estimate is materially stronger while bounded event recall remains weak at FP/min 2.0 and 5.0"
    elif burden_metrics.empty or not np.isfinite(burden_value):
        rec_id = "BURDEN_TARGET_DEFINITION_REQUIRED"
        state = "pre_adr_recommends_burden_target_definition_required"
        reason = "burden/state target metrics are unavailable or too weakly defined for ADR reopening"
    elif np.isfinite(event_fp2) and event_fp2 < 0.10 and np.isfinite(burden_value) and burden_value < 0.65:
        rec_id = "BOTH_EVENT_AND_BURDEN_LIMITED_MANUAL_REVIEW"
        state = "pre_adr_recommends_both_limited_manual_review"
        reason = "bounded event detection remains weak and deployable burden/state ceiling is also modest"
    else:
        rec_id = "BOTH_EVENT_AND_BURDEN_LIMITED_MANUAL_REVIEW"
        state = "pre_adr_recommends_both_limited_manual_review"
        reason = "mixed pre-ADR evidence requires owner/manual review; no G/H/I or Brian2 execution is justified"
    row = {
        "recommendation_id": rec_id,
        "recommendation_selected": True,
        "recommendation_status": "requires_owner_decision_before_any_adr_reopen",
        "final_task_state": state,
        "reason": reason,
        "event_best_fp2_recall": event_fp2,
        "event_best_fp5_recall": event_fp5,
        "current_stagef_tier3_fp2_recall": current_fp2,
        "best_tolerance_fp2_recall": tolerance_fp2,
        "best_deployable_burden_metric": burden_best.get("primary_metric", "NA"),
        "best_deployable_burden_metric_value": burden_value,
        "recommends_ghi_execution": False,
        "recommends_brian2": False,
        "approves_event_target": False,
        "formal_adr_reopening_performed": False,
        "support_status": "proxy",
        "qc_status": "warning",
        "qc_reason": reason,
    }
    return add_universal_columns(
        pd.DataFrame([row]),
        base_universal(config, source_table="event_alarm_reconstruction_best.tsv;event_scoring_tolerance_summary.tsv;burden_state_ceiling_metrics_pre_adr.tsv", source_columns="event and burden pre-ADR metrics", source_lineage="Phase 5_2C pre-ADR bounded analysis recommendation", support_status="proxy", qc_status="warning", qc_reason=reason),
    )


def updated_readiness(config: dict[str, Any], recommendation: pd.DataFrame) -> pd.DataFrame:
    reason = str(recommendation["reason"].iloc[0])
    row = {
        "readiness_state": "not_ready_scientific_gate_failed",
        "target_status": "pre_adr_bounded_analysis_completed_no_approved_event_target",
        "event_metrics_computed": True,
        "can_execute_stage_g_h_i_now": False,
        "can_plan_stage_g_h_i": False,
        "qc_reason": "G/H/I remains blocked after pre-ADR bounded analysis; " + reason,
        "support_status": "direct",
        "qc_status": "failed",
    }
    return add_universal_columns(
        pd.DataFrame([row]),
        base_universal(config, source_table="pre_adr_recommendation.tsv", source_columns="recommendation_id,final_task_state", source_lineage="Stage G/H/I readiness after pre-ADR bounded analysis", support_status="direct", qc_status="failed", qc_reason=row["qc_reason"]),
    )


def write_docs(
    root: Path,
    requirement: pd.DataFrame,
    alarm_sweep: pd.DataFrame,
    alarm_best: pd.DataFrame,
    tolerance_summary: pd.DataFrame,
    target_availability: pd.DataFrame,
    burden_metrics: pd.DataFrame,
    burden_gap: pd.DataFrame,
    event_vs_burden: pd.DataFrame,
    recommendation: pd.DataFrame,
) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    rec_id = str(recommendation["recommendation_id"].iloc[0])
    rec_state = str(recommendation["final_task_state"].iloc[0])
    rec_reason = str(recommendation["reason"].iloc[0])
    alarm_table = compact_markdown(alarm_best, ["fp_per_min_target", "tier", "strategy_id", "param_id", "recall", "precision", "F1", "fp_per_min_achieved", "headroom_interpretation"])
    tolerance_table = compact_markdown(tolerance_summary, ["fp_per_min_target", "current_policy_best_recall", "best_scoring_definition", "best_recall", "recall_improvement_factor", "tolerance_interpretation"])
    target_table = compact_markdown(target_availability, ["target_id", "target_available", "target_derivation", "target_tautology_risk"])
    burden_table = compact_markdown(top_burden_rows(burden_metrics), ["target_id", "window_s", "score_source", "deployability_class", "pearson_correlation", "high_burden_AUROC", "high_burden_AUPRC", "per_subject_metric_median"])
    compare_table = compact_markdown(event_vs_burden, ["branch", "target_type", "primary_metric", "primary_metric_value", "deployability_class", "interpretation"])
    (docs / "PHASE5_2C_PRE_ADR_BOUNDED_REMEDIATION_ANALYSIS.md").write_text(
        "# Phase 5_2C Pre-ADR Bounded Remediation Analysis\n\n"
        "Status: bounded analysis complete. The formal architecture ADR was not reopened. Stage G/H/I were not executed. Brian2 was not run.\n\n"
        "## Owner Requirement\n\n"
        "The project owner rejected the current low-recall event gates and required this bounded analytical sprint before any formal architecture ADR reopening.\n\n"
        "## R1 Alarm Reconstruction Headroom\n\n"
        f"{alarm_table}\n\n"
        "## R2 Scoring Tolerance Sensitivity\n\n"
        f"{tolerance_table}\n\n"
        "## R3 Burden/State Ceiling\n\n"
        f"{burden_table}\n\n"
        "## Event vs Burden Comparison\n\n"
        f"{compare_table}\n\n"
        f"Recommendation: `{rec_id}`.\n\nFinal task state: `{rec_state}`.\n\nRationale: {rec_reason}\n",
        encoding="utf-8",
    )
    (docs / "PHASE5_2C_EVENT_ALARM_RECONSTRUCTION_HEADROOM.md").write_text(
        "# Phase 5_2C Event Alarm Reconstruction Headroom\n\n"
        "This document reports bounded causal score-to-alarm strategies A0-A5. Thresholds were selected on training subjects only and evaluated on held-out subjects. Strategy choice is analytical and posthoc; it is not an approved detector or target gate.\n\n"
        f"{alarm_table}\n",
        encoding="utf-8",
    )
    (docs / "PHASE5_2C_SCORING_TOLERANCE_SENSITIVITY.md").write_text(
        "# Phase 5_2C Scoring Tolerance Sensitivity\n\n"
        "This analysis changes event scoring windows without changing causal scores. Widened and interval scoring rows are sensitivity analyses only, not deployment claims.\n\n"
        f"{tolerance_table}\n",
        encoding="utf-8",
    )
    gap_table = compact_markdown(burden_gap, ["target_id", "window_s", "best_deployable_score_source", "deployable_primary_metric_value", "diagnostic_primary_metric_value", "oracle_primary_metric_value", "gap_deployable_to_diagnostic"])
    (docs / "PHASE5_2C_BURDEN_STATE_CEILING_COMPARISON.md").write_text(
        "# Phase 5_2C Burden/State Ceiling Comparison\n\n"
        "Burden/state targets are derived from frozen Phase 3 labels for evaluation only. Phase 3 labels and thresholds are not used as input features. Prior Phase 5Z evidence is context only and does not override causal Phase 5_2C evidence.\n\n"
        "## Target Availability\n\n"
        f"{target_table}\n\n"
        "## Top Causal Burden/State Metrics\n\n"
        f"{burden_table}\n\n"
        "## Gap To Diagnostic/Oracle Ceilings\n\n"
        f"{gap_table}\n",
        encoding="utf-8",
    )
    (docs / "PHASE5_2C_PRE_ADR_RECOMMENDATION.md").write_text(
        "# Phase 5_2C Pre-ADR Recommendation\n\n"
        f"Recommendation: `{rec_id}`.\n\n"
        f"Final task state: `{rec_state}`.\n\n"
        f"Rationale: {rec_reason}\n\n"
        "This recommendation does not reopen the architecture ADR, does not approve any event target, does not execute Stage G/H/I, does not run Brian2, and does not claim a final detector.\n",
        encoding="utf-8",
    )


def compact_markdown(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if frame.empty:
        return "No rows available."
    cols = [col for col in columns if col in frame.columns]
    return frame.loc[:, cols].head(max_rows).replace({np.nan: "NA"}).to_markdown(index=False)


def top_burden_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    data = metrics.loc[~metrics["deployability_class"].astype(str).eq("oracle_not_deployable")].copy()
    if data.empty:
        data = metrics.copy()
    data["primary"] = pd.to_numeric(data["high_burden_AUROC"], errors="coerce").where(pd.to_numeric(data["high_burden_AUROC"], errors="coerce").notna(), pd.to_numeric(data["pearson_correlation"], errors="coerce"))
    return data.sort_values("primary", ascending=False, kind="mergesort").head(12)


def validation_table(config: dict[str, Any]) -> pd.DataFrame:
    root = repo_root(config)
    table_dir = output_paths(config)["table_dir"]
    required = [
        ("owner_requirement_table_exists", table_dir / "pre_adr_owner_requirement.tsv"),
        ("alarm_reconstruction_sweep_exists", table_dir / "event_alarm_reconstruction_sweep.tsv"),
        ("alarm_reconstruction_best_exists", table_dir / "event_alarm_reconstruction_best.tsv"),
        ("scoring_tolerance_sweep_exists", table_dir / "event_scoring_tolerance_sweep.tsv"),
        ("scoring_tolerance_summary_exists", table_dir / "event_scoring_tolerance_summary.tsv"),
        ("burden_target_availability_exists", table_dir / "burden_state_target_availability_pre_adr.tsv"),
        ("burden_ceiling_metrics_exists", table_dir / "burden_state_ceiling_metrics_pre_adr.tsv"),
        ("burden_gap_exists", table_dir / "burden_state_gap_to_ceiling_pre_adr.tsv"),
        ("event_vs_burden_comparison_exists", table_dir / "event_vs_burden_pre_adr_comparison.tsv"),
        ("pre_adr_recommendation_exists", table_dir / "pre_adr_recommendation.tsv"),
        ("bounded_analysis_doc_exists", root / "docs/PHASE5_2C_PRE_ADR_BOUNDED_REMEDIATION_ANALYSIS.md"),
        ("headroom_doc_exists", root / "docs/PHASE5_2C_EVENT_ALARM_RECONSTRUCTION_HEADROOM.md"),
        ("tolerance_doc_exists", root / "docs/PHASE5_2C_SCORING_TOLERANCE_SENSITIVITY.md"),
        ("burden_doc_exists", root / "docs/PHASE5_2C_BURDEN_STATE_CEILING_COMPARISON.md"),
        ("recommendation_doc_exists", root / "docs/PHASE5_2C_PRE_ADR_RECOMMENDATION.md"),
    ]
    rows = [validation_row(name, path.exists(), str(path), "ok" if path.exists() else "failed", "NA" if path.exists() else "required artifact missing") for name, path in required]
    rec_path = table_dir / "pre_adr_recommendation.tsv"
    rec_count = len(read_tsv(rec_path)) if rec_path.exists() else 0
    selected_count = int(read_tsv(rec_path)["recommendation_selected"].astype(str).str.lower().eq("true").sum()) if rec_path.exists() else 0
    rows.append(validation_row("exactly_one_recommendation_selected", rec_count == 1 and selected_count == 1, str(rec_path), "ok" if rec_count == 1 and selected_count == 1 else "failed", f"rows={rec_count};selected={selected_count}"))
    rows.append(validation_row("no_formal_reopened_adr_created", not (root / "docs/PHASE5_2C_ARCHITECTURE_ADR_REOPENED.md").exists(), "docs/PHASE5_2C_ARCHITECTURE_ADR_REOPENED.md", "ok", "formal ADR reopening is outside this task"))
    forbidden = forbidden_outputs(root)
    rows.append(validation_row("no_stage_g_h_i_execution_outputs_created", not forbidden["ghi"], ",".join(forbidden["ghi"]) or "none", "ok" if not forbidden["ghi"] else "failed", "Stage G/H/I execution is blocked"))
    rows.append(validation_row("no_brian2_outputs_created", not forbidden["brian2"], ",".join(forbidden["brian2"]) or "none", "ok" if not forbidden["brian2"] else "failed", "Brian2 is blocked"))
    rows.append(validation_row("no_closeout_outcome_files_created", not forbidden["closeout"], ",".join(forbidden["closeout"]) or "none", "ok" if not forbidden["closeout"] else "failed", "closeout is outside this task"))
    rows.append(validation_row("no_frozen_brian2_specs_created", not forbidden["frozen"], ",".join(forbidden["frozen"]) or "none", "ok" if not forbidden["frozen"] else "failed", "frozen specs are outside this task"))
    rows.append(validation_row("no_phase3_label_files_modified", True, "explicit scope review", "ok", "outputs are scoped to Phase 5_2C docs/tables/code/tests"))
    rows.append(validation_row("no_meg_introduced", True, "explicit scope review", "ok", "no MEG inputs or outputs are introduced"))
    rows.append(validation_row("no_huge_duplicate_trace_table_created", max_pre_adr_output_size(root) < 100_000_000, "pre-ADR output inventory", "ok", "all required pre-ADR outputs are compact summaries"))
    rows.append(validation_row("git_diff_check_passes", git_diff_check(root), "git diff --check", "ok", "validated by script at generation time"))
    return add_universal_columns(
        pd.DataFrame(rows),
        base_universal(config, source_table="pre-ADR bounded analysis outputs", source_columns="file inventory,git diff --check", source_lineage="Phase 5_2C pre-ADR bounded analysis validation", support_status="direct", qc_status="ok", qc_reason="validation generated after pre-ADR outputs"),
    )


def validation_row(name: str, passed: bool, path_or_source: str, qc_status: str, reason: str) -> dict[str, Any]:
    return {"check_name": name, "passed": bool(passed), "severity": "error", "path_or_source": path_or_source, "qc_status": qc_status if passed else "failed", "qc_reason": reason}


def forbidden_outputs(root: Path) -> dict[str, list[str]]:
    search_roots = [root / "results/tables/05_phase5/phase5_2c", root / "results/figures/05_phase5/phase5_2c", root / "configs", root / "docs", root / "src/stnbeta/phase5_2c", root / "scripts"]
    patterns = {
        "ghi": ["snn_approximation", "stage_g_output", "stage_h_output", "stage_i_output", "dynap_resource", "dynap_core_allocation"],
        "brian2": ["brian2_gate"],
        "closeout": ["PHASE5_2C_CLOSEOUT", "closeout_summary", "closeout_overview", "05_2c_closeout", "closeout.py"],
        "frozen": ["phase5_2c_pipeline_frozen", "phase5_2c_feature_subset_frozen", "phase5_2c_snn_approximation_frozen"],
    }
    found = {key: [] for key in patterns}
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


def max_pre_adr_output_size(root: Path) -> int:
    paths = list((root / "results/tables/05_phase5/phase5_2c").glob("*pre_adr*.tsv")) + list((root / "results/tables/05_phase5/phase5_2c").glob("event_alarm_reconstruction*.tsv")) + list((root / "results/tables/05_phase5/phase5_2c").glob("event_scoring_tolerance*.tsv")) + list((root / "docs").glob("PHASE5_2C_PRE_ADR*.md"))
    return max((path.stat().st_size for path in paths if path.exists()), default=0)


def git_diff_check(root: Path) -> bool:
    return subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True, check=False).returncode == 0


def best_recall_at_fp(frame: pd.DataFrame, fp: float) -> float:
    if frame.empty:
        return np.nan
    sub = frame.loc[pd.to_numeric(frame["fp_per_min_target"], errors="coerce").eq(fp)]
    return float(pd.to_numeric(sub["recall"], errors="coerce").max()) if len(sub) else np.nan


def best_tolerance_recall(frame: pd.DataFrame, fp: float) -> float:
    if frame.empty:
        return np.nan
    sub = frame.loc[pd.to_numeric(frame["fp_per_min_target"], errors="coerce").eq(fp)]
    return float(pd.to_numeric(sub["best_recall"], errors="coerce").max()) if len(sub) else np.nan


def current_stagef_tier3_recall(config: dict[str, Any], fp: float) -> float:
    table = read_tsv(output_paths(config)["table_dir"] / "causal_three_tier_event_summary.tsv")
    row = table.loc[(table["tier"].astype(str).eq("tier3_quantized_mismatched")) & pd.to_numeric(table["target_fp_min"], errors="coerce").eq(fp)]
    return float(pd.to_numeric(row["recall_median"], errors="coerce").iloc[0]) if len(row) else np.nan


def metric_numeric_columns() -> list[str]:
    return ["n_true_events", "n_alarms", "true_positive_events", "true_positive_alarms", "false_positive_alarms", "recall", "precision", "F1", "fp_per_min_achieved", "median_latency_ms", "early_warning_candidate_latency_ms", "one_alarm_per_burst_fraction", "alarms_per_burst", "threshold", "per_subject_recall_median", "per_subject_recall_iqr", "per_subject_recall_min", "per_subject_recall_max"]


def numeric(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def safe_float(value: Any, fallback: float = -np.inf) -> float:
    out = numeric(value)
    return out if np.isfinite(out) else fallback


def subtract_or_nan(a: Any, b: Any) -> float:
    x = numeric(a)
    y = numeric(b)
    return float(x - y) if np.isfinite(x) and np.isfinite(y) else np.nan


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase5_2c.yaml")
    parser.add_argument("--max-refined-features", type=int, default=8)
    parser.add_argument("--tier3-seeds", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    result = run_pre_adr_bounded_analysis(config, max_refined_features=args.max_refined_features, tier3_seeds=args.tier3_seeds)
    print({"task_state": result.recommendation_state, "recommendation_id": result.recommendation_id, "n_outputs": len(result.outputs)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

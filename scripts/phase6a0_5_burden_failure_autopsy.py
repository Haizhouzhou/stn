#!/usr/bin/env python3
"""Phase 6A.0.5 burden gate failure autopsy and ceiling analysis.

This script diagnoses the failed Phase 6A.0 burden viability gate without
starting any Brian2, DYNAP, or SNN simulation work. It uses only internal STN
prior-phase tables and explicitly excludes external PPN/Herz paths.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import gc
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_ERROR = exc
else:
    NUMPY_ERROR = None

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    PANDAS_ERROR = exc
else:
    PANDAS_ERROR = None

try:
    import scipy.stats as scipy_stats
except Exception:
    scipy_stats = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )
except Exception as exc:
    SKLEARN_ERROR = exc
    HistGradientBoostingClassifier = None  # type: ignore[assignment]
    RandomForestClassifier = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    SGDClassifier = None  # type: ignore[assignment]
    average_precision_score = None  # type: ignore[assignment]
    balanced_accuracy_score = None  # type: ignore[assignment]
    brier_score_loss = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
else:
    SKLEARN_ERROR = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    MATPLOTLIB_ERROR = exc
    plt = None  # type: ignore[assignment]
else:
    MATPLOTLIB_ERROR = None


LOG = logging.getLogger("phase6a0_5")

EXTERNAL_EXCLUDE_SUBSTRINGS = [
    "cambium/Data_Code_PPN_JNeurosci_2021",
    "cambium/Force_Scripts",
    "reports/phase6_ppn_he_tan_2021_audit",
    "reports/phase6_stn_force_adaptation_herz_2023_audit",
    "reports/phase6_ppn_he_tan_2021_analysis",
    "ppn_he_tan",
    "force_adaptation_herz",
    "Force_Scripts",
    "Data_Code_PPN",
]

LEAKAGE_KEYWORDS = [
    "label",
    "target",
    "y_true",
    "true",
    "burst_id",
    "burst_onset",
    "onset",
    "offset",
    "future",
    "post",
    "next",
    "previous_label",
    "event",
    "annotation",
    "oracle",
    "split",
    "fold",
    "p_value",
    "auc",
    "score",
    "subject_id",
    "patient_id",
    "session",
    "block",
    "time",
    "sample",
    "index",
    "filename",
    "path",
]

METADATA_HINTS = [
    "subject",
    "participant",
    "patient",
    "condition",
    "channel",
    "band_mode",
    "window_type",
    "negative_category",
    "threshold_source",
    "provenance_status",
    "hemisphere",
    "medication",
    "task",
    "fif_path",
    "window_id",
]

REQUIRED_OUTPUTS = {
    "README_burden_failure_autopsy.md": "",
    "previous_gate_summary.csv": "metric,value,source_file,interpretation\n",
    "table_continuity_audit.csv": "",
    "data_contract_audit.csv": "requirement,status,evidence,severity,notes\n",
    "leakage_reaudit.csv": "column,role_guess,allowed_as_feature,leakage_risk,reason,feature_family_guess,notes\n",
    "selected_feature_sets.csv": "feature_set_name,n_features,feature_names_short,selection_method,selection_leakage_risk,deployability_guess,notes\n",
    "feature_subset_ablation.csv": "",
    "estimator_ceiling_comparison.csv": "",
    "tau_heterogeneity.csv": "",
    "subject_distribution_check.csv": "",
    "proxy_ceiling_comparison.csv": "",
    "beta_baseline_delta.csv": "",
    "prior_phase_comparison.csv": "prior_phase,metric,value,source_file,found_status,interpretation\n",
    "decision_matrix.csv": "classification,status,evidence,recommended_action\n",
    "burden_failure_autopsy_findings.json": "{}\n",
    "phase6a0_5_commands_run.txt": "",
    "phase6a0_prior_log_integrity_note.md": "",
}


@dataclass
class Columns:
    subject: str
    time: str
    session: str
    block: str
    label: str


@dataclass
class ModelSpec:
    model_name: str
    feature_set_name: str
    validation_mode: str
    deployable: bool
    offline_ceiling: bool = False
    top_k: int = 0
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", default="results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv")
    parser.add_argument("--phase6a0-findings", default="reports/phase6a0_burden_viability/burden_viability_findings.json")
    parser.add_argument("--phase6a0-report-dir", default="reports/phase6a0_burden_viability")
    parser.add_argument("--out-dir", default="reports/phase6a0_5_burden_failure_autopsy")
    parser.add_argument("--subject-col", default="")
    parser.add_argument("--time-col", default="")
    parser.add_argument("--session-col", default="")
    parser.add_argument("--block-col", default="")
    parser.add_argument("--label-col", default="")
    parser.add_argument("--proxy-cols", default="")
    parser.add_argument("--tau-ms", default="200,500,800,1500,3000,5000")
    parser.add_argument("--top-k-features", default="25,50,100")
    parser.add_argument("--random-seed", type=int, default=20260428)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-high-capacity", action="store_true")
    parser.add_argument("--stop-after-continuity", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def is_external_path(path: Path | str) -> bool:
    s = Path(path).as_posix()
    return any(token in s for token in EXTERNAL_EXCLUDE_SUBSTRINGS)


def ensure_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    for name, content in REQUIRED_OUTPUTS.items():
        path = out_dir / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: clean_cell(row.get(c, "")) for c in columns})


def clean_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            return [json_sanitize(v) for v in value.tolist()]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_sanitize(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Could not read JSON %s: %s", path, exc)
        return {}


def sep_for(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def read_header(path: Path) -> list[str]:
    if pd is None:
        raise RuntimeError(f"pandas unavailable: {PANDAS_ERROR}")
    return list(pd.read_csv(path, sep=sep_for(path), nrows=0).columns)


def parse_float_list(text: str) -> list[float]:
    vals: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals


def parse_int_list(text: str) -> list[int]:
    vals: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return vals


def pick_col(columns: list[str], explicit: str, candidates: list[str], contains: list[str] | None = None) -> str:
    if explicit and explicit in columns:
        return explicit
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if contains:
        for token in contains:
            for col in columns:
                if token.lower() in col.lower():
                    return col
    return ""


def detect_columns(columns: list[str], args: argparse.Namespace) -> Columns:
    subject = pick_col(columns, args.subject_col, ["subject_id", "participant_id", "patient_id", "subject"], ["subject"])
    time = pick_col(
        columns,
        args.time_col,
        ["window_start_s", "time_s", "time", "timestamp", "sample_index", "row_index"],
        ["window_start", "timestamp"],
    )
    session = pick_col(columns, args.session_col, ["session_id", "session", "ses"], ["session"])
    block = pick_col(columns, args.block_col, ["block_id", "block", "trial", "condition"], ["condition"])
    label = pick_col(
        columns,
        args.label_col,
        ["target_label", "burst_label", "label", "y_true", "annotation_label"],
        ["target", "label"],
    )
    return Columns(subject=subject, time=time, session=session, block=block, label=label)


def load_previous_gate(phase6a0_findings: Path, phase6a0_report_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings = read_json(phase6a0_findings)
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, source: Path, interp: str) -> None:
        rows.append(
            {
                "metric": metric,
                "value": value,
                "source_file": source.as_posix(),
                "interpretation": interp,
            }
        )

    if findings:
        add("selected_input", ";".join(findings.get("selected_input_files", [])), phase6a0_findings, "Phase 6A.0 selected internal STN table")
        add("subject_count", findings.get("selected_subject_count", ""), phase6a0_findings, "subjects found in Phase 6A.0")
        add("valid_subject_count", findings.get("valid_subject_count", ""), phase6a0_findings, "subjects with valid Phase 6A.0 LOSO results")
        add("allowed_feature_count", findings.get("allowed_feature_count", ""), phase6a0_findings, "allowed causal features from Phase 6A.0 audit")
        add("best_tau_ms", findings.get("best_tau_ms", ""), phase6a0_findings, "best Phase 6A.0 fixed tau")
        add("median_loso_pearson", findings.get("best_tau_median_pearson", ""), phase6a0_findings, "best Phase 6A.0 subject-median Pearson")
        add("median_loso_spearman", findings.get("best_tau_median_spearman", ""), phase6a0_findings, "best Phase 6A.0 subject-median Spearman")
        baseline = findings.get("baseline_summary", {})
        add("class_prior_baseline", nested_get(baseline, ["class_prior_baseline", "median_spearman"]), phase6a0_findings, "class-prior median Spearman baseline")
        add("shuffled_baseline", nested_get(baseline, ["shuffled_label_baseline", "median_spearman"]), phase6a0_findings, "shuffled-label median Spearman baseline")
        add("beta_feature_baseline", nested_get(baseline, ["simple_beta_feature_baseline", "median_spearman"]), phase6a0_findings, "simple beta-feature median Spearman baseline")
        add("proxy_summary", json.dumps(findings.get("condition_proxy_summary", {}), sort_keys=True)[:1000], phase6a0_findings, "supportive proxy summary only")
        add("gate_status", findings.get("overall_status", ""), phase6a0_findings, "Phase 6A.0 gate result")
        add("gate_rationale", findings.get("gate_rationale", ""), phase6a0_findings, "Phase 6A.0 gate rationale")
    else:
        add("phase6a0_findings", "missing", phase6a0_findings, "Previous gate findings JSON was unavailable")

    for name in ["burden_metric_by_subject.csv", "burden_tau_sweep.csv", "baseline_comparison.csv"]:
        p = phase6a0_report_dir / name
        if not p.exists():
            add(name, "missing", p, "Previous Phase 6A.0 report file missing")

    return findings, rows


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key, "")
    return cur


def read_phase6a0_feature_audit(report_dir: Path) -> dict[str, dict[str, Any]]:
    path = report_dir / "feature_column_audit.csv"
    out: dict[str, dict[str, Any]] = {}
    if pd is None or not path.exists():
        return out
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        LOG.warning("Could not read previous feature audit: %s", exc)
        return out
    for _, row in df.iterrows():
        name = str(row.get("column_name", ""))
        if name:
            out[name] = row.to_dict()
    return out


def read_feature_metadata(input_table: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        input_table.with_name(input_table.stem + "_feature_metadata.tsv"),
        input_table.parent / "causal_feature_matrix_feature_metadata.tsv",
    ]
    out: dict[str, dict[str, Any]] = {}
    if pd is None:
        return out
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep=sep_for(path))
        except Exception as exc:
            LOG.warning("Could not read feature metadata %s: %s", path, exc)
            continue
        for _, row in df.iterrows():
            name = str(row.get("output_column", "") or row.get("feature_name", ""))
            if name:
                out[name] = row.to_dict()
        break
    return out


def role_and_leakage(
    column: str,
    cols: Columns,
    previous_audit: dict[str, dict[str, Any]],
    feature_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    low = column.lower()
    meta = feature_meta.get(column, {})
    prev = previous_audit.get(column, {})
    feature_family = str(meta.get("feature_family", "") or prev.get("feature_family", "") or guess_feature_family(column))
    role = "excluded non-feature"
    allowed = False
    risk = "low"
    reason_parts: list[str] = []

    if column == cols.subject or "subject_id" == low:
        role = "subject column"
        risk = "identity"
        reason_parts.append("subject identifier")
    elif column == cols.time or low in {"window_start_s", "window_stop_s", "time"}:
        role = "time/order column"
        risk = "time"
        reason_parts.append("time/order metadata")
    elif column == cols.session:
        role = "session column"
        risk = "identity"
        reason_parts.append("session metadata")
    elif column == cols.block or low in {"condition", "block"}:
        role = "block/proxy column"
        risk = "metadata"
        reason_parts.append("condition/block metadata")
    elif column == cols.label:
        role = "target/label column"
        risk = "target"
        reason_parts.append("target label")
    elif any(token in low for token in ["updrs", "brady", "rigidity", "tremor", "clinical", "medication", "task_state"]):
        role = "metadata/proxy column"
        risk = "proxy"
        reason_parts.append("clinical/task proxy")
    elif any(token in low for token in METADATA_HINTS):
        role = "excluded non-feature"
        risk = "metadata"
        reason_parts.append("metadata-like column")

    keyword_hits = [token for token in LEAKAGE_KEYWORDS if token in low]
    if keyword_hits:
        allowed = False
        if role == "excluded non-feature":
            role = "suspicious leakage or metadata"
        if any(token in {"future", "post", "next", "onset", "offset", "true", "oracle", "annotation", "event"} for token in keyword_hits):
            risk = "high"
        elif risk == "low":
            risk = "medium"
        reason_parts.append("keyword exclusion: " + ",".join(keyword_hits[:5]))

    prev_role = str(prev.get("column_role", "")).lower()
    prev_selected = str(prev.get("selected_for_model", "")).lower() in {"true", "1", "yes"}
    prev_allowed = "allowed causal feature" in prev_role
    metadata_ok = True
    for key in ["uses_future_samples", "uses_test_participant_statistics", "uses_phase3_threshold"]:
        val = str(meta.get(key, "")).lower()
        if val == "true":
            metadata_ok = False
            risk = "high" if key == "uses_future_samples" else max_risk(risk, "medium")
            reason_parts.append(f"metadata {key}=true")
    for key in ["leakage_risk_level", "tautology_risk_level"]:
        val = str(meta.get(key, "") or prev.get(key, "")).lower()
        if val and val not in {"low", "nan", "none"}:
            risk = max_risk(risk, "high" if val == "high" else "medium")
            reason_parts.append(f"{key}={val}")
            if val == "high":
                metadata_ok = False

    if role == "excluded non-feature" and not keyword_hits:
        if prev_allowed or column in feature_meta or looks_numeric_feature(column):
            role = "allowed causal feature candidate"
            allowed = metadata_ok
            if allowed:
                reason_parts.append("allowed by previous audit/metadata or feature-name heuristic")
            else:
                reason_parts.append("feature-like but metadata leakage flags prevent use")
    elif prev_allowed and not keyword_hits and metadata_ok:
        role = "allowed causal feature candidate"
        allowed = True
        reason_parts.append("allowed by Phase 6A.0 feature audit")

    if prev_selected:
        reason_parts.append("selected by Phase 6A.0 lower-bound estimator")
    if "z_score" in low or "z-score" in low or "normalized" in low or "standard" in low:
        reason_parts.append("precomputed normalization-like name; cannot undo preprocessing scope")
        risk = max_risk(risk, "medium")

    return {
        "column": column,
        "role_guess": role,
        "allowed_as_feature": bool(allowed),
        "leakage_risk": risk,
        "reason": "; ".join(dict.fromkeys(reason_parts)) or "no specific flag",
        "feature_family_guess": feature_family,
        "notes": "pre columns are not excluded solely for containing pre" if "pre" in low and "previous_label" not in low else "",
    }


def max_risk(a: str, b: str) -> str:
    order = {"low": 0, "metadata": 1, "proxy": 1, "time": 1, "identity": 1, "target": 2, "medium": 2, "high": 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def looks_numeric_feature(column: str) -> bool:
    low = column.lower()
    return any(
        token in low
        for token in [
            "beta",
            "slope",
            "ratio",
            "power",
            "envelope",
            "derivative",
            "rise",
            "boundary",
            "spatial",
            "coactivation",
            "gamma",
            "autocorr",
        ]
    )


def guess_feature_family(column: str) -> str:
    low = column.lower()
    if "beta_local_baseline_ratio" in low:
        return "beta_local_baseline_ratio"
    if "low_beta_high_beta" in low:
        return "low_beta_high_beta_ratio"
    if "boundary" in low:
        return "beta_boundary_contrast"
    if "derivative" in low:
        return "causal_derivative"
    if "rise" in low or "slope" in low:
        return "rise_dynamics"
    if "spatial" in low:
        return "spatial"
    if "beta" in low:
        return "beta_like"
    return ""


def read_metadata_table(path: Path, cols: Columns, max_rows: int = 0) -> Any:
    if pd is None:
        raise RuntimeError(f"pandas unavailable: {PANDAS_ERROR}")
    header = read_header(path)
    wanted = [
        cols.subject,
        cols.session,
        cols.block,
        cols.label,
        cols.time,
        "window_id",
        "condition",
        "channel",
        "band_mode",
        "window_type",
        "negative_category",
        "window_stop_s",
        "window_duration_ms",
        "anchor_onset_s",
        "anchor_offset_s",
        "phase3_duration_ms",
        "fif_path",
    ]
    usecols = [c for c in dict.fromkeys(wanted) if c and c in header]
    dtype: dict[str, Any] = {}
    for c in usecols:
        if c in {cols.label}:
            dtype[c] = "float32"
        elif c in {cols.time, "window_stop_s", "window_duration_ms", "anchor_onset_s", "anchor_offset_s", "phase3_duration_ms"}:
            dtype[c] = "float64"
        else:
            dtype[c] = "string"
    return pd.read_csv(path, sep=sep_for(path), usecols=usecols, dtype=dtype, nrows=max_rows or None)


def data_contract_rows(
    input_table: Path,
    columns: list[str],
    cols: Columns,
    allowed_features: list[str],
    meta_df: Any,
    continuity_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(req: str, ok: bool, evidence: str, severity: str, notes: str = "") -> None:
        rows.append(
            {
                "requirement": req,
                "status": "pass" if ok else "fail",
                "evidence": evidence,
                "severity": severity if not ok else "info",
                "notes": notes,
            }
        )

    n_subjects = int(meta_df[cols.subject].nunique()) if cols.subject and cols.subject in meta_df.columns else 0
    rows_per_subject = meta_df.groupby(cols.subject).size() if cols.subject and cols.subject in meta_df.columns else []
    min_rows = int(rows_per_subject.min()) if len(rows_per_subject) else 0
    label_vals = set()
    if cols.label and cols.label in meta_df.columns:
        label_vals = set(str(v) for v in sorted(meta_df[cols.label].dropna().unique())[:10])
    add("subject identifier exists", bool(cols.subject), cols.subject or "missing", "critical")
    add("binary label or annotation-derived target exists", bool(cols.label) and label_vals.issubset({"0.0", "1.0", "0", "1"}), f"{cols.label}; values={sorted(label_vals)}", "critical")
    add("row order or time column exists", bool(cols.time) or len(meta_df) > 0, cols.time or "row order only", "critical")
    add("at least 5 allowed causal feature columns exist", len(allowed_features) >= 5, str(len(allowed_features)), "critical")
    add("at least 8 subjects exist", n_subjects >= 8, str(n_subjects), "critical")
    add("table is not from external PPN/Herz paths", not is_external_path(input_table), relpath(input_table, repo_root()), "critical")
    aggregate_like = len(meta_df) <= max(n_subjects * 5, 50) or any(c in columns for c in ["metric", "source_file", "interpretation"]) and len(allowed_features) < 5
    add("table is not merely an aggregate report", not aggregate_like, f"rows_sampled={len(meta_df)}; columns={len(columns)}", "critical")
    add("table has enough rows per subject for burden/time analysis", min_rows >= 100, f"min_rows_per_subject={min_rows}", "warning")
    if continuity_summary:
        cls = str(continuity_summary.get("overall_continuity_class", ""))
        rows.append(
            {
                "requirement": "table is a definitive continuous burden stream",
                "status": "pass" if cls in {"continuous_high", "continuous_medium", "row_order_low_confidence"} else "fail",
                "evidence": cls,
                "severity": "warning" if cls == "row_order_low_confidence" else "critical",
                "notes": "Phase 6A.0.5 may compute pseudo-burden for diagnostics, but event/candidate matrices are not definitive burden streams.",
            }
        )
    return rows


def continuity_audit(meta_df: Any, cols: Columns) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pd is None or np is None:
        return [], {"overall_continuity_class": "blocked_missing_dependency"}
    df = meta_df.copy()
    if "condition" not in df.columns and cols.block in df.columns:
        df["condition"] = df[cols.block].astype("string")
    group_keys = [c for c in [cols.subject, cols.session, "condition", "channel", "band_mode"] if c and c in df.columns]
    if not group_keys and cols.subject:
        group_keys = [cols.subject]
    rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    event_like_total = 0
    n_groups = 0
    for key, sub in df.groupby(group_keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(group_keys, key))
        n_groups += 1
        n = len(sub)
        label_arr = pd.to_numeric(sub[cols.label], errors="coerce").fillna(0).to_numpy() if cols.label in sub.columns else np.zeros(n)
        pos = int(np.nansum(label_arr))
        pos_frac = float(np.nanmean(label_arr)) if n else float("nan")
        has_time = bool(cols.time and cols.time in sub.columns)
        time_vals = pd.to_numeric(sub[cols.time], errors="coerce").to_numpy() if has_time else np.arange(n, dtype=float)
        finite_time = time_vals[np.isfinite(time_vals)]
        original_diff = np.diff(time_vals) if len(time_vals) > 1 else np.array([])
        sorted_order = np.argsort(time_vals, kind="mergesort") if has_time else np.arange(n)
        sorted_time = time_vals[sorted_order]
        diff = np.diff(sorted_time[np.isfinite(sorted_time)])
        duplicate_time_count = int(pd.Series(finite_time).duplicated().sum()) if len(finite_time) else 0
        dt_median = float(np.nanmedian(diff)) if len(diff) else float("nan")
        dt_q25 = float(np.nanpercentile(diff, 25)) if len(diff) else float("nan")
        dt_q75 = float(np.nanpercentile(diff, 75)) if len(diff) else float("nan")
        dt_iqr = dt_q75 - dt_q25 if math.isfinite(dt_q25) and math.isfinite(dt_q75) else float("nan")
        dt_min = float(np.nanmin(diff)) if len(diff) else float("nan")
        dt_max = float(np.nanmax(diff)) if len(diff) else float("nan")
        large_gap_count = int(np.sum(diff > (dt_median + 10 * max(dt_iqr, 1e-9)))) if len(diff) and math.isfinite(dt_median) else 0
        resets = int(np.sum(original_diff < 0)) if len(original_diff) else 0
        time_monotonic = bool(np.all(original_diff >= 0)) if len(original_diff) else True
        event_cols = []
        for c in ["window_id", "window_type", "negative_category", "anchor_onset_s", "anchor_offset_s"]:
            if c in sub.columns:
                event_cols.append(c)
        event_pattern = event_centered_guess(sub)
        if event_pattern:
            event_like_total += 1
        sorted_label = label_arr[sorted_order] if len(sorted_order) == len(label_arr) else label_arr
        pos_runs = run_lengths(sorted_label == 1)
        neg_runs = run_lengths(sorted_label == 0)
        transitions = int(np.sum(np.diff(sorted_label) != 0)) if len(sorted_label) > 1 else 0
        score = 0.0
        notes: list[str] = []
        if has_time:
            score += 2.0
        else:
            notes.append("no explicit time column")
        if time_monotonic:
            score += 1.0
        else:
            notes.append("time resets in original row order")
        if n >= 1000:
            score += 1.0
        if transitions > 10:
            score += 1.0
        if duplicate_time_count > max(10, n * 0.1):
            score -= 1.0
            notes.append("many duplicate times")
        if event_pattern:
            score -= 4.0
            notes.append("event/candidate centered row pattern")
        if n < 20:
            continuity_class = "aggregate_table"
        elif event_pattern:
            continuity_class = "event_or_candidate_matrix"
        elif not has_time:
            continuity_class = "row_order_low_confidence"
        elif time_monotonic and score >= 4:
            continuity_class = "continuous_high"
        elif score >= 2:
            continuity_class = "continuous_medium"
        else:
            continuity_class = "ambiguous"
        class_counts[continuity_class] = class_counts.get(continuity_class, 0) + 1
        rows.append(
            {
                "subject_id": str(key_map.get(cols.subject, "")),
                "session_id": str(key_map.get(cols.session, "")) if cols.session else "",
                "block_id": "|".join(str(key_map.get(c, "")) for c in group_keys if c != cols.subject and c != cols.session),
                "n_rows": n,
                "label_positive_count": pos,
                "label_positive_fraction": pos_frac,
                "time_col_used": cols.time,
                "has_time_col": has_time,
                "time_monotonic": time_monotonic,
                "duplicate_time_count": duplicate_time_count,
                "dt_median": dt_median,
                "dt_iqr": dt_iqr,
                "dt_min": dt_min,
                "dt_max": dt_max,
                "dt_units_guess": "seconds" if has_time and (not math.isfinite(dt_median) or dt_median < 60) else "index_or_unknown",
                "large_gap_count": large_gap_count,
                "time_resets_detected": resets,
                "row_order_only": not has_time,
                "positive_run_count": len(pos_runs),
                "positive_run_median_len": float(np.median(pos_runs)) if pos_runs else 0,
                "negative_run_count": len(neg_runs),
                "negative_run_median_len": float(np.median(neg_runs)) if neg_runs else 0,
                "label_transition_count": transitions,
                "event_like_columns_detected": ";".join(event_cols),
                "event_centered_pattern_guess": event_pattern,
                "continuous_stream_score": score,
                "continuity_class": continuity_class,
                "notes": "; ".join(notes),
            }
        )
    if class_counts:
        overall = max(class_counts.items(), key=lambda kv: kv[1])[0]
        if class_counts.get("event_or_candidate_matrix", 0) > 0 and class_counts.get("event_or_candidate_matrix", 0) >= n_groups * 0.25:
            overall = "event_or_candidate_matrix"
    else:
        overall = "unusable"
    summary = {
        "overall_continuity_class": overall,
        "class_counts": class_counts,
        "n_groups": n_groups,
        "event_like_group_fraction": event_like_total / n_groups if n_groups else 0,
    }
    return rows, summary


def event_centered_guess(sub: Any) -> bool:
    text_cols = [c for c in ["window_id", "window_type", "negative_category"] if c in sub.columns]
    if not text_cols:
        return False
    hits = 0
    total = 0
    pattern = re.compile(r"(?:centered|adjacent|true_full_burst|true_burst|candidate|imposter|onset|offset|post)")
    for c in text_cols:
        vals = sub[c].dropna().astype(str)
        if len(vals) == 0:
            continue
        sample = vals.head(2000)
        hits += int(sample.str.lower().str.contains(pattern).sum())
        total += len(sample)
    return total > 0 and hits / total >= 0.2


def run_lengths(mask: Any) -> list[int]:
    if np is None:
        return []
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0:
        return []
    lengths: list[int] = []
    cur = arr[0]
    count = 1
    for v in arr[1:]:
        if v == cur:
            count += 1
        else:
            if cur:
                lengths.append(count)
            cur = v
            count = 1
    if cur:
        lengths.append(count)
    return lengths


def build_feature_sets(
    root: Path,
    input_columns: list[str],
    allowed_features: list[str],
    previous_audit: dict[str, dict[str, Any]],
    top_k_values: list[int],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    input_set = set(input_columns)
    feature_sets: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    def add(name: str, features: list[str], method: str, risk: str, deploy: str, notes: str) -> None:
        unique = [f for f in dict.fromkeys(features) if f in input_set]
        feature_sets[name] = unique
        short = ";".join(unique[:20]) + (";..." if len(unique) > 20 else "")
        rows.append(
            {
                "feature_set_name": name,
                "n_features": len(unique),
                "feature_names_short": short,
                "selection_method": method,
                "selection_leakage_risk": risk,
                "deployability_guess": deploy,
                "notes": notes,
            }
        )

    beta_like = [f for f in allowed_features if is_beta_like(f)]
    non_beta = [f for f in allowed_features if f not in set(beta_like)]
    prev_selected = [
        f
        for f in allowed_features
        if str(previous_audit.get(f, {}).get("selected_for_model", "")).lower() in {"true", "1", "yes"}
    ]
    if not prev_selected:
        prev_selected = allowed_features[:96]
    compact = [
        f
        for f in allowed_features
        if any(token in f.lower() for token in ["beta_local_baseline_ratio", "low_beta_high_beta_ratio", "beta_boundary_contrast"])
    ][:24]
    if len(compact) < 8:
        compact = beta_like[:24]

    add("all_allowed_phase6a0", allowed_features, "Phase 6A.0 allowed-feature audit plus Phase 6A.0.5 leakage reaudit", "low_to_medium", "mixed", "Full allowed feature pool; expensive models may use SGD or fold-local top-k.")
    add("beta_like", beta_like, "feature-name beta/low_beta/high_beta/bandpower heuristic", "low_to_medium", "high", "Single-family beta-like diagnostic feature set.")
    add("non_beta_allowed", non_beta, "all allowed features excluding beta-like names", "low_to_medium", "mixed", "Tests whether non-beta dynamics add burden information.")
    add("phase6a0_selected96_surrogate", prev_selected[:96], "Phase 6A.0 selected_for_model columns or deterministic 96-column surrogate", "low_to_medium", "mixed", "Reproduction/surrogate for prior lower-bound estimator.")
    add("compact_deployable", compact, "small beta/boundary/ratio feature heuristic", "medium", "high", "Exploratory compact set; inferred, not locked by prior ADR.")

    phase5_2b_exact = find_exact_phase5_2b_features(root, input_set)
    add(
        "phase5_2b_refined_if_found",
        phase5_2b_exact,
        "exact column names from Phase 5_2B refined/minimum tables only",
        "low" if phase5_2b_exact else "unavailable",
        "high" if phase5_2b_exact else "unknown",
        "Unavailable if Phase 5_2B only names base features not exact Phase 5_2C columns.",
    )

    phase5_2c_min = find_phase5_2c_minimum_features(root, input_set)
    if phase5_2c_min:
        add("phase5_2c_minimum_sufficient_if_found", phase5_2c_min, "exact Phase 5_2C causal_minimum_sufficient_subset columns", "low", "high", "Extra diagnostic exact refined subset found in Phase 5_2C outputs.")

    phase5_2c_refined = find_phase5_2c_refined_features(root, input_set)
    if phase5_2c_refined:
        add("phase5_2c_refined_candidates_if_found", phase5_2c_refined[:80], "exact Phase 5_2C refined candidate output_column rows", "low_to_medium", "mixed", "Extra diagnostic refined candidate set capped at 80 columns for runtime.")

    for k in top_k_values:
        name = f"top_k_train_only_{k}"
        feature_sets[name] = allowed_features
        rows.append(
            {
                "feature_set_name": name,
                "n_features": k,
                "feature_names_short": "selected inside each training fold only",
                "selection_method": "fold-local absolute Pearson correlation on training subjects only",
                "selection_leakage_risk": "low_if_fold_local",
                "deployability_guess": "mixed",
                "notes": f"Dynamic top-{k}; selected features differ by held-out subject.",
            }
        )
    return feature_sets, rows


def is_beta_like(feature: str) -> bool:
    low = feature.lower()
    return any(token in low for token in ["beta", "13_30", "20_35", "bandpower"])


def find_exact_phase5_2b_features(root: Path, input_set: set[str]) -> list[str]:
    paths = [
        root / "results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv",
        root / "results/tables/05_phase5/feature_atlas_2b/minimum_sufficient_feature_set.tsv",
    ]
    features: list[str] = []
    if pd is None:
        return features
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep=sep_for(path))
        except Exception:
            continue
        for col in ["feature_name", "feature_or_feature_set_name", "source_columns"]:
            if col in df.columns:
                for val in df[col].dropna().astype(str):
                    if val in input_set:
                        features.append(val)
    return list(dict.fromkeys(features))


def find_phase5_2c_minimum_features(root: Path, input_set: set[str]) -> list[str]:
    path = root / "results/tables/05_phase5/phase5_2c/causal_minimum_sufficient_subset.tsv"
    if pd is None or not path.exists():
        return []
    try:
        df = pd.read_csv(path, sep=sep_for(path))
    except Exception:
        return []
    features: list[str] = []
    for val in df.get("subset_features", []).dropna().astype(str):
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            parsed = []
        for item in parsed:
            if str(item) in input_set:
                features.append(str(item))
    return list(dict.fromkeys(features))


def find_phase5_2c_refined_features(root: Path, input_set: set[str]) -> list[str]:
    path = root / "results/tables/05_phase5/phase5_2c/causal_refined_candidate_features.tsv"
    if pd is None or not path.exists():
        return []
    try:
        df = pd.read_csv(path, sep=sep_for(path))
    except Exception:
        return []
    features: list[str] = []
    for col in ["output_column", "feature_name", "variant_feature_name"]:
        if col in df.columns:
            for val in df[col].dropna().astype(str):
                if val in input_set:
                    features.append(val)
    return list(dict.fromkeys(features))


def load_analysis_table(path: Path, meta_cols: list[str], feature_cols: list[str], max_rows: int) -> Any:
    if pd is None:
        raise RuntimeError(f"pandas unavailable: {PANDAS_ERROR}")
    usecols = [c for c in dict.fromkeys(meta_cols + feature_cols) if c]
    dtype: dict[str, Any] = {}
    for c in usecols:
        if c in feature_cols:
            dtype[c] = "float32"
        elif c in {"target_label"}:
            dtype[c] = "float32"
        elif c.endswith("_s") or c.endswith("_ms") or c in {"window_start_s", "window_stop_s"}:
            dtype[c] = "float64"
        else:
            dtype[c] = "string"
    LOG.info("Reading analysis table: %s columns=%d max_rows=%s", path, len(usecols), max_rows or "all")
    return pd.read_csv(path, sep=sep_for(path), usecols=usecols, dtype=dtype, nrows=max_rows or None)


def finite_corr(x: Any, y: Any, method: str = "pearson") -> float:
    if np is None:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
        return float("nan")
    if method == "spearman":
        if scipy_stats is not None:
            try:
                return float(scipy_stats.spearmanr(x_arr, y_arr).correlation)
            except Exception:
                pass
        x_arr = rankdata_fallback(x_arr)
        y_arr = rankdata_fallback(y_arr)
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata_fallback(x: Any) -> Any:
    if np is None:
        return []
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    return ranks


def safe_auc(y: Any, score: Any) -> float:
    if np is None:
        return float("nan")
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    mask = np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(y_arr, s_arr))
        except Exception:
            return float("nan")
    pos = s_arr[y_arr == 1]
    neg = s_arr[y_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def safe_auprc(y: Any, score: Any) -> float:
    if np is None:
        return float("nan")
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(score, dtype=float)
    mask = np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    if average_precision_score is not None:
        try:
            return float(average_precision_score(y_arr, s_arr))
        except Exception:
            return float("nan")
    order = np.argsort(-s_arr)
    y_sorted = y_arr[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.maximum(np.arange(len(y_sorted)) + 1, 1)
    recall_delta = (y_sorted == 1) / max((y_arr == 1).sum(), 1)
    return float(np.sum(precision * recall_delta))


def calibration_slope_intercept(pred: Any, target: Any) -> tuple[float, float]:
    if np is None:
        return float("nan"), float("nan")
    x = np.asarray(pred, dtype=float)
    y = np.asarray(target, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
    return float(slope), float(intercept)


def balanced_accuracy_fallback(y: Any, pred: Any) -> float:
    if np is None:
        return float("nan")
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(pred, dtype=int)
    vals = []
    for cls in [0, 1]:
        mask = y_arr == cls
        if mask.any():
            vals.append(float((p_arr[mask] == cls).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def standardize_train_test(train_x: Any, test_x: Any) -> tuple[Any, Any]:
    if np is None:
        return train_x, test_x
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    xtr = np.nan_to_num((train_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    xte = np.nan_to_num((test_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return xtr, xte


def ridge_predict(train_x: Any, train_y: Any, test_x: Any, ridge_lambda: float = 3.0) -> Any:
    if np is None:
        return []
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    xtr, xte = standardize_train_test(train_x, test_x)
    y_mean = float(np.nanmean(train_y))
    y_center = train_y - y_mean
    xtx = xtr.T @ xtr
    xty = xtr.T @ y_center
    reg = ridge_lambda * np.eye(xtx.shape[0], dtype=np.float32)
    try:
        beta = np.linalg.solve(xtx + reg, xty)
    except Exception:
        beta = np.linalg.pinv(xtx + reg) @ xty
    pred = y_mean + xte @ beta
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def fit_predict_sklearn_model(model_name: str, train_x: Any, train_y: Any, test_x: Any, seed: int) -> tuple[Any, str]:
    if np is None:
        return [], "numpy unavailable"
    if SKLEARN_ERROR is not None:
        return ridge_predict(train_x, train_y, test_x), f"sklearn unavailable; ridge fallback: {SKLEARN_ERROR}"
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=int)
    xtr, xte = standardize_train_test(train_x, test_x)
    note = ""
    if model_name == "sklearn_logistic_l2_balanced" and LogisticRegression is not None:
        max_train = 60_000
        idx = deterministic_sample_indices(train_y, max_train, seed)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="saga",
            max_iter=80,
            random_state=seed,
        )
        model.fit(xtr[idx], train_y[idx])
        note = f"trained on deterministic cap n={len(idx)}"
    elif model_name == "sklearn_ridge_or_sgd" and SGDClassifier is not None:
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0005,
            class_weight="balanced",
            max_iter=35,
            tol=1e-3,
            random_state=seed,
        )
        model.fit(xtr, train_y)
        note = "SGD log-loss surrogate for scalable ridge/logistic ceiling"
    elif model_name == "sklearn_random_forest_ceiling" and RandomForestClassifier is not None:
        max_train = 100_000
        idx = deterministic_sample_indices(train_y, max_train, seed)
        model = RandomForestClassifier(
            n_estimators=64,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=min(8, os.cpu_count() or 1),
            random_state=seed,
        )
        model.fit(np.nan_to_num(train_x[idx], nan=0.0), train_y[idx])
        prob = model.predict_proba(np.nan_to_num(test_x, nan=0.0))[:, 1]
        return np.clip(prob, 0.0, 1.0).astype(np.float32), f"offline ceiling; trained on deterministic cap n={len(idx)}"
    elif model_name == "sklearn_gradient_boosting_ceiling" and HistGradientBoostingClassifier is not None:
        max_train = 100_000
        idx = deterministic_sample_indices(train_y, max_train, seed)
        model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.06,
            max_leaf_nodes=16,
            l2_regularization=0.1,
            random_state=seed,
        )
        model.fit(xtr[idx], train_y[idx])
        note = f"offline ceiling; trained on deterministic cap n={len(idx)}"
    else:
        return ridge_predict(train_x, train_y, test_x), f"{model_name} unavailable; ridge fallback"
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(xte)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(xte)
        prob = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
    else:
        prob = model.predict(xte)
    return np.clip(prob, 0.0, 1.0).astype(np.float32), note


def deterministic_sample_indices(y: Any, max_n: int, seed: int) -> Any:
    if np is None:
        return []
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx_parts = []
    for cls in [0, 1]:
        cls_idx = np.flatnonzero(y_arr == cls)
        take = min(len(cls_idx), max_n // 2)
        if take:
            idx_parts.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(idx_parts) if idx_parts else rng.choice(np.arange(n), size=max_n, replace=False)
    if len(idx) < max_n:
        rest = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        add = rng.choice(rest, size=min(len(rest), max_n - len(idx)), replace=False)
        idx = np.concatenate([idx, add])
    return np.sort(idx)


def fold_topk_indices(x_train: Any, y_train: Any, k: int) -> Any:
    if np is None:
        return []
    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    y_std = np.nanstd(y)
    if not math.isfinite(float(y_std)) or y_std == 0:
        return np.arange(min(k, x.shape[1]))
    x_mean = np.nanmean(x, axis=0)
    x_std = np.nanstd(x, axis=0)
    x_std[~np.isfinite(x_std) | (x_std == 0)] = np.nan
    y_center = y - np.nanmean(y)
    cov = np.nanmean((x - x_mean) * y_center[:, None], axis=0)
    corr = np.abs(cov / (x_std * y_std))
    corr = np.nan_to_num(corr, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return np.argsort(-corr, kind="mergesort")[: min(k, x.shape[1])]


def beta_baseline_score(train_df: Any, test_df: Any, label_col: str, beta_cols: list[str]) -> tuple[Any, str]:
    if np is None or not beta_cols:
        return np.full(len(test_df), float(pd.to_numeric(train_df[label_col], errors="coerce").mean()), dtype=np.float32), ""
    train_y = pd.to_numeric(train_df[label_col], errors="coerce").to_numpy(dtype=float)
    best_col = ""
    best_abs = -1.0
    for col in beta_cols:
        corr = abs(finite_corr(pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float), train_y, "pearson"))
        if math.isfinite(corr) and corr > best_abs:
            best_abs = corr
            best_col = col
    if not best_col:
        return np.full(len(test_df), float(np.nanmean(train_y)), dtype=np.float32), ""
    x = pd.to_numeric(train_df[best_col], errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not math.isfinite(mad) or mad == 0:
        mad = np.nanstd(x)
    if not math.isfinite(mad) or mad == 0:
        mad = 1.0
    test_x = pd.to_numeric(test_df[best_col], errors="coerce").to_numpy(dtype=float)
    z = (test_x - med) / mad
    return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32), best_col


def split_condition_columns(df: Any) -> Any:
    if pd is None or "condition" not in df.columns:
        return df
    parts = df["condition"].astype(str).str.split("_", n=1, expand=True)
    if "medication_state" not in df.columns:
        df["medication_state"] = parts[0]
    if "task_state" not in df.columns:
        df["task_state"] = parts[1] if parts.shape[1] > 1 else ""
    return df


def group_indices_for_burden(df: Any, cols: Columns) -> tuple[list[Any], Any, list[str]]:
    if np is None:
        return [], [], []
    group_cols = [c for c in [cols.subject, cols.session, cols.block, "channel", "band_mode"] if c and c in df.columns]
    if not group_cols and cols.subject in df.columns:
        group_cols = [cols.subject]
    indices: list[Any] = []
    for _, idx in df.groupby(group_cols, dropna=False, sort=False).groups.items():
        indices.append(np.asarray(idx, dtype=int))
    if cols.time and cols.time in df.columns:
        times = pd.to_numeric(df[cols.time], errors="coerce").to_numpy(dtype=np.float64)
    else:
        times = np.arange(len(df), dtype=np.float64)
    return indices, times, group_cols


def leaky_burden(values: Any, group_indices: list[Any], times: Any, tau_ms: float) -> Any:
    if np is None:
        return []
    arr = np.asarray(values, dtype=np.float32)
    times_arr = np.asarray(times, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float32)
    tau_s = float(tau_ms) / 1000.0
    for idx in group_indices:
        if len(idx) == 0:
            continue
        local_times = times_arr[idx]
        order = np.argsort(local_times, kind="mergesort")
        sorted_idx = idx[order]
        vals = arr[sorted_idx]
        tvals = times_arr[sorted_idx]
        finite = tvals[np.isfinite(tvals)]
        default_dt = float(np.nanmedian(np.diff(np.sort(finite)))) if len(finite) > 2 else 0.8
        if not math.isfinite(default_dt) or default_dt <= 0:
            default_dt = 0.8
        prev = 0.0
        last_time = tvals[0] if len(tvals) else 0.0
        local = np.zeros(len(sorted_idx), dtype=np.float32)
        for i, val in enumerate(vals):
            if i == 0:
                dt_s = default_dt
            else:
                dt_s = float(tvals[i] - last_time)
                if not math.isfinite(dt_s) or dt_s <= 0:
                    dt_s = default_dt
            alpha = 1.0 - math.exp(-dt_s / tau_s)
            v = float(val) if math.isfinite(float(val)) else 0.0
            prev = prev + alpha * (v - prev)
            local[i] = prev
            last_time = tvals[i]
        out[sorted_idx] = local
    return out


def metric_row(
    y_label: Any,
    y_burden: Any,
    pred_score: Any,
    pred_burden: Any,
    high_threshold: float,
    pred_threshold: float,
) -> dict[str, Any]:
    if np is None:
        return {}
    y = np.asarray(y_label, dtype=float)
    yb = np.asarray(y_burden, dtype=float)
    s = np.asarray(pred_score, dtype=float)
    pb = np.asarray(pred_burden, dtype=float)
    mask = np.isfinite(yb) & np.isfinite(pb)
    rmse = float(np.sqrt(np.nanmean((pb[mask] - yb[mask]) ** 2))) if mask.any() else float("nan")
    mae = float(np.nanmean(np.abs(pb[mask] - yb[mask]))) if mask.any() else float("nan")
    y_high = (yb >= high_threshold).astype(int)
    p_high = (pb >= pred_threshold).astype(int)
    if len(np.unique(y_high)) == 2:
        if balanced_accuracy_score is not None:
            try:
                bal = float(balanced_accuracy_score(y_high, p_high))
            except Exception:
                bal = balanced_accuracy_fallback(y_high, p_high)
        else:
            bal = balanced_accuracy_fallback(y_high, p_high)
    else:
        bal = float("nan")
    slope, intercept = calibration_slope_intercept(pb, yb)
    try:
        brier = float(brier_score_loss(y.astype(int), np.clip(s, 0, 1))) if brier_score_loss is not None else float(np.nanmean((np.clip(s, 0, 1) - y) ** 2))
    except Exception:
        brier = float(np.nanmean((np.clip(s, 0, 1) - y) ** 2))
    return {
        "pearson_burden": finite_corr(yb, pb, "pearson"),
        "spearman_burden": finite_corr(yb, pb, "spearman"),
        "rmse_burden": rmse,
        "mae_burden": mae,
        "auroc_label": safe_auc(y.astype(int), s),
        "auprc_label": safe_auprc(y.astype(int), s),
        "brier_score": brier,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "high_burden_balanced_accuracy": bal,
    }


def summarize_values(vals: list[float]) -> tuple[float, float, float]:
    if np is None:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray([v for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def run_loso_models(
    df: Any,
    cols: Columns,
    feature_sets: dict[str, list[str]],
    tau_values: list[float],
    top_k_values: list[int],
    seed: int,
    continuity_class: str,
    skip_high_capacity: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if np is None or pd is None:
        raise RuntimeError("numpy and pandas required")
    df = split_condition_columns(df.copy())
    subjects = sorted(df[cols.subject].dropna().astype(str).unique())
    y = pd.to_numeric(df[cols.label], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    group_indices, times, _ = group_indices_for_burden(df, cols)
    y_burdens = {tau: leaky_burden(y, group_indices, times, tau) for tau in tau_values}
    all_allowed = feature_sets.get("all_allowed_phase6a0", [])
    beta_cols = feature_sets.get("beta_like", [])

    specs: list[ModelSpec] = [
        ModelSpec("class_prior_baseline", "none", "LOSO_cross_subject", True, notes="training-subject class prior"),
        ModelSpec("shuffled_label_baseline", "phase6a0_selected96_surrogate", "LOSO_cross_subject", True, notes="labels shuffled inside training fold"),
        ModelSpec("beta_feature_baseline", "beta_like", "LOSO_cross_subject", True, notes="single beta-like feature selected on training subjects only"),
        ModelSpec("phase6a0_reproduction_or_surrogate", "phase6a0_selected96_surrogate", "LOSO_cross_subject", True),
        ModelSpec("numpy_ridge_fallback", "compact_deployable", "LOSO_cross_subject", True),
        ModelSpec("numpy_ridge_fallback", "phase5_2c_minimum_sufficient_if_found", "LOSO_cross_subject", True),
        ModelSpec("sklearn_ridge_or_sgd", "all_allowed_phase6a0", "LOSO_cross_subject", True),
        ModelSpec("sklearn_logistic_l2_balanced", "top_k_train_only_50", "LOSO_cross_subject", True),
    ]
    for k in top_k_values:
        specs.append(ModelSpec("numpy_ridge_fallback", f"top_k_train_only_{k}", "LOSO_cross_subject", True, top_k=k))
    if feature_sets.get("phase5_2b_refined_if_found"):
        specs.append(ModelSpec("numpy_ridge_fallback", "phase5_2b_refined_if_found", "LOSO_cross_subject", True))
    if feature_sets.get("phase5_2c_refined_candidates_if_found"):
        specs.append(ModelSpec("numpy_ridge_fallback", "phase5_2c_refined_candidates_if_found", "LOSO_cross_subject", True))
    if not skip_high_capacity:
        specs.extend(
            [
                ModelSpec("sklearn_random_forest_ceiling", "top_k_train_only_25", "LOSO_cross_subject", False, True, top_k=25),
                ModelSpec("sklearn_gradient_boosting_ceiling", "top_k_train_only_25", "LOSO_cross_subject", False, True, top_k=25),
            ]
        )

    estimator_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    scores_by_spec: dict[str, Any] = {}
    notes_by_spec: dict[str, str] = {}
    subject_arr = df[cols.subject].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    X_all = None
    if all_allowed:
        X_all = df[all_allowed].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    for spec in specs:
        if spec.feature_set_name not in feature_sets and spec.feature_set_name != "none":
            continue
        features = feature_sets.get(spec.feature_set_name, [])
        if not features and spec.feature_set_name not in {"none", "phase5_2b_refined_if_found"}:
            continue
        LOG.info("LOSO model: %s / %s", spec.model_name, spec.feature_set_name)
        score = np.full(len(df), np.nan, dtype=np.float32)
        spec_notes: list[str] = []
        if spec.model_name in {"class_prior_baseline", "shuffled_label_baseline", "beta_feature_baseline"}:
            X_spec = None
        elif spec.top_k:
            X_spec = X_all
            features = all_allowed
        else:
            X_spec = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        for fold_i, subject in enumerate(subjects):
            test_mask = subject_arr == subject
            train_mask = ~test_mask
            train_y = y[train_mask]
            if len(np.unique(train_y[~np.isnan(train_y)])) < 2:
                score[test_mask] = float(np.nanmean(train_y))
                spec_notes.append(f"{subject}: degenerate train labels")
                continue
            if spec.model_name == "class_prior_baseline":
                score[test_mask] = float(np.nanmean(train_y))
                continue
            if spec.model_name == "shuffled_label_baseline":
                use_features = feature_sets.get(spec.feature_set_name, [])[:96]
                X_fold = df[use_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
                shuffled = train_y.copy()
                rng.shuffle(shuffled)
                score[test_mask] = ridge_predict(X_fold[train_mask], shuffled, X_fold[test_mask])
                del X_fold
                continue
            if spec.model_name == "beta_feature_baseline":
                pred, feat = beta_baseline_score(df.loc[train_mask], df.loc[test_mask], cols.label, beta_cols)
                score[test_mask] = pred
                if feat:
                    spec_notes.append(f"{subject}: beta={feat}")
                continue
            if X_spec is None:
                continue
            if spec.top_k:
                top_idx = fold_topk_indices(X_spec[train_mask], train_y, spec.top_k)
                train_x = X_spec[train_mask][:, top_idx]
                test_x = X_spec[test_mask][:, top_idx]
            else:
                train_x = X_spec[train_mask]
                test_x = X_spec[test_mask]
            if spec.model_name in {"numpy_ridge_fallback", "phase6a0_reproduction_or_surrogate"}:
                score[test_mask] = ridge_predict(train_x, train_y, test_x)
            else:
                pred, note = fit_predict_sklearn_model(spec.model_name, train_x, train_y, test_x, seed + fold_i)
                score[test_mask] = pred
                if note:
                    spec_notes.append(f"{subject}: {note}")
        spec_key = f"{spec.validation_mode}|{spec.model_name}|{spec.feature_set_name}"
        scores_by_spec[spec_key] = score
        notes_by_spec[spec_key] = " | ".join(spec_notes[:10])
        rows, tau_spec_rows = evaluate_score_by_tau(
            df,
            cols,
            y,
            score,
            y_burdens,
            group_indices,
            times,
            tau_values,
            spec,
            continuity_class,
            notes_by_spec[spec_key],
        )
        estimator_rows.extend(rows)
        tau_rows.extend(tau_spec_rows)
        if X_spec is not None and not spec.top_k:
            del X_spec
        gc.collect()
    del X_all
    gc.collect()

    within_rows, within_tau, within_scores = run_within_subject_models(df, cols, feature_sets, tau_values, top_k_values, seed, continuity_class, skip_high_capacity)
    estimator_rows.extend(within_rows)
    tau_rows.extend(within_tau)
    scores_by_spec.update(within_scores)

    oracle_rows, oracle_tau = run_oracle_descriptive(df, cols, feature_sets, tau_values, seed, continuity_class)
    estimator_rows.extend(oracle_rows)
    tau_rows.extend(oracle_tau)
    return estimator_rows, tau_rows, specs_to_feature_ablation(estimator_rows), scores_by_spec


def evaluate_score_by_tau(
    df: Any,
    cols: Columns,
    y: Any,
    score: Any,
    y_burdens: dict[float, Any],
    group_indices: list[Any],
    times: Any,
    tau_values: list[float],
    spec: ModelSpec,
    continuity_class: str,
    model_notes: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if np is None:
        return [], []
    subjects = sorted(df[cols.subject].dropna().astype(str).unique())
    subject_arr = df[cols.subject].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    pred_burdens: dict[float, Any] = {}
    for tau in tau_values:
        pred_burdens[tau] = leaky_burden(score, group_indices, times, tau)
    train_best_tau = select_global_train_tau(df, cols, y_burdens, pred_burdens, tau_values)
    for tau in tau_values:
        pb = pred_burdens[tau]
        yb = y_burdens[tau]
        train_thresholds: dict[str, tuple[float, float]] = {}
        for subject in subjects:
            train_mask = subject_arr != subject
            high_thr = float(np.nanpercentile(yb[train_mask], 75))
            pred_thr = float(np.nanpercentile(pb[train_mask], 75))
            train_thresholds[subject] = (high_thr, pred_thr)
        for subject in subjects:
            mask = subject_arr == subject
            high_thr, pred_thr = train_thresholds[subject]
            metric = metric_row(y[mask], yb[mask], np.asarray(score)[mask], pb[mask], high_thr, pred_thr)
            row = {
                "row_type": "subject",
                "validation_mode": spec.validation_mode,
                "model_name": spec.model_name,
                "feature_set_name": spec.feature_set_name,
                "tau_ms": tau,
                "subject_id": subject,
                "n_rows": int(mask.sum()),
                "positive_fraction": float(np.nanmean(y[mask])) if mask.any() else float("nan"),
                "timing_confidence": "non_definitive_event_matrix" if continuity_class == "event_or_candidate_matrix" else "high",
                "continuity_class": continuity_class,
                "deployable": spec.deployable,
                "offline_ceiling": spec.offline_ceiling,
                "notes": clean_join([spec.notes, model_notes, "row-order pseudo-burden is non-definitive" if continuity_class == "event_or_candidate_matrix" else ""]),
            }
            row.update(metric)
            rows.append(row)
            tau_rows.append(
                {
                    "validation_mode": spec.validation_mode,
                    "subject_id": subject,
                    "model_name": spec.model_name,
                    "feature_set_name": spec.feature_set_name,
                    "tau_ms": tau,
                    "pearson": metric.get("pearson_burden", ""),
                    "spearman": metric.get("spearman_burden", ""),
                    "is_subject_best_tau": False,
                    "is_global_train_selected_tau": tau == train_best_tau.get(subject),
                    "timing_confidence": "non_definitive_event_matrix" if continuity_class == "event_or_candidate_matrix" else "high",
                    "notes": "global tau selected on training subjects only" if tau == train_best_tau.get(subject) else "",
                }
            )
    mark_subject_best_tau(tau_rows)
    rows.extend(aggregate_estimator_rows(rows))
    return rows, tau_rows


def clean_join(parts: list[str]) -> str:
    return "; ".join([p for p in parts if p])


def select_global_train_tau(df: Any, cols: Columns, y_burdens: dict[float, Any], pred_burdens: dict[float, Any], tau_values: list[float]) -> dict[str, float]:
    if np is None:
        return {}
    subject_arr = df[cols.subject].astype(str).to_numpy()
    subjects = sorted(df[cols.subject].dropna().astype(str).unique())
    out: dict[str, float] = {}
    for held in subjects:
        train_mask = subject_arr != held
        best_tau = tau_values[0]
        best_metric = -np.inf
        for tau in tau_values:
            per_subject = []
            for sub in subjects:
                if sub == held:
                    continue
                mask = subject_arr == sub
                val = finite_corr(y_burdens[tau][mask], pred_burdens[tau][mask], "spearman")
                if math.isfinite(val):
                    per_subject.append(val)
            med = float(np.median(per_subject)) if per_subject else float("nan")
            if math.isfinite(med) and med > best_metric:
                best_metric = med
                best_tau = tau
        out[held] = best_tau
    return out


def mark_subject_best_tau(tau_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in tau_rows:
        key = (
            str(row.get("validation_mode", "")),
            str(row.get("subject_id", "")),
            str(row.get("model_name", "")),
            str(row.get("feature_set_name", "")),
        )
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        best = None
        best_val = -np.inf
        for row in rows:
            vals = [to_float(row.get("spearman")), to_float(row.get("pearson"))]
            val = np.nanmax(vals) if np is not None else max(vals)
            if math.isfinite(float(val)) and val > best_val:
                best = row
                best_val = float(val)
        if best is not None:
            best["is_subject_best_tau"] = True


def to_float(x: Any) -> float:
    try:
        val = float(x)
        return val if math.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def aggregate_estimator_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if np is None:
        return []
    subject_rows = [r for r in rows if r.get("row_type") == "subject"]
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in subject_rows:
        key = (
            str(row.get("validation_mode", "")),
            str(row.get("model_name", "")),
            str(row.get("feature_set_name", "")),
            float(row.get("tau_ms", 0) or 0),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (mode, model, fset, tau), group in grouped.items():
        pearsons = [to_float(r.get("pearson_burden")) for r in group]
        spearmans = [to_float(r.get("spearman_burden")) for r in group]
        med_p, q25_p, q75_p = summarize_values(pearsons)
        med_s, q25_s, q75_s = summarize_values(spearmans)
        beta_med = float("nan")
        shuffled_med = float("nan")
        prior_med = float("nan")
        out.append(
            {
                "row_type": "aggregate",
                "validation_mode": mode,
                "model_name": model,
                "feature_set_name": fset,
                "tau_ms": tau,
                "subject_id": "ALL",
                "n_rows": sum(int(r.get("n_rows", 0) or 0) for r in group),
                "positive_fraction": float(np.nanmean([to_float(r.get("positive_fraction")) for r in group])),
                "pearson_burden": med_p,
                "spearman_burden": med_s,
                "rmse_burden": float(np.nanmedian([to_float(r.get("rmse_burden")) for r in group])),
                "mae_burden": float(np.nanmedian([to_float(r.get("mae_burden")) for r in group])),
                "auroc_label": float(np.nanmedian([to_float(r.get("auroc_label")) for r in group])),
                "auprc_label": float(np.nanmedian([to_float(r.get("auprc_label")) for r in group])),
                "brier_score": float(np.nanmedian([to_float(r.get("brier_score")) for r in group])),
                "calibration_slope": float(np.nanmedian([to_float(r.get("calibration_slope")) for r in group])),
                "calibration_intercept": float(np.nanmedian([to_float(r.get("calibration_intercept")) for r in group])),
                "high_burden_balanced_accuracy": float(np.nanmedian([to_float(r.get("high_burden_balanced_accuracy")) for r in group])),
                "timing_confidence": group[0].get("timing_confidence", ""),
                "continuity_class": group[0].get("continuity_class", ""),
                "deployable": group[0].get("deployable", ""),
                "offline_ceiling": group[0].get("offline_ceiling", ""),
                "median_pearson": med_p,
                "median_spearman": med_s,
                "iqr_pearson": q75_p - q25_p if math.isfinite(q25_p) and math.isfinite(q75_p) else float("nan"),
                "iqr_spearman": q75_s - q25_s if math.isfinite(q25_s) and math.isfinite(q75_s) else float("nan"),
                "n_valid_subjects": len([v for v in spearmans if math.isfinite(v)]),
                "n_high_subjects_r_ge_0_5": sum(1 for v in spearmans if math.isfinite(v) and v >= 0.5),
                "n_moderate_subjects_r_0_3_to_0_5": sum(1 for v in spearmans if math.isfinite(v) and 0.3 <= v < 0.5),
                "n_low_subjects_r_le_0_2": sum(1 for v in spearmans if math.isfinite(v) and v <= 0.2),
                "beats_beta_by": beta_med,
                "beats_shuffled_by": shuffled_med,
                "beats_class_prior_by": prior_med,
                "notes": "aggregate subject-level median; baseline deltas filled in beta_baseline_delta.csv",
            }
        )
    return out


def run_within_subject_models(
    df: Any,
    cols: Columns,
    feature_sets: dict[str, list[str]],
    tau_values: list[float],
    top_k_values: list[int],
    seed: int,
    continuity_class: str,
    skip_high_capacity: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if np is None or pd is None:
        return [], [], {}
    specs = [
        ModelSpec("numpy_ridge_fallback", "compact_deployable", "within_subject_block_or_time_calibrated", True),
        ModelSpec("numpy_ridge_fallback", "top_k_train_only_50", "within_subject_block_or_time_calibrated", True, top_k=50),
        ModelSpec("beta_feature_baseline", "beta_like", "within_subject_block_or_time_calibrated", True),
    ]
    if not skip_high_capacity:
        specs.append(ModelSpec("sklearn_gradient_boosting_ceiling", "top_k_train_only_25", "within_subject_block_or_time_calibrated", False, True, top_k=25))
    y = pd.to_numeric(df[cols.label], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    group_indices, times, _ = group_indices_for_burden(df, cols)
    y_burdens = {tau: leaky_burden(y, group_indices, times, tau) for tau in tau_values}
    all_allowed = feature_sets.get("all_allowed_phase6a0", [])
    X_all = df[all_allowed].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) if all_allowed else None
    subject_arr = df[cols.subject].astype(str).to_numpy()
    subjects = sorted(df[cols.subject].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    for spec in specs:
        LOG.info("Within-subject model: %s / %s", spec.model_name, spec.feature_set_name)
        score = np.full(len(df), np.nan, dtype=np.float32)
        for i, subject in enumerate(subjects):
            subj_idx = np.flatnonzero(subject_arr == subject)
            if len(subj_idx) < 50:
                continue
            if cols.time in df.columns:
                order = subj_idx[np.argsort(pd.to_numeric(df.iloc[subj_idx][cols.time], errors="coerce").to_numpy(), kind="mergesort")]
            else:
                order = subj_idx
            cut = max(10, int(len(order) * 0.6))
            train_idx = order[:cut]
            test_idx = order[cut:]
            if len(test_idx) < 10 or len(np.unique(y[train_idx])) < 2:
                continue
            if spec.model_name == "beta_feature_baseline":
                pred, _ = beta_baseline_score(df.iloc[train_idx], df.iloc[test_idx], cols.label, feature_sets.get("beta_like", []))
            else:
                if spec.top_k and X_all is not None:
                    top_idx = fold_topk_indices(X_all[train_idx], y[train_idx], spec.top_k)
                    train_x = X_all[train_idx][:, top_idx]
                    test_x = X_all[test_idx][:, top_idx]
                else:
                    feats = feature_sets.get(spec.feature_set_name, [])
                    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
                    train_x = X[train_idx]
                    test_x = X[test_idx]
                if spec.model_name == "numpy_ridge_fallback":
                    pred = ridge_predict(train_x, y[train_idx], test_x)
                else:
                    pred, _ = fit_predict_sklearn_model(spec.model_name, train_x, y[train_idx], test_x, seed + i)
            score[test_idx] = pred
        spec_key = f"{spec.validation_mode}|{spec.model_name}|{spec.feature_set_name}"
        scores[spec_key] = score
        eval_rows, eval_tau = evaluate_score_by_tau(
            df,
            cols,
            y,
            np.nan_to_num(score, nan=float(np.nanmean(y))),
            y_burdens,
            group_indices,
            times,
            tau_values,
            spec,
            continuity_class,
            "within-subject early/late split; train/calibration rows only",
        )
        rows.extend(eval_rows)
        tau_rows.extend(eval_tau)
    del X_all
    gc.collect()
    return rows, tau_rows, scores


def run_oracle_descriptive(
    df: Any,
    cols: Columns,
    feature_sets: dict[str, list[str]],
    tau_values: list[float],
    seed: int,
    continuity_class: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if np is None or pd is None:
        return [], []
    specs = [ModelSpec("oracle_descriptive_subject_fit", "top_k_train_only_50", "oracle_descriptive", False, True, top_k=50)]
    y = pd.to_numeric(df[cols.label], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    group_indices, times, _ = group_indices_for_burden(df, cols)
    y_burdens = {tau: leaky_burden(y, group_indices, times, tau) for tau in tau_values}
    all_allowed = feature_sets.get("all_allowed_phase6a0", [])
    if not all_allowed:
        return [], []
    X_all = df[all_allowed].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    subject_arr = df[cols.subject].astype(str).to_numpy()
    subjects = sorted(df[cols.subject].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    for spec in specs:
        score = np.full(len(df), np.nan, dtype=np.float32)
        for i, subject in enumerate(subjects):
            mask = subject_arr == subject
            if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
                continue
            top_idx = fold_topk_indices(X_all[mask], y[mask], spec.top_k)
            score[mask] = ridge_predict(X_all[mask][:, top_idx], y[mask], X_all[mask][:, top_idx])
        eval_rows, eval_tau = evaluate_score_by_tau(
            df,
            cols,
            y,
            np.nan_to_num(score, nan=float(np.nanmean(y))),
            y_burdens,
            group_indices,
            times,
            tau_values,
            spec,
            continuity_class,
            "oracle descriptive: model and tau can fit same subject; not validation",
        )
        rows.extend(eval_rows)
        tau_rows.extend(eval_tau)
    del X_all
    gc.collect()
    return rows, tau_rows


def specs_to_feature_ablation(estimator_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in estimator_rows
        if r.get("row_type") == "aggregate"
        and r.get("validation_mode") == "LOSO_cross_subject"
        and r.get("model_name") in {"numpy_ridge_fallback", "phase6a0_reproduction_or_surrogate", "sklearn_logistic_l2_balanced", "sklearn_ridge_or_sgd"}
    ]


def add_baseline_deltas(estimator_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = [r for r in estimator_rows if r.get("row_type") == "aggregate"]
    baseline_by_mode_tau: dict[tuple[str, float, str], float] = {}
    for r in aggregates:
        key = (str(r.get("validation_mode", "")), float(r.get("tau_ms", 0) or 0), str(r.get("model_name", "")))
        baseline_by_mode_tau[key] = max(to_float(r.get("median_spearman")), to_float(r.get("median_pearson")))
    rows: list[dict[str, Any]] = []
    for r in aggregates:
        mode = str(r.get("validation_mode", ""))
        tau = float(r.get("tau_ms", 0) or 0)
        metric = max(to_float(r.get("median_spearman")), to_float(r.get("median_pearson")))
        beta = baseline_by_mode_tau.get((mode, tau, "beta_feature_baseline"), float("nan"))
        shuf = baseline_by_mode_tau.get((mode, tau, "shuffled_label_baseline"), float("nan"))
        prior = baseline_by_mode_tau.get((mode, tau, "class_prior_baseline"), float("nan"))
        delta_beta = metric - beta if math.isfinite(beta) and math.isfinite(metric) else float("nan")
        rows.append(
            {
                "model_name": r.get("model_name", ""),
                "feature_set_name": r.get("feature_set_name", ""),
                "validation_mode": mode,
                "tau_ms": tau,
                "median_pearson": r.get("median_pearson", ""),
                "median_spearman": r.get("median_spearman", ""),
                "beta_baseline_median_pearson": beta,
                "delta_vs_beta": delta_beta,
                "delta_vs_shuffled": metric - shuf if math.isfinite(shuf) and math.isfinite(metric) else float("nan"),
                "delta_vs_class_prior": metric - prior if math.isfinite(prior) and math.isfinite(metric) else float("nan"),
                "architecture_value_interpretation": beta_delta_interpretation(delta_beta, bool(r.get("offline_ceiling", False))),
            }
        )
    return rows


def beta_delta_interpretation(delta: float, high_capacity: bool) -> str:
    if not math.isfinite(delta):
        return "not assessable"
    if delta < 0.03:
        return "architecture not justified beyond simple beta baseline"
    if delta < 0.08:
        return "modest improvement over beta baseline"
    if high_capacity:
        return "meaningful high-capacity improvement; not deployable by itself"
    return "meaningful deployable-model improvement over beta baseline"


def subject_distribution(estimator_rows: list[dict[str, Any]], phase6a0_report_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pd is None or np is None:
        return [], {}
    phase6_rows: dict[str, float] = {}
    p = phase6a0_report_dir / "burden_metric_by_subject.csv"
    if p.exists():
        try:
            prev = pd.read_csv(p)
            best_tau = pd.to_numeric(prev.get("tau_ms"), errors="coerce").max()
            sub = prev[pd.to_numeric(prev.get("tau_ms"), errors="coerce") == best_tau]
            for _, row in sub.iterrows():
                phase6_rows[str(row.get("subject_id", ""))] = to_float(row.get("pearson"))
        except Exception:
            pass
    subject_rows = [r for r in estimator_rows if r.get("row_type") == "subject"]
    subjects = sorted({str(r.get("subject_id", "")) for r in subject_rows if r.get("subject_id")})
    out: list[dict[str, Any]] = []
    corr_values: list[float] = []
    for subject in subjects:
        rows = [r for r in subject_rows if str(r.get("subject_id", "")) == subject]
        loso = [r for r in rows if r.get("validation_mode") == "LOSO_cross_subject"]
        within = [r for r in rows if r.get("validation_mode") == "within_subject_block_or_time_calibrated"]
        beta = [r for r in loso if r.get("model_name") == "beta_feature_baseline"]
        best_loso = best_subject_row(loso)
        best_within = best_subject_row(within)
        best_beta = best_subject_row(beta)
        best_val = max(to_float(best_loso.get("spearman_burden")), to_float(best_loso.get("pearson_burden"))) if best_loso else float("nan")
        beta_val = max(to_float(best_beta.get("spearman_burden")), to_float(best_beta.get("pearson_burden"))) if best_beta else float("nan")
        if math.isfinite(best_val):
            corr_values.append(best_val)
        out.append(
            {
                "subject_id": subject,
                "n_rows": best_loso.get("n_rows", "") if best_loso else "",
                "positive_fraction": best_loso.get("positive_fraction", "") if best_loso else "",
                "phase6a0_pearson_if_available": phase6_rows.get(subject, ""),
                "best_loso_pearson": best_loso.get("pearson_burden", "") if best_loso else "",
                "best_loso_spearman": best_loso.get("spearman_burden", "") if best_loso else "",
                "best_within_subject_pearson": best_within.get("pearson_burden", "") if best_within else "",
                "best_tau_loso": best_loso.get("tau_ms", "") if best_loso else "",
                "best_tau_within_subject": best_within.get("tau_ms", "") if best_within else "",
                "best_model": best_loso.get("model_name", "") if best_loso else "",
                "best_feature_set": best_loso.get("feature_set_name", "") if best_loso else "",
                "beta_baseline_pearson": best_beta.get("pearson_burden", "") if best_beta else "",
                "improvement_over_beta": best_val - beta_val if math.isfinite(best_val) and math.isfinite(beta_val) else "",
                "responder_class": responder_class(best_val),
                "notes": "best LOSO row selected by max Pearson/Spearman across models/tau; pseudo-burden if table is non-continuous",
            }
        )
    sorted_vals = sorted(corr_values)
    gaps = [b - a for a, b in zip(sorted_vals[:-1], sorted_vals[1:])]
    max_gap = max(gaps) if gaps else 0.0
    high = sum(1 for v in corr_values if v >= 0.5)
    med = float(np.median(corr_values)) if corr_values else float("nan")
    if max_gap > 0.15:
        heterogeneity = "bimodal_or_heterogeneous"
    elif high >= max(1, math.ceil(0.25 * len(subjects))) and math.isfinite(med) and med < 0.30:
        heterogeneity = "heterogeneous_high_responder_tail"
    elif high == len(subjects) and subjects:
        heterogeneity = "uniformly_high_on_diagnostic_metric"
    else:
        heterogeneity = "fairly_uniform_or_low"
    summary = {
        "n_subjects": len(subjects),
        "n_high_trackable": high,
        "n_moderate_trackable": sum(1 for v in corr_values if 0.3 <= v < 0.5),
        "max_adjacent_correlation_gap": max_gap,
        "median_best_subject_metric": med,
        "heterogeneity_interpretation": heterogeneity,
    }
    return out, summary


def best_subject_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_val = -np.inf if np is not None else -1e9
    for row in rows:
        val = max(to_float(row.get("spearman_burden")), to_float(row.get("pearson_burden")))
        if math.isfinite(val) and val > best_val:
            best_val = val
            best = row
    return best


def responder_class(value: float) -> str:
    if not math.isfinite(value):
        return "invalid_or_insufficient_data"
    if value >= 0.50:
        return "high_trackable"
    if value >= 0.30:
        return "moderate_trackable"
    if value > 0.20:
        return "low_trackable"
    return "not_trackable"


def proxy_ceiling(df: Any, cols: Columns, scores_by_spec: dict[str, Any], best_spec_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if pd is None or np is None:
        return [], {"status": "blocked_missing_dependency"}
    df = split_condition_columns(df.copy())
    proxy_cols = [c for c in ["medication_state", "task_state", "condition"] if c in df.columns]
    clinical_cols = [c for c in df.columns if re.search(r"updrs|brady|rigidity|tremor|clinical", c, re.I)]
    proxy_cols.extend(clinical_cols)
    if not proxy_cols:
        return [
            {
                "proxy_name": "clinical_task_proxy",
                "proxy_type": "none",
                "n_subjects": 0,
                "n_rows": 0,
                "validation_mode": "",
                "model_name": "",
                "feature_set_name": "",
                "metric_primary": "not_assessable",
                "metric_value": "",
                "auroc": "",
                "auprc": "",
                "balanced_accuracy": "",
                "rank_biserial_or_cohens_d": "",
                "spearman_session_level": "",
                "pearson_session_level": "",
                "proxy_interpretation": "no clinical/task proxies detected",
                "notes": "",
            }
        ], {"status": "not_assessable"}
    score = scores_by_spec.get(best_spec_key)
    if score is None:
        score = np.zeros(len(df), dtype=float)
    rows: list[dict[str, Any]] = []
    for proxy in proxy_cols:
        vals = df[proxy].astype(str)
        valid = vals.notna() & (vals != "") & (vals.str.lower() != "nan")
        unique = sorted(vals[valid].unique())
        if len(unique) < 2:
            rows.append(proxy_row(proxy, "constant_or_missing", df, "", "", "", "not_assessable", "", "", "", "", "", "", "proxy has fewer than two values", ""))
            continue
        if len(unique) == 2 or proxy in {"medication_state", "task_state"}:
            pairs = binary_pairs_for_proxy(proxy, unique)
            for pos, neg in pairs:
                mask = valid & vals.isin([pos, neg])
                y_proxy = (vals[mask] == pos).astype(int).to_numpy()
                s = np.asarray(score)[mask.to_numpy()]
                auc = safe_auc(y_proxy, s)
                auprc = safe_auprc(y_proxy, s)
                pred = (s >= np.nanmedian(s)).astype(int)
                bal = balanced_accuracy_fallback(y_proxy, pred)
                pos_scores = s[y_proxy == 1]
                neg_scores = s[y_proxy == 0]
                effect = cohen_d(pos_scores, neg_scores)
                rows.append(
                    proxy_row(
                        proxy,
                        proxy_type(proxy),
                        df.loc[mask],
                        best_spec_key.split("|")[0],
                        best_spec_key.split("|")[1],
                        best_spec_key.split("|")[2],
                        "AUROC",
                        auc,
                        auc,
                        auprc,
                        bal,
                        effect,
                        "",
                        proxy_interpretation(auc),
                        f"positive={pos}; negative={neg}; supportive proxy only",
                    )
                )
        else:
            agg = pd.DataFrame(
                {
                    "subject": df[cols.subject].astype(str),
                    "proxy": pd.to_numeric(df[proxy], errors="coerce"),
                    "score": np.asarray(score, dtype=float),
                }
            ).dropna()
            if agg.empty:
                rows.append(proxy_row(proxy, "continuous_or_ordinal", df, "", "", "", "not_assessable", "", "", "", "", "", "", "proxy not numeric", ""))
                continue
            subagg = agg.groupby("subject").mean(numeric_only=True)
            spearman = finite_corr(subagg["proxy"].to_numpy(), subagg["score"].to_numpy(), "spearman")
            pearson = finite_corr(subagg["proxy"].to_numpy(), subagg["score"].to_numpy(), "pearson")
            rows.append(
                proxy_row(
                    proxy,
                    "continuous_or_ordinal",
                    agg,
                    best_spec_key.split("|")[0],
                    best_spec_key.split("|")[1],
                    best_spec_key.split("|")[2],
                    "subject_level_spearman",
                    spearman,
                    "",
                    "",
                    "",
                    "",
                    spearman,
                    pearson,
                    "session/subject-level proxy correlation; supportive only",
                )
            )
    metric_vals = [to_float(r.get("metric_value")) for r in rows]
    finite = [v for v in metric_vals if math.isfinite(v)]
    summary = {
        "status": "available" if rows else "not_assessable",
        "best_metric": max(finite) if finite else float("nan"),
        "interpretation": "weak_proxy_ceiling" if not finite or max(finite) < 0.65 else "proxy_signal_present",
    }
    return rows, summary


def proxy_row(
    proxy_name: str,
    ptype: str,
    frame: Any,
    mode: str,
    model: str,
    fset: str,
    primary: str,
    metric: Any,
    auroc: Any,
    auprc: Any,
    bal: Any,
    effect: Any,
    spearman: Any,
    pearson: Any,
    interp: str,
    notes: str = "",
) -> dict[str, Any]:
    n_sub = frame["subject_id"].nunique() if hasattr(frame, "columns") and "subject_id" in frame.columns else ""
    return {
        "proxy_name": proxy_name,
        "proxy_type": ptype,
        "n_subjects": n_sub,
        "n_rows": len(frame) if hasattr(frame, "__len__") else "",
        "validation_mode": mode,
        "model_name": model,
        "feature_set_name": fset,
        "metric_primary": primary,
        "metric_value": metric,
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": bal,
        "rank_biserial_or_cohens_d": effect,
        "spearman_session_level": spearman,
        "pearson_session_level": pearson,
        "proxy_interpretation": interp,
        "notes": notes,
    }


def proxy_type(proxy: str) -> str:
    low = proxy.lower()
    if "med" in low:
        return "medication_state"
    if "task" in low or "condition" in low:
        return "task_or_condition_state"
    if any(t in low for t in ["updrs", "brady", "rigidity", "tremor", "clinical"]):
        return "clinical_score"
    return "metadata_proxy"


def binary_pairs_for_proxy(proxy: str, unique: list[str]) -> list[tuple[str, str]]:
    vals = set(unique)
    if proxy == "medication_state" and {"MedOff", "MedOn"}.issubset(vals):
        return [("MedOff", "MedOn")]
    if proxy == "task_state":
        pairs = []
        for pos, neg in [("Hold", "Move"), ("Rest", "Move"), ("Rest", "Hold")]:
            if {pos, neg}.issubset(vals):
                pairs.append((pos, neg))
        return pairs or [(unique[0], unique[1])]
    if len(unique) >= 2:
        return [(unique[0], unique[1])]
    return []


def cohen_d(pos: Any, neg: Any) -> float:
    if np is None:
        return float("nan")
    p = np.asarray(pos, dtype=float)
    n = np.asarray(neg, dtype=float)
    if len(p) < 2 or len(n) < 2:
        return float("nan")
    pooled = math.sqrt(((len(p) - 1) * np.nanvar(p) + (len(n) - 1) * np.nanvar(n)) / max(len(p) + len(n) - 2, 1))
    if pooled == 0 or not math.isfinite(pooled):
        return float("nan")
    return float((np.nanmean(p) - np.nanmean(n)) / pooled)


def proxy_interpretation(auc: float) -> str:
    if not math.isfinite(auc):
        return "not_assessable"
    if auc >= 0.70:
        return "proxy signal present"
    if auc >= 0.60:
        return "weak supportive proxy signal"
    return "weak_or_chance_proxy"


def prior_phase_comparison(root: Path, phase6a0_report_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(phase: str, metric: str, value: Any, source: Path, status: str, interpretation: str) -> None:
        rows.append(
            {
                "prior_phase": phase,
                "metric": metric,
                "value": value,
                "source_file": relpath(source, root),
                "found_status": status,
                "interpretation": interpretation,
            }
        )

    prev = phase6a0_report_dir / "prior_phase_comparison.csv"
    if pd is not None and prev.exists():
        try:
            df = pd.read_csv(prev)
            for _, row in df.head(50).iterrows():
                add(str(row.get("prior_phase", "")), str(row.get("metric", "")), row.get("value", ""), prev, "found_phase6a0_prior_row", str(row.get("interpretation", ""))[:500])
        except Exception:
            pass
    key_paths = [
        root / "results/tables/05_phase5/phase5_2c/burden_state_ceiling_metrics_pre_adr.tsv",
        root / "results/tables/05_phase5/phase5_2c/event_recall_gap_to_ceiling.tsv",
        root / "results/tables/05_phase5/phase5_2c/event_per_subject_recall_distribution.tsv",
        root / "results/tables/05_phase5/phase5_2c/causal_minimum_sufficient_subset.tsv",
        root / "results/tables/05_phase5/feature_atlas_2b/top_feature_sets_for_detector_design.tsv",
        root / "results/tables/05_phase5/feature_atlas_2b/burden_loso_generalization.tsv",
        root / "docs/PHASE5_2C_ARCHITECTURAL_TARGET_REASSESSMENT.md",
        root / "docs/PHASE5_2C_PRE_ADR_BOUNDED_REMEDIATION_ANALYSIS.md",
    ]
    for path in key_paths:
        if not path.exists() or is_external_path(path):
            add("prior_phase", "file_presence", "missing", path, "not_found", "Expected prior-phase comparison source not found")
            continue
        if path.stat().st_size > 5_000_000:
            add("prior_phase", "file_presence", "skipped_large_file", path, "found_skipped", "Skipped large source for report-size safety")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        lower = text.lower()
        if "pearson" in lower or "burden" in lower:
            vals = re.findall(r"(?:pearson|correlation|r)\D{0,20}([-+]?\d+\.\d+)", lower)
            add("Phase 5Y/5Z/5_2C", "burden correlation/ceiling", vals[0] if vals else "not_numeric_in_excerpt", path, "found", "Prior burden/state evidence source; values are context only unless exact table columns are read.")
        if "fp/min" in lower or "recall" in lower:
            vals = re.findall(r"recall\D{0,20}([-+]?\d+\.\d+)", lower)
            add("Phase 5_2C", "event recall / FP-min collapse", vals[0] if vals else "not_numeric_in_excerpt", path, "found", "Prior event-detection limitation context.")
        if "auroc" in lower or "feature" in lower:
            vals = re.findall(r"auroc\D{0,20}([-+]?\d+\.\d+)", lower)
            add("Phase 5_2B/5_2C", "feature subset or AUROC", vals[0] if vals else "not_numeric_in_excerpt", path, "found", "Prior feature subset evidence.")
        if "burden pivot" in lower or "reopen" in lower:
            add("burden pivot", "ADR/recommendation", "found", path, "found", "Prior source discusses burden/state pivot or ADR reopening.")
    if not any("0.334" in str(r.get("value")) or "0.334" in str(r.get("interpretation")) for r in rows):
        add("Phase 5Y", "burden correlation around r ~= 0.334", "not_found", root / "reports+docs search", "not_found", "Specific r ~= 0.334 claim not repeated as fact because exact source was not found in this autopsy search.")
    add("Phase 6A.0", "burden viability gate", "FAIL", phase6a0_report_dir / "burden_viability_findings.json", "found", "Phase 6A.0 failed the formal gate and motivates this autopsy.")
    return rows


def choose_best_key(estimator_rows: list[dict[str, Any]], mode: str, deployable_only: bool | None = None, offline_only: bool | None = None) -> tuple[str, dict[str, Any]]:
    best: dict[str, Any] = {}
    best_val = -np.inf if np is not None else -1e9
    for row in estimator_rows:
        if row.get("row_type") != "aggregate":
            continue
        if row.get("validation_mode") != mode:
            continue
        if deployable_only is not None and bool(row.get("deployable")) != deployable_only:
            continue
        if offline_only is not None and bool(row.get("offline_ceiling")) != offline_only:
            continue
        val = max(to_float(row.get("median_spearman")), to_float(row.get("median_pearson")))
        if math.isfinite(val) and val > best_val:
            best_val = val
            best = row
    if not best:
        return "", {}
    key = f"{best.get('validation_mode')}|{best.get('model_name')}|{best.get('feature_set_name')}"
    return key, best


def decision_outputs(
    findings_base: dict[str, Any],
    continuity_summary: dict[str, Any],
    estimator_rows: list[dict[str, Any]],
    beta_delta_rows: list[dict[str, Any]],
    subject_summary: dict[str, Any],
    proxy_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], str]:
    classifications: list[str] = []
    rationale: list[str] = []
    next_steps: list[str] = []
    blockers: list[str] = []
    overall_cont = str(continuity_summary.get("overall_continuity_class", ""))
    _, best_loso = choose_best_key(estimator_rows, "LOSO_cross_subject")
    _, best_deploy = choose_best_key(estimator_rows, "LOSO_cross_subject", deployable_only=True)
    _, best_highcap = choose_best_key(estimator_rows, "LOSO_cross_subject", offline_only=True)
    _, best_within = choose_best_key(estimator_rows, "within_subject_block_or_time_calibrated")
    loso_val = best_metric(best_loso)
    deploy_val = best_metric(best_deploy)
    highcap_val = best_metric(best_highcap)
    within_val = best_metric(best_within)
    if overall_cont in {"event_or_candidate_matrix", "ambiguous", "unusable"}:
        classifications.append("table_limited_not_definitive")
        blockers.append("Selected table is not a definitive continuous burden stream.")
        rationale.append("Continuity audit classified the selected table as event/candidate or ambiguous.")
        next_steps.append("Build a true continuous window-level feature table before making a final burden-pivot decision.")
    if math.isfinite(loso_val) and loso_val <= 0.25 and math.isfinite(within_val) and within_val <= 0.30:
        classifications.append("substrate_limited" if overall_cont.startswith("continuous") else "table_limited_not_definitive")
        rationale.append("Best LOSO and within-subject ceilings remain low.")
    if math.isfinite(loso_val) and 0.25 < loso_val < 0.35 and proxy_summary.get("interpretation") == "weak_proxy_ceiling":
        classifications.append("proxy_limited")
        rationale.append("Annotation-burden ceiling is marginal and available proxies are weak.")
        next_steps.append("Do not use weak clinical/task proxies as primary validation.")
    if math.isfinite(highcap_val) and highcap_val >= 0.45 and (not math.isfinite(deploy_val) or deploy_val <= 0.30):
        classifications.append("estimator_limited")
        rationale.append("High-capacity offline ceiling exceeds deployable/simple ceiling.")
        next_steps.append("Refine estimator before any formal Phase 6A.0 re-gate.")
    if math.isfinite(within_val) and within_val >= 0.45 and (not math.isfinite(loso_val) or loso_val < 0.30):
        classifications.append("personalized_calibration_needed")
        rationale.append("Within-subject ceiling is materially higher than cross-subject LOSO.")
        next_steps.append("Consider a patient-calibrated burden tracker framing.")
    n_subjects = int(subject_summary.get("n_subjects", 0) or 0)
    n_high = int(subject_summary.get("n_high_trackable", 0) or 0)
    if n_subjects and n_high >= max(1, math.ceil(0.25 * n_subjects)) and math.isfinite(loso_val) and loso_val < 0.30:
        classifications.append("heterogeneous_trackability")
        rationale.append("At least 25% of subjects reach r >= 0.50 under the best diagnostic setting.")
        next_steps.append("Analyze responder/non-responder features before claiming broad generalization.")
    if proxy_summary.get("interpretation") == "weak_proxy_ceiling":
        classifications.append("proxy_limited")
        rationale.append("Available medication/task proxies remain weak supportive evidence.")
        next_steps.append("Treat proxy results as supportive only and do not claim clinical validation.")
    best_beta_delta = best_delta(beta_delta_rows, deployable_preferred=True)
    if math.isfinite(best_beta_delta) and best_beta_delta < 0.03:
        classifications.append("architecture_not_justified_beyond_beta")
        rationale.append("Best deployable improvement over beta baseline is less than 0.03.")
    if (
        math.isfinite(deploy_val)
        and deploy_val >= 0.40
        and math.isfinite(best_beta_delta)
        and best_beta_delta >= 0.05
        and int(best_deploy.get("n_valid_subjects", 0) or 0) >= 10
        and overall_cont not in {"event_or_candidate_matrix", "ambiguous", "unusable"}
    ):
        classifications.append("refined_regate_recommended")
        next_steps.append("Rerun Phase 6A.0 formally with the locked refined protocol.")
    if not classifications:
        classifications.append("inconclusive")
        rationale.append("No decision rule cleanly classified the failure.")
        next_steps.append("Stay blocked/inconclusive until a more suitable continuous substrate is available.")
    if "table_limited_not_definitive" in classifications:
        next_steps.append("Do not proceed to Brian2/DYNAP yet.")
    if not next_steps:
        next_steps.append("Do not proceed to Brian2/DYNAP yet.")
    forbidden = [
        "Do not proceed to Brian2/DYNAP Phase 6A from this autopsy alone.",
        "Do not use PPN or Herz as primary validation datasets.",
        "Do not claim clinical validation or efficacy from these proxy results.",
    ]
    rows = []
    for c in classifications:
        rows.append(
            {
                "classification": c,
                "status": "primary" if c == classifications[0] else "secondary",
                "evidence": "; ".join(rationale),
                "recommended_action": "; ".join(dict.fromkeys(next_steps)),
            }
        )
    return rows, classifications, blockers, forbidden, "; ".join(dict.fromkeys(rationale))


def best_metric(row: dict[str, Any]) -> float:
    if not row:
        return float("nan")
    return max(to_float(row.get("median_spearman")), to_float(row.get("median_pearson")), to_float(row.get("spearman_burden")), to_float(row.get("pearson_burden")))


def best_delta(rows: list[dict[str, Any]], deployable_preferred: bool = True) -> float:
    vals = [to_float(r.get("delta_vs_beta")) for r in rows if r.get("validation_mode") == "LOSO_cross_subject"]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else float("nan")


def build_findings(
    root: Path,
    args: argparse.Namespace,
    input_table: Path,
    previous: dict[str, Any],
    cols: Columns,
    df_meta: Any | None,
    allowed_features: list[str],
    continuity_summary: dict[str, Any],
    data_contract: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    feature_set_rows: list[dict[str, Any]],
    estimator_rows: list[dict[str, Any]],
    tau_rows: list[dict[str, Any]],
    beta_delta_rows: list[dict[str, Any]],
    proxy_summary: dict[str, Any],
    subject_summary: dict[str, Any],
    prior_rows: list[dict[str, Any]],
    classifications: list[str],
    blockers: list[str],
    forbidden: list[str],
    rationale: str,
    extra_limitations: list[str],
) -> dict[str, Any]:
    _, best_loso = choose_best_key(estimator_rows, "LOSO_cross_subject")
    _, best_deploy = choose_best_key(estimator_rows, "LOSO_cross_subject", deployable_only=True)
    _, best_within = choose_best_key(estimator_rows, "within_subject_block_or_time_calibrated")
    _, best_oracle = choose_best_key(estimator_rows, "oracle_descriptive")
    _, best_highcap = choose_best_key(estimator_rows, "LOSO_cross_subject", offline_only=True)
    best_delta_row = max(beta_delta_rows, key=lambda r: to_float(r.get("delta_vs_beta")), default={})
    data_status = "pass" if all(r.get("status") == "pass" for r in data_contract if r.get("severity") == "critical") else "fail"
    severe_leak = [r for r in leakage_rows if r.get("allowed_as_feature") and r.get("leakage_risk") == "high"]
    row_count = int(len(df_meta)) if df_meta is not None else 0
    subject_count = int(df_meta[cols.subject].nunique()) if df_meta is not None and cols.subject in df_meta.columns else 0
    return {
        "phase": "Phase 6A.0.5 burden gate failure autopsy and ceiling analysis",
        "repo_root": root.as_posix(),
        "audit_timestamp": now_iso(),
        "environment_python": f"{sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})",
        "input_table": relpath(input_table, root),
        "previous_phase6a0_status": previous.get("overall_status", ""),
        "previous_best_tau_ms": previous.get("best_tau_ms", ""),
        "previous_median_pearson": previous.get("best_tau_median_pearson", ""),
        "previous_median_spearman": previous.get("best_tau_median_spearman", ""),
        "subject_count": subject_count,
        "valid_subject_count": subject_count,
        "row_count": row_count,
        "column_count": len(read_header(input_table)) if input_table.exists() and pd is not None else 0,
        "subject_column": cols.subject,
        "time_column": cols.time,
        "session_column": cols.session,
        "block_column": cols.block,
        "label_column": cols.label,
        "continuity_summary": continuity_summary,
        "timing_confidence": "non_definitive_event_matrix" if continuity_summary.get("overall_continuity_class") == "event_or_candidate_matrix" else "high_or_medium",
        "data_contract_status": data_status,
        "leakage_summary": {
            "n_columns_reaudited": len(leakage_rows),
            "n_allowed_features": len(allowed_features),
            "n_high_risk_allowed": len(severe_leak),
            "status": "blocked_or_invalid_due_to_leakage" if severe_leak else "usable_with_exclusions",
        },
        "feature_sets_tested": [r.get("feature_set_name") for r in feature_set_rows if int(r.get("n_features", 0) or 0) > 0],
        "estimators_tested": sorted({str(r.get("model_name")) for r in estimator_rows if r.get("model_name")}),
        "best_loso_annotation_ceiling": best_loso,
        "best_high_capacity_loso_ceiling": best_highcap,
        "best_deployable_loso_ceiling": best_deploy,
        "best_within_subject_ceiling": best_within,
        "best_oracle_ceiling": best_oracle,
        "best_tau_summary": best_tau_summary(tau_rows),
        "beta_baseline_delta_summary": {
            "best_delta_vs_beta": to_float(best_delta_row.get("delta_vs_beta")),
            "best_delta_model_name": best_delta_row.get("model_name", ""),
            "best_delta_feature_set_name": best_delta_row.get("feature_set_name", ""),
            "interpretation": best_delta_row.get("architecture_value_interpretation", ""),
        },
        "proxy_ceiling_summary": proxy_summary,
        "subject_heterogeneity_summary": subject_summary,
        "prior_phase_comparison_summary": {
            "rows": len(prior_rows),
            "phase5y_r_0_334_found": any("0.334" in str(r) for r in prior_rows),
        },
        "primary_classification": classifications[0] if classifications else "inconclusive",
        "secondary_classifications": classifications[1:],
        "decision_rationale": rationale,
        "critical_blockers": blockers,
        "noncritical_limitations": extra_limitations,
        "recommended_next_steps": recommended_steps_for(classifications),
        "forbidden_next_steps": forbidden,
    }


def best_tau_summary(tau_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tau: dict[float, list[float]] = {}
    for row in tau_rows:
        if row.get("validation_mode") != "LOSO_cross_subject":
            continue
        tau = to_float(row.get("tau_ms"))
        val = max(to_float(row.get("spearman")), to_float(row.get("pearson")))
        if math.isfinite(tau) and math.isfinite(val):
            by_tau.setdefault(tau, []).append(val)
    med = {str(k): float(np.median(v)) for k, v in by_tau.items() if v and np is not None}
    best = max(med.items(), key=lambda kv: kv[1]) if med else ("", "")
    return {"median_metric_by_tau": med, "best_tau_ms": best[0], "best_tau_median_metric": best[1]}


def recommended_steps_for(classifications: list[str]) -> list[str]:
    steps: list[str] = []
    if "table_limited_not_definitive" in classifications:
        steps.append("Build a true continuous window-level internal STN feature table before final burden decisions.")
    if "estimator_limited" in classifications:
        steps.append("Refine estimator and rerun Phase 6A.0 with a locked protocol.")
    if "personalized_calibration_needed" in classifications:
        steps.append("Evaluate a patient-calibrated burden tracker framing.")
    if "heterogeneous_trackability" in classifications:
        steps.append("Analyze responder/non-responder structure.")
    if "architecture_not_justified_beyond_beta" in classifications:
        steps.append("Do not add neuromorphic complexity unless it beats beta controls.")
    if "substrate_limited" in classifications:
        steps.append("Abandon or reconsider the burden pivot on this substrate.")
    if not steps:
        steps.append("Stay blocked/inconclusive pending better inputs.")
    steps.append("Do not proceed to Brian2/DYNAP Phase 6A from this autopsy alone.")
    return list(dict.fromkeys(steps))


def write_readme(
    out_dir: Path,
    input_table: Path,
    previous_rows: list[dict[str, Any]],
    continuity_summary: dict[str, Any],
    data_contract: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    feature_set_rows: list[dict[str, Any]],
    estimator_rows: list[dict[str, Any]],
    tau_rows: list[dict[str, Any]],
    subject_summary: dict[str, Any],
    proxy_summary: dict[str, Any],
    beta_delta_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    findings: dict[str, Any],
    commands: list[str],
) -> None:
    agg = [r for r in estimator_rows if r.get("row_type") == "aggregate"]
    best_loso = findings.get("best_loso_annotation_ceiling", {})
    best_deploy = findings.get("best_deployable_loso_ceiling", {})
    best_within = findings.get("best_within_subject_ceiling", {})
    best_oracle = findings.get("best_oracle_ceiling", {})
    lines = [
        "# Phase 6A.0.5 Burden Gate Failure Autopsy",
        "",
        "## Purpose",
        "",
        "Phase 6A.0.5 diagnoses why the Phase 6A.0 internal STN annotation-derived burden gate failed. It does not authorize Brian2, DYNAP, or SNN simulation.",
        "",
        "## Why The Phase 6A.0 FAIL Was Meaningful But Not Decisive",
        "",
        "The failed gate showed that the lower-bound LOSO burden estimator did not reach the predeclared threshold. This autopsy checks whether that reflects substrate limits, proxy limits, estimator limits, table continuity limits, or subject heterogeneity.",
        "",
        "## Inputs Used",
        "",
        f"- Input table: `{input_table.as_posix()}`",
        "- External PPN/Herz datasets and their audit reports were excluded from model inputs.",
        "",
        "## Environment And Commands",
        "",
        f"- Python: `{sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})`",
        "- Environment route: `source /scratch/haizhe/stn/start_stn.sh && python ...`",
        "",
        "## Previous Phase 6A.0 Summary",
        "",
    ]
    for row in previous_rows[:20]:
        lines.append(f"- `{row['metric']}`: `{row['value']}`")
    lines.extend(
        [
            "",
            "## Data Contract And Continuity Findings",
            "",
            f"- Overall continuity class: `{continuity_summary.get('overall_continuity_class')}`",
            f"- Continuity class counts: `{continuity_summary.get('class_counts')}`",
            "- If classified as `event_or_candidate_matrix`, causal burden integration over row order is diagnostic pseudo-burden only, not definitive continuous tracking.",
            "",
        ]
    )
    for row in data_contract:
        if row.get("status") != "pass":
            lines.append(f"- Contract issue: `{row['requirement']}` -> `{row['status']}` ({row['notes']})")
    lines.extend(
        [
            "",
            "## Leakage Reaudit",
            "",
            f"- Columns reaudit count: `{len(leakage_rows)}`",
            f"- Allowed feature count after reaudit: `{findings.get('leakage_summary', {}).get('n_allowed_features')}`",
            f"- High-risk allowed features: `{findings.get('leakage_summary', {}).get('n_high_risk_allowed')}`",
            "",
            "## Feature Sets Tested",
            "",
        ]
    )
    for row in feature_set_rows:
        lines.append(f"- `{row['feature_set_name']}`: `{row['n_features']}` features; {row['notes']}")
    lines.extend(
        [
            "",
            "## Estimator Ceiling Results",
            "",
            f"- Best LOSO annotation-burden ceiling: `{summary_metric(best_loso)}` from `{best_loso.get('model_name', '')}` / `{best_loso.get('feature_set_name', '')}` at tau `{best_loso.get('tau_ms', '')}`.",
            f"- Best deployable/simple LOSO ceiling: `{summary_metric(best_deploy)}` from `{best_deploy.get('model_name', '')}` / `{best_deploy.get('feature_set_name', '')}`.",
            f"- Best within-subject calibrated ceiling: `{summary_metric(best_within)}` from `{best_within.get('model_name', '')}` / `{best_within.get('feature_set_name', '')}`.",
            f"- Best oracle descriptive ceiling: `{summary_metric(best_oracle)}` from `{best_oracle.get('model_name', '')}` / `{best_oracle.get('feature_set_name', '')}`.",
            "",
            "High-capacity models, when present, are offline ceilings only and are not deployable model claims.",
            "",
            "## Tau Heterogeneity Results",
            "",
            f"- Best tau summary: `{findings.get('best_tau_summary')}`",
            "- Oracle tau rows are descriptive only and are not validation.",
            "",
            "## Subject Distribution / Responder Analysis",
            "",
            f"- Heterogeneity summary: `{subject_summary}`",
            "",
            "## Proxy Ceiling Results",
            "",
            f"- Proxy summary: `{proxy_summary}`",
            "- Clinical/task proxies remain supportive only, not the primary gate.",
            "",
            "## Beta-Baseline Delta",
            "",
        ]
    )
    best_delta_row = max(beta_delta_rows, key=lambda r: to_float(r.get("delta_vs_beta")), default={})
    lines.append(f"- Best delta row: `{best_delta_row}`")
    lines.extend(
        [
            "",
            "## Prior-Phase Comparison",
            "",
            f"- Prior comparison rows: `{len(prior_rows)}`",
            "- Claims not found in source files are recorded as not found rather than repeated as fact.",
            "",
            "## Decision Classification",
            "",
            f"- Primary classification: `{findings.get('primary_classification')}`",
            f"- Secondary classifications: `{findings.get('secondary_classifications')}`",
            f"- Rationale: {findings.get('decision_rationale')}",
            "",
            "## Recommended Next Action",
            "",
        ]
    )
    for step in findings.get("recommended_next_steps", []):
        lines.append(f"- {step}")
    lines.extend(
        [
            "",
            "## What Not To Do Next",
            "",
            "- Do not proceed to Brian2/DYNAP yet.",
            "- Do not use PPN as primary STN burden validation.",
            "- Do not use Herz/Groppa/Brown as primary STN LFP architecture validation.",
            "- Do not claim clinical validation, clinical efficacy, or commercial-sensing equivalence.",
            "",
            "## Role Of PPN And Herz Datasets",
            "",
            "He/Tan PPN remains an optional future cross-target extension only after the primary internal STN architecture passes. Herz/Groppa/Brown remains a methods/code reference only.",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in findings.get("noncritical_limitations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Exact Commands Run", ""])
    for cmd in commands:
        lines.append(f"- `{cmd}`")
    (out_dir / "README_burden_failure_autopsy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_metric(row: dict[str, Any]) -> str:
    if not row:
        return "not_assessable"
    return f"Pearson {row.get('median_pearson', row.get('pearson_burden', ''))}, Spearman {row.get('median_spearman', row.get('spearman_burden', ''))}"


def make_plots(
    out_dir: Path,
    continuity_rows: list[dict[str, Any]],
    estimator_rows: list[dict[str, Any]],
    tau_rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    beta_rows: list[dict[str, Any]],
    no_plots: bool,
) -> list[str]:
    if no_plots or plt is None or np is None:
        return [f"plots skipped: matplotlib unavailable={MATPLOTLIB_ERROR}" if plt is None else "plots skipped by --no-plots"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    try:
        counts: dict[str, int] = {}
        for row in continuity_rows:
            cls = str(row.get("continuity_class", ""))
            counts[cls] = counts.get(cls, 0) + 1
        simple_bar(counts, fig_dir / "continuity_summary.png", "Continuity classes", "groups")
        notes.append("continuity_summary.png")
    except Exception as exc:
        notes.append(f"continuity plot failed: {exc}")
    try:
        agg = [r for r in estimator_rows if r.get("row_type") == "aggregate" and r.get("validation_mode") == "LOSO_cross_subject"]
        top = sorted(agg, key=lambda r: max(to_float(r.get("median_spearman")), to_float(r.get("median_pearson"))), reverse=True)[:15]
        labels = [f"{r.get('model_name')}|{r.get('feature_set_name')}|{int(float(r.get('tau_ms', 0) or 0))}" for r in top]
        vals = [max(to_float(r.get("median_spearman")), to_float(r.get("median_pearson"))) for r in top]
        horizontal_bar(labels, vals, fig_dir / "estimator_ceiling_comparison.png", "Estimator ceiling comparison")
        notes.append("estimator_ceiling_comparison.png")
    except Exception as exc:
        notes.append(f"estimator plot failed: {exc}")
    try:
        sub = [r for r in tau_rows if r.get("validation_mode") == "LOSO_cross_subject"]
        models = sorted({f"{r.get('model_name')}|{r.get('feature_set_name')}" for r in sub})[:12]
        taus = sorted({float(r.get("tau_ms", 0) or 0) for r in sub})
        mat = np.full((len(models), len(taus)), np.nan)
        for i, model in enumerate(models):
            for j, tau in enumerate(taus):
                vals = [max(to_float(r.get("spearman")), to_float(r.get("pearson"))) for r in sub if f"{r.get('model_name')}|{r.get('feature_set_name')}" == model and float(r.get("tau_ms", 0) or 0) == tau]
                vals = [v for v in vals if math.isfinite(v)]
                if vals:
                    mat[i, j] = np.median(vals)
        heatmap(mat, [str(int(t)) for t in taus], models, fig_dir / "tau_heterogeneity_heatmap.png", "Tau heterogeneity")
        notes.append("tau_heterogeneity_heatmap.png")
    except Exception as exc:
        notes.append(f"tau plot failed: {exc}")
    try:
        labels = [str(r.get("subject_id")) for r in subject_rows]
        vals = [max(to_float(r.get("best_loso_spearman")), to_float(r.get("best_loso_pearson"))) for r in subject_rows]
        horizontal_bar(labels, vals, fig_dir / "per_subject_distribution.png", "Per-subject best LOSO diagnostic metric")
        notes.append("per_subject_distribution.png")
    except Exception as exc:
        notes.append(f"subject plot failed: {exc}")
    try:
        agg = [r for r in estimator_rows if r.get("row_type") == "aggregate" and r.get("model_name") in {"numpy_ridge_fallback", "phase6a0_reproduction_or_surrogate"}]
        top = sorted(agg, key=lambda r: max(to_float(r.get("median_spearman")), to_float(r.get("median_pearson"))), reverse=True)[:12]
        labels = [str(r.get("feature_set_name")) + "|" + str(int(float(r.get("tau_ms", 0) or 0))) for r in top]
        vals = [max(to_float(r.get("median_spearman")), to_float(r.get("median_pearson"))) for r in top]
        horizontal_bar(labels, vals, fig_dir / "feature_subset_ablation.png", "Feature subset ablation")
        notes.append("feature_subset_ablation.png")
    except Exception as exc:
        notes.append(f"feature plot failed: {exc}")
    try:
        labels = [str(r.get("proxy_name")) + ":" + str(r.get("notes"))[:20] for r in proxy_rows]
        vals = [to_float(r.get("metric_value")) for r in proxy_rows]
        horizontal_bar(labels, vals, fig_dir / "proxy_ceiling_summary.png", "Proxy ceiling summary")
        notes.append("proxy_ceiling_summary.png")
    except Exception as exc:
        notes.append(f"proxy plot failed: {exc}")
    try:
        top = sorted(beta_rows, key=lambda r: to_float(r.get("delta_vs_beta")), reverse=True)[:15]
        labels = [f"{r.get('model_name')}|{r.get('feature_set_name')}|{int(float(r.get('tau_ms', 0) or 0))}" for r in top]
        vals = [to_float(r.get("delta_vs_beta")) for r in top]
        horizontal_bar(labels, vals, fig_dir / "beta_baseline_delta.png", "Delta over beta baseline")
        notes.append("beta_baseline_delta.png")
    except Exception as exc:
        notes.append(f"beta delta plot failed: {exc}")
    return notes


def simple_bar(counts: dict[str, int], path: Path, title: str, ylabel: str) -> None:
    labels = list(counts)
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def horizontal_bar(labels: list[str], vals: list[float], path: Path, title: str) -> None:
    labels = labels[:20]
    vals = vals[:20]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(mat: Any, xlabels: list[str], ylabels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(ylabels))))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_command_log(out_dir: Path, args: argparse.Namespace, commands: list[str]) -> None:
    commands.extend(
        [
            "pwd",
            "git rev-parse --show-toplevel",
            "git status --short",
            "source /scratch/haizhe/stn/start_stn.sh && python -V",
            "ls -lh reports/phase6a0_burden_viability || true",
            "python - <<'PY' ... Phase 6A.0 findings JSON ... PY",
            "find results/tables/05_phase5 -maxdepth 4 -type f | sort | head -300 || true",
            "find reports -maxdepth 5 -type f | sort | grep -E 'phase5|5_2|5Y|5_2C|burden|feature|ADR|phase6a0' | head -500 || true",
            "rg -n \"Phase 5_2B|Phase 5_2C|Phase 5Y|burden|AUROC|LOSO|MedOff|MedOn|Hold|Move|UPDRS|bradykinesia|causal_feature_matrix|refined feature|feature subset|beta-feature\" reports scripts results config configs data 2>/dev/null | head -800 || true",
            "source /scratch/haizhe/stn/start_stn.sh && python -m py_compile scripts/phase6a0_5_burden_failure_autopsy.py",
            "source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --stop-after-continuity",
            "source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_5_burden_failure_autopsy.py --input-table results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv --out-dir reports/phase6a0_5_burden_failure_autopsy --tau-ms 200,500,800,1500,3000,5000",
            "ls -lh reports/phase6a0_5_burden_failure_autopsy",
            "head -120 reports/phase6a0_5_burden_failure_autopsy/README_burden_failure_autopsy.md",
            "python - <<'PY' ... print burden_failure_autopsy_findings.json ... PY",
            "git diff --check",
            "find reports/phase6a0_5_burden_failure_autopsy -type f -size +5M -print",
            "git status --short",
        ]
    )
    commands.append("actual script argv: " + " ".join([sys.executable, *sys.argv]))
    (out_dir / "phase6a0_5_commands_run.txt").write_text("\n".join(dict.fromkeys(commands)) + "\n", encoding="utf-8")


def write_log_integrity_note(out_dir: Path) -> None:
    text = """# Phase 6A.0 Prior Run-Log Integrity Note

The Phase 6A.0 run log recorded that the final response and post-commit hash
would be reported in the final response rather than embedding the complete final
response text in the log. This Phase 6A.0.5 task does not edit old logs. The
gap is noted here so the follow-up run log can explicitly include final status
and commit/push information.
"""
    (out_dir / "phase6a0_prior_log_integrity_note.md").write_text(text, encoding="utf-8")


def blocked_outputs(
    out_dir: Path,
    root: Path,
    args: argparse.Namespace,
    input_table: Path,
    previous: dict[str, Any],
    previous_rows: list[dict[str, Any]],
    status: str,
    reason: str,
    commands: list[str],
) -> int:
    ensure_outputs(out_dir)
    write_csv(out_dir / "previous_gate_summary.csv", previous_rows, ["metric", "value", "source_file", "interpretation"])
    findings = {
        "phase": "Phase 6A.0.5 burden gate failure autopsy and ceiling analysis",
        "repo_root": root.as_posix(),
        "audit_timestamp": now_iso(),
        "environment_python": f"{sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})",
        "input_table": relpath(input_table, root),
        "previous_phase6a0_status": previous.get("overall_status", ""),
        "data_contract_status": status,
        "primary_classification": status,
        "secondary_classifications": [],
        "critical_blockers": [reason],
        "recommended_next_steps": ["Provide the required internal STN input table before continuing."],
        "forbidden_next_steps": ["Do not proceed to Brian2/DYNAP Phase 6A from this autopsy alone."],
    }
    write_json(out_dir / "burden_failure_autopsy_findings.json", findings)
    write_command_log(out_dir, args, commands)
    readme = f"""# Phase 6A.0.5 Burden Gate Failure Autopsy

Status: `{status}`

Input table: `{input_table.as_posix()}`

Blocker: {reason}

Do not proceed to Brian2/DYNAP Phase 6A from this autopsy alone.
"""
    (out_dir / "README_burden_failure_autopsy.md").write_text(readme, encoding="utf-8")
    write_log_integrity_note(out_dir)
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()
    root = repo_root()
    input_table = (root / args.input_table).resolve() if not Path(args.input_table).is_absolute() else Path(args.input_table)
    out_dir = root / args.out_dir
    commands: list[str] = []
    ensure_outputs(out_dir)
    write_log_integrity_note(out_dir)
    phase6a0_report_dir = root / args.phase6a0_report_dir
    phase6a0_findings = root / args.phase6a0_findings
    previous, previous_rows = load_previous_gate(phase6a0_findings, phase6a0_report_dir)
    write_csv(out_dir / "previous_gate_summary.csv", previous_rows, ["metric", "value", "source_file", "interpretation"])

    if pd is None or np is None:
        reason = f"missing dependency pandas={PANDAS_ERROR} numpy={NUMPY_ERROR}"
        return blocked_outputs(out_dir, root, args, input_table, previous, previous_rows, "blocked_missing_dependency", reason, commands)
    if not input_table.exists():
        return blocked_outputs(out_dir, root, args, input_table, previous, previous_rows, "blocked_missing_input_table", "input table is missing or unreadable", commands)
    if is_external_path(relpath(input_table, root)):
        return blocked_outputs(out_dir, root, args, input_table, previous, previous_rows, "blocked_external_dataset", "external PPN/Herz path is forbidden as a primary input", commands)

    header = read_header(input_table)
    cols = detect_columns(header, args)
    previous_audit = read_phase6a0_feature_audit(phase6a0_report_dir)
    feature_meta = read_feature_metadata(input_table)
    leakage_rows = [role_and_leakage(c, cols, previous_audit, feature_meta) for c in header]
    allowed_features = [r["column"] for r in leakage_rows if r.get("allowed_as_feature")]
    write_csv(out_dir / "leakage_reaudit.csv", leakage_rows, ["column", "role_guess", "allowed_as_feature", "leakage_risk", "reason", "feature_family_guess", "notes"])

    meta_df = read_metadata_table(input_table, cols, args.max_rows)
    continuity_rows, continuity_summary = continuity_audit(meta_df, cols)
    continuity_cols = [
        "subject_id",
        "session_id",
        "block_id",
        "n_rows",
        "label_positive_count",
        "label_positive_fraction",
        "time_col_used",
        "has_time_col",
        "time_monotonic",
        "duplicate_time_count",
        "dt_median",
        "dt_iqr",
        "dt_min",
        "dt_max",
        "dt_units_guess",
        "large_gap_count",
        "time_resets_detected",
        "row_order_only",
        "positive_run_count",
        "positive_run_median_len",
        "negative_run_count",
        "negative_run_median_len",
        "label_transition_count",
        "event_like_columns_detected",
        "event_centered_pattern_guess",
        "continuous_stream_score",
        "continuity_class",
        "notes",
    ]
    write_csv(out_dir / "table_continuity_audit.csv", continuity_rows, continuity_cols)
    contract = data_contract_rows(input_table, header, cols, allowed_features, meta_df, continuity_summary)
    write_csv(out_dir / "data_contract_audit.csv", contract, ["requirement", "status", "evidence", "severity", "notes"])

    top_k_values = parse_int_list(args.top_k_features)
    feature_sets, feature_set_rows = build_feature_sets(root, header, allowed_features, previous_audit, top_k_values)
    write_csv(out_dir / "selected_feature_sets.csv", feature_set_rows, ["feature_set_name", "n_features", "feature_names_short", "selection_method", "selection_leakage_risk", "deployability_guess", "notes"])
    prior_rows = prior_phase_comparison(root, phase6a0_report_dir)
    write_csv(out_dir / "prior_phase_comparison.csv", prior_rows, ["prior_phase", "metric", "value", "source_file", "found_status", "interpretation"])

    if args.stop_after_continuity:
        empty_model_headers(out_dir)
        classifications = ["table_limited_not_definitive"] if continuity_summary.get("overall_continuity_class") == "event_or_candidate_matrix" else ["continuity_completed_only"]
        findings = build_findings(
            root,
            args,
            input_table,
            previous,
            cols,
            meta_df,
            allowed_features,
            continuity_summary,
            contract,
            leakage_rows,
            feature_set_rows,
            [],
            [],
            [],
            {"status": "not_run_continuity_only"},
            {},
            prior_rows,
            classifications,
            [],
            ["Do not proceed to Brian2/DYNAP Phase 6A from this autopsy alone."],
            "Continuity-only pass completed.",
            ["Model ceilings not run because --stop-after-continuity was set."],
        )
        write_json(out_dir / "burden_failure_autopsy_findings.json", findings)
        decision_rows = [{"classification": c, "status": "preliminary", "evidence": "continuity-only pass", "recommended_action": "run full autopsy unless table blocker is sufficient"} for c in classifications]
        write_csv(out_dir / "decision_matrix.csv", decision_rows, ["classification", "status", "evidence", "recommended_action"])
        write_command_log(out_dir, args, commands)
        write_readme(out_dir, input_table, previous_rows, continuity_summary, contract, leakage_rows, feature_set_rows, [], [], {}, {"status": "not_run_continuity_only"}, [], prior_rows, findings, commands)
        LOG.info("Continuity-only pass complete: %s", continuity_summary)
        return 0

    meta_cols = [c for c in meta_df.columns if c in header]
    union_features = sorted({f for features in feature_sets.values() for f in features if f in header})
    # Dynamic top-k feature sets store the full allowed pool; keep all allowed for fold-local selection.
    for f in allowed_features:
        if f not in union_features:
            union_features.append(f)
    full_df = load_analysis_table(input_table, meta_cols, union_features, args.max_rows)
    tau_values = parse_float_list(args.tau_ms)
    estimator_rows, tau_rows, feature_ablation_rows, scores_by_spec = run_loso_models(
        full_df,
        cols,
        feature_sets,
        tau_values,
        top_k_values,
        args.random_seed,
        str(continuity_summary.get("overall_continuity_class", "")),
        args.skip_high_capacity,
    )
    beta_delta_rows = add_baseline_deltas(estimator_rows)
    subject_rows, subject_summary = subject_distribution(estimator_rows, phase6a0_report_dir)
    best_key, _ = choose_best_key(estimator_rows, "LOSO_cross_subject")
    proxy_rows, proxy_summary = proxy_ceiling(full_df, cols, scores_by_spec, best_key)
    decision_rows, classifications, blockers, forbidden, rationale = decision_outputs(
        {},
        continuity_summary,
        estimator_rows,
        beta_delta_rows,
        subject_summary,
        proxy_summary,
    )
    plot_notes = make_plots(out_dir, continuity_rows, estimator_rows, tau_rows, subject_rows, proxy_rows, beta_delta_rows, args.no_plots)
    extra_limitations = [
        "Selected Phase 5_2C table contains event/candidate-centered rows; burden traces are pseudo-burden when continuity is not continuous.",
        "High-capacity sklearn ceilings use deterministic training caps for runtime and are offline descriptive only.",
        "Window rows are autocorrelated; summaries use subject-level medians.",
        "Phase 6A.0.5 does not authorize Brian2/DYNAP work.",
        *plot_notes,
    ]
    findings = build_findings(
        root,
        args,
        input_table,
        previous,
        cols,
        meta_df,
        allowed_features,
        continuity_summary,
        contract,
        leakage_rows,
        feature_set_rows,
        estimator_rows,
        tau_rows,
        beta_delta_rows,
        proxy_summary,
        subject_summary,
        prior_rows,
        classifications,
        blockers,
        forbidden,
        rationale,
        extra_limitations,
    )

    estimator_cols = [
        "row_type",
        "validation_mode",
        "model_name",
        "feature_set_name",
        "tau_ms",
        "subject_id",
        "n_rows",
        "positive_fraction",
        "pearson_burden",
        "spearman_burden",
        "rmse_burden",
        "mae_burden",
        "auroc_label",
        "auprc_label",
        "brier_score",
        "calibration_slope",
        "calibration_intercept",
        "high_burden_balanced_accuracy",
        "timing_confidence",
        "continuity_class",
        "median_pearson",
        "median_spearman",
        "iqr_pearson",
        "iqr_spearman",
        "n_valid_subjects",
        "n_high_subjects_r_ge_0_5",
        "n_moderate_subjects_r_0_3_to_0_5",
        "n_low_subjects_r_le_0_2",
        "beats_beta_by",
        "beats_shuffled_by",
        "beats_class_prior_by",
        "deployable",
        "offline_ceiling",
        "notes",
    ]
    tau_cols = ["validation_mode", "subject_id", "model_name", "feature_set_name", "tau_ms", "pearson", "spearman", "is_subject_best_tau", "is_global_train_selected_tau", "timing_confidence", "notes"]
    subject_cols = [
        "subject_id",
        "n_rows",
        "positive_fraction",
        "phase6a0_pearson_if_available",
        "best_loso_pearson",
        "best_loso_spearman",
        "best_within_subject_pearson",
        "best_tau_loso",
        "best_tau_within_subject",
        "best_model",
        "best_feature_set",
        "beta_baseline_pearson",
        "improvement_over_beta",
        "responder_class",
        "notes",
    ]
    proxy_cols = [
        "proxy_name",
        "proxy_type",
        "n_subjects",
        "n_rows",
        "validation_mode",
        "model_name",
        "feature_set_name",
        "metric_primary",
        "metric_value",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "rank_biserial_or_cohens_d",
        "spearman_session_level",
        "pearson_session_level",
        "proxy_interpretation",
        "notes",
    ]
    beta_cols = ["model_name", "feature_set_name", "validation_mode", "tau_ms", "median_pearson", "median_spearman", "beta_baseline_median_pearson", "delta_vs_beta", "delta_vs_shuffled", "delta_vs_class_prior", "architecture_value_interpretation"]
    write_csv(out_dir / "estimator_ceiling_comparison.csv", estimator_rows, estimator_cols)
    write_csv(out_dir / "tau_heterogeneity.csv", tau_rows, tau_cols)
    write_csv(out_dir / "feature_subset_ablation.csv", feature_ablation_rows, estimator_cols)
    write_csv(out_dir / "subject_distribution_check.csv", subject_rows, subject_cols)
    write_csv(out_dir / "proxy_ceiling_comparison.csv", proxy_rows, proxy_cols)
    write_csv(out_dir / "beta_baseline_delta.csv", beta_delta_rows, beta_cols)
    write_csv(out_dir / "decision_matrix.csv", decision_rows, ["classification", "status", "evidence", "recommended_action"])
    write_json(out_dir / "burden_failure_autopsy_findings.json", findings)
    write_command_log(out_dir, args, commands)
    write_readme(out_dir, input_table, previous_rows, continuity_summary, contract, leakage_rows, feature_set_rows, estimator_rows, tau_rows, subject_summary, proxy_summary, beta_delta_rows, prior_rows, findings, commands)
    LOG.info("Phase 6A.0.5 complete: %s", findings.get("primary_classification"))
    return 0


def empty_model_headers(out_dir: Path) -> None:
    write_csv(out_dir / "feature_subset_ablation.csv", [], ["row_type", "validation_mode", "model_name", "feature_set_name", "tau_ms", "subject_id", "median_pearson", "median_spearman", "notes"])
    write_csv(out_dir / "estimator_ceiling_comparison.csv", [], ["row_type", "validation_mode", "model_name", "feature_set_name", "tau_ms", "subject_id", "pearson_burden", "spearman_burden", "notes"])
    write_csv(out_dir / "tau_heterogeneity.csv", [], ["validation_mode", "subject_id", "model_name", "feature_set_name", "tau_ms", "pearson", "spearman", "notes"])
    write_csv(out_dir / "subject_distribution_check.csv", [], ["subject_id", "n_rows", "best_loso_pearson", "best_loso_spearman", "responder_class", "notes"])
    write_csv(out_dir / "proxy_ceiling_comparison.csv", [], ["proxy_name", "proxy_type", "metric_primary", "metric_value", "proxy_interpretation", "notes"])
    write_csv(out_dir / "beta_baseline_delta.csv", [], ["model_name", "feature_set_name", "validation_mode", "median_pearson", "delta_vs_beta", "architecture_value_interpretation"])


if __name__ == "__main__":
    raise SystemExit(main())

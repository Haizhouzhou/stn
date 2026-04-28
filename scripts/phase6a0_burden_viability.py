#!/usr/bin/env python3
"""Phase 6A.0 burden-target viability gate for the internal STN substrate.

This script is intentionally gate-oriented. It discovers existing internal STN
feature/label tables, excludes external PPN/Herz Phase 6 audits from candidate
selection, audits causal features and leakage risk, and runs a subject-held-out
burden tracking check only when the input contract is satisfied.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    NUMPY_ERROR = ""

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    PANDAS_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    PANDAS_ERROR = ""

try:
    import scipy.stats as scipy_stats
except Exception as exc:  # pragma: no cover
    scipy_stats = None  # type: ignore[assignment]
    SCIPY_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    SCIPY_ERROR = ""

try:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        precision_recall_curve,
        roc_auc_score,
    )
except Exception as exc:  # pragma: no cover
    average_precision_score = None  # type: ignore[assignment]
    balanced_accuracy_score = None  # type: ignore[assignment]
    brier_score_loss = None  # type: ignore[assignment]
    precision_recall_curve = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    SKLEARN_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    SKLEARN_ERROR = ""

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    MATPLOTLIB_ERROR = ""


LOG = logging.getLogger("phase6a0_burden_viability")

EXCLUDED_INPUT_PATTERNS = [
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

CANDIDATE_EXTS = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".npz", ".pkl", ".pickle", ".mat"}
CANDIDATE_KEYWORDS = [
    "phase3",
    "phase5",
    "phase5_2a",
    "phase5_2b",
    "phase5_2c",
    "phase5y",
    "burst",
    "burden",
    "feature",
    "features",
    "label",
    "labels",
    "stn",
    "subject",
    "window",
    "causal",
]

DISCOVERY_SKIP_DIR_TERMS = {
    "__pycache__",
    "bursts",
    "cache",
    "causal_feature_matrix_chunks",
    "checkpoints",
    "chunks",
    "figures",
    "logs",
    "plots",
    "probes",
    "progress",
    "tmp",
}

DISCOVERY_SKIP_PATH_TERMS = [
    "/probes/",
    "/progress/",
    "/figures/",
    "/plots/",
    "/logs/",
    "/causal_feature_matrix_chunks/",
    "/bursts/",
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

IDENTITY_COL_HINTS = {
    "subject": ["subject_id", "subject", "participant", "patient_id", "subj"],
    "time": ["window_start_s", "time", "timestamp", "sample", "row_time", "start_s"],
    "session": ["session", "ses", "visit"],
    "block": ["block", "run", "trial", "condition"],
    "label": ["target_label", "label", "burst_label", "y", "is_burst", "annotation"],
}

METADATA_HINTS = [
    "condition",
    "channel",
    "band_mode",
    "fif_path",
    "window_type",
    "negative_category",
    "window_duration_ms",
    "threshold_source",
    "is_beta_active_channel",
    "long_burst_category",
    "provenance_status",
    "med",
    "task",
    "updrs",
    "brady",
    "tremor",
    "rigidity",
]

MIN_REQUIRED_SUBJECTS = 8
MIN_REQUIRED_FEATURES = 5
DEFAULT_MODEL_FEATURE_CAP = 96
READ_CHUNKSIZE = 100_000
LARGE_DISCOVERY_TABLE_BYTES = 100_000_000


@dataclass
class Columns:
    subject: str = ""
    time: str = ""
    session: str = ""
    block: str = ""
    label: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path)
    parser.add_argument("--features-table", type=Path)
    parser.add_argument("--labels-table", type=Path)
    parser.add_argument("--metadata-table", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/phase6a0_burden_viability"))
    parser.add_argument("--subject-col")
    parser.add_argument("--time-col")
    parser.add_argument("--session-col")
    parser.add_argument("--block-col")
    parser.add_argument("--label-col")
    parser.add_argument("--tau-ms", default="200,500,800,1500,3000")
    parser.add_argument("--window-ms", type=float, default=800.0)
    parser.add_argument("--step-ms", default="auto")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--random-seed", type=int, default=20260427)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--stop-after-discovery", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def clean_join(values: Iterable[Any], sep: str = ";") -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return sep.join(out)


def bool_cell(value: Any) -> str:
    return "true" if bool(value) else "false"


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if np is not None and isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_tau_ms(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        stripped = part.strip()
        if stripped:
            values.append(float(stripped))
    return values


def is_excluded_external_path(path: Path | str) -> bool:
    text = str(path)
    return any(pattern in text for pattern in EXCLUDED_INPUT_PATTERNS)


def file_extension(path: Path) -> str:
    return path.suffix.lower()


def discover_candidate_paths(repo_root: Path) -> list[Path]:
    roots = [repo_root / name for name in ["data", "results", "reports", "outputs", "artifacts"]]
    candidates: list[Path] = []
    explicit_candidates = [
        repo_root / "results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv",
        repo_root / "results/tables/05_phase5/phase5_2c/causal_feature_matrix_validation.tsv",
        repo_root / "results/phase5_2c/causal_feature_matrix.tsv",
        repo_root / "results/phase5_2a_feature_atlas/feature_matrix.tsv",
        repo_root / "results/tables/05_phase5/feature_atlas/window_index.tsv",
    ]
    for path in explicit_candidates:
        if path.exists():
            candidates.append(path)
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dir_rel = relpath(Path(dirpath), repo_root).lower()
            dirnames[:] = [
                name
                for name in dirnames
                if name.lower() not in DISCOVERY_SKIP_DIR_TERMS
                and ".git" not in name.lower()
            ]
            for filename in filenames:
                path = Path(dirpath) / filename
                if not path.is_file():
                    continue
                if path.suffix.lower() not in CANDIDATE_EXTS:
                    continue
                rel = relpath(path, repo_root)
                lower = rel.lower()
                if any(term in lower for term in DISCOVERY_SKIP_PATH_TERMS):
                    continue
                if is_excluded_external_path(rel):
                    candidates.append(path)
                    continue
                if any(keyword in lower for keyword in CANDIDATE_KEYWORDS):
                    candidates.append(path)
    return sorted(set(candidates))


def read_header_and_sample(path: Path, max_rows: int = 1000) -> tuple[list[str], Any | None, str, str]:
    ext = path.suffix.lower()
    if pd is None:
        return [], None, "unreadable", f"pandas unavailable: {PANDAS_ERROR}"
    try:
        if ext in {".tsv", ".csv"}:
            delimiter = "\t" if ext == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                header = next(reader, [])
            size_note = "large table" if path.stat().st_size > LARGE_DISCOVERY_TABLE_BYTES else "table"
            return [str(c) for c in header], None, "readable_header_only", f"{size_note} sampled by header only during discovery"
        if ext == ".tsv":
            sample = pd.read_csv(path, sep="\t", nrows=max_rows, low_memory=False)
        elif ext == ".csv":
            sample = pd.read_csv(path, nrows=max_rows, low_memory=False)
        elif ext == ".jsonl":
            sample = pd.read_json(path, lines=True, nrows=max_rows)
        elif ext == ".json":
            if path.stat().st_size > 20_000_000:
                return [], None, "not_read_large_json", "large JSON skipped during discovery"
            loaded = pd.read_json(path)
            sample = loaded.head(max_rows)
        elif ext == ".parquet":
            return [], None, "not_deeply_read_parquet", "parquet files are not opened during discovery; pass explicitly if needed"
        elif ext in {".pkl", ".pickle"}:
            if path.stat().st_size > 20_000_000:
                return [], None, "not_read_large_pickle", "large pickle skipped during discovery"
            sample = pd.read_pickle(path)
            if hasattr(sample, "head"):
                sample = sample.head(max_rows)
            else:
                return [], None, "read_non_table_pickle", type(sample).__name__
        else:
            return [], None, "not_read_schema", f"schema read not implemented for {ext}"
        return [str(c) for c in sample.columns], sample, "readable_sample", ""
    except Exception as exc:
        return [], None, "unreadable", f"{exc.__class__.__name__}: {exc}"


def detect_columns(columns: list[str]) -> Columns:
    lower_map = {c.lower(): c for c in columns}

    def pick(hints: list[str]) -> str:
        for hint in hints:
            if hint.lower() in lower_map:
                return lower_map[hint.lower()]
        for c in columns:
            lc = c.lower()
            if any(hint.lower() in lc for hint in hints):
                return c
        return ""

    return Columns(
        subject=pick(IDENTITY_COL_HINTS["subject"]),
        time=pick(IDENTITY_COL_HINTS["time"]),
        session=pick(IDENTITY_COL_HINTS["session"]),
        block=pick(IDENTITY_COL_HINTS["block"]),
        label=pick(IDENTITY_COL_HINTS["label"]),
    )


def feature_like_columns(columns: list[str]) -> list[str]:
    detected: list[str] = []
    for col in columns:
        lower = col.lower()
        if any(key in lower for key in ["beta", "causal", "ratio", "slope", "power", "rms", "envelope", "spatial"]):
            if not any(leak in lower for leak in ["target", "label", "subject_id", "window_id", "fif_path"]):
                detected.append(col)
    return detected


def metadata_like_columns(columns: list[str]) -> list[str]:
    out = []
    for col in columns:
        lower = col.lower()
        if any(hint in lower for hint in METADATA_HINTS):
            out.append(col)
    return out


def label_like_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if any(hint in c.lower() for hint in IDENTITY_COL_HINTS["label"])]


def time_like_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if any(hint in c.lower() for hint in IDENTITY_COL_HINTS["time"])]


def subject_like_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if any(hint in c.lower() for hint in IDENTITY_COL_HINTS["subject"])]


def count_unique_sample(sample: Any | None, col: str) -> int | str:
    if sample is None or not col or col not in sample.columns:
        return ""
    try:
        return int(sample[col].nunique(dropna=True))
    except Exception:
        return ""


def score_candidate(
    repo_root: Path,
    path: Path,
    columns: list[str],
    sample: Any | None,
    readable_status: str,
) -> tuple[int, str, str]:
    rel = relpath(path, repo_root)
    lower = rel.lower()
    reasons: list[str] = []
    rejection: list[str] = []
    score = 0
    if is_excluded_external_path(rel):
        return -100, "excluded_external_dataset_or_report", "critical path exclusion rule"

    cols = detect_columns(columns)
    features = feature_like_columns(columns)
    if cols.subject:
        score += 4
        reasons.append("subject column detected")
    else:
        score -= 5
        rejection.append("no subject column")
    if cols.label:
        score += 4
        reasons.append("label/target column detected")
    else:
        score -= 5
        rejection.append("no label column")
    if cols.time:
        score += 3
        reasons.append("time/order column detected")
    if len(features) >= MIN_REQUIRED_FEATURES:
        score += 5
        reasons.append(f"{len(features)} feature-like columns detected")
    else:
        score -= 3
        rejection.append("fewer than five feature-like columns")

    sample_subjects = count_unique_sample(sample, cols.subject)
    if isinstance(sample_subjects, int) and sample_subjects >= 8:
        score += 3
        reasons.append("sample shows at least 8 subjects")
    elif isinstance(sample_subjects, int) and sample_subjects > 0:
        reasons.append(f"sample shows {sample_subjects} subject(s)")

    if "phase5_2c" in lower:
        score += 8
        reasons.append("Phase 5_2C path")
    if "causal_feature_matrix" in lower:
        score += 8
        reasons.append("causal feature matrix path")
    if "phase5_2a" in lower or "feature_atlas" in lower:
        score += 3
        reasons.append("Phase 5_2A/feature atlas path")
    if "phase5y" in lower or "burden" in lower:
        score += 3
        reasons.append("burden/Phase 5Y keyword")
    if "phase3" in lower or "burst" in lower:
        score += 2
        reasons.append("Phase 3/burst keyword")
    if path.suffix.lower() == ".parquet" and "/results/bursts/" in rel:
        score -= 4
        rejection.append("per-channel Phase 3 burst parquet, not a combined feature table")
    if any(term in lower for term in ["summary", "validation", "audit", "manifest", "report"]):
        score -= 4
        rejection.append("likely aggregate/report metadata rather than window-level feature table")
    if readable_status.startswith("unreadable"):
        score -= 6
        rejection.append("unreadable during discovery")

    return score, clean_join(reasons), clean_join(rejection)


def discover_inputs(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover_candidate_paths(repo_root):
        rel = relpath(path, repo_root)
        ext = path.suffix.lower()
        columns, sample, readable_status, notes = read_header_and_sample(path)
        detected = detect_columns(columns)
        features = feature_like_columns(columns)
        metadata_cols = metadata_like_columns(columns)
        score, reasons, rejection = score_candidate(repo_root, path, columns, sample, readable_status)
        n_rows = ""
        n_cols = len(columns) if columns else ""
        if sample is not None:
            n_rows = f"sample={len(sample)}"
            n_cols = len(sample.columns)
        # Known prior validation rows provide exact row counts for major internal matrices.
        if rel.endswith("results/tables/05_phase5/phase5_2c/causal_feature_matrix.tsv"):
            n_rows = "known_prior_validation=690088"
        elif rel.endswith("results/phase5_2a_feature_atlas/feature_matrix.tsv"):
            n_rows = "known_prior_validation=690088"
        likely_use = "candidate_primary_input" if score >= 20 else "candidate_weak_or_context"
        if score < 0:
            likely_use = "rejected"
        rows.append(
            {
                "relative_path": rel,
                "extension": ext,
                "size_mb": f"{path.stat().st_size / (1024 * 1024):.6f}",
                "readable_status": readable_status,
                "n_rows_if_known": n_rows,
                "n_cols_if_known": n_cols,
                "detected_subject_cols": clean_join(subject_like_columns(columns)),
                "detected_time_cols": clean_join(time_like_columns(columns)),
                "detected_label_cols": clean_join(label_like_columns(columns)),
                "detected_feature_cols": clean_join(features[:60]),
                "detected_metadata_cols": clean_join(metadata_cols[:40]),
                "candidate_score": score,
                "likely_use": likely_use,
                "acceptance_reasons": reasons,
                "rejection_reasons": rejection,
                "notes": notes,
            }
        )
    rows.sort(key=lambda r: (int(r.get("candidate_score", 0)), -float(r.get("size_mb", "0") or 0)), reverse=True)
    viable = [r for r in rows if int(r.get("candidate_score", 0)) >= 20 and r["likely_use"] != "rejected"]
    selected = ""
    confidence = "none"
    if viable:
        top = viable[0]
        second_score = int(viable[1]["candidate_score"]) if len(viable) > 1 else -999
        top_score = int(top["candidate_score"])
        if top_score >= 28 and top_score - second_score >= 5:
            selected = top["relative_path"]
            confidence = "high"
        elif top_score >= 24 and top_score - second_score >= 3:
            selected = top["relative_path"]
            confidence = "medium"
        else:
            confidence = "low"
    recommendation = {
        "selected_input_path": selected,
        "selection_confidence": confidence,
        "alternative_candidates": [r["relative_path"] for r in viable[1:8]],
        "rejection_reasons": {r["relative_path"]: r["rejection_reasons"] for r in rows[:50] if r["rejection_reasons"]},
        "required_user_action_if_blocked": (
            "Provide --input-table, --labels-table, and/or --metadata-table for a window-level internal STN feature/label table."
            if confidence in {"none", "low"}
            else ""
        ),
    }
    return rows, recommendation


def classify_column(col: str, selected_feature_cols: set[str], allowed_feature_cols: set[str], identity: Columns) -> tuple[str, str]:
    lower = col.lower()
    if col in {identity.subject, identity.time, identity.session, identity.block}:
        return "subject/session/block/time columns", "identity/order column"
    if col == identity.label:
        return "target/label columns", "annotation-derived target"
    if col in allowed_feature_cols:
        role = "allowed causal feature columns"
        reason = "metadata or naming indicates causal online-compatible feature"
        if col not in selected_feature_cols:
            reason += "; not selected for model due deterministic runtime cap"
        return role, reason
    for leak in LEAKAGE_KEYWORDS:
        if leak in lower:
            return "suspicious leakage columns", f"contains leakage keyword '{leak}'"
    if any(hint in lower for hint in METADATA_HINTS):
        return "metadata/proxy columns", "metadata/proxy or grouping column"
    return "excluded non-feature columns", "not selected as causal feature"


def infer_allowed_features(input_path: Path, columns: list[str], repo_root: Path) -> tuple[list[str], dict[str, dict[str, Any]], list[str]]:
    metadata: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    meta_candidates = [
        input_path.parent / "causal_feature_matrix_feature_metadata.tsv",
        repo_root / "results/tables/05_phase5/phase5_2c/causal_feature_matrix_feature_metadata.tsv",
    ]
    meta_path = next((p for p in meta_candidates if p.exists()), None)
    if meta_path is not None and pd is not None:
        try:
            meta_df = pd.read_csv(meta_path, sep="\t")
            for _, row in meta_df.iterrows():
                out_col = str(row.get("output_column") or row.get("feature_name") or "")
                if not out_col:
                    continue
                metadata[out_col] = row.to_dict()
            allowed = []
            for col in columns:
                row = metadata.get(col)
                if not row:
                    continue
                leakage = str(row.get("leakage_risk_level", "")).lower()
                taut = str(row.get("tautology_risk_level", "")).lower()
                qc = str(row.get("qc_status", "")).lower()
                future = str(row.get("uses_future_samples", "")).lower() == "true"
                causal = str(row.get("causal_frontend_confirmed", "")).lower() == "true"
                if qc == "ok" and leakage in {"low", ""} and taut in {"low", ""} and not future and causal:
                    allowed.append(col)
            notes.append(f"allowed features from metadata {relpath(meta_path, repo_root)}")
            return allowed, metadata, notes
        except Exception as exc:
            notes.append(f"metadata read failed: {exc.__class__.__name__}: {exc}")
    allowed = []
    for col in columns:
        lower = col.lower()
        if col in feature_like_columns([col]) and not any(key in lower for key in LEAKAGE_KEYWORDS):
            allowed.append(col)
    notes.append("allowed features inferred from column names")
    return allowed, metadata, notes


def select_model_features(allowed: list[str], metadata: dict[str, dict[str, Any]], cap: int = DEFAULT_MODEL_FEATURE_CAP) -> list[str]:
    if len(allowed) <= cap:
        return allowed
    by_family: dict[str, list[str]] = defaultdict(list)
    for col in allowed:
        family = str(metadata.get(col, {}).get("feature_family", "") or col.split("__", 1)[0])
        by_family[family].append(col)
    selected: list[str] = []
    family_order = sorted(by_family, key=lambda fam: (-len(by_family[fam]), fam))
    while len(selected) < cap:
        added = False
        for family in family_order:
            items = by_family[family]
            if items:
                selected.append(items.pop(0))
                added = True
                if len(selected) >= cap:
                    break
        if not added:
            break
    return selected


def read_selected_table(
    input_path: Path,
    usecols: list[str],
    max_rows: int | None,
) -> Any:
    if pd is None:
        raise RuntimeError(f"pandas unavailable: {PANDAS_ERROR}")
    ext = input_path.suffix.lower()
    if ext == ".tsv":
        return pd.read_csv(input_path, sep="\t", usecols=usecols, nrows=max_rows, low_memory=False, na_values=["NA", "NaN", "nan"])
    if ext == ".csv":
        return pd.read_csv(input_path, usecols=usecols, nrows=max_rows, low_memory=False, na_values=["NA", "NaN", "nan"])
    if ext == ".parquet":
        df = pd.read_parquet(input_path, columns=usecols)
        if max_rows is not None:
            df = df.head(max_rows)
        return df
    raise RuntimeError(f"full analysis currently supports csv/tsv/parquet only, got {ext}")


def split_condition_columns(df: Any) -> Any:
    if pd is None or "condition" not in df.columns:
        return df
    parts = df["condition"].astype(str).str.split("_", n=1, expand=True)
    df["medication_state"] = parts[0]
    df["task_state"] = parts[1] if parts.shape[1] > 1 else ""
    return df


def finite_corr(x: Any, y: Any, method: str = "pearson") -> float:
    if np is None:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    if method == "spearman":
        if scipy_stats is not None:
            return float(scipy_stats.spearmanr(x, y).correlation)
        xr = pd.Series(x).rank().to_numpy() if pd is not None else rankdata_fallback(x)
        yr = pd.Series(y).rank().to_numpy() if pd is not None else rankdata_fallback(y)
        return float(np.corrcoef(xr, yr)[0, 1])
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_fallback(x: Any) -> Any:
    if np is None:
        return x
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def ridge_fit_predict(
    train_x: Any,
    train_y: Any,
    test_x: Any,
    ridge_lambda: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    if np is None:
        raise RuntimeError(f"numpy unavailable: {NUMPY_ERROR}")
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    xtr = np.nan_to_num((train_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    xte = np.nan_to_num((test_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    y_mean = float(np.nanmean(train_y))
    y_center = train_y - y_mean
    xtx = xtr.T @ xtr
    xty = xtr.T @ y_center
    reg = ridge_lambda * np.eye(xtx.shape[0])
    try:
        beta = np.linalg.solve(xtx + reg, xty)
    except Exception:
        beta = np.linalg.pinv(xtx + reg) @ xty
    raw = y_mean + xte @ beta
    return np.clip(raw, 0.0, 1.0), {"train_mean_label": y_mean, "ridge_lambda": ridge_lambda}


def direct_beta_score(train_df: Any, test_df: Any, label_col: str, beta_cols: list[str]) -> tuple[Any, str]:
    if np is None or not beta_cols:
        return np.full(len(test_df), float(train_df[label_col].mean())), ""
    best_col = ""
    best_abs = -1.0
    for col in beta_cols:
        corr = abs(finite_corr(train_df[col].to_numpy(), train_df[label_col].to_numpy(), "pearson"))
        if math.isfinite(corr) and corr > best_abs:
            best_abs = corr
            best_col = col
    if not best_col:
        return np.full(len(test_df), float(train_df[label_col].mean())), ""
    x = train_df[best_col].to_numpy(dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not math.isfinite(mad) or mad == 0:
        mad = np.nanstd(x)
    if not math.isfinite(mad) or mad == 0:
        mad = 1.0
    z = (test_df[best_col].to_numpy(dtype=float) - med) / mad
    score = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return score, best_col


def leaky_burden_for_df(df: Any, value_col: str, group_cols: list[str], time_col: str, tau_ms: float) -> Any:
    if np is None or pd is None:
        return []
    out = np.zeros(len(df), dtype=float)
    tau_s = tau_ms / 1000.0
    for _, idx in df.groupby(group_cols, dropna=False, sort=False).groups.items():
        positions = np.asarray(idx)
        sub = df.loc[positions]
        order = np.argsort(sub[time_col].to_numpy(dtype=float), kind="mergesort") if time_col in sub.columns else np.arange(len(sub))
        sorted_pos = positions[order]
        values = df.loc[sorted_pos, value_col].to_numpy(dtype=float)
        times = df.loc[sorted_pos, time_col].to_numpy(dtype=float) if time_col in df.columns else np.arange(len(sorted_pos), dtype=float)
        burden = np.zeros(len(sorted_pos), dtype=float)
        prev = 0.0
        last_time = times[0] if len(times) else 0.0
        for i, value in enumerate(values):
            if i == 0:
                dt_s = max(float(np.nanmedian(np.diff(times[np.isfinite(times)]))) if len(times) > 2 else 0.8, 1e-6)
            else:
                dt_s = float(times[i] - last_time)
                if not math.isfinite(dt_s) or dt_s <= 0:
                    dt_s = 0.8
            alpha = 1.0 - math.exp(-dt_s / tau_s)
            prev = prev + alpha * (float(value) - prev)
            burden[i] = prev
            last_time = times[i]
        out[sorted_pos] = burden
    return out


def metric_values(y_burden: Any, pred_burden: Any, y_binary: Any, pred_score: Any, high_threshold: float, pred_threshold: float) -> dict[str, Any]:
    if np is None:
        return {}
    yb = np.asarray(y_burden, dtype=float)
    pb = np.asarray(pred_burden, dtype=float)
    y = np.asarray(y_binary, dtype=float)
    score = np.asarray(pred_score, dtype=float)
    mask = np.isfinite(yb) & np.isfinite(pb)
    rmse = float(np.sqrt(np.nanmean((pb[mask] - yb[mask]) ** 2))) if mask.any() else float("nan")
    mae = float(np.nanmean(np.abs(pb[mask] - yb[mask]))) if mask.any() else float("nan")
    out = {
        "pearson": finite_corr(yb, pb, "pearson"),
        "spearman": finite_corr(yb, pb, "spearman"),
        "rmse": rmse,
        "mae": mae,
        "brier_score": float(np.mean((np.clip(score, 0, 1) - y) ** 2)) if len(y) else float("nan"),
        "auc_row_score": safe_auc(y, score),
        "auprc_row_score": safe_auprc(y, score),
        "high_burden_threshold": high_threshold,
        "predicted_high_burden_threshold": pred_threshold,
    }
    y_high = (yb >= high_threshold).astype(int)
    p_high = (pb >= pred_threshold).astype(int)
    out["auc_high_burden"] = safe_auc(y_high, pb)
    out["auprc_high_burden"] = safe_auprc(y_high, pb)
    if len(np.unique(y_high)) == 2:
        if balanced_accuracy_score is not None:
            out["balanced_accuracy_high_burden"] = float(balanced_accuracy_score(y_high, p_high))
        else:
            out["balanced_accuracy_high_burden"] = balanced_accuracy_fallback(y_high, p_high)
    else:
        out["balanced_accuracy_high_burden"] = float("nan")
    slope, intercept = calibration_slope_intercept(pb, yb)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    return out


def safe_auc(y: Any, score: Any) -> float:
    if np is None:
        return float("nan")
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(y, score))
        except Exception:
            return float("nan")
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def safe_auprc(y: Any, score: Any) -> float:
    if np is None:
        return float("nan")
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    if average_precision_score is not None:
        try:
            return float(average_precision_score(y, score))
        except Exception:
            return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall_delta = (y_sorted == 1) / max((y == 1).sum(), 1)
    return float(np.sum(precision * recall_delta))


def balanced_accuracy_fallback(y: Any, pred: Any) -> float:
    if np is None:
        return float("nan")
    y = np.asarray(y)
    pred = np.asarray(pred)
    vals = []
    for cls in [0, 1]:
        mask = y == cls
        if mask.any():
            vals.append(float((pred[mask] == cls).mean()))
    return float(np.mean(vals)) if vals else float("nan")


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


def summarize_iqr(values: list[float]) -> tuple[float, float, float]:
    vals = [v for v in values if isinstance(v, (float, int)) and math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def run_loso_analysis(
    df: Any,
    cols: Columns,
    selected_features: list[str],
    tau_values: list[float],
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if np is None or pd is None:
        raise RuntimeError("numpy/pandas required for full analysis")
    rng = np.random.default_rng(random_seed)
    df = split_condition_columns(df.copy())
    label = cols.label
    time_col = cols.time
    group_cols = [c for c in [cols.subject, cols.session, cols.block, "channel", "band_mode"] if c and c in df.columns]
    if cols.block == "condition" and "condition" not in group_cols:
        group_cols.append("condition")
    subjects = sorted(df[cols.subject].dropna().astype(str).unique())
    all_metric_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    separability_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    pred_cols_by_tau: dict[float, str] = {}
    anno_cols_by_tau: dict[float, str] = {}
    model_score = np.full(len(df), np.nan)
    prior_score = np.full(len(df), np.nan)
    shuffled_score = np.full(len(df), np.nan)
    beta_score = np.full(len(df), np.nan)
    beta_cols = [c for c in selected_features if "beta" in c.lower()]
    fold_notes: dict[str, str] = {}

    y = df[label].astype(float).to_numpy()
    X = df[selected_features].apply(pd.to_numeric, errors="coerce")
    for subject in subjects:
        test_mask = df[cols.subject].astype(str).to_numpy() == subject
        train_mask = ~test_mask
        train_y = y[train_mask]
        test_y = y[test_mask]
        if len(np.unique(train_y[~np.isnan(train_y)])) < 2 or len(np.unique(test_y[~np.isnan(test_y)])) < 2:
            fold_notes[subject] = "fold has degenerate train or test labels"
        pred, _ = ridge_fit_predict(X.loc[train_mask].to_numpy(), train_y, X.loc[test_mask].to_numpy())
        model_score[test_mask] = pred
        shuffled_y = train_y.copy()
        rng.shuffle(shuffled_y)
        shuf_pred, _ = ridge_fit_predict(X.loc[train_mask].to_numpy(), shuffled_y, X.loc[test_mask].to_numpy())
        shuffled_score[test_mask] = shuf_pred
        beta_pred, beta_feature = direct_beta_score(df.loc[train_mask], df.loc[test_mask], label, beta_cols)
        beta_score[test_mask] = beta_pred
        if beta_feature:
            fold_notes[subject] = clean_join([fold_notes.get(subject, ""), f"beta_baseline_feature={beta_feature}"])
        prior_score[test_mask] = float(np.nanmean(train_y))

    df["_phase6a0_model_score"] = model_score
    df["_phase6a0_class_prior_score"] = prior_score
    df["_phase6a0_shuffled_score"] = shuffled_score
    df["_phase6a0_beta_score"] = beta_score

    for tau in tau_values:
        anno_col = f"_phase6a0_annotation_burden_tau_{int(tau)}"
        pred_col = f"_phase6a0_predicted_burden_tau_{int(tau)}"
        prior_col = f"_phase6a0_prior_burden_tau_{int(tau)}"
        shuffled_col = f"_phase6a0_shuffled_burden_tau_{int(tau)}"
        beta_col = f"_phase6a0_beta_burden_tau_{int(tau)}"
        df[anno_col] = leaky_burden_for_df(df, label, group_cols, time_col, tau)
        df[pred_col] = leaky_burden_for_df(df, "_phase6a0_model_score", group_cols, time_col, tau)
        df[prior_col] = leaky_burden_for_df(df, "_phase6a0_class_prior_score", group_cols, time_col, tau)
        df[shuffled_col] = leaky_burden_for_df(df, "_phase6a0_shuffled_score", group_cols, time_col, tau)
        df[beta_col] = leaky_burden_for_df(df, "_phase6a0_beta_score", group_cols, time_col, tau)
        pred_cols_by_tau[tau] = pred_col
        anno_cols_by_tau[tau] = anno_col

        train_thresholds: dict[str, tuple[float, float]] = {}
        for subject in subjects:
            train_df = df[df[cols.subject].astype(str) != subject]
            high_thr = float(np.nanpercentile(train_df[anno_col], 75))
            pred_thr = float(np.nanpercentile(train_df[pred_col], 75))
            train_thresholds[subject] = (high_thr, pred_thr)
        for subject in subjects:
            sub = df[df[cols.subject].astype(str) == subject]
            high_thr, pred_thr = train_thresholds[subject]
            metrics = metric_values(
                sub[anno_col].to_numpy(),
                sub[pred_col].to_numpy(),
                sub[label].to_numpy(),
                sub["_phase6a0_model_score"].to_numpy(),
                high_thr,
                pred_thr,
            )
            row = {
                "subject_id": subject,
                "tau_ms": tau,
                "n_windows": len(sub),
                "n_positive_labels": int(sub[label].sum()),
                "positive_label_fraction": float(sub[label].mean()),
                "timing_confidence": "high" if time_col else "low",
                "notes": fold_notes.get(subject, ""),
            }
            row.update(metrics)
            all_metric_rows.append(row)

        for baseline_name, score_col, burden_col in [
            ("class_prior_baseline", "_phase6a0_class_prior_score", prior_col),
            ("shuffled_label_baseline", "_phase6a0_shuffled_score", shuffled_col),
            ("simple_beta_feature_baseline", "_phase6a0_beta_score", beta_col),
            ("model_based_burden_estimate", "_phase6a0_model_score", pred_col),
        ]:
            per_subject = []
            for subject in subjects:
                sub = df[df[cols.subject].astype(str) == subject]
                per_subject.append(finite_corr(sub[anno_col].to_numpy(), sub[burden_col].to_numpy(), "spearman"))
            median, q1, q3 = summarize_iqr(per_subject)
            baseline_rows.append(
                {
                    "baseline_name": baseline_name,
                    "tau_ms": tau,
                    "median_spearman": median,
                    "iqr25_spearman": q1,
                    "iqr75_spearman": q3,
                    "n_valid_subjects": sum(1 for v in per_subject if isinstance(v, (float, int)) and math.isfinite(float(v))),
                    "notes": "subject-level median; no pooled-window gate",
                }
            )

    best_tau = select_best_tau(all_metric_rows)
    if best_tau is None:
        best_tau = tau_values[0]
    best_pred_col = pred_cols_by_tau[best_tau]
    best_anno_col = anno_cols_by_tau[best_tau]

    condition_group_cols = [cols.subject]
    for c in ["condition", "medication_state", "task_state"]:
        if c in df.columns:
            condition_group_cols.append(c)
    grouped = df.groupby(condition_group_cols, dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(condition_group_cols, keys))
        condition_rows.append(
            {
                **key_map,
                "tau_ms": best_tau,
                "n_windows": len(sub),
                "mean_predicted_burden": float(sub[best_pred_col].mean()),
                "mean_annotation_burden": float(sub[best_anno_col].mean()),
                "positive_label_fraction": float(sub[label].mean()),
            }
        )

    condition_df = pd.DataFrame(condition_rows)
    if "medication_state" in condition_df.columns and condition_df["medication_state"].nunique(dropna=True) >= 2:
        separability_rows.extend(condition_separability(condition_df, "medication_state", "MedOff", "MedOn", "mean_predicted_burden"))
    if "task_state" in condition_df.columns and condition_df["task_state"].nunique(dropna=True) >= 2:
        for a, b in [("Hold", "Move"), ("Rest", "Move"), ("Rest", "Hold")]:
            if a in set(condition_df["task_state"]) and b in set(condition_df["task_state"]):
                separability_rows.extend(condition_separability(condition_df, "task_state", a, b, "mean_predicted_burden"))

    clinical_cols = [c for c in df.columns if any(k in c.lower() for k in ["updrs", "brady", "tremor", "rigidity"])]
    if not clinical_cols:
        proxy_rows.append(
            {
                "proxy_name": "clinical_scores",
                "status": "not_assessable",
                "n_units": 0,
                "spearman": "",
                "pearson": "",
                "notes": "No UPDRS/bradykinesia/tremor/rigidity score columns in selected table.",
            }
        )

    analysis_state = {
        "best_tau": best_tau,
        "subjects": subjects,
        "dataframe_for_plots": df,
        "best_pred_col": best_pred_col,
        "best_anno_col": best_anno_col,
    }
    return all_metric_rows, condition_rows, separability_rows, proxy_rows, baseline_rows, analysis_state


def select_best_tau(metric_rows: list[dict[str, Any]]) -> float | None:
    by_tau: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_tau[float(row["tau_ms"])].append(row)
    best_tau = None
    best_value = -float("inf")
    for tau, rows in by_tau.items():
        p, _, _ = summarize_iqr([float(r.get("pearson", float("nan"))) for r in rows])
        s, _, _ = summarize_iqr([float(r.get("spearman", float("nan"))) for r in rows])
        value = max(p if math.isfinite(p) else -float("inf"), s if math.isfinite(s) else -float("inf"))
        if value > best_value:
            best_value = value
            best_tau = tau
    return best_tau


def condition_separability(df: Any, col: str, positive_name: str, negative_name: str, value_col: str) -> list[dict[str, Any]]:
    if np is None or pd is None:
        return []
    subset = df[df[col].astype(str).isin([positive_name, negative_name])].copy()
    if subset.empty:
        return []
    y = (subset[col].astype(str) == positive_name).astype(int).to_numpy()
    x = subset[value_col].to_numpy(dtype=float)
    pos = subset[subset[col].astype(str) == positive_name][value_col].to_numpy(dtype=float)
    neg = subset[subset[col].astype(str) == negative_name][value_col].to_numpy(dtype=float)
    if len(pos) < 2 or len(neg) < 2:
        d = float("nan")
    else:
        pooled = math.sqrt((np.nanvar(pos, ddof=1) + np.nanvar(neg, ddof=1)) / 2)
        d = float((np.nanmean(pos) - np.nanmean(neg)) / pooled) if pooled else float("nan")
    return [
        {
            "proxy_type": col,
            "positive_condition": positive_name,
            "negative_condition": negative_name,
            "value_column": value_col,
            "n_units": len(subset),
            "positive_mean": float(np.nanmean(pos)) if len(pos) else float("nan"),
            "negative_mean": float(np.nanmean(neg)) if len(neg) else float("nan"),
            "cohens_d": d,
            "rank_biserial_or_auc": safe_auc(y, x),
            "notes": "subject-condition units; supportive proxy only",
        }
    ]


def summarize_tau(metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[float, float], dict[float, float]]:
    by_tau = defaultdict(list)
    for row in metric_rows:
        by_tau[float(row["tau_ms"])].append(row)
    rows = []
    med_p = {}
    med_s = {}
    for tau, items in sorted(by_tau.items()):
        p_med, p_q1, p_q3 = summarize_iqr([float(r.get("pearson", float("nan"))) for r in items])
        s_med, s_q1, s_q3 = summarize_iqr([float(r.get("spearman", float("nan"))) for r in items])
        rmse_med, _, _ = summarize_iqr([float(r.get("rmse", float("nan"))) for r in items])
        med_p[tau] = p_med
        med_s[tau] = s_med
        valid = sum(1 for r in items if math.isfinite(float(r.get("pearson", float("nan")))) or math.isfinite(float(r.get("spearman", float("nan")))))
        rows.append(
            {
                "tau_ms": tau,
                "median_pearson": p_med,
                "iqr25_pearson": p_q1,
                "iqr75_pearson": p_q3,
                "median_spearman": s_med,
                "iqr25_spearman": s_q1,
                "iqr75_spearman": s_q3,
                "median_rmse": rmse_med,
                "n_valid_subjects": valid,
                "n_failed_or_invalid_subjects": len(items) - valid,
                "notes": "subject-level median/IQR; windows not treated as independent",
            }
        )
    return rows, med_p, med_s


def audit_feature_columns(columns: list[str], allowed: list[str], selected: list[str], identity: Columns, metadata: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_set = set(allowed)
    selected_set = set(selected)
    feature_rows = []
    leakage_rows = []
    for col in columns:
        role, reason = classify_column(col, selected_set, allowed_set, identity)
        meta = metadata.get(col, {})
        row = {
            "column_name": col,
            "column_role": role,
            "selected_for_model": bool_cell(col in selected_set),
            "feature_family": meta.get("feature_family", ""),
            "support_status": meta.get("support_status", ""),
            "qc_status": meta.get("qc_status", ""),
            "uses_future_samples": meta.get("uses_future_samples", ""),
            "uses_test_participant_statistics": meta.get("uses_test_participant_statistics", ""),
            "tautology_risk_level": meta.get("tautology_risk_level", ""),
            "leakage_risk_level": meta.get("leakage_risk_level", ""),
            "causal_frontend_confirmed": meta.get("causal_frontend_confirmed", ""),
            "reason": reason,
        }
        feature_rows.append(row)
        if role == "suspicious leakage columns" or str(meta.get("leakage_risk_level", "")).lower() not in {"", "low"} or str(meta.get("uses_future_samples", "")).lower() == "true":
            severity = "high" if "future" in reason or str(meta.get("uses_future_samples", "")).lower() == "true" else "medium"
            leakage_rows.append(
                {
                    "column_name": col,
                    "risk_type": role,
                    "severity": severity,
                    "evidence": reason,
                    "action": "excluded from model features",
                }
            )
    return feature_rows, leakage_rows


def metadata_proxy_audit(df: Any, cols: Columns) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pd is None or df is None or len(df) == 0:
        return rows
    for col in ["condition", "medication_state", "task_state", "channel", "band_mode"]:
        if col in df.columns:
            vals = df[col].astype(str).value_counts(dropna=False).head(30)
            rows.append(
                {
                    "proxy_column": col,
                    "proxy_type": "task_or_metadata",
                    "n_unique": int(df[col].nunique(dropna=False)),
                    "top_values": clean_join([f"{idx}:{val}" for idx, val in vals.items()]),
                    "status": "available",
                    "notes": "supportive proxy only; not primary gate",
                }
            )
    clinical_cols = [c for c in df.columns if any(k in c.lower() for k in ["updrs", "brady", "tremor", "rigidity"])]
    if clinical_cols:
        for col in clinical_cols:
            rows.append(
                {
                    "proxy_column": col,
                    "proxy_type": "clinical",
                    "n_unique": int(df[col].nunique(dropna=True)),
                    "top_values": "",
                    "status": "available",
                    "notes": "correlation only at subject/session level if usable",
                }
            )
    else:
        rows.append(
            {
                "proxy_column": "clinical_scores",
                "proxy_type": "clinical",
                "n_unique": 0,
                "top_values": "",
                "status": "not_assessable",
                "notes": "No UPDRS/bradykinesia/tremor/rigidity columns in selected input.",
            }
        )
    return rows


def prior_phase_comparison(repo_root: Path) -> list[dict[str, Any]]:
    roots = [repo_root / "docs", repo_root / "reports", repo_root / "results/tables/05_phase5"]
    patterns = [
        ("Phase 5Y burden correlation", re.compile(r"(Phase 5Y|phase5y|burden).*?(r\s*[=≈]\s*0\.\d+|0\.334)", re.IGNORECASE)),
        ("Phase 5_2C event recall / FP-min", re.compile(r"(Phase 5_2C|phase5_2c|FP/min|recall collapsed|event recovery gap|low-FP)", re.IGNORECASE)),
        ("Phase 5_2B AUROC/subset", re.compile(r"(Phase 5_2B|phase5_2b|AUROC|AUPRC|LOSO)", re.IGNORECASE)),
        ("burden pivot ADR", re.compile(r"(burden pivot|burden/state|state tracking|reconsider burden)", re.IGNORECASE)),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.stat().st_size > 5_000_000 or is_excluded_external_path(relpath(path, repo_root)):
                continue
            if path.suffix.lower() not in {".md", ".txt", ".tsv", ".csv", ".json"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for metric, regex in patterns:
                match = regex.search(text)
                if match:
                    key = (metric, relpath(path, repo_root))
                    if key in seen:
                        continue
                    seen.add(key)
                    excerpt = re.sub(r"\s+", " ", text[max(match.start() - 120, 0) : match.end() + 180]).strip()
                    value_match = re.search(r"(r\s*[=≈]\s*-?\d+\.\d+|-?\d+\.\d+|AUROC\s*[=≈:]?\s*\d+\.\d+|AUPRC\s*[=≈:]?\s*\d+\.\d+)", excerpt, re.IGNORECASE)
                    rows.append(
                        {
                            "prior_phase": metric.split()[0] + " " + metric.split()[1],
                            "metric": metric,
                            "value": value_match.group(0) if value_match else "not_numeric_in_excerpt",
                            "source_file": relpath(path, repo_root),
                            "interpretation": excerpt[:500],
                        }
                    )
    if not rows:
        rows.append(
            {
                "prior_phase": "not_found",
                "metric": "prior Phase 5Y/5_2B/5_2C comparison",
                "value": "not found",
                "source_file": "",
                "interpretation": "No matching prior-phase comparison phrases found by heuristic text search.",
            }
        )
    return rows[:80]


def gate_decision(
    tau_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    separability_rows: list[dict[str, Any]],
    selected_subject_count: int,
    allowed_feature_count: int,
    leakage_rows: list[dict[str, Any]],
) -> tuple[str, str, list[str], list[str], float, float, float]:
    critical: list[str] = []
    limitations: list[str] = []
    severe_leakage = [r for r in leakage_rows if r.get("severity") == "high" and r.get("action") != "excluded from model features"]
    if selected_subject_count < MIN_REQUIRED_SUBJECTS:
        critical.append(f"Only {selected_subject_count} subjects available; at least {MIN_REQUIRED_SUBJECTS} required.")
    if allowed_feature_count < MIN_REQUIRED_FEATURES:
        critical.append(f"Only {allowed_feature_count} allowed causal features; at least {MIN_REQUIRED_FEATURES} required.")
    best = None
    best_score = -float("inf")
    for row in tau_rows:
        p = float(row.get("median_pearson", float("nan")))
        s = float(row.get("median_spearman", float("nan")))
        score = max(p if math.isfinite(p) else -float("inf"), s if math.isfinite(s) else -float("inf"))
        if score > best_score:
            best = row
            best_score = score
    if not best:
        critical.append("No valid tau results.")
        return "BLOCKED", "No valid annotation-derived burden metrics were computed.", critical, limitations, float("nan"), float("nan"), float("nan")
    best_tau = float(best["tau_ms"])
    best_p = float(best["median_pearson"])
    best_s = float(best["median_spearman"])
    valid_subjects = int(best["n_valid_subjects"])
    model_rows = [r for r in baseline_rows if r["baseline_name"] == "model_based_burden_estimate" and float(r["tau_ms"]) == best_tau]
    prior_rows = [r for r in baseline_rows if r["baseline_name"] == "class_prior_baseline" and float(r["tau_ms"]) == best_tau]
    shuf_rows = [r for r in baseline_rows if r["baseline_name"] == "shuffled_label_baseline" and float(r["tau_ms"]) == best_tau]
    model_med = float(model_rows[0]["median_spearman"]) if model_rows else float("nan")
    prior_med = float(prior_rows[0]["median_spearman"]) if prior_rows else float("nan")
    shuf_med = float(shuf_rows[0]["median_spearman"]) if shuf_rows else float("nan")
    beats_baseline = math.isfinite(model_med) and (
        (not math.isfinite(prior_med) or model_med > prior_med + 0.05)
        and (not math.isfinite(shuf_med) or model_med > shuf_med + 0.05)
    )
    proxy_positive = any(
        math.isfinite(float(row.get("rank_biserial_or_auc", float("nan"))))
        and abs(float(row.get("rank_biserial_or_auc", 0.5)) - 0.5) >= 0.05
        for row in separability_rows
    )
    proxy_available = bool(separability_rows)
    if critical:
        return "BLOCKED", clean_join(critical), critical, limitations, best_tau, best_p, best_s
    if severe_leakage:
        critical.append("Severe leakage risk in selected model features.")
        return "fail_or_blocked_due_to_leakage_risk", "Severe leakage risk prevents valid evaluation.", critical, limitations, best_tau, best_p, best_s
    if not beats_baseline:
        critical.append("Model-based burden estimate does not clearly beat class-prior and shuffled-label baselines.")
        return "FAIL", clean_join(critical), critical, limitations, best_tau, best_p, best_s
    if valid_subjects >= 10 and max(best_p, best_s) >= 0.50 and (proxy_positive or not proxy_available):
        return "PASS", "At least 10 valid subjects, median LOSO correlation >= 0.50, and baselines beaten.", critical, limitations, best_tau, best_p, best_s
    if valid_subjects >= 8 and 0.30 <= max(best_p, best_s) < 0.50 and (proxy_positive or not proxy_available):
        return "CONDITIONAL_PASS", "At least 8 valid subjects, median LOSO correlation is 0.30-0.50, and baselines beaten.", critical, limitations, best_tau, best_p, best_s
    if max(best_p, best_s) <= 0.20:
        critical.append("Median LOSO burden correlation is <= 0.20.")
    else:
        critical.append("Gate thresholds for PASS/CONDITIONAL_PASS were not met.")
    return "FAIL", clean_join(critical), critical, limitations, best_tau, best_p, best_s


def write_readme(
    path: Path,
    findings: dict[str, Any],
    discovery_rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
    tau_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    separability_rows: list[dict[str, Any]],
    commands: list[str],
) -> None:
    lines = [
        "# Phase 6A.0 Burden-Target Viability Gate",
        "",
        "## Purpose And Corrected Phase 6 Framing",
        "",
        "Phase 6 starts with an internal STN-LFP burden/state viability gate. The claim being tested is a neuromorphic approximation of a clinically relevant STN-LFP burden/state-tracking policy, not clinical efficacy or FDA-grade validity.",
        "",
        "## Why Not PPN Or Herz First",
        "",
        "The He/Tan PPN dataset is an optional cross-target extension only after the primary STN architecture works. The Herz/Groppa/Brown force-adaptation package is a methods/code reference with minimum example data; it is not used as a primary STN burden-validation input. External PPN/Herz paths were explicitly excluded from candidate input selection.",
        "",
        "## Dataset/Input Files Discovered And Selected",
        "",
        f"- Selected input recommendation: `{recommendation.get('selected_input_path') or 'none'}`",
        f"- Selection confidence: `{recommendation.get('selection_confidence')}`",
        f"- Selected input files used: `{clean_join(findings.get('selected_input_files', [])) or 'none'}`",
        f"- Discovery candidates recorded: `{len(discovery_rows)}`",
        "",
        "Top discovery candidates:",
        "",
        "| path | score | use | notes |",
        "| --- | --- | --- | --- |",
    ]
    for row in discovery_rows[:12]:
        lines.append(f"| `{row['relative_path']}` | {row['candidate_score']} | {row['likely_use']} | {row['notes'] or row['acceptance_reasons']} |")
    lines.extend(
        [
            "",
            "## Environment And Commands",
            "",
            f"- Python: `{findings.get('environment_python')}`",
            "- Environment route: `source /scratch/haizhe/stn/start_stn.sh && python ...`",
            "",
            "## Feature And Leakage Audit",
            "",
            f"- Allowed causal feature count: `{findings.get('allowed_feature_count')}`",
            f"- Excluded/leakage feature count: `{findings.get('excluded_feature_count')}`",
            f"- Leakage risk summary: `{findings.get('leakage_risk_summary')}`",
            "- Feature names suggesting labels, targets, future/post windows, split/fold, onset/offset, oracle outputs, file paths, or identity fields were excluded.",
            "- Any precomputed global or subject-level normalization suspicion is documented in `leakage_risk_audit.csv`; training normalization for the model is fitted on training subjects only.",
            "",
            "## Burden Target Construction",
            "",
            "Annotation-derived burden uses the binary label column and a causal leaky integrator: `burden_t = burden_{t-1} + alpha * (label_t - burden_{t-1})`, `alpha = 1 - exp(-dt / tau)`. Time deltas are estimated from the selected time/order column within subject/session/condition/channel streams. No centered windows or future labels are used.",
            "",
            "## Model And LOSO Validation",
            "",
            "The primary model is a deterministic subject-held-out ridge least-squares probability scorer on selected allowed causal features. Features are standardized using training subjects only, then applied to the held-out subject. Gate metrics are summarized per subject first, then by subject-level median/IQR.",
            "",
            "## Tau Sweep Results",
            "",
            "| tau_ms | median_pearson | median_spearman | valid_subjects |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in tau_rows:
        lines.append(f"| {row.get('tau_ms')} | {row.get('median_pearson')} | {row.get('median_spearman')} | {row.get('n_valid_subjects')} |")
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            "| baseline | tau_ms | median_spearman | notes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in baseline_rows:
        if row.get("tau_ms") == findings.get("best_tau_ms"):
            lines.append(f"| {row.get('baseline_name')} | {row.get('tau_ms')} | {row.get('median_spearman')} | {row.get('notes')} |")
    lines.extend(
        [
            "",
            "## Clinical/Task Proxy Results",
            "",
        ]
    )
    if separability_rows:
        lines.extend(["| proxy | positive | negative | AUROC/rank-biserial | Cohen d |", "| --- | --- | --- | --- | --- |"])
        for row in separability_rows:
            lines.append(f"| {row.get('proxy_type')} | {row.get('positive_condition')} | {row.get('negative_condition')} | {row.get('rank_biserial_or_auc')} | {row.get('cohens_d')} |")
    else:
        lines.append("Clinical/task proxies were not assessable or not separable from the selected input. This does not by itself fail the gate; the primary gate is annotation-derived burden tracking.")
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            f"- Overall status: `{findings.get('overall_status')}`",
            f"- Gate rationale: {findings.get('gate_rationale')}",
            "",
            "## Interpretation",
            "",
            "This is an internal STN-substrate technical viability result. It is not clinical validation, not PKG validation, and not evidence of clinical efficacy.",
            "",
            "## Next Actions",
            "",
        ]
    )
    for step in findings.get("recommended_next_steps", []):
        lines.append(f"- {step}")
    lines.extend(
        [
            "",
            "If the gate passes, proceed to Phase 6A Brian2/state-machine simulation on this internal STN substrate. If it conditionally passes, add ablations and robustness checks before simulation. If it fails, stop and reconsider the burden pivot. If blocked, provide the required internal STN feature/label table columns.",
            "",
            "## PPN And Herz Dataset Handling",
            "",
            "PPN remains optional cross-target generalization only after the primary STN burden architecture works. Herz remains a methods/code reference only.",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in findings.get("noncritical_limitations", []):
        lines.append(f"- {item}")
    if findings.get("critical_blockers"):
        for item in findings.get("critical_blockers", []):
            lines.append(f"- Critical blocker: {item}")
    lines.extend(["", "## Exact Commands Run", ""])
    for command in commands:
        lines.append(f"- `{command}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_adr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# ADR: Corrected Phase 6 Strategy",
                "",
                "## Status",
                "",
                "Accepted for Phase 6 start.",
                "",
                "## Decision",
                "",
                "The primary Phase 6 path is: existing internal STN dataset -> burden viability gate -> Brian2/state-machine simulation -> DYNAP/hardware-ready demonstration.",
                "",
                "The He/Tan PPN dataset is optional cross-target extension only after the primary STN architecture passes. The Herz/Groppa/Brown force-adaptation package is a methods/code reference only.",
                "",
                "Phase 6 must not claim FDA-grade validity, FDA validation, clinical efficacy, equivalence to commercial sensing systems, or therapeutic superiority.",
                "",
                "Allowed framing: neuromorphic implementation/evaluation of a clinically relevant STN-LFP burden/state-tracking policy and hardware-compatible evaluation of physiologically motivated aDBS-style control substrate.",
                "",
                "## Gate Outcomes",
                "",
                "- PASS: proceed to Phase 6A Brian2/state-machine simulation.",
                "- CONDITIONAL_PASS: proceed only with explicit limitations and added ablations before simulation code expansion.",
                "- FAIL: stop and reconsider the burden pivot.",
                "- BLOCKED: provide the required internal STN feature/label substrate before Phase 6 can continue.",
                "",
                "## Longer-Horizon Workstream",
                "",
                "Clinical, kinematic, PKG, or therapeutic validation is a separate workstream if a proper STN kinematic/clinical cohort becomes available.",
                "",
                "## Consequences",
                "",
                "Phase 6A.0 is the first corrected Phase 6 deliverable and controls whether Brian2/DYNAP-style work is justified.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def plot_outputs(out_dir: Path, tau_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], analysis_state: dict[str, Any], condition_rows: list[dict[str, Any]], no_plots: bool) -> list[str]:
    if no_plots or plt is None or pd is None:
        return []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    try:
        taus = [float(r["tau_ms"]) for r in tau_rows]
        medp = [float(r["median_pearson"]) for r in tau_rows]
        meds = [float(r["median_spearman"]) for r in tau_rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(taus, medp, marker="o", label="Pearson")
        ax.plot(taus, meds, marker="o", label="Spearman")
        ax.set_xlabel("Tau (ms)")
        ax.set_ylabel("Median subject correlation")
        ax.legend()
        ax.set_title("Tau Sweep Median Correlation")
        fig.tight_layout()
        path = fig_dir / "tau_sweep_median_correlation.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(path.as_posix())
    except Exception as exc:
        LOG.warning("plot failed: %s", exc)
    try:
        best_tau = analysis_state.get("best_tau")
        rows = [r for r in metric_rows if float(r["tau_ms"]) == float(best_tau)]
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [r["subject_id"] for r in rows]
        vals = [float(r["spearman"]) for r in rows]
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("Spearman")
        ax.set_title("Per-Subject Burden Correlation")
        fig.tight_layout()
        path = fig_dir / "per_subject_burden_correlation.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(path.as_posix())
    except Exception as exc:
        LOG.warning("plot failed: %s", exc)
    try:
        df = analysis_state.get("dataframe_for_plots")
        pred_col = analysis_state.get("best_pred_col")
        anno_col = analysis_state.get("best_anno_col")
        rows = [r for r in metric_rows if float(r["tau_ms"]) == float(analysis_state.get("best_tau")) and math.isfinite(float(r.get("spearman", float("nan"))))]
        sorted_rows = sorted(rows, key=lambda r: float(r["spearman"]))
        if sorted_rows and df is not None:
            examples = [
                ("predicted_vs_annotation_burden_example_best_subject.png", sorted_rows[-1]["subject_id"]),
                ("predicted_vs_annotation_burden_example_median_subject.png", sorted_rows[len(sorted_rows) // 2]["subject_id"]),
            ]
            for filename, subject in examples:
                sub = df[df["subject_id"].astype(str) == str(subject)].sort_values("window_start_s").head(2000)
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(sub["window_start_s"], sub[anno_col], label="annotation burden", linewidth=1)
                ax.plot(sub["window_start_s"], sub[pred_col], label="predicted burden", linewidth=1)
                ax.set_title(subject)
                ax.legend()
                fig.tight_layout()
                path = fig_dir / filename
                fig.savefig(path, dpi=140)
                plt.close(fig)
                created.append(path.as_posix())
    except Exception as exc:
        LOG.warning("plot failed: %s", exc)
    try:
        best_tau = analysis_state.get("best_tau")
        rows = [r for r in baseline_rows if float(r["tau_ms"]) == float(best_tau)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([r["baseline_name"] for r in rows], [float(r["median_spearman"]) for r in rows])
        ax.set_ylabel("Median Spearman")
        ax.set_title("Baseline Comparison")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = fig_dir / "baseline_comparison.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(path.as_posix())
    except Exception as exc:
        LOG.warning("plot failed: %s", exc)
    if condition_rows:
        try:
            cond = pd.DataFrame(condition_rows)
            if "condition" in cond.columns:
                summ = cond.groupby("condition")["mean_predicted_burden"].mean().sort_values()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(summ.index.astype(str), summ.values)
                ax.set_ylabel("Mean predicted burden")
                ax.set_title("Condition Burden Summary")
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                path = fig_dir / "condition_burden_summary.png"
                fig.savefig(path, dpi=140)
                plt.close(fig)
                created.append(path.as_posix())
        except Exception as exc:
            LOG.warning("plot failed: %s", exc)
    return created


def commands_run() -> list[str]:
    return [
        "pwd",
        "git rev-parse --show-toplevel",
        "git status --short",
        "source /scratch/haizhe/stn/start_stn.sh && python -V",
        "find . -maxdepth 3 -type d | sort | head -300",
        "find . -maxdepth 3 -type f | sort | head -300",
        "find reports -maxdepth 4 -type f | sort | grep -E 'phase3|phase4|phase5|5_2|5Y|burden|burst|ADR|audit' | head -300 || true",
        "find data -maxdepth 5 -type f | sort | grep -E 'phase3|phase4|phase5|burst|burden|feature|label|stn|parquet|csv|json|pkl|npz|mat' | head -300 || true",
        "rg -n \"Phase 5_2B|Phase 5_2C|burden|AUROC|268,959|LOSO|MedOff|MedOn|Hold|Move|UPDRS|bradykinesia|burst\" reports scripts data config configs pyproject.toml README* 2>/dev/null | head -500 || true",
        "source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_burden_viability.py --out-dir reports/phase6a0_burden_viability --stop-after-discovery",
        "source /scratch/haizhe/stn/start_stn.sh && python -m py_compile scripts/phase6a0_burden_viability.py",
        "~/bin/claim_best_immediate_resource.sh --mode cpu \"cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_burden_viability.py --out-dir reports/phase6a0_burden_viability --tau-ms 200,500,800,1500,3000\"",
        "sinfo -o '%P %a %l %D %c %m %G' | head -80",
        "sacctmgr -nP show assoc user=$USER format=Account,Partition,QOS 2>/dev/null | head -80 || true",
        "sinfo -p teaching -o '%P %a %l %D %c %m %G' || true",
        "~/bin/claim_best_immediate_resource.sh --mode cpu --candidate \"--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00\" \"cd /scratch/haizhe/stn && source /scratch/haizhe/stn/start_stn.sh && python scripts/phase6a0_burden_viability.py --out-dir reports/phase6a0_burden_viability --tau-ms 200,500,800,1500,3000\"",
        "sacct -j 2577074 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW",
        "seff 2577074 2>/dev/null || true",
        "ls -lh reports/phase6a0_burden_viability",
        "head -100 reports/phase6a0_burden_viability/README_burden_viability.md",
        "python - <<'PY' ... print burden_viability_findings.json ... PY",
        "git diff --check",
        "find reports/phase6a0_burden_viability -type f -size +5M -print",
        "git status --short",
    ]


def empty_deliverables(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "feature_column_audit.csv", [])
    write_csv(out_dir / "metadata_proxy_audit.csv", [])
    write_csv(out_dir / "burden_tau_sweep.csv", [])
    write_csv(out_dir / "burden_metric_by_subject.csv", [])
    write_csv(out_dir / "burden_metric_by_condition.csv", [])
    write_csv(out_dir / "condition_separability.csv", [])
    write_csv(out_dir / "clinical_proxy_correlation.csv", [])
    write_csv(out_dir / "baseline_comparison.csv", [])
    write_csv(out_dir / "leakage_risk_audit.csv", [])
    write_csv(out_dir / "prior_phase_comparison.csv", [])


def main() -> int:
    setup_logging()
    args = parse_args()
    if np is None or pd is None:
        LOG.error("numpy/pandas unavailable; writing blocked report")
    repo_root = Path.cwd().resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    timestamp = now_iso()
    tau_values = parse_tau_ms(args.tau_ms)
    commands = commands_run()
    (out_dir / "phase6a0_commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_adr(repo_root / "reports/phase6_publication_strategy/ADR_phase6_corrected_strategy.md")

    discovery_rows, recommendation = discover_inputs(repo_root)
    write_csv(out_dir / "input_discovery.csv", discovery_rows)
    write_json(out_dir / "selected_input_recommendation.json", recommendation)
    prior_rows = prior_phase_comparison(repo_root)
    write_csv(out_dir / "prior_phase_comparison.csv", prior_rows)

    selected_rel = relpath(args.input_table.resolve(), repo_root) if args.input_table else recommendation.get("selected_input_path", "")
    selected_path = (repo_root / selected_rel).resolve() if selected_rel else None

    base_findings: dict[str, Any] = {
        "phase": "Phase 6A.0 burden-target viability gate",
        "repo_root": str(repo_root),
        "audit_timestamp": timestamp,
        "environment_python": f"{sys.executable} ({sys.version.split()[0]})",
        "selected_input_files": [selected_rel] if selected_rel else [],
        "selected_subject_count": 0,
        "valid_subject_count": 0,
        "target_label_column": args.label_col or "",
        "subject_column": args.subject_col or "",
        "time_column": args.time_col or "",
        "session_column": args.session_col or "",
        "allowed_feature_count": 0,
        "excluded_feature_count": 0,
        "leakage_risk_summary": "",
        "tau_ms_values": tau_values,
        "best_tau_ms": None,
        "median_pearson_by_tau": {},
        "median_spearman_by_tau": {},
        "best_tau_median_pearson": None,
        "best_tau_median_spearman": None,
        "baseline_summary": {},
        "condition_proxy_summary": {"status": "not_assessable"},
        "clinical_proxy_summary": {"status": "not_assessable"},
        "overall_status": "discovery_only" if args.stop_after_discovery else "BLOCKED",
        "gate_rationale": "",
        "critical_blockers": [],
        "noncritical_limitations": [],
        "recommended_next_steps": [],
    }

    if args.stop_after_discovery:
        empty_deliverables(out_dir)
        status = "blocked_no_existing_stn_feature_dataset_found" if recommendation.get("selection_confidence") in {"none", "low"} else "discovery_only_candidate_selected"
        base_findings["overall_status"] = status
        base_findings["gate_rationale"] = "Discovery pass only; modeling intentionally not run."
        base_findings["recommended_next_steps"] = [
            "Run full Phase 6A.0 analysis with the selected internal STN table or provide --input-table if selection confidence is not high.",
        ]
        write_json(out_dir / "burden_viability_findings.json", base_findings)
        write_readme(out_dir / "README_burden_viability.md", base_findings, discovery_rows, recommendation, [], [], [], commands)
        LOG.info("Discovery-only pass complete: %s", status)
        return 0

    if selected_path is None or not selected_path.exists():
        empty_deliverables(out_dir)
        base_findings["overall_status"] = "blocked_no_existing_stn_feature_dataset_found"
        base_findings["gate_rationale"] = "No usable internal STN feature/label table was selected."
        base_findings["critical_blockers"] = ["No selected input table."]
        base_findings["recommended_next_steps"] = [
            "Provide a window-level internal STN table with subject, time/order, binary label, and at least five causal feature columns.",
        ]
        write_json(out_dir / "burden_viability_findings.json", base_findings)
        write_readme(out_dir / "README_burden_viability.md", base_findings, discovery_rows, recommendation, [], [], [], commands)
        return 0

    if not args.input_table and recommendation.get("selection_confidence") not in {"high", "medium"}:
        empty_deliverables(out_dir)
        base_findings["overall_status"] = "blocked_ambiguous_input_candidates"
        base_findings["gate_rationale"] = "More than one plausible input or low confidence selection; explicit --input-table required."
        base_findings["critical_blockers"] = ["Ambiguous input candidates."]
        base_findings["recommended_next_steps"] = ["Rerun with explicit --input-table, --labels-table, and/or --metadata-table."]
        write_json(out_dir / "burden_viability_findings.json", base_findings)
        write_readme(out_dir / "README_burden_viability.md", base_findings, discovery_rows, recommendation, [], [], [], commands)
        return 0

    if is_excluded_external_path(relpath(selected_path, repo_root)):
        empty_deliverables(out_dir)
        base_findings["overall_status"] = "blocked_external_dataset_input_excluded"
        base_findings["gate_rationale"] = "Selected input violates external PPN/Herz exclusion rule."
        base_findings["critical_blockers"] = ["External PPN/Herz path cannot be used as Phase 6A.0 input."]
        write_json(out_dir / "burden_viability_findings.json", base_findings)
        write_readme(out_dir / "README_burden_viability.md", base_findings, discovery_rows, recommendation, [], [], [], commands)
        return 0

    columns, sample, readable, notes = read_header_and_sample(selected_path, max_rows=1000)
    if not columns:
        empty_deliverables(out_dir)
        base_findings["overall_status"] = "blocked_missing_required_columns"
        base_findings["gate_rationale"] = f"Selected table schema unreadable: {readable} {notes}"
        base_findings["critical_blockers"] = ["Selected input schema unreadable."]
        write_json(out_dir / "burden_viability_findings.json", base_findings)
        write_readme(out_dir / "README_burden_viability.md", base_findings, discovery_rows, recommendation, [], [], [], commands)
        return 0

    detected = detect_columns(columns)
    identity = Columns(
        subject=args.subject_col or detected.subject,
        time=args.time_col or detected.time,
        session=args.session_col or detected.session,
        block=args.block_col or detected.block,
        label=args.label_col or detected.label,
    )
    allowed_features, feature_metadata, feature_notes = infer_allowed_features(selected_path, columns, repo_root)
    selected_features = select_model_features(allowed_features, feature_metadata)
    feature_rows, leakage_rows = audit_feature_columns(columns, allowed_features, selected_features, identity, feature_metadata)
    write_csv(out_dir / "feature_column_audit.csv", feature_rows)
    write_csv(out_dir / "leakage_risk_audit.csv", leakage_rows)

    missing_contract = []
    if not identity.subject:
        missing_contract.append("subject identifier")
    if not identity.time:
        missing_contract.append("row order/time column")
    if not identity.label:
        missing_contract.append("binary burst/event label")
    if len(allowed_features) < MIN_REQUIRED_FEATURES:
        missing_contract.append("at least 5 allowed causal feature columns")
    if missing_contract:
        empty_only = ["metadata_proxy_audit.csv", "burden_tau_sweep.csv", "burden_metric_by_subject.csv", "burden_metric_by_condition.csv", "condition_separability.csv", "clinical_proxy_correlation.csv", "baseline_comparison.csv"]
        for name in empty_only:
            write_csv(out_dir / name, [])
        base_findings.update(
            {
                "overall_status": "blocked_missing_required_columns",
                "gate_rationale": f"Missing required data contract fields: {clean_join(missing_contract)}",
                "critical_blockers": missing_contract,
                "allowed_feature_count": len(allowed_features),
                "excluded_feature_count": len(columns) - len(allowed_features),
                "subject_column": identity.subject,
                "time_column": identity.time,
                "session_column": identity.session,
                "block_column": identity.block,
                "target_label_column": identity.label,
            }
        )
        write_json(out_dir / "burden_viability_findings.json", base_findings)
        write_readme(out_dir / "README_burden_viability.md", base_findings, discovery_rows, recommendation, [], [], [], commands)
        return 0

    usecols = list(dict.fromkeys([identity.subject, identity.time, identity.session, identity.block, identity.label, "condition", "channel", "band_mode"] + selected_features))
    usecols = [c for c in usecols if c and c in columns]
    LOG.info("Reading selected table %s with %d columns (%d selected model features of %d allowed)", selected_path, len(usecols), len(selected_features), len(allowed_features))
    df = read_selected_table(selected_path, usecols, args.max_rows)
    df = df.reset_index(drop=True)
    df = split_condition_columns(df)
    metadata_rows = metadata_proxy_audit(df, identity)
    write_csv(out_dir / "metadata_proxy_audit.csv", metadata_rows)
    subject_count = int(df[identity.subject].nunique(dropna=True)) if identity.subject in df.columns else 0
    if subject_count < MIN_REQUIRED_SUBJECTS:
        base_findings.update(
            {
                "overall_status": "blocked_missing_required_columns",
                "gate_rationale": f"Only {subject_count} subjects found; at least {MIN_REQUIRED_SUBJECTS} required.",
                "critical_blockers": [f"Only {subject_count} subjects found."],
                "selected_subject_count": subject_count,
                "allowed_feature_count": len(allowed_features),
                "excluded_feature_count": len(columns) - len(allowed_features),
            }
        )
        write_csv(out_dir / "burden_tau_sweep.csv", [])
        write_csv(out_dir / "burden_metric_by_subject.csv", [])
        write_csv(out_dir / "burden_metric_by_condition.csv", [])
        write_csv(out_dir / "condition_separability.csv", [])
        write_csv(out_dir / "clinical_proxy_correlation.csv", [])
        write_csv(out_dir / "baseline_comparison.csv", [])
        write_json(out_dir / "burden_viability_findings.json", base_findings)
        write_readme(out_dir / "README_burden_viability.md", base_findings, discovery_rows, recommendation, [], [], [], commands)
        return 0

    metric_rows, condition_rows, separability_rows, clinical_proxy_rows, baseline_rows, analysis_state = run_loso_analysis(
        df, identity, selected_features, tau_values, args.random_seed
    )
    tau_rows, med_p, med_s = summarize_tau(metric_rows)
    write_csv(out_dir / "burden_tau_sweep.csv", tau_rows)
    write_csv(out_dir / "burden_metric_by_subject.csv", metric_rows)
    write_csv(out_dir / "burden_metric_by_condition.csv", condition_rows)
    write_csv(out_dir / "condition_separability.csv", separability_rows)
    write_csv(out_dir / "clinical_proxy_correlation.csv", clinical_proxy_rows)
    write_csv(out_dir / "baseline_comparison.csv", baseline_rows)
    plot_outputs(out_dir, tau_rows, metric_rows, baseline_rows, analysis_state, condition_rows, args.no_plots)

    status, rationale, critical, limitations, best_tau, best_p, best_s = gate_decision(
        tau_rows, baseline_rows, separability_rows, subject_count, len(allowed_features), leakage_rows
    )
    if len(allowed_features) > len(selected_features):
        limitations.append(f"Model used deterministic subset of {len(selected_features)} selected allowed features from {len(allowed_features)} allowed causal features to keep Phase 6A.0 lightweight and report-sized.")
    if not separability_rows:
        limitations.append("Clinical/task proxy support not assessable from selected input; primary gate remains annotation-derived burden tracking.")
    baseline_summary = {}
    if best_tau is not None:
        baseline_summary = {
            row["baseline_name"]: row
            for row in baseline_rows
            if float(row["tau_ms"]) == float(best_tau)
        }
    findings = base_findings | {
        "selected_input_files": [relpath(selected_path, repo_root)],
        "selected_subject_count": subject_count,
        "valid_subject_count": int(max((r.get("n_valid_subjects", 0) for r in tau_rows), default=0)),
        "target_label_column": identity.label,
        "subject_column": identity.subject,
        "time_column": identity.time,
        "session_column": identity.session,
        "block_column": identity.block,
        "allowed_feature_count": len(allowed_features),
        "selected_model_feature_count": len(selected_features),
        "excluded_feature_count": len(columns) - len(allowed_features),
        "leakage_risk_summary": f"{len(leakage_rows)} excluded/suspicious columns; selected model features restricted to allowed causal feature list.",
        "best_tau_ms": best_tau,
        "median_pearson_by_tau": {str(int(k)): v for k, v in med_p.items()},
        "median_spearman_by_tau": {str(int(k)): v for k, v in med_s.items()},
        "best_tau_median_pearson": best_p,
        "best_tau_median_spearman": best_s,
        "baseline_summary": baseline_summary,
        "condition_proxy_summary": {"status": "available" if separability_rows else "not_assessable", "rows": separability_rows},
        "clinical_proxy_summary": clinical_proxy_rows[0] if clinical_proxy_rows else {"status": "not_assessable"},
        "overall_status": status,
        "gate_rationale": rationale,
        "critical_blockers": critical,
        "noncritical_limitations": limitations + feature_notes,
        "recommended_next_steps": recommended_next_steps(status),
    }
    write_json(out_dir / "burden_viability_findings.json", findings)
    write_readme(out_dir / "README_burden_viability.md", findings, discovery_rows, recommendation, tau_rows, baseline_rows, separability_rows, commands)

    if status == "CONDITIONAL_PASS":
        next_dir = repo_root / "reports/phase6a_snn_burden_state_machine"
        next_dir.mkdir(parents=True, exist_ok=True)
        (next_dir / "README_phase6a_next_steps.md").write_text(
            "# Phase 6A State-Machine Next Steps\n\nPhase 6A.0 returned CONDITIONAL_PASS. Do not create simulation code yet. Add robustness checks, feature ablations, and leakage/normalization sensitivity analyses before Brian2/state-machine implementation.\n",
            encoding="utf-8",
        )
    LOG.info("Phase 6A.0 complete: %s", status)
    return 0


def recommended_next_steps(status: str) -> list[str]:
    if status == "PASS":
        return [
            "Proceed to Phase 6A Brian2/state-machine simulation using the selected internal STN input and best tau.",
            "Keep PPN as optional cross-target extension and Herz as methods/code reference only.",
        ]
    if status == "CONDITIONAL_PASS":
        return [
            "Add ablations, leakage sensitivity checks, and feature-family robustness analyses before simulation code.",
            "Proceed to Phase 6A simulation only after limitations are accepted explicitly.",
        ]
    if status == "FAIL":
        return [
            "Stop Phase 6 simulation work and reconsider the burden/state-machine pivot.",
            "Inspect whether target construction, feature families, or annotation burden definition should be revised.",
        ]
    return [
        "Provide a window-level internal STN feature/label table with subject, time/order, binary labels, and at least five causal feature columns.",
        "Do not use external PPN/Herz datasets as primary Phase 6A.0 inputs.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())

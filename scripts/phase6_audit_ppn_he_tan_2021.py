#!/usr/bin/env python3
"""Phase 6A metadata/data audit for the He/Tan 2021 PPN gait dataset.

This script is intentionally audit-only. It inventories files, MATLAB schemas,
channels, markers, and lightweight signal sanity metrics without preprocessing,
filtering, modeling, or writing inside raw patient folders.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - exercised only in broken envs
    np = None  # type: ignore[assignment]
    NUMPY_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    NUMPY_IMPORT_ERROR = ""

try:
    import scipy.io as scipy_io
except Exception as exc:  # pragma: no cover - exercised only if scipy missing
    scipy_io = None  # type: ignore[assignment]
    SCIPY_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    SCIPY_IMPORT_ERROR = ""

try:
    import h5py
except Exception as exc:  # pragma: no cover - h5py is optional
    h5py = None  # type: ignore[assignment]
    H5PY_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"
else:
    H5PY_IMPORT_ERROR = ""


DATASET_ID = "ppn_he_tan_2021"
EXPECTED_PATIENTS = [f"PD{i:02d}" for i in range(1, 8)] + [f"MSA{i:02d}" for i in range(1, 5)]
PROTOCOLS = ["RestSitting", "RestStanding", "StepSitting", "StepStanding", "FreeWalking"]
EXPECTED_PROTOCOLS = {
    "RestSitting": set(EXPECTED_PATIENTS),
    "RestStanding": {f"PD{i:02d}" for i in range(1, 8)},
    "StepSitting": {f"MSA{i:02d}" for i in range(1, 5)},
    "StepStanding": {"MSA01", "MSA02"},
    "FreeWalking": {"PD01", "PD02"},
}

SCAN_EXTENSIONS = {
    ".dcm",
    ".nii",
    ".nii.gz",
    ".mri",
    ".ima",
    ".nrrd",
    ".mgz",
    ".mgh",
    ".hdr",
    ".img",
}
CODE_EXTENSIONS = {".m", ".mlx", ".py", ".sh", ".ipynb"}
LARGE_FILE_BYTES = 100_000_000
PATIENT_RE = re.compile(r"\b(PD\d{2}|MSA\d{2})\b", re.IGNORECASE)

FS_KEY_RE = re.compile(
    r"(^fs$|^srate$|sample[_-]?rate|sampling[_-]?(rate|freq|frequency)|"
    r"^sfreq$|^sr$|^Fs$|^FS$)",
    re.IGNORECASE,
)

EEG_NAMES = {
    "fz",
    "cz",
    "c3",
    "c4",
    "f3",
    "f4",
    "pz",
    "oz",
    "cp3",
    "cp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="cambium/Data_Code_PPN_JNeurosci_2021",
        help="Dataset root to audit, relative to repo root by default.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/phase6_ppn_he_tan_2021_audit",
        help="Output report directory, relative to repo root by default.",
    )
    parser.add_argument("--paper-fs", type=float, default=2048.0)
    parser.add_argument("--max-sanity-channels", type=int, default=256)
    return parser.parse_args()


def utc_local_timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def clean_join(values: Iterable[Any]) -> str:
    cleaned = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return ";".join(cleaned)


def json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def bool_cell(value: Any) -> str:
    return "true" if bool(value) else "false"


def file_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower() or "[none]"


def infer_patient_id(path_text: str) -> str:
    match = PATIENT_RE.search(path_text)
    return match.group(1).upper() if match else ""


def patient_group(patient_id: str) -> str:
    if patient_id.startswith("PD"):
        return "PD"
    if patient_id.startswith("MSA"):
        return "MSA"
    return ""


def normalize_for_protocol(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def guess_protocol(path_or_name: str) -> str:
    text = normalize_for_protocol(path_or_name)
    if not text:
        return ""
    if "stepstanding" in text or "step_standing" in text or "stepping_standing" in text:
        return "StepStanding"
    if "stepsitting" in text or "step_sitting" in text or "stepping_sitting" in text:
        return "StepSitting"
    if "freewalking" in text or "free_walking" in text or re.search(r"(^|_)walk(ing)?($|_)", text):
        return "FreeWalking"
    if "reststanding" in text or "rest_standing" in text or "reststand" in text:
        return "RestStanding"
    if re.search(r"(^|_)(stand|standing|stand\d+)($|_)", text) and "step" not in text:
        return "RestStanding"
    if "restsitting" in text or "rest_sitting" in text or "restsit" in text:
        return "RestSitting"
    if re.search(r"(^|_)(rest|rst)($|_)", text) and "stand" not in text and "step" not in text:
        return "RestSitting"
    if re.search(r"(^|_)sitting(_\d+)?($|_)", text) and "step" not in text:
        return "RestSitting"
    return ""


def is_possible_code_file(path: Path) -> bool:
    return file_extension(path) in CODE_EXTENSIONS


def is_possible_scan_or_identifiable(path: Path) -> bool:
    return file_extension(path) in SCAN_EXTENSIONS


def read_text_limited(path: Path, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        data = handle.read(max_bytes)
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_referenced_mat_stems(text: str) -> list[str]:
    stems: list[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        for token in re.split(r"[\s,;]+", line):
            token = token.strip().strip("\"'`")
            if not token:
                continue
            token = token.replace("\\", "/").split("/")[-1]
            if token.lower().endswith(".mat"):
                token = token[:-4]
            if not token or token.lower() in {"mat", "txt"}:
                continue
            if token not in seen:
                stems.append(token)
                seen.add(token)
    return stems


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, bool):
                    out[column] = bool_cell(value)
                elif isinstance(value, (list, tuple, set)):
                    out[column] = clean_join(value)
                elif isinstance(value, dict):
                    out[column] = json_cell(value)
                elif isinstance(value, float):
                    out[column] = "" if math.isnan(value) else value
                else:
                    out[column] = value
            writer.writerow(out)


def inventory_files(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
        stat = path.stat()
        extension = file_extension(path)
        rel = relpath(path, data_root)
        rows.append(
            {
                "relative_path": rel,
                "parent_folder": relpath(path.parent, data_root),
                "extension": extension,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / 1_000_000, 3),
                "inferred_patient_id": infer_patient_id(rel),
                "possible_protocol": guess_protocol(path.name),
                "is_mat_file": extension == ".mat",
                "is_txt_file": extension == ".txt",
                "is_possible_code_file": is_possible_code_file(path),
                "is_possible_scan_or_identifiable_file": is_possible_scan_or_identifiable(path),
                "notes": "large_file" if stat.st_size >= LARGE_FILE_BYTES else "",
            }
        )
    return rows


def build_patient_task_matrix(data_root: Path, manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    observed_patients = sorted(
        {
            p.name.upper()
            for p in data_root.rglob("*")
            if p.is_dir() and re.fullmatch(r"(PD|MSA)\d{2}", p.name, flags=re.IGNORECASE)
        }
    )
    extras = sorted(set(observed_patients) - set(EXPECTED_PATIENTS))
    patient_ids = EXPECTED_PATIENTS + extras
    by_patient: dict[str, dict[str, Any]] = {}
    for patient in patient_ids:
        by_patient[patient] = {
            "patient_id": patient,
            "group": patient_group(patient),
            **{f"expected_{protocol}": patient in EXPECTED_PROTOCOLS[protocol] for protocol in PROTOCOLS},
            **{f"observed_{protocol}": False for protocol in PROTOCOLS},
            "txt_files": [],
            "referenced_mat_files": [],
            "missing_referenced_mat_files": [],
            "orphan_mat_files": [],
            "notes": [],
        }

    referenced_paths_by_patient: dict[str, set[str]] = defaultdict(set)
    all_patient_mat_paths: dict[str, set[str]] = defaultdict(set)

    for row in manifest_rows:
        patient = row["inferred_patient_id"]
        if patient:
            rel = row["relative_path"]
            protocol = row["possible_protocol"]
            if row["is_mat_file"]:
                all_patient_mat_paths[patient].add(rel)
                if protocol:
                    by_patient.setdefault(patient, {}).setdefault(f"observed_{protocol}", False)
                    by_patient[patient][f"observed_{protocol}"] = True

    for txt_path in sorted(data_root.rglob("*.txt")):
        rel = relpath(txt_path, data_root)
        patient = infer_patient_id(rel)
        if not patient:
            continue
        protocol = guess_protocol(txt_path.name)
        if patient not in by_patient:
            by_patient[patient] = {
                "patient_id": patient,
                "group": patient_group(patient),
                **{f"expected_{p}": patient in EXPECTED_PROTOCOLS[p] for p in PROTOCOLS},
                **{f"observed_{p}": False for p in PROTOCOLS},
                "txt_files": [],
                "referenced_mat_files": [],
                "missing_referenced_mat_files": [],
                "orphan_mat_files": [],
                "notes": [],
            }
        by_patient[patient]["txt_files"].append(rel)
        if protocol:
            by_patient[patient][f"observed_{protocol}"] = True
        try:
            refs = extract_referenced_mat_stems(read_text_limited(txt_path))
        except Exception as exc:
            by_patient[patient]["notes"].append(f"could_not_read_{rel}: {exc.__class__.__name__}")
            continue
        for stem in refs:
            candidate = txt_path.parent / f"{stem}.mat"
            if candidate.exists():
                candidate_rel = relpath(candidate, data_root)
                by_patient[patient]["referenced_mat_files"].append(candidate_rel)
                referenced_paths_by_patient[patient].add(candidate_rel)
                ref_protocol = protocol or guess_protocol(stem)
                if ref_protocol:
                    by_patient[patient][f"observed_{ref_protocol}"] = True
            else:
                missing = f"{rel} -> {stem}.mat"
                by_patient[patient]["missing_referenced_mat_files"].append(missing)

    for patient, mat_paths in all_patient_mat_paths.items():
        orphan_paths = sorted(mat_paths - referenced_paths_by_patient.get(patient, set()))
        by_patient[patient]["orphan_mat_files"].extend(orphan_paths)

    for patient, row in by_patient.items():
        notes = row["notes"]
        for protocol in PROTOCOLS:
            if row[f"expected_{protocol}"] and not row[f"observed_{protocol}"]:
                notes.append(f"missing_expected_{protocol}")
            if row[f"observed_{protocol}"] and not row[f"expected_{protocol}"]:
                notes.append(f"observed_unexpected_{protocol}")
        if row["missing_referenced_mat_files"]:
            notes.append("missing_referenced_mat_files")
        if row["orphan_mat_files"]:
            notes.append(f"{len(row['orphan_mat_files'])}_orphan_mat_files")

    rows = []
    for patient in patient_ids:
        row = by_patient[patient]
        rows.append(
            {
                **{k: row[k] for k in ["patient_id", "group"]},
                **{f"expected_{p}": row[f"expected_{p}"] for p in PROTOCOLS},
                **{f"observed_{p}": row[f"observed_{p}"] for p in PROTOCOLS},
                "txt_files": sorted(set(row["txt_files"])),
                "referenced_mat_files": sorted(set(row["referenced_mat_files"])),
                "missing_referenced_mat_files": sorted(set(row["missing_referenced_mat_files"])),
                "orphan_mat_files": sorted(set(row["orphan_mat_files"])),
                "notes": sorted(set(row["notes"])),
            }
        )

    missing_refs = sorted({item for row in rows for item in row.get("missing_referenced_mat_files", [])})
    orphan_mats = sorted({item for row in rows for item in row.get("orphan_mat_files", [])})
    return rows, missing_refs, orphan_mats


def is_hdf5_mat(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(8).startswith(b"\x89HDF")
    except OSError:
        return False


def shape_to_text(shape: Any) -> str:
    if not shape:
        return ""
    try:
        return "x".join(str(int(dim)) for dim in shape)
    except Exception:
        return str(shape)


def safe_loadmat(path: Path, variable_names: list[str] | None = None) -> dict[str, Any]:
    if scipy_io is None:
        raise RuntimeError(f"scipy unavailable: {SCIPY_IMPORT_ERROR}")
    kwargs = {
        "squeeze_me": True,
        "struct_as_record": False,
    }
    if variable_names is not None:
        kwargs["variable_names"] = variable_names
    return scipy_io.loadmat(str(path), **kwargs)


def mat_whos(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    if scipy_io is None:
        raise RuntimeError(f"scipy unavailable: {SCIPY_IMPORT_ERROR}")
    return scipy_io.whosmat(str(path))


def array_size_from_shape(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def matlab_to_strings(value: Any) -> list[str]:
    if value is None or np is None:
        return []
    out: list[str] = []

    def add(text: Any) -> None:
        if text is None:
            return
        if isinstance(text, bytes):
            decoded = text.decode("utf-8", errors="replace")
        else:
            decoded = str(text)
        decoded = decoded.strip()
        if decoded:
            out.append(decoded)

    def visit(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, bytes):
            add(obj)
        elif isinstance(obj, str):
            add(obj)
        elif np is not None and isinstance(obj, np.ndarray):
            if obj.dtype.kind in {"U", "S"}:
                if obj.ndim == 0:
                    add(obj.item())
                elif obj.ndim == 1:
                    vals = [str(x) for x in obj.tolist()]
                    if vals and all(len(v) == 1 for v in vals):
                        add("".join(vals))
                    else:
                        for item in vals:
                            add(item)
                elif obj.ndim == 2:
                    for row in obj:
                        vals = [str(x) for x in np.asarray(row).tolist()]
                        add("".join(vals).strip())
                else:
                    for item in obj.flat:
                        add(item)
            elif obj.dtype == object:
                for item in obj.flat:
                    visit(item)
            elif obj.ndim == 0:
                add(obj.item())
            else:
                for item in obj.flat:
                    add(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        else:
            add(obj)

    visit(value)
    return out


def get_mat_field(obj: Any, field: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    if hasattr(obj, field):
        return getattr(obj, field)
    if np is not None and isinstance(obj, np.ndarray):
        if obj.dtype.names and field in obj.dtype.names:
            return obj[field]
        if obj.size == 1:
            try:
                return get_mat_field(obj.item(), field)
            except Exception:
                return None
    return None


def numeric_array(value: Any) -> Any:
    if np is None or value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
        return None
    return arr


def infer_axes(shape: tuple[int, ...], channel_count: int | None, max_sanity_channels: int) -> tuple[int | None, int | None, int | None, int | None]:
    if not shape:
        return None, None, None, None
    dims = tuple(int(x) for x in shape)
    channel_axis = None
    time_axis = None
    if channel_count:
        matches = [idx for idx, dim in enumerate(dims) if dim == channel_count]
        if len(matches) == 1:
            channel_axis = matches[0]
    if channel_axis is None and len(dims) >= 2:
        small_axes = [idx for idx, dim in enumerate(dims) if 1 < dim <= max_sanity_channels]
        if len(small_axes) == 1:
            channel_axis = small_axes[0]
        elif len(small_axes) > 1:
            channel_axis = min(small_axes, key=lambda idx: dims[idx])
    if channel_axis is not None:
        candidates = [idx for idx in range(len(dims)) if idx != channel_axis]
        if candidates:
            time_axis = max(candidates, key=lambda idx: dims[idx])
    elif len(dims) == 1:
        time_axis = 0
    else:
        time_axis = max(range(len(dims)), key=lambda idx: dims[idx])
        channel_candidates = [idx for idx in range(len(dims)) if idx != time_axis]
        if channel_candidates:
            channel_axis = min(channel_candidates, key=lambda idx: dims[idx])
    n_channels = dims[channel_axis] if channel_axis is not None else None
    n_samples = dims[time_axis] if time_axis is not None else None
    return channel_axis, time_axis, n_channels, n_samples


def find_fs_value(keys: list[str], loaded: dict[str, Any]) -> tuple[bool, str, str]:
    if np is None:
        return False, "", ""
    for key in keys:
        if not FS_KEY_RE.search(key):
            continue
        if key not in loaded:
            continue
        arr = numeric_array(loaded.get(key))
        if arr is None or arr.size != 1:
            continue
        try:
            value = float(np.ravel(arr)[0])
        except Exception:
            continue
        if value > 0:
            return True, f"{value:g}", f"top_level:{key}"
    return False, "", ""


def channel_modality_guess(channel_name: str, channel_type: str) -> tuple[str, dict[str, bool]]:
    name = channel_name.strip()
    ctype = channel_type.strip()
    low = f"{name} {ctype}".lower()
    compact_name = re.sub(r"[^a-z0-9]", "", name.lower())
    eeg = (
        "eeg" in low
        or compact_name in EEG_NAMES
        or any(re.fullmatch(pattern, compact_name) for pattern in EEG_NAMES)
    )
    force = any(token in low for token in ["force", "frc", "pressure", "biometrics", "plate", "foot", "heel", "toe"])
    accel = any(token in low for token in ["acc", "accel", "accelerometer", "trunk"])
    marker = any(token in low for token in ["marker", "trigger", "event"])
    lfp = (
        "lfp" in low
        or "ppn" in low
        or "dbs" in low
        or bool(re.search(r"\b[lr]?\d+[lr]?\d*\b", low))
        or bool(re.search(r"\b[lr](?:eft|ight)?[_ -]?\d", low))
    )
    if eeg:
        modality = "EEG"
    elif force:
        modality = "force_pressure"
    elif accel:
        modality = "accelerometer"
    elif lfp:
        modality = "PPN_LFP"
    elif marker:
        modality = "marker_like"
    else:
        modality = "unknown"
    flags = {
        "is_lfp_guess": lfp,
        "is_eeg_guess": eeg,
        "is_force_guess": force,
        "is_accelerometer_guess": accel,
        "is_marker_like_guess": marker,
        "is_cz_or_fz_guess": compact_name in {"cz", "fz"},
    }
    return modality, flags


def marker_summary(
    marker_obj: Any,
    marker_name: str,
    duration_sec: float | None,
    task_guess: str,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    marks = get_mat_field(marker_obj, "Marks")
    times = get_mat_field(marker_obj, "Times")
    marks_arr = numeric_array(marks)
    times_arr = numeric_array(times)
    has_marks = marks_arr is not None
    has_times = times_arr is not None

    codes: list[int] = []
    code_counts: dict[str, int] = {}
    n_marks = 0
    n_times = 0
    if marks_arr is not None:
        arr = np.asarray(marks_arr)
        n_marks = int(arr.shape[0]) if arr.ndim > 0 else int(arr.size)
        if arr.ndim == 0:
            code_values = [arr.item()]
        elif arr.ndim == 1:
            code_values = arr
        else:
            code_values = arr[:, 0]
        for value in np.ravel(code_values):
            try:
                if np.isfinite(value):
                    codes.append(int(value))
            except Exception:
                continue
        code_counts = {str(k): int(v) for k, v in sorted(Counter(codes).items())}
    if times_arr is not None:
        n_times = int(np.asarray(times_arr).size)
    times_flat = np.ravel(times_arr.astype(float)) if times_arr is not None and times_arr.size else np.asarray([])
    times_monotonic = bool(times_flat.size < 2 or np.all(np.diff(times_flat) >= 0))
    if not times_monotonic:
        issues.append(f"{marker_name}_times_not_monotonic")
    times_within_duration = ""
    if duration_sec is not None and times_flat.size:
        times_within = bool(np.nanmin(times_flat) >= -1e-6 and np.nanmax(times_flat) <= duration_sec + 1e-6)
        times_within_duration = bool_cell(times_within)
        if not times_within:
            issues.append(f"{marker_name}_times_outside_duration")

    start_count = int(code_counts.get("0", 0))
    end_count = int(code_counts.get("1", 0))
    code16_count = int(code_counts.get("16", 0))
    balanced = start_count == end_count
    block_durations: list[float] = []
    if marker_name == "Marker" and marks_arr is not None and times_flat.size:
        if len(codes) == len(times_flat):
            pending_start: float | None = None
            for code, timestamp in zip(codes, times_flat):
                if code == 0:
                    pending_start = float(timestamp)
                elif code == 1 and pending_start is not None:
                    duration = float(timestamp) - pending_start
                    if duration >= 0:
                        block_durations.append(duration)
                    pending_start = None
    if marker_name == "Marker" and task_guess in {"StepSitting", "StepStanding", "FreeWalking"}:
        if not has_marks or not has_times:
            issues.append("movement_marker_missing_marks_or_times")
        if not balanced:
            issues.append("marker_start_end_unbalanced")
    if marker_name == "MarkerWalk" and task_guess == "FreeWalking" and code16_count == 0:
        issues.append("markerwalk_no_code_16")

    return (
        {
            "marker_object": marker_name,
            "has_Marks": has_marks,
            "has_Times": has_times,
            "n_marks": n_marks,
            "n_times": n_times,
            "marker_code_counts": code_counts,
            "marker_code_0_count": start_count,
            "marker_code_1_count": end_count,
            "marker_code_16_count": code16_count,
            "n_start_end_blocks": len(block_durations),
            "start_end_balanced": balanced,
            "times_monotonic": times_monotonic,
            "times_within_data_duration": times_within_duration,
            "block_duration_min_sec": min(block_durations) if block_durations else "",
            "block_duration_median_sec": float(np.median(block_durations)) if block_durations else "",
            "block_duration_max_sec": max(block_durations) if block_durations else "",
            "walk_phase_timestamp_count": code16_count if marker_name == "MarkerWalk" else "",
        },
        issues,
    )


def orient_data_channels(data: Any, channel_axis: int | None, time_axis: int | None) -> Any:
    if np is None:
        return None
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if channel_axis is None:
        channel_axis = 0 if arr.shape[0] <= arr.shape[-1] else arr.ndim - 1
    if time_axis is None:
        candidates = [idx for idx in range(arr.ndim) if idx != channel_axis]
        time_axis = max(candidates, key=lambda idx: arr.shape[idx]) if candidates else 0
    moved = np.moveaxis(arr, [channel_axis, time_axis], [0, 1])
    if moved.ndim > 2:
        moved = moved.reshape(moved.shape[0], moved.shape[1], -1)
        moved = moved[:, :, 0]
    return moved


def signal_sanity(
    path: Path,
    mat_row: dict[str, Any],
    channel_names: list[str],
    max_sanity_channels: int,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    base = {
        "relative_path": mat_row["relative_path"],
        "patient_id": mat_row["patient_id"],
        "task_guess": mat_row["task_guess"],
        "n_channels": mat_row["inferred_n_channels"],
        "n_samples": mat_row["inferred_n_samples"],
        "duration_sec_if_possible": mat_row["duration_sec_if_possible"],
        "data_numeric": "",
        "nan_count": "",
        "nan_percent": "",
        "inf_count": "",
        "inf_percent": "",
        "all_zero_channel_count": "",
        "all_zero_channels": "",
        "constant_channel_count": "",
        "constant_channels": "",
        "aggregate_mean": "",
        "aggregate_std": "",
        "aggregate_min": "",
        "aggregate_max": "",
        "channels_summarized": "",
        "notes": "",
    }
    if np is None:
        base["notes"] = f"numpy_unavailable:{NUMPY_IMPORT_ERROR}"
        issues.append("numpy_unavailable_for_signal_sanity")
        return base, issues
    try:
        loaded = safe_loadmat(path, ["data"])
        data = loaded.get("data")
    except Exception as exc:
        base["notes"] = f"data_load_failed:{exc.__class__.__name__}: {exc}"
        issues.append(f"data_load_failed:{mat_row['relative_path']}")
        return base, issues
    arr = np.asarray(data)
    if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
        base["data_numeric"] = False
        base["notes"] = f"data_not_numeric:{arr.dtype}"
        issues.append(f"data_not_numeric:{mat_row['relative_path']}")
        return base, issues
    base["data_numeric"] = True
    if arr.size == 0:
        base["notes"] = "empty_data"
        issues.append(f"empty_data:{mat_row['relative_path']}")
        return base, issues
    channel_axis = mat_row.get("_channel_axis")
    time_axis = mat_row.get("_time_axis")
    channel_data = orient_data_channels(arr, channel_axis, time_axis)
    if channel_data is None:
        base["notes"] = "could_not_orient_data"
        issues.append(f"could_not_orient_data:{mat_row['relative_path']}")
        return base, issues
    finite_mask = np.isfinite(channel_data)
    nan_count = int(np.isnan(channel_data).sum()) if channel_data.dtype.kind in {"f", "c"} else 0
    inf_count = int(np.isinf(channel_data).sum()) if channel_data.dtype.kind in {"f", "c"} else 0
    total = int(channel_data.size)
    base["nan_count"] = nan_count
    base["nan_percent"] = round(100.0 * nan_count / total, 6) if total else ""
    base["inf_count"] = inf_count
    base["inf_percent"] = round(100.0 * inf_count / total, 6) if total else ""
    finite_values = channel_data[finite_mask]
    if finite_values.size:
        base["aggregate_mean"] = float(np.mean(finite_values))
        base["aggregate_std"] = float(np.std(finite_values))
        base["aggregate_min"] = float(np.min(finite_values))
        base["aggregate_max"] = float(np.max(finite_values))
    else:
        issues.append(f"all_values_nonfinite:{mat_row['relative_path']}")
    n_channels = int(channel_data.shape[0])
    summarize_n = min(n_channels, max_sanity_channels)
    all_zero = []
    constant = []
    entirely_nonfinite = []
    for idx in range(summarize_n):
        values = np.asarray(channel_data[idx])
        finite = values[np.isfinite(values)]
        label = channel_names[idx] if idx < len(channel_names) else str(idx)
        if finite.size == 0:
            entirely_nonfinite.append(label)
            continue
        if np.all(finite == 0):
            all_zero.append(label)
        if finite.size and np.nanmax(finite) == np.nanmin(finite):
            constant.append(label)
    base["all_zero_channel_count"] = len(all_zero)
    base["all_zero_channels"] = all_zero
    base["constant_channel_count"] = len(constant)
    base["constant_channels"] = constant
    base["channels_summarized"] = summarize_n
    notes = []
    if n_channels > max_sanity_channels:
        notes.append(f"per_channel_stats_limited_to_{max_sanity_channels}_of_{n_channels}")
    if nan_count:
        notes.append("contains_nan")
        issues.append(f"contains_nan:{mat_row['relative_path']}")
    if inf_count:
        notes.append("contains_inf")
        issues.append(f"contains_inf:{mat_row['relative_path']}")
    if all_zero and len(all_zero) == summarize_n == n_channels:
        notes.append("all_summarized_channels_zero")
        issues.append(f"all_channels_zero:{mat_row['relative_path']}")
    elif all_zero:
        notes.append("has_all_zero_channels")
    if entirely_nonfinite:
        notes.append("has_entirely_nonfinite_channels")
        issues.append(f"entirely_nonfinite_channels:{mat_row['relative_path']}")
    if constant:
        notes.append("has_constant_channels")
    if not mat_row["duration_sec_if_possible"]:
        notes.append("duration_impossible_to_infer")
        issues.append(f"duration_impossible:{mat_row['relative_path']}")
    else:
        try:
            duration = float(mat_row["duration_sec_if_possible"])
            if mat_row["task_guess"] in {"RestSitting", "RestStanding"} and duration < 60:
                notes.append("rest_recording_implausibly_short_under_60s")
                issues.append(f"rest_recording_short:{mat_row['relative_path']}")
        except Exception:
            pass
    base["notes"] = notes
    return base, issues


def inspect_hdf5_mat(path: Path, rel: str, patient: str, task_guess: str, size_mb: float, paper_fs: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[str], list[str]]:
    mat_issues: list[str] = []
    channel_issues: list[str] = []
    marker_issues: list[str] = []
    signal_issues: list[str] = []
    if h5py is None:
        row = base_mat_row(rel, patient, path.stem, size_mb, task_guess, paper_fs)
        row.update(
            {
                "mat_format_guess": "matlab_v7.3_hdf5",
                "load_status": "unreadable",
                "error_message": f"h5py unavailable: {H5PY_IMPORT_ERROR}",
                "notes": "hdf5_mat_requires_h5py",
            }
        )
        mat_issues.append(f"hdf5_unreadable_without_h5py:{rel}")
        return row, [], [], [], mat_issues, channel_issues, marker_issues + signal_issues
    try:
        with h5py.File(path, "r") as handle:
            keys = sorted(handle.keys())
            data_obj = handle.get("data")
            data_shape = tuple(data_obj.shape) if data_obj is not None and hasattr(data_obj, "shape") else ()
            data_dtype = str(data_obj.dtype) if data_obj is not None and hasattr(data_obj, "dtype") else ""
    except Exception as exc:
        row = base_mat_row(rel, patient, path.stem, size_mb, task_guess, paper_fs)
        row.update(
            {
                "mat_format_guess": "matlab_v7.3_hdf5",
                "load_status": "unreadable",
                "error_message": f"{exc.__class__.__name__}: {exc}",
            }
        )
        mat_issues.append(f"hdf5_open_failed:{rel}")
        return row, [], [], [], mat_issues, channel_issues, marker_issues + signal_issues
    row = base_mat_row(rel, patient, path.stem, size_mb, task_guess, paper_fs)
    has_channel_name = "ChannelName" in keys
    has_channel_type = "ChannelType" in keys
    has_data = "data" in keys
    row.update(
        {
            "mat_format_guess": "matlab_v7.3_hdf5",
            "load_status": "metadata_ok",
            "top_level_keys": clean_join(keys),
            "has_ChannelName": has_channel_name,
            "has_ChannelType": has_channel_type,
            "has_data": has_data,
            "has_Marker": "Marker" in keys,
            "has_MarkerWalk": "MarkerWalk" in keys,
            "data_shape": shape_to_text(data_shape),
            "data_dtype": data_dtype,
            "likely_raw_or_result": "raw_like" if has_channel_name and has_channel_type and has_data else "result_or_metadata",
            "notes": "hdf5_metadata_only; h5py_string_reference_decoding_not_deeply_processed",
        }
    )
    return row, [], [], [], mat_issues, channel_issues, marker_issues + signal_issues


def base_mat_row(rel: str, patient: str, stem: str, size_mb: float, task_guess: str, paper_fs: float) -> dict[str, Any]:
    return {
        "relative_path": rel,
        "patient_id": patient,
        "task_guess": task_guess,
        "file_stem": stem,
        "size_mb": size_mb,
        "mat_format_guess": "",
        "load_status": "",
        "error_message": "",
        "top_level_keys": "",
        "has_ChannelName": False,
        "has_ChannelType": False,
        "has_data": False,
        "has_Marker": False,
        "has_MarkerWalk": False,
        "data_shape": "",
        "data_dtype": "",
        "inferred_n_channels": "",
        "inferred_n_samples": "",
        "inferred_time_axis": "",
        "explicit_fs_found": False,
        "fs_value": f"{paper_fs:g}",
        "fs_source": "paper default / assumed",
        "duration_sec_if_possible": "",
        "likely_raw_or_result": "",
        "notes": "",
        "_channel_axis": None,
        "_time_axis": None,
    }


def inspect_mat_file(
    path: Path,
    data_root: Path,
    paper_fs: float,
    max_sanity_channels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[str], list[str]]:
    rel = relpath(path, data_root)
    patient = infer_patient_id(rel)
    task_guess = guess_protocol(path.name)
    size_mb = round(path.stat().st_size / 1_000_000, 3)
    row = base_mat_row(rel, patient, path.stem, size_mb, task_guess, paper_fs)
    channel_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    mat_issues: list[str] = []
    channel_issues: list[str] = []
    marker_signal_issues: list[str] = []

    if is_hdf5_mat(path):
        return inspect_hdf5_mat(path, rel, patient, task_guess, size_mb, paper_fs)
    if scipy_io is None:
        row.update(
            {
                "mat_format_guess": "classic_or_unknown",
                "load_status": "unreadable",
                "error_message": f"scipy unavailable: {SCIPY_IMPORT_ERROR}",
            }
        )
        mat_issues.append(f"scipy_unavailable:{rel}")
        return row, channel_rows, marker_rows, signal_rows, mat_issues, channel_issues, marker_signal_issues

    try:
        whos = mat_whos(path)
    except Exception as exc:
        message = str(exc)
        if "HDF" in message or "v7.3" in message:
            return inspect_hdf5_mat(path, rel, patient, task_guess, size_mb, paper_fs)
        row.update(
            {
                "mat_format_guess": "classic_or_unknown",
                "load_status": "unreadable",
                "error_message": f"{exc.__class__.__name__}: {exc}",
            }
        )
        mat_issues.append(f"mat_unreadable:{rel}")
        return row, channel_rows, marker_rows, signal_rows, mat_issues, channel_issues, marker_signal_issues

    info = {name: {"shape": tuple(shape), "class": mat_class} for name, shape, mat_class in whos}
    keys = sorted(info)
    row["mat_format_guess"] = "matlab_classic"
    row["load_status"] = "metadata_ok"
    row["top_level_keys"] = clean_join(keys)
    row["has_ChannelName"] = "ChannelName" in info
    row["has_ChannelType"] = "ChannelType" in info
    row["has_data"] = "data" in info
    row["has_Marker"] = "Marker" in info
    row["has_MarkerWalk"] = "MarkerWalk" in info
    if "data" in info:
        data_shape = tuple(int(x) for x in info["data"]["shape"])
        row["data_shape"] = shape_to_text(data_shape)
        row["data_dtype"] = info["data"]["class"]

    fs_candidate_keys = [key for key in keys if FS_KEY_RE.search(key)]
    small_load_keys = [key for key in ["ChannelName", "ChannelType", "Marker", "MarkerWalk"] if key in info]
    for key in fs_candidate_keys:
        if key not in small_load_keys:
            small_load_keys.append(key)

    loaded_small: dict[str, Any] = {}
    if small_load_keys:
        try:
            loaded_small = safe_loadmat(path, small_load_keys)
        except Exception as exc:
            row["load_status"] = "metadata_only"
            row["error_message"] = f"small_variable_load_failed:{exc.__class__.__name__}: {exc}"
            mat_issues.append(f"small_variable_load_failed:{rel}")

    explicit_fs, fs_value, fs_source = find_fs_value(fs_candidate_keys, loaded_small)
    if explicit_fs:
        row["explicit_fs_found"] = True
        row["fs_value"] = fs_value
        row["fs_source"] = fs_source
    elif fs_candidate_keys:
        row["notes"] = clean_join([row["notes"], "fs_like_key_present_but_value_not_scalar_numeric"])

    channel_names = matlab_to_strings(loaded_small.get("ChannelName")) if row["has_ChannelName"] else []
    channel_types = matlab_to_strings(loaded_small.get("ChannelType")) if row["has_ChannelType"] else []
    channel_count = len(channel_names) or len(channel_types) or None
    if "data" in info:
        data_shape = tuple(int(x) for x in info["data"]["shape"])
        channel_axis, time_axis, n_channels, n_samples = infer_axes(data_shape, channel_count, max_sanity_channels)
        row["_channel_axis"] = channel_axis
        row["_time_axis"] = time_axis
        row["inferred_n_channels"] = n_channels if n_channels is not None else ""
        row["inferred_n_samples"] = n_samples if n_samples is not None else ""
        row["inferred_time_axis"] = f"axis_{time_axis}" if time_axis is not None else ""
        try:
            fs = float(row["fs_value"])
            if n_samples and fs > 0:
                row["duration_sec_if_possible"] = round(float(n_samples) / fs, 6)
        except Exception:
            pass

    filename_low = path.name.lower()
    result_hint = any(token in filename_low for token in ["_psd", "gaitmodul", "_perm", "hmm", "ic2019", "cluster"])
    raw_like = bool(row["has_ChannelName"] and row["has_ChannelType"] and row["has_data"])
    row["likely_raw_or_result"] = "raw_like" if raw_like else ("result" if result_hint else "metadata_or_result")
    notes = [row["notes"]] if row["notes"] else []
    if path.stat().st_size >= LARGE_FILE_BYTES:
        notes.append("large_file")
    if not row["explicit_fs_found"]:
        notes.append("no_file_level_fs_found_using_top_level_fs_key_search")
    if raw_like and not channel_names:
        notes.append("raw_like_missing_decodable_ChannelName")
        channel_issues.append(f"missing_decodable_ChannelName:{rel}")
    if raw_like and not channel_types:
        notes.append("raw_like_missing_decodable_ChannelType")
        channel_issues.append(f"missing_decodable_ChannelType:{rel}")
    row["notes"] = clean_join(notes)

    if raw_like:
        max_channels = max(len(channel_names), len(channel_types), int(row["inferred_n_channels"] or 0))
        if max_channels == 0:
            max_channels = 1
        duplicate_names = sorted([name for name, count in Counter(channel_names).items() if count > 1 and name])
        data_channels = int(row["inferred_n_channels"] or 0)
        file_notes: list[str] = []
        if duplicate_names:
            file_notes.append(f"duplicate_channel_names:{clean_join(duplicate_names)}")
            channel_issues.append(f"duplicate_channel_names:{rel}")
        if channel_names and data_channels and len(channel_names) != data_channels:
            file_notes.append(f"ChannelName_count_{len(channel_names)}_vs_data_channels_{data_channels}")
            channel_issues.append(f"channel_name_data_mismatch:{rel}")
        if channel_types and data_channels and len(channel_types) != data_channels:
            file_notes.append(f"ChannelType_count_{len(channel_types)}_vs_data_channels_{data_channels}")
            channel_issues.append(f"channel_type_data_mismatch:{rel}")
        modalities_in_file: Counter[str] = Counter()
        has_cz_fz = False
        has_force = False
        has_accel = False
        for idx in range(max_channels):
            name = channel_names[idx] if idx < len(channel_names) else ""
            ctype = channel_types[idx] if idx < len(channel_types) else ""
            modality, flags = channel_modality_guess(name, ctype)
            modalities_in_file[modality] += 1
            has_cz_fz = has_cz_fz or flags["is_cz_or_fz_guess"]
            has_force = has_force or flags["is_force_guess"]
            has_accel = has_accel or flags["is_accelerometer_guess"]
            channel_rows.append(
                {
                    "relative_path": rel,
                    "patient_id": patient,
                    "task_guess": task_guess,
                    "channel_index": idx,
                    "channel_name": name,
                    "channel_type": ctype,
                    "modality_guess": modality,
                    **flags,
                    "notes": file_notes if idx == 0 else "",
                }
            )
        if not has_cz_fz:
            channel_issues.append(f"no_Cz_or_Fz_guess:{rel}")
            if channel_rows:
                channel_rows[0]["notes"] = clean_join([channel_rows[0].get("notes", ""), "no_Cz_or_Fz_guess"])
        if patient.startswith("MSA") and task_guess in {"StepSitting", "StepStanding"} and not has_force:
            channel_issues.append(f"no_force_guess_in_msa_stepping:{rel}")
            if channel_rows:
                channel_rows[0]["notes"] = clean_join([channel_rows[0].get("notes", ""), "no_force_guess_in_msa_stepping"])
        if patient.startswith("PD") and task_guess == "FreeWalking" and not (has_accel or row["has_MarkerWalk"]):
            channel_issues.append(f"no_accelerometer_or_MarkerWalk_in_pd_freewalking:{rel}")
            if channel_rows:
                channel_rows[0]["notes"] = clean_join(
                    [channel_rows[0].get("notes", ""), "no_accelerometer_or_MarkerWalk_in_pd_freewalking"]
                )
        if not row["duration_sec_if_possible"]:
            marker_signal_issues.append(f"duration_impossible:{rel}")

        signal_row, signal_issues = signal_sanity(path, row, channel_names, max_sanity_channels)
        signal_rows.append(signal_row)
        marker_signal_issues.extend(signal_issues)

    duration = None
    if row["duration_sec_if_possible"] != "":
        try:
            duration = float(row["duration_sec_if_possible"])
        except Exception:
            duration = None
    for marker_name in ("Marker", "MarkerWalk"):
        if not row[f"has_{marker_name}"]:
            continue
        summary, issues = marker_summary(loaded_small.get(marker_name), marker_name, duration, task_guess)
        marker_rows.append(
            {
                "relative_path": rel,
                "patient_id": patient,
                "task_guess": task_guess,
                **summary,
                "notes": issues,
            }
        )
        marker_signal_issues.extend(f"{issue}:{rel}" for issue in issues)
    if raw_like and task_guess in {"StepSitting", "StepStanding", "FreeWalking"} and not row["has_Marker"]:
        marker_signal_issues.append(f"movement_file_missing_Marker:{rel}")
    if raw_like and task_guess == "FreeWalking" and not row["has_MarkerWalk"]:
        marker_signal_issues.append(f"freewalking_file_missing_MarkerWalk:{rel}")

    return row, channel_rows, marker_rows, signal_rows, mat_issues, channel_issues, marker_signal_issues


def code_audit(data_root: Path, manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    code_paths = [data_root / row["relative_path"] for row in manifest_rows if row["is_possible_code_file"]]
    matlab_paths = [p for p in code_paths if file_extension(p) in {".m", ".mlx"}]
    readme_paths = [
        data_root / row["relative_path"]
        for row in manifest_rows
        if Path(str(row["relative_path"])).name.lower().startswith("readme")
        or Path(str(row["relative_path"])).name.lower() in {"license", "description.txt"}
    ]
    main_scripts = []
    toolbox_dirs = set()
    feature_hits = {
        "preprocessing": False,
        "psd": False,
        "coherence": False,
        "gait_phase_detection": False,
        "modulation_index": False,
        "permutation_statistics": False,
        "figure_generation": False,
    }
    keyword_map = {
        "preprocessing": ["preprocess", "filter", "downsample", "filtfilt"],
        "psd": ["psd", "pwelch", "power spectral"],
        "coherence": ["coherence", "mscohere", "coh_fft", "coh_wavelet"],
        "gait_phase_detection": ["markerwalk", "gait", "phase", "force", "zero-crossing", "zerocross"],
        "modulation_index": ["modulationindex", "modulation index"],
        "permutation_statistics": ["permutation", "perm", "clustermass", "cluster mass"],
        "figure_generation": ["figure(", "saveas", "print(", "export_fig"],
    }
    for path in matlab_paths:
        rel = relpath(path, data_root)
        rel_low = rel.lower()
        if "/toolbox/" in rel_low:
            parts = rel.split("/")
            try:
                toolbox_idx = [part.lower() for part in parts].index("toolbox")
                if toolbox_idx + 1 < len(parts):
                    toolbox_dirs.add("/".join(parts[: toolbox_idx + 2]))
            except ValueError:
                pass
        if "/toolbox/" not in rel_low and (path.name.lower().endswith(".m") or "analysis" in path.name.lower()):
            main_scripts.append(rel)
        try:
            text = read_text_limited(path, 131072).lower()
        except Exception:
            continue
        for feature, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                feature_hits[feature] = True
    return {
        "total_code_files": len(code_paths),
        "total_matlab_scripts_functions": len(matlab_paths),
        "main_scripts": sorted(main_scripts),
        "readme_or_description_files": sorted(relpath(p, data_root) for p in readme_paths),
        "toolbox_dirs": sorted(toolbox_dirs),
        "feature_hits": feature_hits,
    }


def summarize_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(str(row.get(key, "")) for row in rows).items())}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_readme(
    out_dir: Path,
    data_root: Path,
    audit_timestamp: str,
    manifest_rows: list[dict[str, Any]],
    patient_rows: list[dict[str, Any]],
    mat_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    marker_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    findings: dict[str, Any],
    code_summary: dict[str, Any],
    commands: list[str],
) -> str:
    extension_counts = summarize_counts(manifest_rows, "extension")
    patient_counts = summarize_counts([r for r in manifest_rows if r.get("inferred_patient_id")], "inferred_patient_id")
    raw_like_count = sum(1 for row in mat_rows if row.get("likely_raw_or_result") == "raw_like")
    unreadable_count = sum(1 for row in mat_rows if row.get("load_status") == "unreadable")
    modality_counts = summarize_counts(channel_rows, "modality_guess")
    marker_issue_rows = [row for row in marker_rows if row.get("notes")]
    signal_issue_rows = [row for row in signal_rows if row.get("notes")]
    coverage_rows = []
    for row in patient_rows:
        coverage_rows.append(
            [
                row["patient_id"],
                row["group"],
                "Y" if row["observed_RestSitting"] else "N",
                "Y" if row["observed_RestStanding"] else "N",
                "Y" if row["observed_StepSitting"] else "N",
                "Y" if row["observed_StepStanding"] else "N",
                "Y" if row["observed_FreeWalking"] else "N",
                clean_join(row["notes"]),
            ]
        )
    large_files = [row for row in manifest_rows if int(row["size_bytes"]) >= LARGE_FILE_BYTES]
    scan_files = findings["possible_scan_or_identifiable_files"]
    recommendations = findings["recommended_next_steps"]
    lines = [
        "# Phase 6A Dataset Audit: PPN He/Tan 2021",
        "",
        "## Scope",
        "",
        (
            "Dataset: LFPs and EEGs from patients with Parkinson's disease or multiple system "
            "atrophy during gait, associated with He et al. 2021, "
            "Gait-Phase Modulates Alpha and Beta Oscillations in the Pedunculopontine Nucleus."
        ),
        "",
        (
            "This is a Phase 6A audit only. It inventories files, expected protocol coverage, "
            "MATLAB schemas, channel/modalities, markers, lightweight signal sanity, Matlab code, "
            "and privacy/governance risks. It does not run preprocessing, modeling, PSD, coherence, "
            "gait-phase analysis, or figure reproduction."
        ),
        "",
        "## Dataset Root And Environment",
        "",
        f"- Dataset root audited: `{data_root}`",
        f"- Audit timestamp: `{audit_timestamp}`",
        "- Environment: existing repo `stn_env` only. `conda run -n stn_env python -V` was attempted but the named conda env was not registered; the repo-local `/scratch/haizhe/stn/stn_env` Python was used via `start_stn.sh` or direct path.",
        f"- Optional readers: scipy `{('available' if scipy_io is not None else 'missing')}`, h5py `{('available' if h5py is not None else 'missing')}`.",
        "",
        "## Patients And Protocol Coverage",
        "",
        f"- Expected patients: `{', '.join(EXPECTED_PATIENTS)}`",
        f"- Observed patients: `{', '.join(findings['observed_patients'])}`",
        f"- Missing expected patient folders: `{', '.join(findings['missing_patient_folders']) or 'none'}`",
        "",
        markdown_table(
            ["patient", "group", "RestSitting", "RestStanding", "StepSitting", "StepStanding", "FreeWalking", "notes"],
            coverage_rows,
        ),
        "",
        "## File Inventory Summary",
        "",
        f"- Total files: `{findings['total_files']}`",
        f"- MAT files: `{findings['total_mat_files']}`",
        f"- TXT files: `{findings['total_txt_files']}`",
        f"- Code files: `{findings['total_code_files']}`",
        f"- Files >= {LARGE_FILE_BYTES} bytes: `{len(large_files)}`",
        "",
        markdown_table(["extension", "count"], [[k, v] for k, v in extension_counts.items()]),
        "",
        "Patient-level file counts:",
        "",
        markdown_table(["patient", "file_count"], [[k, v] for k, v in patient_counts.items()]),
        "",
        "## Raw-Like MAT Schema Summary",
        "",
        f"- Total MAT files inspected: `{len(mat_rows)}`",
        f"- Raw-like MAT files with `ChannelName`, `ChannelType`, and `data`: `{raw_like_count}`",
        f"- Unreadable MAT files: `{unreadable_count}`",
        f"- Explicit sampling-rate variables found: `{sum(1 for row in mat_rows if row.get('explicit_fs_found'))}`",
        "- When no file-level sampling-rate variable was found, `2048 Hz` was recorded as `paper default / assumed` only.",
        "",
        "## Channel And Modality Summary",
        "",
        f"- Channel rows inventoried: `{len(channel_rows)}`",
        "",
        markdown_table(["modality_guess", "channel_rows"], [[k, v] for k, v in modality_counts.items()]),
        "",
        f"- Channel issues recorded: `{len(findings['channel_issues'])}`",
        "",
        "## Marker Summary",
        "",
        f"- Marker objects inventoried: `{len(marker_rows)}`",
        f"- Marker rows with issues: `{len(marker_issue_rows)}`",
        "- Rest recordings may have no meaningful markers; missing movement markers are flagged only for stepping/free-walking raw-like files.",
        "",
        "## Signal Sanity Summary",
        "",
        f"- Raw-like files with signal sanity rows: `{len(signal_rows)}`",
        f"- Signal rows with notes/issues: `{len(signal_issue_rows)}`",
        "- Metrics are aggregate/bounded sanity checks only: shapes, missingness, Inf/NaN counts, all-zero/constant channel flags, and aggregate finite-value ranges. No raw time-series samples are written.",
        "",
        "## Matlab Code And Reproducibility Audit",
        "",
        f"- Matlab scripts/functions: `{code_summary['total_matlab_scripts_functions']}`",
        f"- Obvious main scripts: `{clean_join(code_summary['main_scripts']) or 'none detected'}`",
        f"- Readme/description files: `{clean_join(code_summary['readme_or_description_files']) or 'none detected'}`",
        f"- Toolbox-like directories: `{clean_join(code_summary['toolbox_dirs']) or 'none detected'}`",
        "",
        "Feature keywords observed in Matlab code:",
        "",
        markdown_table(
            ["feature", "observed"],
            [[feature, "Y" if observed else "N"] for feature, observed in code_summary["feature_hits"].items()],
        ),
        "",
        "## Privacy And Governance Findings",
        "",
        f"- DICOM/NIfTI/MRI/CT-like files flagged: `{len(scan_files)}`",
        f"- Flagged files: `{clean_join(scan_files) or 'none'}`",
        "- No scan-like files were opened or processed deeply by this audit.",
        "- Patient identifiers observed in paths are summarized at the folder/filename level only; this audit does not inspect private imaging metadata.",
        "",
        "## Known Issues And Ambiguities",
        "",
        f"- Missing referenced MAT files: `{clean_join(findings['missing_referenced_mat_files']) or 'none'}`",
        f"- Orphan MAT files not referenced by protocol TXT files: `{len(findings['orphan_mat_files'])}`. Many are expected generated results or toolbox outputs; review `patient_task_matrix.csv` before using them.",
        f"- Channel issues: `{len(findings['channel_issues'])}`; see `channel_inventory.csv` and `audit_findings.json`.",
        f"- Marker issues: `{len(findings['marker_issues'])}`; see `marker_inventory.csv` and `audit_findings.json`.",
        f"- Signal issues: `{len(findings['signal_issues'])}`; see `signal_sanity_summary.csv` and `audit_findings.json`.",
        f"- Overall status: `{findings['overall_status']}`",
        "",
        "## Recommendations For Phase 6B",
        "",
    ]
    for item in recommendations:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Exact Commands Run",
            "",
            "See also `phase6a_commands_run.txt`.",
            "",
        ]
    )
    for command in commands:
        lines.append(f"- `{command}`")
    lines.append("")
    return "\n".join(lines)


def command_log_default(argv: list[str]) -> list[str]:
    return [
        "pwd",
        "git rev-parse --show-toplevel",
        "git status --short",
        "conda run -n stn_env python -V",
        "find . -maxdepth 3 -type d | sort | head -200",
        "find . -maxdepth 3 -type f | sort | head -200",
        "source /scratch/haizhe/stn/start_stn.sh && python -V",
        "/scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6_audit_ppn_he_tan_2021.py",
        " ".join([sys.executable, *argv]),
        "/scratch/haizhe/stn/stn_env/bin/python -m py_compile scripts/phase6_audit_ppn_he_tan_2021.py",
        "ls -lh reports/phase6_ppn_he_tan_2021_audit",
        "head -50 reports/phase6_ppn_he_tan_2021_audit/README_dataset_audit.md",
        "git status --short",
    ]


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    data_root = data_root.resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_timestamp = utc_local_timestamp()
    commands = command_log_default(sys.argv)

    if not data_root.exists():
        raise SystemExit(f"data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise SystemExit(f"data root is not a directory: {data_root}")

    manifest_rows = inventory_files(data_root)
    patient_rows, missing_refs, orphan_mats = build_patient_task_matrix(data_root, manifest_rows)

    mat_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    mat_issues: list[str] = []
    channel_issues: list[str] = []
    marker_signal_issues: list[str] = []

    mat_paths = [data_root / row["relative_path"] for row in manifest_rows if row["is_mat_file"]]
    for mat_path in mat_paths:
        mat_row, ch_rows, mk_rows, sig_rows, mi, ci, msi = inspect_mat_file(
            mat_path,
            data_root,
            args.paper_fs,
            args.max_sanity_channels,
        )
        mat_rows.append({k: v for k, v in mat_row.items() if not k.startswith("_")})
        channel_rows.extend(ch_rows)
        marker_rows.extend(mk_rows)
        signal_rows.extend(sig_rows)
        mat_issues.extend(mi)
        channel_issues.extend(ci)
        marker_signal_issues.extend(msi)

    code_summary = code_audit(data_root, manifest_rows)

    observed_patients = sorted(
        {
            p.name.upper()
            for p in data_root.rglob("*")
            if p.is_dir() and re.fullmatch(r"(PD|MSA)\d{2}", p.name, flags=re.IGNORECASE)
        }
    )
    missing_patient_folders = sorted(set(EXPECTED_PATIENTS) - set(observed_patients))
    extra_patient_like_folders = sorted(set(observed_patients) - set(EXPECTED_PATIENTS))
    scan_files = sorted(
        row["relative_path"] for row in manifest_rows if row["is_possible_scan_or_identifiable_file"]
    )
    marker_issues = sorted({issue for issue in marker_signal_issues if "marker" in issue.lower()})
    signal_issues = sorted({issue for issue in marker_signal_issues if "marker" not in issue.lower()})
    raw_like_count = sum(1 for row in mat_rows if row.get("likely_raw_or_result") == "raw_like")

    recommended_next_steps = [
        "Use only raw-like MAT files with ChannelName, ChannelType, data, and valid movement markers as Phase 6B candidates.",
        "Resolve orphan MAT files by separating raw recordings from generated PSD, gait modulation, permutation, HMM, and IC2019 outputs.",
        "Treat 2048 Hz as paper default only until a file-level sampling-rate source is confirmed or the original acquisition metadata is located.",
        "Review channel/modality flags before choosing PPN LFP, EEG Cz/Fz, force, and accelerometer channels for Phase 6B.",
        "Validate Marker and MarkerWalk timing against data duration before any gait-phase segmentation.",
        "Port/reuse Matlab preprocessing only after Phase 6B defines a reproducible Python/Brian2-compatible input contract.",
    ]
    if scan_files:
        recommended_next_steps.insert(0, "Quarantine and governance-review scan-like files before any further processing.")
    if missing_refs:
        recommended_next_steps.insert(0, "Resolve missing protocol TXT to MAT references before Phase 6B.")

    if scan_files:
        overall_status = "completed_with_privacy_flags"
    elif missing_refs or marker_issues or channel_issues or signal_issues:
        overall_status = "completed_with_audit_issues"
    else:
        overall_status = "completed_no_critical_issues"

    findings = {
        "dataset_root": str(data_root),
        "audit_timestamp": audit_timestamp,
        "expected_patients": EXPECTED_PATIENTS,
        "observed_patients": observed_patients,
        "missing_patient_folders": missing_patient_folders,
        "extra_patient_like_folders": extra_patient_like_folders,
        "total_files": len(manifest_rows),
        "total_mat_files": sum(1 for row in manifest_rows if row["is_mat_file"]),
        "total_txt_files": sum(1 for row in manifest_rows if row["is_txt_file"]),
        "total_code_files": code_summary["total_code_files"],
        "total_raw_like_mat_files": raw_like_count,
        "possible_scan_or_identifiable_files": scan_files,
        "missing_referenced_mat_files": missing_refs,
        "orphan_mat_files": orphan_mats,
        "channel_issues": sorted(set(channel_issues)),
        "marker_issues": marker_issues,
        "signal_issues": signal_issues,
        "mat_issues": sorted(set(mat_issues)),
        "overall_status": overall_status,
        "recommended_next_steps": recommended_next_steps,
    }

    write_csv(
        out_dir / "dataset_manifest.csv",
        manifest_rows,
        [
            "relative_path",
            "parent_folder",
            "extension",
            "size_bytes",
            "size_mb",
            "inferred_patient_id",
            "possible_protocol",
            "is_mat_file",
            "is_txt_file",
            "is_possible_code_file",
            "is_possible_scan_or_identifiable_file",
            "notes",
        ],
    )
    write_csv(
        out_dir / "patient_task_matrix.csv",
        patient_rows,
        [
            "patient_id",
            "group",
            "expected_RestSitting",
            "observed_RestSitting",
            "expected_RestStanding",
            "observed_RestStanding",
            "expected_StepSitting",
            "observed_StepSitting",
            "expected_StepStanding",
            "observed_StepStanding",
            "expected_FreeWalking",
            "observed_FreeWalking",
            "txt_files",
            "referenced_mat_files",
            "missing_referenced_mat_files",
            "orphan_mat_files",
            "notes",
        ],
    )
    write_csv(
        out_dir / "mat_file_inventory.csv",
        mat_rows,
        [
            "relative_path",
            "patient_id",
            "file_stem",
            "size_mb",
            "mat_format_guess",
            "load_status",
            "error_message",
            "top_level_keys",
            "has_ChannelName",
            "has_ChannelType",
            "has_data",
            "has_Marker",
            "has_MarkerWalk",
            "data_shape",
            "data_dtype",
            "inferred_n_channels",
            "inferred_n_samples",
            "inferred_time_axis",
            "explicit_fs_found",
            "fs_value",
            "fs_source",
            "duration_sec_if_possible",
            "likely_raw_or_result",
            "notes",
        ],
    )
    write_csv(
        out_dir / "channel_inventory.csv",
        channel_rows,
        [
            "relative_path",
            "patient_id",
            "task_guess",
            "channel_index",
            "channel_name",
            "channel_type",
            "modality_guess",
            "is_lfp_guess",
            "is_eeg_guess",
            "is_force_guess",
            "is_accelerometer_guess",
            "is_marker_like_guess",
            "is_cz_or_fz_guess",
            "notes",
        ],
    )
    write_csv(
        out_dir / "marker_inventory.csv",
        marker_rows,
        [
            "relative_path",
            "patient_id",
            "task_guess",
            "marker_object",
            "has_Marks",
            "has_Times",
            "n_marks",
            "n_times",
            "marker_code_counts",
            "marker_code_0_count",
            "marker_code_1_count",
            "marker_code_16_count",
            "n_start_end_blocks",
            "start_end_balanced",
            "times_monotonic",
            "times_within_data_duration",
            "block_duration_min_sec",
            "block_duration_median_sec",
            "block_duration_max_sec",
            "walk_phase_timestamp_count",
            "notes",
        ],
    )
    write_csv(
        out_dir / "signal_sanity_summary.csv",
        signal_rows,
        [
            "relative_path",
            "patient_id",
            "task_guess",
            "n_channels",
            "n_samples",
            "duration_sec_if_possible",
            "data_numeric",
            "nan_count",
            "nan_percent",
            "inf_count",
            "inf_percent",
            "all_zero_channel_count",
            "all_zero_channels",
            "constant_channel_count",
            "constant_channels",
            "aggregate_mean",
            "aggregate_std",
            "aggregate_min",
            "aggregate_max",
            "channels_summarized",
            "notes",
        ],
    )
    with (out_dir / "audit_findings.json").open("w", encoding="utf-8") as handle:
        json.dump(findings, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    readme = make_readme(
        out_dir,
        data_root,
        audit_timestamp,
        manifest_rows,
        patient_rows,
        mat_rows,
        channel_rows,
        marker_rows,
        signal_rows,
        findings,
        code_summary,
        commands,
    )
    (out_dir / "README_dataset_audit.md").write_text(readme, encoding="utf-8")
    (out_dir / "phase6a_commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    print(json.dumps({k: findings[k] for k in [
        "overall_status",
        "total_files",
        "total_mat_files",
        "total_txt_files",
        "total_code_files",
        "total_raw_like_mat_files",
    ]}, indent=2, sort_keys=True))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

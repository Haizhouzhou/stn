"""Read-only helpers for consuming frozen Phase 3 labels in Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from stnbeta.ground_truth.pipeline import get_epoch_mask


@dataclass(frozen=True)
class RealConditionCase:
    """One condition/channel dev case aligned to a single extracted fif file."""

    subject_id: str
    condition: str
    channel: str
    band_mode: str
    fif_path: Path
    sfreq_hz: float
    signal: np.ndarray
    task_mask: np.ndarray
    burst_mask: np.ndarray
    events: pd.DataFrame


def _condition_suffix(description: str) -> str | None:
    lowered = description.lower()
    if lowered == "rest":
        return "Rest"
    if lowered.startswith("hold"):
        return "Hold"
    if lowered.startswith("move"):
        return "Move"
    return None


def _task_mask_for_condition(raw: mne.io.BaseRaw, condition_suffix: str) -> np.ndarray:
    mask = np.zeros(len(raw.times), dtype=bool)
    descriptions = sorted({ann["description"] for ann in raw.annotations})
    for description in descriptions:
        if _condition_suffix(description) == condition_suffix:
            mask |= get_epoch_mask(raw, description)
    return mask


def _parse_entities(name: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in name.replace("_lfp.fif", "").split("_"):
        if "-" in token:
            key, value = token.split("-", 1)
            entities[key] = value
    return entities


def _matching_condition_files(subject_id: str, extracted_root: Path, condition: str) -> list[Path]:
    acq, suffix = condition.split("_", 1)
    fif_dir = extracted_root / subject_id / "ses-PeriOp" / "meg"
    files: list[Path] = []
    for fif_path in sorted(fif_dir.glob("*_lfp.fif")):
        entities = _parse_entities(fif_path.name)
        if entities.get("acq") != acq:
            continue
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
        if _task_mask_for_condition(raw, suffix).any():
            files.append(fif_path)
    return files


def event_table_to_mask(events: pd.DataFrame, n_samples: int, sfreq_hz: float) -> np.ndarray:
    """Convert Phase 3 onset/offset rows to a sample mask."""
    mask = np.zeros(n_samples, dtype=bool)
    for row in events.itertuples(index=False):
        start = max(0, int(round(float(row.onset_s) * sfreq_hz)))
        stop = min(n_samples, int(round(float(row.offset_s) * sfreq_hz)))
        if stop > start:
            mask[start:stop] = True
    return mask


def load_real_condition_case(
    *,
    subject_id: str,
    condition: str,
    channel: str,
    extracted_root: Path,
    bursts_root: Path,
    band_mode: str = "fixed_13_30",
) -> RealConditionCase:
    """Load one single-file condition case aligned to frozen Phase 3 events."""
    matches = _matching_condition_files(subject_id, extracted_root, condition)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one fif file for {subject_id} {condition}, found {len(matches)}"
        )

    fif_path = matches[0]
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    signal = raw.get_data(picks=[channel])[0].astype(np.float32)
    task_mask = _task_mask_for_condition(raw, condition.split("_", 1)[1])

    event_path = bursts_root / subject_id / band_mode / f"{condition}_{channel}.parquet"
    events = pd.read_parquet(event_path)
    burst_mask = event_table_to_mask(events, len(signal), float(raw.info["sfreq"]))
    return RealConditionCase(
        subject_id=subject_id,
        condition=condition,
        channel=channel,
        band_mode=band_mode,
        fif_path=fif_path,
        sfreq_hz=float(raw.info["sfreq"]),
        signal=signal,
        task_mask=task_mask,
        burst_mask=burst_mask,
        events=events,
    )

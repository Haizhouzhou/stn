"""State occupancy and stable readout helpers for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReadoutSummary:
    """Continuous and event-level readout derived from state occupancy."""

    score: np.ndarray
    stable_mask: np.ndarray
    event_onsets_s: np.ndarray
    event_offsets_s: np.ndarray
    state_masks: dict[str, np.ndarray]


def state_active_masks(
    occupancy: np.ndarray,
    state_names: tuple[str, ...],
    *,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Return per-state active masks from occupancy traces."""
    occupancy_array = np.asarray(occupancy, dtype=np.float32)
    return {
        state_name: occupancy_array[index] >= threshold
        for index, state_name in enumerate(state_names)
    }


def _apply_dwell(mask: np.ndarray, dwell_samples: int) -> np.ndarray:
    if dwell_samples <= 1:
        return np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= dwell_samples:
            out[index] = True
    return out


def _bridge_short_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
    if gap_samples <= 0:
        return np.asarray(mask, dtype=bool)
    out = np.asarray(mask, dtype=bool).copy()
    start = None
    for index, value in enumerate(out):
        if value:
            if start is not None and 0 < index - start <= gap_samples:
                out[start:index] = True
            start = None
        elif start is None and out[:index].any():
            start = index
    return out


def _event_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_int = np.asarray(mask, dtype=np.int8)
    padded = np.pad(mask_int, (1, 1))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    return starts, stops


def events_from_mask(mask: np.ndarray, *, dt_ms: float) -> pd.DataFrame:
    """Convert a boolean mask to onset/offset events."""
    starts, stops = _event_edges(mask)
    return pd.DataFrame(
        {
            "onset_s": starts * dt_ms / 1000.0,
            "offset_s": stops * dt_ms / 1000.0,
            "duration_ms": (stops - starts) * dt_ms,
        }
    )


def detect_stable_events(
    score: np.ndarray,
    *,
    threshold: float,
    dwell_ms: float,
    dt_ms: float,
    gap_bridge_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the stable readout mask plus onset/offset times."""
    score_array = np.asarray(score, dtype=np.float32)
    dwell_samples = max(1, int(round(dwell_ms / dt_ms)))
    stable_mask = _apply_dwell(score_array >= threshold, dwell_samples)
    stable_mask = _bridge_short_gaps(stable_mask, max(0, int(round(gap_bridge_ms / dt_ms))))
    starts, stops = _event_edges(stable_mask)
    return (
        stable_mask,
        starts * dt_ms / 1000.0,
        stops * dt_ms / 1000.0,
    )


def build_readout_summary(result: Any, config: Any) -> ReadoutSummary:
    """Derive stable burst-detected events from occupancy plus readout trace."""
    occupancy = np.asarray(result.occupancy, dtype=np.float32)
    threshold_state_index = int(config.readout_bucket_index) + 1
    state_score = occupancy[threshold_state_index:].sum(axis=0)
    readout_trace = np.asarray(getattr(result, "readout_trace", np.zeros_like(state_score)), dtype=np.float32)
    score = np.maximum(state_score, readout_trace)
    stable_mask, event_onsets_s, event_offsets_s = detect_stable_events(
        score,
        threshold=config.occupancy_active_threshold,
        dwell_ms=config.readout_dwell_ms,
        dt_ms=config.dt_ms,
        gap_bridge_ms=getattr(config, "quiet_holdoff_ms", 0.0),
    )
    return ReadoutSummary(
        score=score,
        stable_mask=stable_mask,
        event_onsets_s=event_onsets_s,
        event_offsets_s=event_offsets_s,
        state_masks=state_active_masks(
            occupancy,
            tuple(result.state_names),
            threshold=config.occupancy_active_threshold,
        ),
    )

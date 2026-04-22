"""Lateralized MDS-UPDRS Part III subscores from participants_updrs_off.tsv.

Column schema (verified from data):
  participant_id, MEG_UPDRS_SAMEDAY,
  3_1, 3_2,                                    — axial: speech, facial expression
  3_3_a,                                        — axial: neck rigidity
  3_3_b, 3_3_c, 3_3_d, 3_3_e,                  — rigidity: R/L upper / R/L lower
  3_4_a/b … 3_8_a/b,                            — finger tap, hand mvt, pro-sup, toe tap, leg agility (R/L)
  3_9 … 3_14,                                   — axial: arising, gait, freezing, posture, global
  3_15_a/b, 3_16_a/b,                           — postural/kinetic tremor (R/L) — NOT counted in contralateral
  3_17_a, 3_17_b, 3_17_c, 3_17_d, 3_17_e,      — rest tremor amplitude: RUE/LUE/RLE/LLE/jaw
  3_18,                                          — constancy of rest tremor (bilateral)
  SUM, AR right, AR left, trem right, trem left, AR sum, trem sum, axial

Per-side items used for contralateral subscore (Tinkhauser 2017 / spec):
  3.3 (rigidity), 3.4 (finger tapping), 3.5 (hand movements),
  3.6 (pronation-supination), 3.7 (toe tapping), 3.8 (leg agility),
  3.17 (rest tremor amplitude — upper and lower extremity only, not jaw).

Verification: AR right = 3_3_b + 3_3_d + 3_4_a + 3_5_a + 3_6_a + 3_7_a + 3_8_a (confirmed).
So contralateral = AR_contra + 3_17_ue_contra + 3_17_le_contra.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pre-computed akinesia+rigidity columns (exclude neck 3_3_a which is bilateral)
_AR_COL = {"right": "AR right", "left": "AR left"}

# Rest tremor amplitude: upper + lower extremity (jaw 3_17_e is bilateral)
_TREMOR_17_COLS = {
    "right": ["3_17_a", "3_17_c"],   # RUE, RLE
    "left":  ["3_17_b", "3_17_d"],   # LUE, LLE
}


def load_updrs(tsv_path: Path) -> pd.DataFrame:
    """Load participants_updrs_off.tsv, returning a DataFrame indexed by participant_id."""
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.set_index("participant_id")
    return df


def get_updrs_lateralized(
    participant_id: str,
    side: str,
    tsv_df: pd.DataFrame,
) -> dict[str, float | None]:
    """Return UPDRS-III subscores for one (subject, STN hemisphere) pair.

    side: 'left' or 'right' — the STN hemisphere recorded from.
    Contralateral is the body side opposite to *side*
    (left STN → right body symptoms; right STN → left body symptoms).

    Returns:
        total:          SUM column (full MDS-UPDRS Part III)
        contralateral:  AR_contra + rest-tremor-amplitude_contra (per-side items only)
        axial:          pre-computed 'axial' column
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    contra_side = "left" if side == "right" else "right"

    if participant_id not in tsv_df.index:
        logger.warning(
            "participant_id %s not in UPDRS TSV — returning None for all scores",
            participant_id,
        )
        return {"total": None, "contralateral": None, "axial": None}

    row = tsv_df.loc[participant_id]

    total = _safe_float(row.get("SUM"))

    ar_contra = _safe_float(row.get(_AR_COL[contra_side]))
    trem_cols = _TREMOR_17_COLS[contra_side]
    trem_vals = [_safe_float(row.get(c)) for c in trem_cols]

    if ar_contra is None or any(v is None for v in trem_vals):
        logger.warning(
            "%s: missing lateralized UPDRS items for %s side — contralateral=None",
            participant_id, contra_side,
        )
        contralateral = None
    else:
        contralateral = ar_contra + float(np.nansum(trem_vals))

    axial = _safe_float(row.get("axial"))

    return {"total": total, "contralateral": contralateral, "axial": axial}


def _safe_float(val) -> float | None:
    """Convert to float, returning None for NaN / missing."""
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None

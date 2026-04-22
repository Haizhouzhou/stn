"""Per-subject beta-burst pipeline: load → bandpass → envelope → threshold → label → save.

Key correctness invariants (v2):
- Threshold is derived exclusively from 'rest' annotation epochs within MedOff recordings.
  Every MedOff file (task-Rest, task-HoldL, task-MoveL, …) embeds ~5 min of rest at the
  start; these are found via events.tsv annotations attached to each fif by
  06_attach_annotations.py.  18 of 20 subjects have no task-Rest file; their rest data
  lives inside the Hold/Move files.
- BAD_* annotations (uppercased from bad_lfp by 06_attach_annotations.py) are excluded
  from every mask: threshold computation, FOOOF fitting, and burst labeling.
- Burst labeling is restricted to the relevant task epoch within each condition file;
  samples outside the epoch window are zeroed before label_bursts().
- Thresholds are computed from the concatenated rest-masked envelope across ALL MedOff
  files (not just the first one).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from stnbeta.ground_truth.bursts import (
    bandpass,
    burst_stats,
    hilbert_envelope,
    label_bursts,
)
from stnbeta.ground_truth.fooof_band import fit_individual_beta

logger = logging.getLogger(__name__)

LONG_BURST_MS = 500.0
_BIPOLAR_RE = re.compile(r"^LFP-(left|right)-\d{2}$")
MIN_REST_S = 60.0          # exclude subject if total clean rest < this
MIN_REST_S_PER_FILE = 10.0  # skip a single file's rest contribution if < this


# ── Small helpers ─────────────────────────────────────────────────────────────

def _bipolar_picks(raw: mne.io.BaseRaw) -> list[str]:
    return [c for c in raw.ch_names if _BIPOLAR_RE.match(c)]


def _parse_entities(fname: str) -> dict:
    out: dict[str, str] = {}
    for token in fname.replace("_lfp.fif", "").split("_"):
        if "-" in token:
            k, v = token.split("-", 1)
            out[k] = v
    return out


# ── Epoch masking (public — tested in tests/test_ground_truth.py) ─────────────

def get_epoch_mask(raw: mne.io.BaseRaw, description: str) -> np.ndarray:
    """Boolean mask over raw samples: True where *description* epoch is active AND no BAD_* overlaps."""
    n = len(raw.times)
    sfreq = raw.info["sfreq"]
    in_epoch = np.zeros(n, dtype=bool)
    is_bad = np.zeros(n, dtype=bool)
    for ann in raw.annotations:
        s = int(round(ann["onset"] * sfreq))
        e = int(round((ann["onset"] + ann["duration"]) * sfreq))
        s, e = max(0, s), min(n, e)
        if ann["description"].upper().startswith("BAD"):
            is_bad[s:e] = True
        elif ann["description"] == description:
            in_epoch[s:e] = True
    return in_epoch & ~is_bad


def _epoch_type_to_cond_suffix(description: str) -> str | None:
    """Map annotation description to condition suffix (Rest/Hold/Move), or None."""
    d = description.strip()
    if d.upper().startswith("BAD"):
        return None
    if d.lower() == "rest":
        return "Rest"
    if re.match(r"(?i)^hold", d):
        return "Hold"
    if re.match(r"(?i)^move", d):
        return "Move"
    return None


def _discover_epoch_types(raw: mne.io.BaseRaw) -> list[tuple[str, str]]:
    """Return (annotation_description, cond_suffix) for non-BAD task epochs present in raw."""
    seen: dict[str, str] = {}
    for ann in raw.annotations:
        suffix = _epoch_type_to_cond_suffix(ann["description"])
        if suffix is not None and ann["description"] not in seen:
            seen[ann["description"]] = suffix
    return list(seen.items())


# ── Main subject pipeline ─────────────────────────────────────────────────────

def run_subject(
    subject_id: str,
    bids_root: Path,
    extracted_root: Path,
    out_dir: Path,
    band_mode: str = "fixed_13_30",
) -> dict:
    """Extract burst stats for all epoch conditions and band modes for one subject.

    Returns a dict with keys:
        subject_id, thresh_cond, n_bipolar, rest_duration_s, rows (list of stat dicts).
    """
    fif_dir = extracted_root / subject_id / "ses-PeriOp" / "meg"
    if not fif_dir.exists():
        logger.error("No fif dir for %s at %s", subject_id, fif_dir)
        return {"subject_id": subject_id, "error": "no_fif_dir", "rows": []}

    all_fifs = sorted(fif_dir.glob("*_lfp.fif"))
    if not all_fifs:
        logger.error("No _lfp.fif files for %s", subject_id)
        return {"subject_id": subject_id, "error": "no_fifs", "rows": []}

    medoff_fifs = [
        f for f in all_fifs
        if _parse_entities(f.name).get("acq", "") == "MedOff"
    ]
    if not medoff_fifs:
        logger.error("No MedOff fifs for %s", subject_id)
        return {"subject_id": subject_id, "error": "no_medoff_data", "rows": []}

    # Reference channel list and sfreq from the first available fif
    ref_raw = mne.io.read_raw_fif(all_fifs[0], preload=False, verbose="ERROR")
    bipolar_picks = _bipolar_picks(ref_raw)
    sfreq = float(ref_raw.info["sfreq"])
    del ref_raw

    if not bipolar_picks:
        logger.error("No bipolar LFP channels for %s", subject_id)
        return {"subject_id": subject_id, "error": "no_bipolar", "rows": []}

    band_modes_to_run = (
        ["fixed_13_30", "individualized"] if band_mode == "both" else [band_mode]
    )

    # ── Step 1: fit individualized bands (FOOOF) from first MedOff file with ≥30 s rest ──
    individualized_bands: dict[str, tuple[float, float]] = {}
    if "individualized" in band_modes_to_run:
        fooof_raw = fooof_mask = None
        for fif_path in medoff_fifs:
            _r = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
            _m = get_epoch_mask(_r, "rest")
            if _m.sum() >= sfreq * 30:
                fooof_raw, fooof_mask = _r, _m
                break
            del _r
        if fooof_raw is None:
            logger.warning(
                "%s: no MedOff file has ≥30 s clean rest for FOOOF; defaulting to 13–30 Hz",
                subject_id,
            )
            for ch in bipolar_picks:
                individualized_bands[ch] = (13.0, 30.0)
        else:
            for ch in bipolar_picks:
                lo, hi = fit_individual_beta(fooof_raw, ch, mask=fooof_mask)
                individualized_bands[ch] = (lo, hi)
            del fooof_raw

    # ── Step 2: accumulate rest-masked envelope across all MedOff files for threshold ──
    # rest_segs[bm][ch_idx] → list of 1-D arrays (clean rest envelope samples)
    rest_segs: dict[str, list[list]] = {
        bm: [[] for _ in bipolar_picks] for bm in band_modes_to_run
    }
    total_rest_s = 0.0

    for fif_path in medoff_fifs:
        raw_mo = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
        rest_mask = get_epoch_mask(raw_mo, "rest")
        rest_s = float(rest_mask.sum()) / sfreq

        if rest_s < MIN_REST_S_PER_FILE:
            logger.warning(
                "%s: only %.1f s clean rest in %s — skipping for threshold",
                subject_id, rest_s, fif_path.name,
            )
            del raw_mo
            continue

        total_rest_s += rest_s
        file_picks_set = set(_bipolar_picks(raw_mo))

        if "fixed_13_30" in band_modes_to_run:
            common = [p for p in bipolar_picks if p in file_picks_set]
            if common:
                bp = bandpass(raw_mo, 13.0, 30.0, common)   # (n_common, n_samples)
                env = hilbert_envelope(bp, sfreq)
                for i_c, ch in enumerate(common):
                    rest_segs["fixed_13_30"][bipolar_picks.index(ch)].append(
                        env[i_c, rest_mask]
                    )

        if "individualized" in band_modes_to_run:
            for i, ch in enumerate(bipolar_picks):
                if ch not in file_picks_set:
                    continue
                lo, hi = individualized_bands.get(ch, (13.0, 30.0))
                bp_ch = bandpass(raw_mo, lo, hi, [ch])      # (1, n_samples)
                env_ch = hilbert_envelope(bp_ch, sfreq)
                rest_segs["individualized"][i].append(env_ch[0, rest_mask])

        del raw_mo

    if total_rest_s < MIN_REST_S:
        logger.error(
            "%s: only %.1f s total clean MedOff rest — excluding subject "
            "(need ≥%.0f s)",
            subject_id, total_rest_s, MIN_REST_S,
        )
        return {
            "subject_id": subject_id,
            "error": "insufficient_rest",
            "rest_duration_s": round(total_rest_s, 1),
            "rows": [],
        }

    logger.info(
        "%s: %.1f s clean MedOff rest | %d bipolar channels | threshold=MedOff_Rest",
        subject_id, total_rest_s, len(bipolar_picks),
    )

    # Compute 75th-percentile threshold from concatenated rest data
    thresholds_by_bm: dict[str, np.ndarray] = {}
    for bm in band_modes_to_run:
        thresh = []
        for segs in rest_segs[bm]:
            if segs:
                thresh.append(float(np.percentile(np.concatenate(segs), 75)))
            else:
                thresh.append(1e-7)   # channel absent from all MedOff files
        thresholds_by_bm[bm] = np.array(thresh)

    # ── Step 3: burst extraction per (fif, acq, epoch_type, band_mode) ─────────
    # Accumulate burst events keyed by (bm, cond_label, channel)
    # cond_accum[(bm, cond_label)][ch] = {"events": list[DataFrame], "total_s": float}
    cond_accum: dict[tuple[str, str], dict[str, dict]] = {}

    for fif_path in all_fifs:
        ents = _parse_entities(fif_path.name)
        acq = ents.get("acq", "")
        if acq not in ("MedOff", "MedOn"):
            continue

        raw_cond = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
        sfreq_c = float(raw_cond.info["sfreq"])
        file_picks_set = set(_bipolar_picks(raw_cond))

        epoch_types = _discover_epoch_types(raw_cond)
        if not epoch_types:
            logger.warning(
                "%s: no task epoch annotations in %s "
                "— run 06_attach_annotations.py first",
                subject_id, fif_path.name,
            )
            del raw_cond
            continue

        # Pre-compute (mask, total_s) for each epoch type in this file
        epoch_masks: list[tuple[str, str, np.ndarray, float]] = []
        for epoch_desc, cond_suffix in epoch_types:
            task_mask = get_epoch_mask(raw_cond, epoch_desc)
            total_s = float(task_mask.sum()) / sfreq_c
            if total_s >= 1.0:
                epoch_masks.append((epoch_desc, cond_suffix, task_mask, total_s))

        if not epoch_masks:
            del raw_cond
            continue

        # Ensure accumulator entries exist
        for _, cond_suffix, _, _ in epoch_masks:
            cond_label = f"{acq}_{cond_suffix}"
            for bm in band_modes_to_run:
                key = (bm, cond_label)
                if key not in cond_accum:
                    cond_accum[key] = {
                        ch: {"events": [], "total_s": 0.0} for ch in bipolar_picks
                    }

        for bm in band_modes_to_run:
            thresholds = thresholds_by_bm[bm]

            if bm == "fixed_13_30":
                # Compute bandpass once for all common channels, apply each epoch mask
                common = [p for p in bipolar_picks if p in file_picks_set]
                if not common:
                    continue
                bp_all = bandpass(raw_cond, 13.0, 30.0, common)  # (n_common, n_samples)
                env_all = hilbert_envelope(bp_all, sfreq_c)

                for epoch_desc, cond_suffix, task_mask, total_s in epoch_masks:
                    cond_label = f"{acq}_{cond_suffix}"
                    key = (bm, cond_label)
                    for i_c, ch in enumerate(common):
                        i = bipolar_picks.index(ch)
                        masked_env = env_all[i_c : i_c + 1].copy()  # (1, n_samples)
                        masked_env[:, ~task_mask] = 0.0
                        _, ev = label_bursts(
                            masked_env, np.array([thresholds[i]]), sfreq_c
                        )
                        ev = _tag_channel(ev, ch)
                        cond_accum[key][ch]["events"].append(ev)
                        cond_accum[key][ch]["total_s"] += total_s

            else:  # individualized — one bandpass per channel, multiple epoch masks
                for i, ch in enumerate(bipolar_picks):
                    if ch not in file_picks_set:
                        continue
                    lo, hi = individualized_bands.get(ch, (13.0, 30.0))
                    bp_ch = bandpass(raw_cond, lo, hi, [ch])      # (1, n_samples)
                    env_ch = hilbert_envelope(bp_ch, sfreq_c)

                    for epoch_desc, cond_suffix, task_mask, total_s in epoch_masks:
                        cond_label = f"{acq}_{cond_suffix}"
                        key = (bm, cond_label)
                        masked_env = env_ch.copy()
                        masked_env[:, ~task_mask] = 0.0
                        _, ev = label_bursts(
                            masked_env, np.array([thresholds[i]]), sfreq_c
                        )
                        ev = _tag_channel(ev, ch)
                        cond_accum[key][ch]["events"].append(ev)
                        cond_accum[key][ch]["total_s"] += total_s

        del raw_cond

    # ── Step 4: combine events per condition × channel, save parquets, build rows ──
    summary_rows: list[dict] = []

    for (bm, cond_label), ch_dict in sorted(cond_accum.items()):
        bm_dir = out_dir / subject_id / bm
        bm_dir.mkdir(parents=True, exist_ok=True)

        for i, ch in enumerate(bipolar_picks):
            acc = ch_dict.get(ch, {"events": [], "total_s": 0.0})
            if not acc["events"]:
                continue

            all_events = pd.concat(acc["events"], ignore_index=True)
            total_s = acc["total_s"]

            # Save combined parquet for this condition × channel
            parquet_path = bm_dir / f"{cond_label}_{ch}.parquet"
            all_events.to_parquet(parquet_path, index=False)

            if len(all_events) == 0:
                continue

            stats = burst_stats(all_events, total_s)
            n_long = int((all_events["duration_ms"] > LONG_BURST_MS).sum())
            long_frac = n_long / max(stats["n_bursts"], 1)

            if bm == "fixed_13_30":
                fmin_h, fmax_h = 13.0, 30.0
            else:
                fmin_h, fmax_h = individualized_bands.get(ch, (13.0, 30.0))

            summary_rows.append({
                "subject_id": subject_id,
                "hemi": "left" if "left" in ch else "right",
                "channel": ch,
                "condition": cond_label,
                "band_mode": bm,
                "fmin_hz": fmin_h,
                "fmax_hz": fmax_h,
                "threshold_condition_used": "MedOff_Rest",
                "rest_duration_s_used_for_threshold": round(total_rest_s, 1),
                "bipolar_count": len(bipolar_picks),
                **stats,
                "long_burst_fraction": long_frac,
            })

    logger.info(
        "%s: %d stat rows across %d (band_mode, condition) pairs",
        subject_id, len(summary_rows), len(cond_accum),
    )
    return {
        "subject_id": subject_id,
        "thresh_cond": "MedOff_Rest",
        "n_bipolar": len(bipolar_picks),
        "rest_duration_s": round(total_rest_s, 1),
        "rows": summary_rows,
    }


def _tag_channel(events: pd.DataFrame, ch: str) -> pd.DataFrame:
    """Return events DataFrame with the channel column set."""
    if len(events) > 0:
        events = events.copy()
        events["channel"] = ch
        return events
    return pd.DataFrame(
        columns=["channel", "onset_s", "offset_s", "duration_ms", "peak_amp", "mean_amp"]
    )

"""
Extract LFP + EMG + EOG + events for ds004998, downsample to 1 kHz, save compact .fif.

Input:  full BIDS tree at --bids-root (162 GB with MEG)
Output: compact .fif tree at --out  (~3-5 GB, MEG dropped)

Strategy:
  - For every valid (non-noise, primary-split) run of each included subject:
      1. Load raw fif (preload=True; we need to resample)
      2. Mark bad channels from channels.tsv
      3. Rename LFP channels using montage.tsv (EEG### -> LFP-<side>-<contact>)
      4. Compute bipolar re-references (adjacent contacts per hemisphere)
      5. Keep only: (monopolar LFP, bipolar LFP, EMG, EOG, stim)
      6. Resample to 1000 Hz
      7. Save to extracted/lfp/sub-XXX/ses-PeriOp/<same basename>_lfp.fif
      8. Save sidecar JSON with provenance

Usage (HPC, SLURM):
    python 02_extract_lfp.py \\
        --bids-root ~/scratch/stn/raw \\
        --cohort-summary ~/scratch/stn/audit/cohort_summary.tsv \\
        --out ~/scratch/stn/extracted \\
        --resample-hz 1000 \\
        --subject sub-0cGdk9    # optional: process one subject (for parallel jobs)

Or omit --subject to process all INCLUDED subjects serially (~2-4h total).

With --jobs N, subjects are processed in parallel using joblib (loky backend).
Requires >= 12 GB RAM per worker; 256 GB / 10 workers = 25.6 GB each.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


def _lazy_mne():
    import mne
    mne.set_log_level("WARNING")
    return mne


def parse_entities(fif_name: str) -> dict:
    out = {}
    for tok in fif_name.replace(".fif", "").split("_"):
        if "-" in tok:
            k, v = tok.split("-", 1)
            out[k] = v
    return out


def build_lfp_rename_map(montage_tsv: Path, subject_id: str | None = None) -> dict[str, str]:
    """Parse montage.tsv → mapping from raw EEG### label to LFP-<side>-<contact>.

    Tries the explicit ds004998 schema first (left/right_contacts_old →
    left/right_contacts_new), then falls back to a single src/dst column heuristic.
    Logs which path was taken and the final mapping count.
    """
    if not montage_tsv.exists():
        return {}
    df = pd.read_csv(montage_tsv, sep="\t")

    # Filter to this subject's rows when participant_id column is present
    # (guards against cohort-wide montage TSVs where all subjects are interleaved)
    if subject_id and "participant_id" in df.columns:
        sub_df = df[df["participant_id"].astype(str) == subject_id]
        if not sub_df.empty:
            df = sub_df
        else:
            logging.warning(
                f"participant_id={subject_id!r} not found in {montage_tsv.name}; using all rows"
            )

    # --- Path 1: explicit ds004998 schema ---
    has_left  = {"left_contacts_old",  "left_contacts_new"}.issubset(df.columns)
    has_right = {"right_contacts_old", "right_contacts_new"}.issubset(df.columns)
    if has_left or has_right:
        mapping: dict[str, str] = {}
        if has_left:
            for old, new in zip(df["left_contacts_old"].astype(str),
                                df["left_contacts_new"].astype(str)):
                if old not in ("nan", "") and new not in ("nan", ""):
                    mapping[old.strip()] = new.strip()
        if has_right:
            for old, new in zip(df["right_contacts_old"].astype(str),
                                df["right_contacts_new"].astype(str)):
                if old not in ("nan", "") and new not in ("nan", ""):
                    mapping[old.strip()] = new.strip()
        preview = ", ".join(f"{k}→{v}" for k, v in list(mapping.items())[:4])
        logging.info(
            f"Montage {montage_tsv.name}: explicit ds004998 schema → "
            f"{len(mapping)} mappings [{preview}{'…' if len(mapping) > 4 else ''}]"
        )
        return mapping

    # --- Path 2: fallback heuristic (single src/dst column pair) ---
    src_col = next((c for c in ["name", "ch_name", "EEG", "eeg_name", "raw_name"]
                    if c in df.columns), None)
    dst_col = next((c for c in ["new_name", "bids_name", "label", "contact", "LFP"]
                    if c in df.columns), None)
    if src_col is None or dst_col is None:
        logging.warning(
            f"Montage TSV columns not recognized in {montage_tsv}: {list(df.columns)}. "
            "No rename map built."
        )
        return {}
    mapping = {
        k.strip(): v.strip()
        for k, v in zip(df[src_col].astype(str), df[dst_col].astype(str))
        if k not in ("nan", "") and v not in ("nan", "")
    }
    logging.info(
        f"Montage {montage_tsv.name}: fallback heuristic ({src_col}→{dst_col}) → "
        f"{len(mapping)} mappings"
    )
    return mapping


def make_bipolar_pairs(lfp_names: list[str]) -> list[tuple[str, str, str]]:
    """From renamed LFP channel names, make adjacent-contact bipolar pairs per hemisphere.

    Returns list of (anode, cathode, new_name) tuples.
    Assumes names like 'LFP-left-0', 'LFP-left-1', ... (case-insensitive).
    """
    import re
    pairs: list[tuple[str, str, str]] = []
    # Group by hemisphere
    per_hemi: dict[str, list[tuple[int, str]]] = {"left": [], "right": []}
    pattern = re.compile(r"(LFP|lfp)[-_](left|right)[-_](\d+)", re.IGNORECASE)
    for n in lfp_names:
        m = pattern.match(n)
        if not m:
            continue
        hemi = m.group(2).lower()
        idx = int(m.group(3))
        per_hemi[hemi].append((idx, n))

    for hemi, items in per_hemi.items():
        items.sort(key=lambda t: t[0])
        for (i0, n0), (i1, n1) in zip(items[:-1], items[1:]):
            if i1 - i0 == 1:
                new_name = f"LFP-{hemi}-{i0}{i1}"
                pairs.append((n0, n1, new_name))
    return pairs


def _events_tsv_to_annotations(events_tsv_path: Path, first_time: float = 0.0, meas_date=None):
    """Build MNE Annotations from a BIDS events.tsv.

    BIDS onset values are seconds from the first sample of the recording
    (raw.first_time from meas_date).  To place annotations correctly inside
    MNE's absolute-time frame, onset_abs = bids_onset + first_time, and
    orig_time = meas_date.  If meas_date is None we fall back to orig_time=None
    (works only when raw.first_time == 0, i.e. recording started at time 0).

    bad_* descriptions are uppercased (→ BAD_*) so MNE auto-excludes them.
    Returns None if the file is missing or empty.
    """
    mne = _lazy_mne()
    if not events_tsv_path.exists():
        return None
    df = pd.read_csv(events_tsv_path, sep="\t")
    if df.empty:
        return None
    desc = df["trial_type"].astype(str).copy()
    bad_mask = desc.str.lower().str.startswith("bad")
    desc.loc[bad_mask] = desc.loc[bad_mask].str.upper()  # bad_lfp → BAD_LFP
    return mne.Annotations(
        onset=df["onset"].to_numpy(dtype=float) + first_time,
        duration=df["duration"].to_numpy(dtype=float),
        description=desc.to_numpy(dtype=str),
        orig_time=meas_date,
    )


def attach_events_to_raw(raw, events_tsv_path: Path):
    """Attach BIDS events.tsv annotations to raw in-place. Returns raw."""
    ann = _events_tsv_to_annotations(
        events_tsv_path,
        first_time=float(raw.first_time),
        meas_date=raw.info.get("meas_date"),
    )
    if ann is not None and len(ann) > 0:
        raw.set_annotations(ann)
    return raw


def process_one_fif(fif_path: Path, montage_tsv: Path, out_fif: Path,
                    resample_hz: float, subject_id: str | None = None) -> dict[str, Any]:
    """Returns a provenance dict; raises on unrecoverable errors."""
    mne = _lazy_mne()
    prov: dict[str, Any] = {"source": str(fif_path), "output": str(out_fif)}

    raw = mne.io.read_raw_fif(fif_path, preload=True, allow_maxshield=True, verbose=False)
    prov["orig_sfreq_hz"] = float(raw.info["sfreq"])
    prov["orig_duration_s"] = float(raw.times[-1])
    prov["orig_n_channels"] = len(raw.ch_names)

    # Mark bad channels
    base = fif_path.name.replace("_meg.fif", "")
    ch_tsv = fif_path.parent / f"{base}_channels.tsv"
    if ch_tsv.exists():
        chdf = pd.read_csv(ch_tsv, sep="\t")
        if "status" in chdf.columns:
            bads = chdf.loc[chdf["status"] == "bad", "name"].astype(str).tolist()
            # Only keep bads that are actually in the raw
            bads = [b for b in bads if b in raw.ch_names]
            raw.info["bads"] = bads
            prov["bad_channels"] = bads

    # Rename LFP (EEG-type) channels using montage
    rename_map = build_lfp_rename_map(montage_tsv, subject_id=subject_id)
    applicable = {k: v for k, v in rename_map.items() if k in raw.ch_names}
    if not applicable:
        raise RuntimeError(
            f"LFP rename produced no applicable mappings for {fif_path.name}. "
            f"Rename map had {len(rename_map)} entries "
            f"(keys: {list(rename_map.keys())[:4]}); "
            f"raw eeg-type ch_names: "
            f"{[c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == 'eeg'][:6]}"
        )
    raw.rename_channels(applicable)
    prov["lfp_rename_applied"] = applicable

    # Only keep channels that were successfully mapped by the montage TSV.
    # Any eeg-type channel NOT in the rename map is by definition not a valid LFP
    # contact (e.g., the unused 9th connector pin on directional leads).
    lfp_names = [new for old, new in applicable.items() if new in raw.ch_names]
    if not lfp_names:
        raise RuntimeError(
            f"No eeg-type (LFP) channels remain after renaming in {fif_path.name}. "
            f"Check that the source file contains EEG-type channels."
        )

    # Pick only: LFP (eeg type) + EMG + EOG + STIM; drop MEG
    raw.pick(picks=lfp_names
             + [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "emg"]
             + [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eog"]
             + [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "stim"])
    prov["kept_channels"] = raw.ch_names.copy()

    # Hardening: every eeg-type channel in the output must be a valid LFP-* name.
    # If this fails, the montage TSV and the raw channel list disagree in a way
    # that will silently corrupt downstream analysis.
    survivors = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    bad_survivors = [ch for ch in survivors if not ch.startswith("LFP-")]
    if bad_survivors:
        raise RuntimeError(
            f"Unmapped eeg-type channels survived pick for {fif_path.name}: "
            f"{bad_survivors}. Check montage TSV coverage for this subject."
        )

    # Bipolar re-reference (keep originals AND add bipolar as new channels)
    pairs = make_bipolar_pairs(lfp_names)
    if pairs:
        anodes = [a for a, _, _ in pairs]
        cathodes = [c for _, c, _ in pairs]
        new_names = [n for _, _, n in pairs]
        try:
            raw = mne.set_bipolar_reference(
                raw, anode=anodes, cathode=cathodes, ch_name=new_names,
                drop_refs=False, copy=True, verbose=False,
            )
            prov["bipolar_pairs"] = [{"anode": a, "cathode": c, "name": n}
                                     for a, c, n in pairs]
        except Exception as e:
            # Non-fatal — downstream can still compute bipolars itself
            logging.warning(f"Bipolar reref failed for {fif_path.name}: {e}")
            prov["bipolar_error"] = str(e)
    else:
        prov["bipolar_pairs"] = []
        logging.warning(f"No adjacent bipolar pairs recognized for {fif_path.name}")

    # Resample (only if target is lower than source)
    if resample_hz < raw.info["sfreq"]:
        raw.resample(resample_hz, npad="auto", verbose=False)
    prov["final_sfreq_hz"] = float(raw.info["sfreq"])
    prov["final_n_channels"] = len(raw.ch_names)

    # Attach events.tsv annotations before saving so task/bad epochs ride with the fif
    ev_tsv_src = fif_path.parent / f"{fif_path.name.replace('_meg.fif', '')}_events.tsv"
    raw = attach_events_to_raw(raw, ev_tsv_src)
    prov["annotations_attached"] = ev_tsv_src.exists()

    # Save
    out_fif.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_fif, overwrite=True, verbose=False)

    # Copy events.tsv sidecar if present (still valid since we kept timing)
    ev_tsv = fif_path.parent / f"{base}_events.tsv"
    if ev_tsv.exists():
        out_ev = out_fif.parent / f"{out_fif.stem.replace('_lfp', '')}_events.tsv"
        out_ev.write_text(ev_tsv.read_text())
        prov["events_copied"] = str(out_ev)

    return prov


def iter_runs_for_subject(subject_dir: Path):
    """Yield fif paths that should be processed (non-noise, primary split)."""
    for fif in sorted(subject_dir.rglob("*_meg.fif")):
        ents = parse_entities(fif.name)
        if ents.get("task", "").lower() == "noise":
            continue
        if ents.get("split", "01") != "01":
            continue
        yield fif


def _process_subject_worker(
    subject_dir: Path, out_root: Path, resample_hz: float, log_dir: Path
) -> dict:
    """Joblib worker: process one subject and write its per-subject JSON log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    sub_log = process_subject(subject_dir, out_root, resample_hz)
    with open(log_dir / f"{subject_dir.name}.json", "w") as f:
        json.dump(sub_log, f, indent=2, default=str)
    return sub_log


def process_subject(subject_dir: Path, out_root: Path, resample_hz: float) -> dict:
    """Process all runs for one subject. Returns a subject-level log dict."""
    subject_id = subject_dir.name
    log: dict = {"subject_id": subject_id, "runs": [], "errors": []}

    # Find montage TSV
    montage_files = list(subject_dir.rglob("*_montage.tsv"))
    montage_tsv = montage_files[0] if montage_files else Path("/nonexistent")
    log["montage_tsv"] = str(montage_tsv)

    for fif in iter_runs_for_subject(subject_dir):
        rel = fif.relative_to(subject_dir.parent)  # e.g. sub-XXX/ses-PeriOp/meg/..._meg.fif
        # Replace ..._meg.fif -> ..._lfp.fif
        new_name = fif.name.replace("_meg.fif", "_lfp.fif")
        out_fif = out_root / rel.parent / new_name

        try:
            prov = process_one_fif(fif, montage_tsv, out_fif, resample_hz, subject_id=subject_id)
            log["runs"].append(prov)
            logging.info(f"    OK  {fif.name} -> {out_fif.relative_to(out_root)}  "
                         f"[{prov['final_n_channels']} ch @ {prov['final_sfreq_hz']:.0f} Hz]")
        except Exception as e:
            tb = traceback.format_exc()
            log["errors"].append({"fif": str(fif), "error": str(e), "traceback": tb})
            logging.error(f"    FAIL {fif.name}: {e}")

    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bids-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cohort-summary", type=Path, default=None,
                    help="If given, only process subjects where include=True")
    ap.add_argument("--resample-hz", type=float, default=1000.0)
    ap.add_argument("--subject", type=str, default=None,
                    help="Optional: only process this subject_id (enables trivial SLURM array parallelism)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of parallel workers (subject-level). Default 1 (serial). "
                         "Uses joblib loky backend. Requires >= 12 GB RAM per worker.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "extract_logs"
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / (args.subject + ".log" if args.subject else "extract_all.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

    # Determine subject list
    if args.subject:
        subjects = [args.bids_root / args.subject]
    elif args.cohort_summary and args.cohort_summary.exists():
        cohort = pd.read_csv(args.cohort_summary, sep="\t")
        included = cohort.loc[cohort["include"].astype(bool), "subject_id"].tolist()
        subjects = [args.bids_root / s for s in included]
        logging.info(f"Including {len(subjects)} subjects from cohort summary")
    else:
        subjects = sorted(p for p in args.bids_root.glob("sub-*") if p.is_dir())
        logging.info(f"No cohort summary given; processing all {len(subjects)} subjects")

    # Memory guard for parallel mode
    if args.jobs > 1:
        import psutil
        total_mem_gb = psutil.virtual_memory().total / 1e9
        mem_per_worker_gb = total_mem_gb / args.jobs
        if mem_per_worker_gb < 12:
            logging.error(
                f"Insufficient memory per worker: {mem_per_worker_gb:.1f} GB "
                f"(total {total_mem_gb:.1f} GB / {args.jobs} workers). "
                f"Need >= 12 GB per worker. Reduce --jobs or run on a larger node."
            )
            return 1
        logging.info(
            f"Parallel mode: {args.jobs} workers, "
            f"{mem_per_worker_gb:.1f} GB/worker ({total_mem_gb:.1f} GB total)"
        )

    if args.jobs > 1:
        from joblib import Parallel, delayed
        # Warn about missing dirs before launching workers
        valid_subjects = []
        for sd in subjects:
            if not sd.exists():
                logging.warning(f"Subject dir missing: {sd}")
            else:
                valid_subjects.append(sd)
        all_logs = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(_process_subject_worker)(sd, args.out, args.resample_hz, log_dir)
            for sd in valid_subjects
        )
    else:
        all_logs = []
        for sd in subjects:
            if not sd.exists():
                logging.warning(f"Subject dir missing: {sd}")
                continue
            logging.info(f"== {sd.name} ==")
            sub_log = process_subject(sd, args.out, args.resample_hz)
            all_logs.append(sub_log)

            # Per-subject dump
            with open(log_dir / f"{sd.name}.json", "w") as f:
                json.dump(sub_log, f, indent=2, default=str)

    # Aggregate log
    agg_name = f"extract_{args.subject}.json" if args.subject else "extract_all.json"
    with open(log_dir / agg_name, "w") as f:
        json.dump(all_logs, f, indent=2, default=str)

    # Size report
    total_bytes = sum(p.stat().st_size for p in args.out.rglob("*_lfp.fif"))
    logging.info("=" * 60)
    logging.info("EXTRACTION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Subjects processed : {len(all_logs)}")
    logging.info(f"  Runs succeeded     : {sum(len(l['runs']) for l in all_logs)}")
    logging.info(f"  Runs failed        : {sum(len(l['errors']) for l in all_logs)}")
    total_bipolar = sum(
        len(r.get("bipolar_pairs", []))
        for l in all_logs
        for r in l["runs"]
    )
    logging.info(f"  Total bipolar pairs: {total_bipolar}")
    if total_bipolar == 0 and sum(len(l["runs"]) for l in all_logs) > 0:
        logging.error(
            "FATAL: Zero bipolar pairs produced across all runs. "
            "Check montage TSV column recognition in build_lfp_rename_map."
        )
        return 1
    logging.info(f"  Output size        : {total_bytes/1e9:.2f} GB")
    logging.info(f"  Output path        : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

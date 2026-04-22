"""Attach events.tsv annotations to existing extracted _lfp.fif files (in-place re-save).

This is a one-time patch for fifs extracted before annotation attachment was added to
process_one_fif().  It is purely additive: same channel data, same timing, new annotations.

Usage:
    python scripts/06_attach_annotations.py \\
        --extracted-root ~/scratch/stn/extracted \\
        --audit-tsv ~/scratch/stn/audit/cohort_summary.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mne
import pandas as pd

from stnbeta.preprocessing.extract import attach_events_to_raw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("06_attach_annotations")


def process_subject(subject_id: str, extracted_root: Path) -> dict:
    fif_dir = extracted_root / subject_id / "ses-PeriOp" / "meg"
    if not fif_dir.exists():
        logger.warning("%s: no fif dir at %s", subject_id, fif_dir)
        return {"subject_id": subject_id, "n_ok": 0, "n_missing_tsv": 0, "n_fail": 0}

    n_ok = n_missing_tsv = n_fail = 0
    for fif_path in sorted(fif_dir.glob("*_lfp.fif")):
        ev_tsv = fif_path.parent / fif_path.name.replace("_lfp.fif", "_events.tsv")
        if not ev_tsv.exists():
            logger.warning("%s: no events.tsv for %s — skipping", subject_id, fif_path.name)
            n_missing_tsv += 1
            continue
        try:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
            raw = attach_events_to_raw(raw, ev_tsv)
            raw.save(fif_path, overwrite=True, verbose="ERROR")
            ann_count = len(raw.annotations)
            logger.info(
                "%s: %s — attached %d annotations and resaved",
                subject_id, fif_path.name, ann_count,
            )
            n_ok += 1
        except Exception as exc:
            logger.error("%s: FAILED %s — %s", subject_id, fif_path.name, exc)
            n_fail += 1

    return {"subject_id": subject_id, "n_ok": n_ok, "n_missing_tsv": n_missing_tsv, "n_fail": n_fail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted-root", type=Path, required=True)
    ap.add_argument("--audit-tsv", type=Path, required=True)
    ap.add_argument("--subject", type=str, default=None,
                    help="Process a single subject only (for verification).")
    args = ap.parse_args()

    audit_df = pd.read_csv(args.audit_tsv, sep="\t")
    included = audit_df[audit_df["include"] == True]["subject_id"].tolist()

    if args.subject is not None:
        if args.subject not in included:
            logger.error("--subject %s not in included cohort", args.subject)
            return 1
        subjects = [args.subject]
    else:
        subjects = included

    logger.info("Attaching annotations to %d subjects", len(subjects))
    total_ok = total_missing = total_fail = 0
    for subj in subjects:
        res = process_subject(subj, args.extracted_root)
        total_ok += res["n_ok"]
        total_missing += res["n_missing_tsv"]
        total_fail += res["n_fail"]

    logger.info("=" * 60)
    logger.info("DONE  ok=%d  missing_tsv=%d  failed=%d", total_ok, total_missing, total_fail)
    if total_fail > 0:
        logger.error("Some files failed — check logs above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

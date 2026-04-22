"""Unit tests for get_epoch_mask() in pipeline.py and annotation attachment.

These use synthetic MNE RawArray objects — no file I/O, safe on login node.
"""

import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest

from stnbeta.ground_truth.pipeline import get_epoch_mask
from stnbeta.preprocessing.extract import _events_tsv_to_annotations


def _make_raw(duration_s: float, sfreq: float = 100.0) -> mne.io.RawArray:
    n = int(duration_s * sfreq)
    data = np.zeros((1, n))
    info = mne.create_info(["LFP-left-01"], sfreq, ch_types=["eeg"])
    return mne.io.RawArray(data, info, verbose=False)


def test_mask_excludes_bad():
    """rest epoch overlapping a BAD_LFP annotation: the overlap must be False in the mask."""
    sfreq = 100.0
    raw = _make_raw(10.0, sfreq)
    # rest: 0–5 s   (samples 0–499)
    # BAD_LFP: 2–3 s (samples 200–299) overlaps rest
    ann = mne.Annotations(
        onset=[0.0, 2.0],
        duration=[5.0, 1.0],
        description=["rest", "BAD_LFP"],
    )
    raw.set_annotations(ann)

    mask = get_epoch_mask(raw, "rest")

    # Entire rest window exists
    assert mask[:200].all(), "Samples before bad region should be in rest"
    # Bad overlap excluded
    assert not mask[200:300].any(), "Samples in BAD_LFP overlap must be excluded"
    # Rest resumes after bad
    assert mask[300:500].all(), "Samples after bad, still in rest, should be True"
    # Outside rest is False
    assert not mask[500:].any(), "Samples outside rest epoch should be False"
    # Total clean rest: 200 + 200 = 400 samples
    assert mask.sum() == 400


def test_mask_selects_only_requested_epoch():
    """Given rest and HoldL epochs, mask('rest') must not include HoldL samples and vice versa."""
    sfreq = 100.0
    raw = _make_raw(15.0, sfreq)
    # rest: 0–5 s   (samples 0–499)
    # HoldL: 6–11 s (samples 600–1099)
    ann = mne.Annotations(
        onset=[0.0, 6.0],
        duration=[5.0, 5.0],
        description=["rest", "HoldL"],
    )
    raw.set_annotations(ann)

    rest_mask = get_epoch_mask(raw, "rest")
    hold_mask = get_epoch_mask(raw, "HoldL")

    assert rest_mask[:500].all()
    assert not rest_mask[500:].any()
    assert not hold_mask[:600].any()
    assert hold_mask[600:1100].all()
    assert not hold_mask[1100:].any()
    # No sample is True in both masks
    assert not (rest_mask & hold_mask).any()


def test_mask_all_bad_returns_empty():
    """If the entire rest epoch is marked BAD, the mask should be all False."""
    sfreq = 100.0
    raw = _make_raw(10.0, sfreq)
    ann = mne.Annotations(
        onset=[0.0, 0.0],
        duration=[5.0, 5.0],
        description=["rest", "BAD_LFP"],
    )
    raw.set_annotations(ann)
    mask = get_epoch_mask(raw, "rest")
    assert not mask.any()


def test_mask_no_annotations_returns_empty():
    """Raw with no annotations should give all-False mask for any description."""
    raw = _make_raw(5.0)
    mask = get_epoch_mask(raw, "rest")
    assert not mask.any()
    mask2 = get_epoch_mask(raw, "HoldL")
    assert not mask2.any()


def test_annotation_first_time_offset():
    """_events_tsv_to_annotations must offset BIDS onsets by first_time.

    When a raw recording starts at first_time=257.2 s (non-zero first_samp),
    a BIDS onset=0.0 must be shifted to 257.2 in the Annotations so that after
    MNE subtracts raw.first_time the recovered onset is 0.0 (start of recording).
    """
    import pandas as pd

    first_time = 257.2
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        f.write("onset\tduration\ttrial_type\n")
        f.write(f"0.0\t287.2\trest\n")
        f.write(f"87.155\t4.4\tbad_lfp\n")
        tsv_path = Path(f.name)

    ann = _events_tsv_to_annotations(tsv_path, first_time=first_time, meas_date=None)
    tsv_path.unlink()

    assert ann is not None
    assert len(ann) == 2
    # Onset should be shifted: BIDS onset 0.0 + first_time 257.2 = 257.2
    assert abs(ann.onset[0] - (0.0 + first_time)) < 1e-6
    # bad_lfp should be uppercased
    assert ann.description[1] == "BAD_LFP"
    # Duration should be unchanged
    assert abs(ann.duration[0] - 287.2) < 1e-6

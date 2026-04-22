# Architecture Decision Records

This file records design decisions made during implementation that are not
self-evident from the code. One paragraph per decision.

---

## ADR-001: Minimum burst duration set to 100 ms
**Date:** 2026-04-21
**Decision:** Use 100 ms as the minimum burst duration cutoff in `label_bursts()`.
**Rationale:** Tinkhauser et al. (2017, Brain) use 100 ms as their primary cutoff,
defining a beta burst as a continuous period of supra-threshold beta power lasting
at least 100 ms. This is the most widely adopted threshold in the STN literature
and provides the best comparability with published results.

---

## ADR-002: Threshold source: MedOff Rest > MedOff Hold > MedOff Move (fallback chain)
**Date:** 2026-04-21
**Decision:** The burst amplitude threshold is computed from the best available
MedOff condition in the priority order: Rest > Hold > Move. Only 2 of 20 subjects
have a Rest MedOff recording; the remaining 18 use MedOff Hold.
**Rationale:** MedOff Rest is preferred (canonical pathological baseline per Tinkhauser
2017). When unavailable, MedOff Hold is the next-best resting-like condition in this
dataset. MedOff Move is the fallback of last resort. The threshold condition label
is recorded in the pipeline output for traceability.

---

## ADR-003: Joblib parallel burst extraction (revised from serial)
**Date:** 2026-04-21
**Decision:** `03_extract_bursts.py` uses `joblib.Parallel` with `--jobs 10`
(SLURM provides 12 CPUs).
**Rationale:** The original ADR-003 said serial was sufficient, but the master plan
spec explicitly requires `--jobs 10` and `joblib`. With 20 subjects × 2 band_modes
× ~6 conditions × 6 channels each, parallel processing saves ~20 min wall-clock
vs. serial on the 45-min allocation.

---

## ADR-004: Use specparam (not fooof) for individualized beta band fitting
**Date:** 2026-04-21
**Decision:** `fooof_band.py` imports `specparam` first; `fooof` is a deprecated
fallback. Both are installed in the SSN venv.
**Rationale:** The `fooof` package (v1.1.1) itself prints a deprecation warning on
import, directing users to `specparam`. `specparam` v2.0.0rc6 is the active
maintainer-recommended replacement with identical API for our use case. Installing
both ensures the code still runs if only one is present.

---

## ADR-005: Extracted LFP fifs are at 1000 Hz (not the original 2000 Hz)
**Date:** 2026-04-21
**Decision:** All signal processing uses sfreq=1000 Hz as read from the fif files,
ignoring the `sfreq_hz=2000` in the audit TSV.
**Rationale:** Phase 2 extraction downsampled the original 2000 Hz MEG recordings
to 1000 Hz when writing the LFP fif files. The audit TSV's `sfreq_hz` reflects the
original raw file sfreq, not the extracted file sfreq. 1000 Hz is more than adequate
for beta-band analysis (Nyquist 500 Hz >> 30 Hz). No resampling is needed.

---

## ADR-006: Long burst threshold at 500 ms
**Date:** 2026-04-21
**Decision:** A burst is classified as "long" (for `long_burst_fraction`) if its
duration exceeds 500 ms.
**Rationale:** The sanity check requires the group median MedOff burst duration to
fall in [150, 500] ms. Defining long bursts at 500 ms cleanly separates the
pathologically prolonged tail (above the expected median range) from typical bursts.
This aligns with Tinkhauser et al. (2017) who distinguish "long" bursts (>500 ms)
as a specific clinical marker.

---

## ADR-007: SSN environment is a uv venv, not a conda env
**Date:** 2026-04-21
**Decision:** `source activate SSN` in SLURM scripts activates a uv virtual
environment at `/home/haizhe/conda/envs/SSN/` (contains `pyvenv.cfg`). Conda
commands (`conda run -n SSN`) do not work for it.
**Rationale:** The SSN environment was created with `uv` (v0.11.7, Python 3.11.15)
but placed in a conda-like path. The `source activate SSN` command works because
the shell profile maps it. Using `/home/haizhe/conda/envs/SSN/bin/pip3` for
package installation targets the correct Python 3.11 environment.

---

## ADR-008: Saponati feedback-control optimizer deferred
**Date:** 2026-04-21
**Decision:** The Saponati feedback-control optimizer is not implemented in Phase 5.
**Rationale:** Specified as OPTIONAL and OFF THE CRITICAL PATH. Will only be
attempted if Gate 3 passes cleanly and there is remaining wall-clock budget.

---

## ADR-009: Burst threshold uses longest available MedOff segment (Rest > Hold > Move)
**Date:** 2026-04-21
**Decision:** Use the best available MedOff condition for burst threshold computation,
in priority order Rest > Hold > Move. The condition actually used is recorded per
subject in the `threshold_condition_used` column of `cohort_burst_stats.tsv`.
**Rationale:** Validated in the Phase 3 dry-run (job 2463939): sub-0cGdk9 used
MedOff_Hold because no MedOff_Rest recording was available, and produced physiologically
plausible burst statistics (median duration 158 ms, rate ~61 bursts/min). The fallback
chain is acceptable per the master plan; traceability via the TSV column ensures
reviewer reproducibility without recomputing.

---

## ADR-010: Individualized band defaults to fixed 13–30 Hz when no beta peak is found
**Date:** 2026-04-21
**Decision:** When specparam/fooof finds no peak in the 13–30 Hz range for a given
bipolar channel, `fit_individual_beta` returns (13.0, 30.0) as the band. The
`individualized` column in `cohort_burst_stats.tsv` is then effectively identical
to `fixed_13_30` for those channels.
**Rationale:** Dry-run showed 2 of 6 channels in sub-0cGdk9 (LFP-right-01,
LFP-right-23) had no detectable beta peak. Falling back to the fixed band is the
conservative choice: it avoids arbitrary band assignment and keeps the channel in
the analysis rather than excluding it. Affected channels are logged to stderr at
run time (WARNING level) to enable post-hoc identification. This is a documented
limitation, not a bug.

---

## ADR-011: Per-subject bipolar count reflects bad-channel exclusions
**Date:** 2026-04-21
**Decision:** Only bipolar pairs formed from adjacent, non-bad contacts are included
in analysis. The `bipolar_count` column in `cohort_burst_stats.tsv` records how many
bipolar channels each subject contributed. The theoretical maximum for the DBS
electrode geometry is 14 (7 contacts per hemisphere × 2 hemispheres minus edge pairs),
but the effective count varies per subject depending on their bad-channel list.
**Rationale:** Dry-run showed sub-0cGdk9 had 6 of a possible 14 bipolar channels.
Recording the per-subject count is cheap at run time and is necessary for the reviewer
to assess whether subject-level variance in burst statistics is confounded by the
number of channels averaged over.

---

## ADR-012: events.tsv annotations are authoritative for task epochs and bad segments
**Date:** 2026-04-21
**Decision:** Each extracted `_lfp.fif` carries its full BIDS events.tsv as MNE Annotations
(attached by `scripts/06_attach_annotations.py`; also written by `process_one_fif` for
future extractions). `bad_*` entries are uppercased to `BAD_*` on import so MNE's
`reject_by_annotation` machinery excludes them automatically. All downstream analysis
(threshold computation, burst labeling) derives epoch boundaries and bad-segment masks
exclusively from these annotations.
**Rationale:** Without embedded annotations, Phase 3 treated every file as a single
undifferentiated signal, causing (1) threshold computation over non-rest task segments,
(2) burst detection across `bad_lfp` stretches (producing 15,000–268,000 ms "bursts"),
and (3) incorrect condition labeling (file name ≠ epoch content).

---

## ADR-013: Rest baseline is sourced from in-file rest epochs uniformly across all subjects
**Date:** 2026-04-21
**Decision:** The 75th-percentile threshold is computed from the `trial_type=rest` epoch
present in every MedOff file, not from a dedicated task-Rest recording. Rest-masked
envelope data is concatenated across ALL MedOff files per subject before computing the
threshold. The `rest_duration_s_used_for_threshold` column in `cohort_burst_stats.tsv`
records the total clean rest seconds that informed the threshold.
**Rationale:** Only 2 of 20 subjects have a standalone `task-Rest` file. Every file in
the dataset starts with ~5 min of rest (Rassoulou et al., Sci Data 2024). Phase 3 v1
fell back to Hold/Move for threshold computation for 18 subjects, incorrectly treating
entire multi-condition recordings as the baseline and ignoring the embedded rest epoch.

---

## ADR-014: Phase 3 Gate 1 results from job 2467898 are retracted
**Date:** 2026-04-21
**Decision:** The Gate 1 outcome reported for SLURM job 2467898 is invalid and must not
be cited. All output artifacts (`results/bursts/`, `results/figures/03_bursts/`,
`results/tables/03_bursts/`) from that run were deleted and regenerated after the three
bugs described in ADR-012/ADR-013 were fixed.
**Rationale:** Three compounding bugs produced pathological outputs: (1) thresholds were
computed over entire files including non-rest epochs and BAD_LFP segments; (2) burst
labeling ran on unmasked signals, producing multi-minute "bursts" for sub-dCsWjQ,
sub-i4oK0F, and sub-oLNpHd; (3) CHECK 3 (UPDRS correlation) failed with r = −0.35
due to corrupted MedOn statistics and incorrect MedOff threshold baselines.

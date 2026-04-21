# Architecture Decision Records

This file records design decisions made during implementation that are not
self-evident from the code. One paragraph per decision.

---

## ADR-001: Minimum burst duration set to 100 ms
**Date:** (to be filled when Phase 3 is implemented)
**Decision:** Use 100 ms as the minimum burst duration cutoff in `label_bursts()`.
**Rationale:** Tinkhauser et al. (2017, Brain) use 100 ms as their primary cutoff,
defining a beta burst as a continuous period of supra-threshold beta power lasting
at least 100 ms. This is the most widely adopted threshold in the STN literature
and provides the best comparability with published results.

---

## ADR-002: Threshold computed on MedOff Rest only
**Date:** (to be filled when Phase 3 is implemented)
**Decision:** The burst amplitude threshold is computed from MedOff Rest epochs only,
not pooled across conditions.
**Rationale:** MedOff Rest represents the "natural" pathological state with the highest
beta power. Computing the threshold from this condition ensures the threshold is
calibrated to the pathological signal and is condition-independent, matching standard
practice in the literature.

---

## ADR-003: Serial (no joblib) burst extraction
**Date:** (to be filled when Phase 3 is implemented)
**Decision:** `03_extract_bursts.py` uses a plain for-loop over subjects, no joblib.
**Rationale:** Burst labeling is I/O-bound and requires <2 min per subject on the
extracted 7 GB dataset. The total 20-subject runtime is ~30 min, well within the 45-min
SLURM allocation. Adding joblib would complicate logging without meaningful speedup.

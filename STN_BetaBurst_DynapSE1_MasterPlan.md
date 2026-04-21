# STN Beta-Burst Detection on DYNAP-SE1 — Master Plan

> **Note:** This file is a placeholder. The full master plan document was not found
> on disk at the time of repo initialization (2026-04-21). Replace this file with the
> complete master plan before Phase 3.

---

## Project Summary

Detect subthalamic nucleus (STN) beta bursts in perioperative LFP recordings and
implement a low-latency burst detector on the DYNAP-SE1 neuromorphic chip using
a spiking neural network (SNN) encoding pipeline.

**Dataset:** ds004998 (OpenNeuro) — 20 subjects, perioperative STN-LFP, MedOff/MedOn,
Rest/Hold/Move conditions.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset acquisition and BIDS audit | ✅ Complete |
| 2 | LFP extraction (monopolar + bipolar, 1 kHz) | ✅ Complete |
| 3 | Tinkhauser beta-burst ground truth | 🔲 Next |
| 4 | ADM eventization sweep | 🔲 Pending |
| 5 | SNN architecture exploration (simulation) | 🔲 Pending |
| 6 | DYNAP-SE1 hardware bring-up | 🔲 Pending |

---

## §3 Phase 3 — Beta-Burst Ground Truth

See `src/stnbeta/ground_truth/` for implementation.

Key parameters:
- Band: fixed 13–30 Hz + individualized FOOOF peak ± 3 Hz
- Threshold: 75th percentile of MedOff Rest envelope
- Min duration: 100 ms
- Long burst: > 400 ms

## §3 Phase 4 — ADM Encoder Sweep

See `src/stnbeta/encoding/` for implementation.

## §3 Phase 5 — SNN Architecture Comparison

See `src/stnbeta/snn/` for implementation.

Architectures: A (heterogeneous phase-coincidence), B (homogeneous),
C (random reservoir), D (recurrent SG).

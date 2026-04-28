# ADR: Corrected Phase 6 Strategy

## Status

Accepted for Phase 6 start.

## Decision

The primary Phase 6 path is: existing internal STN dataset -> burden viability gate -> Brian2/state-machine simulation -> DYNAP/hardware-ready demonstration.

The He/Tan PPN dataset is optional cross-target extension only after the primary STN architecture passes. The Herz/Groppa/Brown force-adaptation package is a methods/code reference only.

Phase 6 must not claim FDA-grade validity, FDA validation, clinical efficacy, equivalence to commercial sensing systems, or therapeutic superiority.

Allowed framing: neuromorphic implementation/evaluation of a clinically relevant STN-LFP burden/state-tracking policy and hardware-compatible evaluation of physiologically motivated aDBS-style control substrate.

## Gate Outcomes

- PASS: proceed to Phase 6A Brian2/state-machine simulation.
- CONDITIONAL_PASS: proceed only with explicit limitations and added ablations before simulation code expansion.
- FAIL: stop and reconsider the burden pivot.
- BLOCKED: provide the required internal STN feature/label substrate before Phase 6 can continue.

## Longer-Horizon Workstream

Clinical, kinematic, PKG, or therapeutic validation is a separate workstream if a proper STN kinematic/clinical cohort becomes available.

## Consequences

Phase 6A.0 is the first corrected Phase 6 deliverable and controls whether Brian2/DYNAP-style work is justified.

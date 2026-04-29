# Phase 5_2C Robustness-Family Protocol

## Purpose

This protocol upgrades the Phase 5_2C state-machine follow-up from a single-config
Gaussian-noise check into a literature-grounded neuromorphic robustness audit.

The intended claim is:

> We identify a robust operating region of a duration-sensitive spiking state-machine
> family under structured input noise, event noise, quantization, and mixed-signal
> hardware mismatch.

The intended claim is not:

> We solved clinical beta-burst detection.

## Scientific Boundary

The local repository snapshot does not contain the runnable legacy `snn_brian2`
implementation. The local audit runner is therefore a deterministic topology surrogate
and table-schema contract. It locks the cases, metrics, perturbation battery, and
acceptance gates before the same protocol is rerun in the teammate/cluster checkout
that contains the full Brian2 implementation listed in
`docs/BRIAN2_STATUS_MACHINE_TEAM_HANDOFF.md`.

No clinical, DYNAP-SE1, or real-data deployment claim may be made from the local
surrogate outputs alone.

## Hypotheses

- H1: The duration state-machine has a clean operating region, not just one hand-tuned
  configuration.
- H2: The operating region remains stable under 5-10% mixed-signal hardware mismatch.
- H3: Structured noise exposes failure modes that a single additive Gaussian current
  test would miss.
- H4: Sustained-only and power-shift negative controls distinguish beta-state tracking
  from transient burst detection.

## Locked Synthetic Cases

- `no_burst`
- `short_40ms`
- `near_threshold_90ms`
- `threshold_crossing_120ms`
- `sustained_200ms`
- `long_400ms_plus`
- `two_bursts_with_quiet_gap`
- `interrupted_burst`
- `power_shift_no_burst`
- `sustained_only`

## Locked Parameter Family

The clean sweep covers:

- population size: `(8,2)`, `(16,4)`, `(32,8)`, `(64,16)`
- duration thresholds: `[40,80,160,320]`, `[50,100,200,400]`,
  `[75,150,300,600]`, `[100,200,350,500]`
- sustain weight multipliers: `0.5`, `0.75`, `1.0`, `1.25`, `1.5`
- forward transition weight multipliers: `0.5`, `0.75`, `1.0`, `1.25`, `1.5`
- reset strength multipliers: `0.5`, `0.75`, `1.0`, `1.25`, `1.5`, `2.0`
- quiet gain multipliers: `0.5`, `0.75`, `1.0`, `1.25`, `1.5`

The publication unit is the selected family of 10-30 clean-pass configurations, not a
single best configuration.

## Locked Perturbation Battery

- input-current additive Gaussian noise
- input-current multiplicative gain noise
- slow current gain drift
- 1/f-like colored current noise
- 50 Hz line contamination
- event boundary jitter
- event dropout
- false background beta evidence
- impulse artifacts
- amplitude clipping
- hardware mismatch on duration thresholds, sustain/forward/reset/quiet parameters
- 8-bit, 6-bit, and 4-bit quantization

Noise is applied at the state-machine contract boundary (`encoder_currents` or synthetic
event/current evidence), not blindly to raw LFP. Raw-LFP perturbations are only valid for
a separate end-to-end front-end audit.

## Hard Gates

- progression violations: `0`
- `no_burst` false alarm/min: `0`
- short and near-threshold stable readout: `0`
- sustained positive recall: `>= 0.95`
- reset failure rate: `<= 0.05`

Family-level paper gates:

- 5% mismatch family pass rate: `>= 0.90`
- 10% mismatch family pass rate: `>= 0.75`
- 20% mismatch family pass rate: `>= 0.50`
- 4-bit quantization must retain a nontrivial family, not only one surviving config
- `power_shift_no_burst` must not become a transient-burst success
- `sustained_only` may support sustained beta-state tracking, but not transient onset
  detection

## Output Contract

The audit writes:

- `clean_baseline_audit.csv`
- `parameter_sweep_summary.csv`
- `robust_family_configs.csv`
- `noise_robustness_seed_distribution.csv`
- `family_robustness_summary.csv`
- `negative_control_summary.csv`
- `literature_perturbation_battery.csv`
- `manifest.json`

## Literature Anchors

- Büchel et al., Scientific Reports 2021: mixed-signal SNN deployment must account for
  mismatch in thresholds, weights, biases, synaptic and neuronal time constants, plus
  low-bit quantization.
- DYNAP-SE recurrent vision work: DYNAP-SE imposes 64 afferent connections per neuron,
  shared core biases, limited weight types, and measured variability across sessions.
- DYNAP-SE2 hardware-aware training: parameter noise injection and quantization-aware
  simulation are used to improve deployment robustness.
- Langford and Wilson 2025: threshold beta-burst detection can misidentify sustained
  signals and power shifts as transient bursts.
- Lofredi et al. 2023: STN beta burst duration is clinically relevant, but robust claims
  require multiple definitions and calibrated interpretation.

# Experiment Protocol: Phase 5_2C Robustness Family

## Status

CONFIRMATORY protocol for the synthetic/topology audit contract. The local runner is a
surrogate because this snapshot does not include the full Brian2 implementation.

## Research Question

Does the duration-sensitive state-machine mechanism have a robust operating region
under structured current noise, event noise, quantization, and hardware-style mismatch?

## Predictions

- Clean sweep should identify multiple configurations that pass the hard gates.
- Mild mismatch (`5-10%`) should degrade metrics gradually rather than collapse all
  family members.
- Structured perturbations should expose different failure modes than additive Gaussian
  current noise.
- Negative controls should prevent transient-burst overclaiming, especially for
  `power_shift_no_burst` and `sustained_only`.

## Primary Metrics

- family pass rate
- robustness-volume proxy
- positive recall mean and 5th percentile
- false alarm/min
- progression violation count
- reset failure rate
- sustained-only transient-claim rate

## Confirmatory Command

```bash
python scripts/05_2c_robustness_family_audit.py \
  --out-dir reports/phase5_2c_robustness_family_audit \
  --family-size 20 \
  --noise-seeds 5 \
  --max-configs 384
```

Use `--max-configs 0` only in the full Brian2/cluster checkout or when a full local
grid is acceptable.

## Interpretation Rule

Local outputs can support protocol readiness and figure/table design. Final
neuromorphic paper claims require rerunning the same contract with the full Brian2 or
Brian2-equivalent state-machine runner.

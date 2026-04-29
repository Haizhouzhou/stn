"""Synthetic beta-burst traces for Phase 4 validation."""

from .beta_burst_generator import (
    BurstSpec,
    SyntheticTrace,
    SyntheticTraceConfig,
    duration_bucket_index,
    generate_trace,
    generate_trace_suite,
)

__all__ = [
    "BurstSpec",
    "SyntheticTrace",
    "SyntheticTraceConfig",
    "duration_bucket_index",
    "generate_trace",
    "generate_trace_suite",
]

"""Phase 5 duration-bucket SNN utilities."""

from .grid import expand_grid_points, filter_grid_points, split_rebuild_overrides
from .metrics import (
    evaluate_real_case,
    evaluate_synthetic_case,
    summarize_real_metrics,
    summarize_synthetic_metrics,
)
from .readout import build_readout_summary, detect_stable_events, state_active_masks
from .synthetic_suite import (
    SyntheticDurationCase,
    generate_end_to_end_suite,
    generate_topology_suite,
)

__all__ = [
    "SyntheticDurationCase",
    "build_readout_summary",
    "detect_stable_events",
    "evaluate_real_case",
    "evaluate_synthetic_case",
    "expand_grid_points",
    "filter_grid_points",
    "generate_end_to_end_suite",
    "generate_topology_suite",
    "split_rebuild_overrides",
    "state_active_masks",
    "summarize_real_metrics",
    "summarize_synthetic_metrics",
]

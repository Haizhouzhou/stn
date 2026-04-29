"""Grid expansion helpers for Phase 4 sweeps."""

from __future__ import annotations

import itertools
from collections.abc import Mapping


OVERRIDE_ALIASES = {
    "refractory_ms": "encoder_refractory_ms",
}


def normalize_override_keys(overrides: Mapping[str, object]) -> dict[str, object]:
    """Normalize legacy sweep keys to dataclass field names."""
    normalized: dict[str, object] = {}
    for key, value in overrides.items():
        normalized[OVERRIDE_ALIASES.get(key, key)] = value
    return normalized


def expand_grid_points(grid_cfg: Mapping[str, object], section: str) -> list[dict[str, object]]:
    """Expand one sweep section plus any rebuild axes into flat override dicts."""
    axes: dict[str, object] = {}
    axes.update(dict(grid_cfg.get(section, {})))
    axes.update(dict(grid_cfg.get("rebuild_axes", {})))
    if not axes:
        return [{}]

    keys = list(axes)
    values = [list(axes[key]) for key in keys]
    return [
        normalize_override_keys(dict(zip(keys, combo)))
        for combo in itertools.product(*values)
    ]


def split_rebuild_overrides(
    overrides: Mapping[str, object],
    rebuild_axes: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split override keys into rebuild-sensitive and run-argument subsets."""
    rebuild_names = set(normalize_override_keys({key: None for key in rebuild_axes}))
    build_overrides = {key: value for key, value in overrides.items() if key in rebuild_names}
    run_overrides = {key: value for key, value in overrides.items() if key not in rebuild_names}
    return build_overrides, run_overrides


def filter_grid_points(
    grid_points: list[dict[str, object]],
    indices: list[int] | None,
) -> list[tuple[int, dict[str, object]]]:
    """Return ``(grid_index, overrides)`` pairs filtered to *indices* when provided."""
    indexed = list(enumerate(grid_points))
    if indices is None:
        return indexed
    index_set = set(indices)
    return [(index, overrides) for index, overrides in indexed if index in index_set]

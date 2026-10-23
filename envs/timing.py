from __future__ import annotations

import math
import warnings
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedEnvTiming:
    timestep: float
    decision_steps: int
    decision_dt: float
    max_decision_steps: int
    episode_time_seconds: float


def _whole_multiple(duration: float, unit: float, *, name: str, unit_name: str) -> int:
    """Number of ``unit`` intervals in ``duration``, rounding up (with a warning) if inexact."""
    if duration <= 0 or unit <= 0:
        raise ValueError(f"{name} and {unit_name} must be positive")
    nearest = round(duration / unit)
    if nearest > 0 and math.isclose(nearest * unit, duration, rel_tol=1e-9, abs_tol=1e-12):
        return nearest
    count = math.ceil(duration / unit)
    warnings.warn(
        f"{name} is not an integer multiple of {unit_name}; rounding up from "
        f"{duration:.12g}s to {count * unit:.12g}s ({count} x {unit_name}).",
        stacklevel=3,
    )
    return count


def decision_dt_to_steps(decision_dt: float, timestep: float) -> int:
    return _whole_multiple(decision_dt, timestep, name="decision_dt", unit_name="timestep")


def resolve_env_timing(
    *,
    timestep: float,
    decision_steps: int | None,
    decision_dt: float,
    episode_time_seconds: float,
) -> ResolvedEnvTiming:
    """``decision_steps`` (physics steps per decision) takes precedence over ``decision_dt``."""
    if decision_steps is None:
        decision_steps = decision_dt_to_steps(decision_dt, timestep)
    elif decision_steps <= 0:
        raise ValueError("decision_steps must be positive")
    decision_dt = decision_steps * timestep
    max_decision_steps = _whole_multiple(
        episode_time_seconds, decision_dt, name="episode_time_seconds", unit_name="decision_dt"
    )
    return ResolvedEnvTiming(
        timestep=timestep,
        decision_steps=decision_steps,
        decision_dt=decision_dt,
        max_decision_steps=max_decision_steps,
        episode_time_seconds=max_decision_steps * decision_dt,
    )

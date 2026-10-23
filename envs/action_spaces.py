"""The walk7_groom discrete action set: left/right descending drives, or grooming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_SET = "walk7_groom"
POLICY_ACTION_SPACE = "discrete"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    flat_index: int
    name: str
    coarse_label: str
    forces: tuple[float, float]

    @property
    def groom(self) -> bool:
        return self.name == "groom"


_ACTION_SPECS = tuple(
    ActionSpec(index, name, label, forces)
    for index, (name, label, forces) in enumerate(
        (
            ("left", "left", (-1.0, 1.2)),
            ("forward", "forward", (1.0, 1.0)),
            ("right", "right", (1.2, -1.0)),
            ("stop", "stop", (0.0, 0.0)),
            ("groom", "groom", (0.0, 0.0)),
            ("hard_left", "left", (-1.2, 1.2)),
            ("hard_right", "right", (1.2, -1.2)),
        )
    )
)
# Clip range of the descending drives (the Warp CPG kernel clamps to it).
ACTION_FORCE_RANGE = (
    min(min(spec.forces) for spec in _ACTION_SPECS),
    max(max(spec.forces) for spec in _ACTION_SPECS),
)


def discrete_action_specs(action_set: str = ACTION_SET) -> tuple[ActionSpec, ...]:
    if action_set != ACTION_SET:
        raise ValueError(f"action_set must be {ACTION_SET}; got {action_set!r}")
    return _ACTION_SPECS


def resolve_discrete_action(action: Any) -> ActionSpec:
    value = np.asarray(action)
    if value.size != 1 or value.dtype.kind not in "iu":
        raise ValueError("discrete action must be an integer index")
    index = int(value.item())
    if not 0 <= index < len(_ACTION_SPECS):
        raise ValueError(f"discrete action index must be in [0, {len(_ACTION_SPECS) - 1}]")
    return _ACTION_SPECS[index]


def validate_action_config(policy_action_space: str, action_set: str) -> None:
    if policy_action_space != POLICY_ACTION_SPACE:
        raise ValueError(f"policy_action_space must be {POLICY_ACTION_SPACE}")
    discrete_action_specs(action_set)

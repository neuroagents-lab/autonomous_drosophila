"""Policy observations: component contract, spaces, and the odor/proprioception math.

Both simulators compute observations with the batched functions below (the CPU env
passes a batch of one), so CPU and Warp observations follow the same definitions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from gymnasium import spaces

ODOR_OBSERVATION = "odor"
FLY_VISION_OBSERVATION = "fly_vision"
PROPRIOCEPTION_OBSERVATION = "proprioception"

OBSERVATION_COMPONENT_ORDER = (
    ODOR_OBSERVATION,
    FLY_VISION_OBSERVATION,
    PROPRIOCEPTION_OBSERVATION,
)
DEFAULT_OBSERVATION_COMPONENTS = OBSERVATION_COMPONENT_ORDER
DEFAULT_RGB_SIZE = (32, 32)
FLY_VISION_EYE_COUNT = 2
FLY_VISION_CHANNEL_COUNT = 2
# Weights of the maxillary palps vs. antennae when averaging each side's odor sensors.
ODOR_PALPS_ANTENNAE_WEIGHTS = (1.0, 9.0)


def normalize_observation_components(value: Any) -> tuple[str, ...]:
    """Validate components and return them in stable canonical order."""
    if value is None:
        items = list(DEFAULT_OBSERVATION_COMPONENTS)
    elif isinstance(value, str):
        stripped = value.strip().strip("[]")
        items = [part.strip() for part in stripped.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        items = [str(item).strip() for item in value]
    else:
        raise TypeError("observation_components must be a sequence of strings")

    if not items:
        raise ValueError("observation_components must not be empty")
    if len(items) != len(set(items)):
        raise ValueError("observation_components must not contain duplicates")
    unknown = sorted(set(items) - set(OBSERVATION_COMPONENT_ORDER))
    if unknown:
        raise ValueError(
            "Unsupported observation components: "
            f"{unknown}; expected values from {sorted(OBSERVATION_COMPONENT_ORDER)}"
        )
    return tuple(item for item in OBSERVATION_COMPONENT_ORDER if item in items)


def normalize_rgb_size(value: Any) -> tuple[int, int]:
    """Normalize an RGB size to ``(height, width)``."""
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"rgb_size must have length 2, got {value!r}")
    height, width = int(items[0]), int(items[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"rgb_size values must be positive, got {value!r}")
    return (height, width)


def make_multimodal_observation_space(
    components: Any,
    *,
    rgb_size: tuple[int, int],
    proprioception_dim: int,
) -> spaces.Dict:
    height, width = normalize_rgb_size(rgb_size)
    vision_shape = (FLY_VISION_EYE_COUNT, FLY_VISION_CHANNEL_COUNT, height, width)
    all_spaces = {
        ODOR_OBSERVATION: spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float32),
        FLY_VISION_OBSERVATION: spaces.Box(low=0.0, high=1.0, shape=vision_shape, dtype=np.float32),
        PROPRIOCEPTION_OBSERVATION: spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(proprioception_dim),), dtype=np.float32
        ),
    }
    return spaces.Dict(
        {
            component: all_spaces[component]
            for component in normalize_observation_components(components)
        }
    )


def finite_float32(value) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def odor_observation(
    sensor_positions: np.ndarray,
    odor_sources: np.ndarray,
    peak_odor_intensity: np.ndarray,
    *,
    log_transform: bool,
) -> np.ndarray:
    """Left/right odor intensity, shape ``(worlds, 2)``.

    ``sensor_positions`` is ``(worlds, 4, 3)`` for the left/right palps then the left/right
    antennae; ``odor_sources`` is ``(worlds, sources, 3)``. Intensity falls off with the
    inverse square distance and sums over sources and odor dimensions.
    """
    distance = np.linalg.norm(
        sensor_positions[:, np.newaxis, np.newaxis, :, :]
        - odor_sources[:, :, np.newaxis, np.newaxis, :],
        axis=-1,
    )
    peak = np.asarray(peak_odor_intensity, dtype=np.float32)[np.newaxis, :, :, np.newaxis]
    intensity = finite_float32((peak * distance**-2).sum(axis=1))  # (worlds, dims, sensors)
    by_sensor = intensity.reshape(intensity.shape[0], -1, 2, 2)  # (worlds, dims, organ, side)
    odor = np.average(by_sensor, axis=2, weights=ODOR_PALPS_ANTENNAE_WEIGHTS).sum(axis=1)
    if log_transform:
        odor = np.log1p(np.maximum(odor, 0.0))
    return odor.astype(np.float32)


def world_vectors_to_body(quats: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quats.T
    rotations = np.empty((quats.shape[0], 3, 3), dtype=np.float32)
    rotations[:, 0] = np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1
    )
    rotations[:, 1] = np.stack(
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1
    )
    rotations[:, 2] = np.stack(
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1
    )
    return np.einsum("nji,nj->ni", rotations, vectors).astype(np.float32)


def proprioception_observation(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    root_qposadr: int,
    root_qveladr: int,
    joint_qposadrs: np.ndarray,
    joint_qveladrs: np.ndarray,
) -> np.ndarray:
    """Joint angles and velocities, root height, and gravity/linear/angular velocity in the
    body frame; ``qpos``/``qvel`` are ``(worlds, nq)``/``(worlds, nv)``."""
    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    quats = qpos[:, root_qposadr + 3 : root_qposadr + 7]
    gravity = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(qpos), 1))
    return finite_float32(
        np.concatenate(
            [
                qpos[:, joint_qposadrs],
                qvel[:, joint_qveladrs],
                qpos[:, root_qposadr + 2 : root_qposadr + 3],
                world_vectors_to_body(quats, gravity),
                world_vectors_to_body(quats, qvel[:, root_qveladr : root_qveladr + 3]),
                world_vectors_to_body(quats, qvel[:, root_qveladr + 3 : root_qveladr + 6]),
            ],
            axis=1,
        )
    )

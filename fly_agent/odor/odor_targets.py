from __future__ import annotations

import numpy as np


def reward_odor_source_indices_from_config(config) -> tuple[int, ...]:
    configured = config.reward_odor_source_indices
    if configured is not None:
        return tuple(int(index) for index in configured)
    return (int(config.target_odor_source_index),)


def odor_source_distances(
    fly_positions: np.ndarray,
    odor_sources: np.ndarray,
    source_indices: tuple[int, ...],
) -> np.ndarray:
    indices = np.asarray(source_indices, dtype=np.int64)
    sources = np.asarray(odor_sources, dtype=np.float32)
    fly_xy = np.asarray(fly_positions, dtype=np.float32)[..., :2]
    if sources.ndim == 2:
        return np.linalg.norm(sources[indices, :2] - fly_xy, axis=1)
    return np.linalg.norm(sources[:, indices, :2] - fly_xy[:, None, :], axis=2)


def distance_to_nearest_odor_source(
    fly_position: np.ndarray,
    odor_sources: np.ndarray,
    source_indices: tuple[int, ...],
) -> float:
    return float(np.min(odor_source_distances(fly_position, odor_sources, source_indices)))


def reached_odor_source_indices(
    fly_position: np.ndarray,
    odor_sources: np.ndarray,
    source_indices: tuple[int, ...],
    target_radius: float,
) -> tuple[int, ...]:
    indices = np.asarray(source_indices, dtype=np.int64)
    distances = odor_source_distances(fly_position, odor_sources, source_indices)
    reached = indices[distances <= float(target_radius)]
    return tuple(int(source_index) for source_index in reached)

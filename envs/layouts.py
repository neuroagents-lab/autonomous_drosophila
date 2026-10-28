"""Arena geometry and per-episode layout sampling (shared by the CPU and Warp envs)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

SINGLE_ODOR_SOURCE = np.array([[25.0, 0.0, 2.0]], dtype=np.float32)
SINGLE_ODOR_PEAK_INTENSITY = np.array([[1.0]], dtype=np.float32)
# Obstacle mocap slots of the single-source arena; positions are resampled every episode.
DEFAULT_OBSTACLE_POSITIONS = np.array(
    [(7.5, 0.0), (12.5, 5.0), (12.5, -5.0), (17.5, -5.0), (22.5, 5.0)], dtype=np.float32
)

THREE_ODOR_SOURCE = np.array(
    [[28.0, 0.0, 2.0], [-14.0, 24.25, 2.0], [-14.0, -24.25, 2.0]], dtype=np.float32
)
THREE_ODOR_PEAK_INTENSITY = np.ones((3, 1), dtype=np.float32)
THREE_ODOR_MARKER_COLORS = (
    (1.0, 1.0, 1.0, 1.0),
    (0.2, 0.6, 1.0, 1.0),
    (1.0, 0.5, 0.2, 1.0),
)


def spread_obstacle_slot_positions(
    count: int = 50, *, half: float = 49.0, step: float = 5.0, spawn_clear: float = 4.0
) -> np.ndarray:
    """Placeholder XY for each obstacle mocap slot of the wide arena."""
    coords = np.arange(-half, half + 1e-9, step, dtype=np.float32)
    grid = np.array(np.meshgrid(coords, coords), dtype=np.float32).reshape(2, -1).T
    grid = grid[np.linalg.norm(grid, axis=1) >= spawn_clear]
    if grid.shape[0] < count:
        raise ValueError(f"Could not place {count} obstacle slots with step={step} in ±{half} mm.")
    return grid[:count]


SPREAD_OBSTACLE_POSITIONS = spread_obstacle_slot_positions()


def resolve_odor_layout(odor_layout: str):
    """Return ``(odor_source, peak_odor_intensity, marker_colors)`` for a named layout."""
    if odor_layout == "single_source":
        return SINGLE_ODOR_SOURCE.copy(), SINGLE_ODOR_PEAK_INTENSITY.copy(), None
    if odor_layout == "three_sources":
        return THREE_ODOR_SOURCE.copy(), THREE_ODOR_PEAK_INTENSITY.copy(), THREE_ODOR_MARKER_COLORS
    raise ValueError(f"Unsupported odor layout: {odor_layout}")


def resolve_obstacle_positions(*, odor_layout: str, arena_mode: str) -> np.ndarray:
    if arena_mode == "flat_odor":
        return np.empty((0, 2), dtype=np.float32)
    if odor_layout == "three_sources":
        return SPREAD_OBSTACLE_POSITIONS.copy()
    return DEFAULT_OBSTACLE_POSITIONS.copy()


def layout_episode_seed(base: int, env_idx: int, episode_index: int, num_envs: int) -> int:
    """Seed of one episode's layout; independent of how episodes are spread over envs."""
    return int(base) + int(env_idx) + int(episode_index) * int(num_envs)


class EpisodeLayout(NamedTuple):
    spawn_position: np.ndarray
    spawn_yaw: float
    odor_source: np.ndarray
    obstacle_positions: np.ndarray
    active_obstacle_count: int

    @property
    def active_obstacle_positions(self) -> np.ndarray:
        return self.obstacle_positions[: self.active_obstacle_count]


def episode_layout_is_valid(config, spawn_position, odor_source, obstacles, active_count) -> bool:
    """Check the clearance constraints of ``config`` (a ``FlyBodyTrainEnvConfig``)."""
    spawn_xy = np.asarray(spawn_position[:2], dtype=np.float32)
    odor_xy = np.asarray(odor_source[:, :2], dtype=np.float32)
    if np.any(np.linalg.norm(odor_xy - spawn_xy, axis=1) < config.min_spawn_to_odor_distance):
        return False
    if config.randomize_odor_source and _too_close(odor_xy, config.min_odor_source_spacing):
        return False
    if active_count == 0:
        return True
    active = np.asarray(obstacles[:active_count], dtype=np.float32)
    if np.any(np.linalg.norm(active - spawn_xy, axis=1) < config.min_spawn_to_obstacle_distance):
        return False
    obstacle_to_odor = np.linalg.norm(active[:, None, :] - odor_xy[None, :, :], axis=2)
    if np.any(obstacle_to_odor < config.min_obstacle_to_odor_distance):
        return False
    return not _too_close(active, config.min_obstacle_spacing)


def _too_close(points: np.ndarray, min_spacing: float) -> bool:
    if points.shape[0] < 2 or min_spacing <= 0:
        return False
    return any(
        np.any(np.linalg.norm(points[idx + 1 :] - points[idx], axis=1) < min_spacing)
        for idx in range(points.shape[0] - 1)
    )


def sample_greedy_obstacle_positions(
    rng: np.random.Generator,
    config,
    *,
    slot_count: int,
    spawn_xy: np.ndarray,
    odor_xy: np.ndarray,
    per_obstacle_attempts: int = 256,
    candidate_batch_size: int = 64,
) -> tuple[np.ndarray, int] | None:
    """Place obstacles one by one; unused slots park at ``inactive_obstacle_position``."""
    low, high = config.obstacle_count_range
    target = int(rng.integers(low, high + 1))
    obstacle_positions = np.tile(
        np.asarray(config.inactive_obstacle_position, dtype=np.float32), (slot_count, 1)
    )
    if target == 0:
        return obstacle_positions, 0

    low_xy = np.array([config.obstacle_x_range[0], config.obstacle_y_range[0]], dtype=np.float32)
    high_xy = np.array([config.obstacle_x_range[1], config.obstacle_y_range[1]], dtype=np.float32)
    batch_size = max(1, min(candidate_batch_size, per_obstacle_attempts))
    placed: list[np.ndarray] = []

    def valid_candidates(candidates: np.ndarray) -> np.ndarray:
        valid = (
            np.linalg.norm(candidates - spawn_xy, axis=1) >= config.min_spawn_to_obstacle_distance
        )
        if odor_xy.size:
            odor_dist = np.linalg.norm(odor_xy[:, None, :] - candidates[None, :, :], axis=2)
            valid &= (odor_dist >= config.min_obstacle_to_odor_distance).all(axis=0)
        if placed and config.min_obstacle_spacing > 0:
            stacked = np.asarray(placed, dtype=np.float32)
            spacing = np.linalg.norm(stacked[:, None, :] - candidates[None, :, :], axis=2)
            valid &= (spacing >= config.min_obstacle_spacing).all(axis=0)
        return valid

    for _ in range(target):
        accepted = None
        remaining = per_obstacle_attempts
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            candidates = rng.uniform(low_xy, high_xy, size=(current_batch, 2)).astype(np.float32)
            valid = valid_candidates(candidates)
            if np.any(valid):
                accepted = candidates[int(np.argmax(valid))]
                break
            remaining -= current_batch
        if accepted is None:
            return None
        placed.append(accepted)

    obstacle_positions[:target] = np.asarray(placed, dtype=np.float32)
    return obstacle_positions, target


def sample_episode_layout(config, rng: np.random.Generator) -> EpisodeLayout:
    """Sample spawn pose, odor sources and obstacles for ``config`` (a ``FlyBodyTrainEnvConfig``)."""
    base = config.base_env_config
    with_obstacles = base.arena_mode == "flat_odor_obstacles"
    for _ in range(config.layout_sampling_attempts):
        spawn_position = np.asarray(base.spawn_pos, dtype=np.float32).copy()
        spawn_yaw = float(base.spawn_orientation[2])
        if config.spawn_xy_jitter > 0:
            spawn_position[:2] += rng.uniform(
                -config.spawn_xy_jitter, config.spawn_xy_jitter, size=2
            ).astype(np.float32)
        if config.spawn_yaw_jitter > 0:
            spawn_yaw += float(rng.uniform(-config.spawn_yaw_jitter, config.spawn_yaw_jitter))
        odor_source = np.asarray(base.odor_source, dtype=np.float32).copy()
        if config.randomize_odor_source:
            n_sources = odor_source.shape[0]
            odor_source[:, 0] = rng.uniform(*config.odor_source_x_range, size=n_sources)
            odor_source[:, 1] = rng.uniform(*config.odor_source_y_range, size=n_sources)
        obstacles = np.asarray(base.obstacle_positions, dtype=np.float32).copy()
        if not episode_layout_is_valid(config, spawn_position, odor_source, obstacles, 0):
            continue
        active_count = obstacles.shape[0] if with_obstacles else 0
        if with_obstacles and config.randomize_obstacles and obstacles.size:
            sampled = sample_greedy_obstacle_positions(
                rng,
                config,
                slot_count=obstacles.shape[0],
                spawn_xy=spawn_position[:2],
                odor_xy=odor_source[:, :2],
            )
            if sampled is None:
                continue
            obstacles, active_count = sampled
        elif not episode_layout_is_valid(
            config, spawn_position, odor_source, obstacles, active_count
        ):
            continue
        return EpisodeLayout(spawn_position, spawn_yaw, odor_source, obstacles, active_count)
    raise RuntimeError("Failed to sample a valid fly/odor/obstacle layout")

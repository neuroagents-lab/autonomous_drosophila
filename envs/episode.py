"""Per-episode reward, success and termination, shared by the CPU env and every Warp world."""

from __future__ import annotations

import numpy as np

from envs.layouts import EpisodeLayout
from fly_agent.odor.odor_targets import (
    distance_to_nearest_odor_source,
    reached_odor_source_indices,
    reward_odor_source_indices_from_config,
)

ODOR_TOUCH_REWARD_MODES = ("per_episode", "per_visit")


def obstacle_collision(fly_xy: np.ndarray, obstacles: np.ndarray, radius: float) -> bool:
    """True when the thorax XY lies within ``radius`` of any active obstacle center."""
    if len(obstacles) == 0:
        return False
    return bool(np.any(np.linalg.norm(obstacles - fly_xy[:2], axis=1) <= radius))


class EpisodeTracker:
    """Turns fly positions into rewards, termination flags and info dicts for one env.

    Odor sources pay ``success_bonus`` when the fly enters their ``target_radius``:
    once per episode (``per_episode``) or on every re-entry (``per_visit``). In
    ``per_episode`` mode the reward also has a distance-progress and a time term.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.reward_sources = reward_odor_source_indices_from_config(config)
        base = config.base_env_config
        self.decision_dt = base.decision_steps * base.timestep

    def reset(self, fly_position: np.ndarray, layout: EpisodeLayout) -> dict:
        self.layout = layout
        self.steps = 0
        self.distance = self.min_distance = self._distance(fly_position)
        self.low_heights: list[float] = []
        self.inside: set[int] = set()
        self.touched: set[int] = set()
        terms = {"progress": 0.0, "time": 0.0, "success": 0.0, "total": 0.0}
        return self._info(fly_position, terms, entered=(), failure_reason=None)

    def step(self, fly_position: np.ndarray) -> tuple[float, bool, bool, dict]:
        cfg = self.config
        self.steps += 1
        previous_distance = self.distance
        self.distance = self._distance(fly_position)
        self.min_distance = min(self.min_distance, self.distance)

        reached = reached_odor_source_indices(
            fly_position, self.layout.odor_source, self.reward_sources, cfg.target_radius
        )
        entered = set(reached) - self.inside
        if cfg.odor_touch_reward_mode == "per_episode":
            entered -= self.touched
        entered = tuple(sorted(entered))
        self.inside = set(reached)
        self.touched.update(entered)

        fly_z = float(fly_position[2])
        self.low_heights = ([] if fly_z >= cfg.min_height else [*self.low_heights, fly_z])[
            -cfg.low_height_consecutive_decisions :
        ]
        collided = obstacle_collision(
            fly_position,
            self.layout.active_obstacle_positions,
            cfg.base_env_config.obstacle_radius,
        )

        terms = {"progress": 0.0, "time": 0.0, "success": float(cfg.success_bonus * len(entered))}
        if cfg.odor_touch_reward_mode == "per_episode":
            terms["progress"] = float(
                cfg.progress_reward_weight * (previous_distance - self.distance)
            )
            terms["time"] = -float(cfg.time_penalty)
            if collided and cfg.obstacle_collision_penalty != 0.0:
                terms["collision"] = float(cfg.obstacle_collision_penalty)
        reward = float(sum(terms.values()))
        terms["total"] = reward

        failure_reason = None
        if reached and cfg.terminate_on_success:
            terminated = True
        elif collided and cfg.terminate_on_obstacle_collision:
            terminated, failure_reason = True, "obstacle_collision"
        elif len(self.low_heights) >= cfg.low_height_consecutive_decisions:
            terminated, failure_reason = True, "low_height"
        elif abs(float(fly_position[0])) > cfg.max_abs_x:
            terminated, failure_reason = True, "x_out_of_bounds"
        elif abs(float(fly_position[1])) > cfg.max_abs_y:
            terminated, failure_reason = True, "y_out_of_bounds"
        else:
            terminated = False
        truncated = not terminated and self.steps >= cfg.base_env_config.max_decision_steps
        info = self._info(fly_position, terms, entered=entered, failure_reason=failure_reason)
        info["terminated"], info["truncated"] = terminated, truncated
        return reward, terminated, truncated, info

    def _distance(self, fly_position: np.ndarray) -> float:
        return distance_to_nearest_odor_source(
            fly_position, self.layout.odor_source, self.reward_sources
        )

    def _info(self, fly_position, reward_terms, *, entered, failure_reason) -> dict:
        cfg = self.config
        success = bool(self.touched)
        return {
            "sim_time": self.steps * self.decision_dt,
            "episode_step_count": self.steps,
            "fly_position": np.array(fly_position, dtype=np.float32),
            "fly_z": float(fly_position[2]),
            "target_position": self.layout.odor_source[cfg.target_odor_source_index].copy(),
            "active_obstacle_count": int(self.layout.active_obstacle_count),
            "active_obstacle_positions": self.layout.active_obstacle_positions.copy(),
            "distance_to_target": self.distance,
            "min_distance_to_target": float(self.min_distance),
            "reward_terms": reward_terms,
            "newly_reached_odor_source_indices": entered,
            "reached_odor_source_indices": tuple(sorted(self.touched)),
            "success": success,
            "is_success": success,
            "failure_reason": failure_reason,
            "terminated": False,
            "truncated": False,
            "low_height_streak_values": list(self.low_heights),
            "low_height_threshold": float(cfg.min_height),
            "low_height_consecutive_decisions": int(cfg.low_height_consecutive_decisions),
        }


def sanitize_nonfinite_transition(
    config, observation: dict, reward: float, info: dict, *, physics_invalid: bool = False
) -> tuple[dict, float, bool]:
    """End the episode with finite values when physics or the transition went non-finite.

    Returns ``(observation, reward, sanitized)``; on sanitizing, ``info`` is updated in place
    to a terminal (not truncated) transition.
    """
    nonfinite = not np.isfinite(reward) or not all(
        np.isfinite(value).all() for value in observation.values()
    )
    if not physics_invalid and not (config.sanitize_nonfinite_rollout and nonfinite):
        return observation, reward, False
    fill = float(config.nonfinite_rollout_obs_fill)
    observation = {
        key: np.nan_to_num(value, nan=fill, posinf=fill, neginf=fill)
        for key, value in observation.items()
    }
    reward = float(config.nonfinite_rollout_reward)
    info["reward_terms"] = {**info["reward_terms"], "total": reward}
    info["failure_reason"] = "nonfinite_physics" if physics_invalid else "nonfinite_transition"
    info["terminated"], info["truncated"] = True, False
    return observation, reward, True

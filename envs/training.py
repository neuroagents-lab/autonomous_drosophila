"""Hydra ``envs`` config -> env configs, and the CPU training env (one fly, discrete actions)."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from envs.action_spaces import (
    ACTION_SET,
    discrete_action_specs,
    resolve_discrete_action,
    validate_action_config,
)
from envs.episode import ODOR_TOUCH_REWARD_MODES, EpisodeTracker, sanitize_nonfinite_transition
from envs.layouts import (
    layout_episode_seed,
    resolve_obstacle_positions,
    resolve_odor_layout,
    sample_episode_layout,
)
from envs.navigation import FlyBodyEnvConfig, FlyBodyNavigationEnv
from envs.observations import ODOR_OBSERVATION
from envs.timing import ResolvedEnvTiming, resolve_env_timing

_DEFAULT_ENV_CONFIG = FlyBodyEnvConfig()
_DEFAULT_DECISION_DT = _DEFAULT_ENV_CONFIG.decision_steps * _DEFAULT_ENV_CONFIG.timestep


@dataclass(slots=True)
class FlyBodyTrainEnvConfig:
    """Task settings on top of the simulator: actions, rewards, termination, layouts."""

    base_env_config: FlyBodyEnvConfig = field(default_factory=FlyBodyEnvConfig)
    action_set: str = ACTION_SET
    policy_action_space: str = "discrete"
    # Rewards: odor sources pay success_bonus on entering target_radius (see envs.episode).
    target_odor_source_index: int = 0
    reward_odor_source_indices: tuple[int, ...] | None = None
    target_radius: float = 2.0
    odor_touch_reward_mode: str = "per_episode"
    progress_reward_weight: float = 1.0
    time_penalty: float = 0.01
    success_bonus: float = 10.0
    obstacle_collision_penalty: float = 0.0
    # Termination.
    terminate_on_success: bool = True
    terminate_on_obstacle_collision: bool = False
    min_height: float = 0.03
    low_height_consecutive_decisions: int = 5
    max_abs_x: float = 60.0
    max_abs_y: float = 60.0
    sanitize_nonfinite_rollout: bool = False
    nonfinite_rollout_reward: float = 0.0
    nonfinite_rollout_obs_fill: float = 0.0
    # Per-episode layout randomization (see envs.layouts.sample_episode_layout).
    spawn_xy_jitter: float = 1.0
    spawn_yaw_jitter: float = math.pi / 4.0
    randomize_odor_source: bool = True
    randomize_obstacles: bool = True
    obstacle_count_range: tuple[int, int] = (3, 5)
    odor_source_x_range: tuple[float, float] = (20.0, 30.0)
    odor_source_y_range: tuple[float, float] = (-6.0, 6.0)
    obstacle_x_range: tuple[float, float] = (6.0, 20.0)
    obstacle_y_range: tuple[float, float] = (-8.0, 8.0)
    inactive_obstacle_position: tuple[float, float] = (1_000.0, 0.0)
    min_spawn_to_odor_distance: float = 10.0
    min_spawn_to_obstacle_distance: float = 3.0
    min_obstacle_to_odor_distance: float = 4.0
    min_obstacle_spacing: float = 4.0
    min_odor_source_spacing: float = 0.0
    layout_sampling_attempts: int = 128

    def __post_init__(self) -> None:
        validate_action_config(self.policy_action_space, self.action_set)
        if self.odor_touch_reward_mode not in ODOR_TOUCH_REWARD_MODES:
            raise ValueError(f"odor_touch_reward_mode must be one of {ODOR_TOUCH_REWARD_MODES}")
        if self.base_env_config.arena_mode == "flat_odor":
            self.randomize_obstacles = False
            self.obstacle_count_range = (0, 0)
        self.obstacle_count_range = tuple(int(v) for v in self.obstacle_count_range)
        self.inactive_obstacle_position = tuple(float(v) for v in self.inactive_obstacle_position)
        for name in (
            "odor_source_x_range",
            "odor_source_y_range",
            "obstacle_x_range",
            "obstacle_y_range",
        ):
            low, high = (float(v) for v in getattr(self, name))
            if low > high:
                raise ValueError(f"{name} must be ordered as (min, max)")
            setattr(self, name, (low, high))
        low, high = self.obstacle_count_range
        if not 0 <= low <= high <= self.base_env_config.obstacle_positions.shape[0]:
            raise ValueError("obstacle_count_range must fit within the configured obstacle slots")
        self.low_height_consecutive_decisions = int(self.low_height_consecutive_decisions)
        num_sources = self.base_env_config.odor_source.shape[0]
        if self.reward_odor_source_indices is not None:
            self.reward_odor_source_indices = tuple(int(i) for i in self.reward_odor_source_indices)
        indices = (self.target_odor_source_index, *(self.reward_odor_source_indices or ()))
        if not all(0 <= index < num_sources for index in indices):
            raise ValueError("odor source indices must reference configured odor sources")


def make_flygym_train_config(config) -> FlyBodyTrainEnvConfig:
    """Build env configs from the Hydra ``envs`` group (configs/envs/*.yaml are the defaults).

    Keys named like a ``FlyBodyEnvConfig`` or ``FlyBodyTrainEnvConfig`` field are copied over;
    timing (``decision_dt``, ``episode_time_seconds``) and the arena (``odor_layout``,
    ``arena_mode``) are resolved here.
    """
    if OmegaConf.is_config(config):
        values = OmegaConf.to_container(config, resolve=True)
    else:
        values = dict(vars(config))
    values = {key: value for key, value in values.items() if value is not None}
    timing = resolved_env_timing_from_ssrl_envs(config)
    odor_layout = values.get("odor_layout", "single_source")
    odor_source, peak_odor_intensity, marker_colors = resolve_odor_layout(odor_layout)
    base_env_config = FlyBodyEnvConfig(
        **_field_values(values, FlyBodyEnvConfig),
        timestep=timing.timestep,
        decision_steps=timing.decision_steps,
        max_decision_steps=timing.max_decision_steps,
        odor_source=odor_source,
        peak_odor_intensity=peak_odor_intensity,
        marker_colors=marker_colors,
        obstacle_positions=resolve_obstacle_positions(
            odor_layout=odor_layout, arena_mode=values.get("arena_mode", "flat_odor_obstacles")
        ),
    )
    return FlyBodyTrainEnvConfig(
        base_env_config=base_env_config, **_field_values(values, FlyBodyTrainEnvConfig)
    )


_RESOLVED_FIELDS = {
    "base_env_config",
    "timestep",
    "decision_steps",
    "max_decision_steps",
    "odor_source",
    "peak_odor_intensity",
    "marker_colors",
    "obstacle_positions",
}


def _field_values(values: dict[str, Any], cls) -> dict[str, Any]:
    return {
        f.name: values[f.name]
        for f in fields(cls)
        if f.name in values and f.name not in _RESOLVED_FIELDS
    }


def resolved_env_timing_from_ssrl_envs(config) -> ResolvedEnvTiming:
    """Resolve simulator/decision/episode timing from the Hydra ``envs`` config."""
    decision_steps = getattr(config, "decision_steps", None)
    return resolve_env_timing(
        timestep=float(getattr(config, "timestep", None) or _DEFAULT_ENV_CONFIG.timestep),
        decision_steps=None if decision_steps is None else int(decision_steps),
        decision_dt=float(getattr(config, "decision_dt", None) or _DEFAULT_DECISION_DT),
        episode_time_seconds=float(getattr(config, "episode_time_seconds", None) or 5.0),
    )


def make_flygym_navigation_env(
    config,
    *,
    seed: int,
    worker_rank: int = 0,
    layout_seed_base: int | None = None,
    layout_num_envs: int = 1,
) -> FlyBodyNavigationTrainEnv:
    if layout_seed_base is None:
        config_seed = getattr(config, "seed", None)
        layout_seed_base = int(seed if config_seed is None else config_seed)
    return FlyBodyNavigationTrainEnv(
        config=make_flygym_train_config(config),
        worker_rank=worker_rank,
        layout_seed_base=layout_seed_base,
        layout_num_envs=layout_num_envs,
    )


def action_info(spec, observation: dict[str, np.ndarray]) -> dict[str, Any]:
    """Info entries describing the chosen action (shared with the Warp env)."""
    odor = observation.get(ODOR_OBSERVATION)
    return {
        "policy_action": spec.flat_index,
        "policy_action_label": spec.coarse_label,
        "low_level_action": {
            "left_force": float(spec.forces[0]),
            "right_force": float(spec.forces[1]),
            "groom": spec.groom,
        },
        "target_odor_total": 0.0 if odor is None else float(np.sum(odor)),
    }


class FlyBodyNavigationTrainEnv(gym.Env):
    """CPU training env: discrete walk/groom actions, sampled layouts, odor-touch rewards."""

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: FlyBodyTrainEnvConfig | None = None,
        render_mode: str | None = None,
        worker_rank: int | None = None,
        layout_seed_base: int | None = None,
        layout_num_envs: int = 1,
    ) -> None:
        self.train_config = deepcopy(config) if config is not None else FlyBodyTrainEnvConfig()
        self.render_mode = render_mode
        self.sim = FlyBodyNavigationEnv(self.train_config.base_env_config, render_mode=render_mode)
        self.observation_space = self.sim.observation_space
        self.action_space = gym.spaces.Discrete(
            len(discrete_action_specs(self.train_config.action_set))
        )
        self.episode = EpisodeTracker(self.train_config)
        self.worker_rank = worker_rank
        self.layout_seed_base = layout_seed_base
        self.layout_num_envs = max(1, int(layout_num_envs))
        self._episode_index = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is None and self.layout_seed_base is not None and self.worker_rank is not None:
            seed = layout_episode_seed(
                self.layout_seed_base, self.worker_rank, self._episode_index, self.layout_num_envs
            )
            self._episode_index += 1
        layout = sample_episode_layout(self.train_config, np.random.default_rng(seed))
        observation = self.sim.reset(layout, seed=seed)
        return observation, self.episode.reset(self.sim.fly_position(), layout)

    def step(self, action):
        spec = resolve_discrete_action(action)
        observation = self.sim.step(spec.forces, groom=spec.groom)
        reward, terminated, truncated, info = self.episode.step(self.sim.fly_position())
        info.update(action_info(spec, observation))
        observation, reward, sanitized = sanitize_nonfinite_transition(
            self.train_config, observation, reward, info
        )
        if sanitized:
            terminated, truncated = True, False
        return observation, reward, terminated, truncated, info

    def save_recordings(self, video_path, stabilization_time: float = 0.02) -> None:
        del stabilization_time  # Accepted for scripts/eval_odor.py; recordings need no settling.
        self.sim.save_video(video_path)

    def reset_recordings(self) -> None:
        self.sim.reset_recordings()

    def close(self) -> None:
        self.sim.close()

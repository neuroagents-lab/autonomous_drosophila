"""SB3 ``VecEnv`` over the batched Warp simulator: the GPU counterpart of the CPU train env."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from envs.action_spaces import discrete_action_specs, resolve_discrete_action
from envs.episode import EpisodeTracker, sanitize_nonfinite_transition
from envs.layouts import layout_episode_seed, sample_episode_layout
from envs.training import FlyBodyTrainEnvConfig, action_info, make_flygym_train_config
from envs.warp_runner import WarpNavigationTaskRunner


class FlyGymWarpVecEnv(VecEnv):
    """``num_envs`` Warp worlds, each with its own layout RNG and ``EpisodeTracker``.

    Done worlds are reset right away (SB3 convention: ``terminal_observation`` in info).
    With ``layout_seed_base``, episode ``k`` of world ``i`` uses layout seed
    ``layout_episode_seed(base, i, k, num_envs)``, so layouts are reproducible.
    """

    def __init__(
        self,
        train_config: FlyBodyTrainEnvConfig,
        num_envs: int,
        *,
        layout_seed_base: int | None = None,
        render_mode: str | None = None,
        record_dir: Path | None = None,
        eval_camera: str = "navigation_birdeye",
        collection_memory_policy: str = "existing",
        reset_render_capacity: int = 0,
    ) -> None:
        if render_mode not in {None, "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.train_config = deepcopy(train_config)
        self.train_config.base_env_config.eval_camera = eval_camera
        self.runner = WarpNavigationTaskRunner(
            self.train_config.base_env_config, num_envs, render=render_mode is not None
        )
        action_count = len(discrete_action_specs(self.train_config.action_set))
        self.render_mode = render_mode  # Read by VecEnv.__init__ through get_attr.
        super().__init__(
            num_envs=self.runner.num_envs,
            observation_space=self.runner.observation_space,
            action_space=spaces.Discrete(action_count),
        )
        self._layout_seed_base = layout_seed_base
        self._record_dir = record_dir
        self._recording_episode_idx = 0
        self._episode_counters = np.zeros(self.num_envs, dtype=np.int64)
        self._rngs = [np.random.default_rng() for _ in range(self.num_envs)]
        self._episodes = [EpisodeTracker(self.train_config) for _ in range(self.num_envs)]
        self._actions = None
        self._collection_startup = None
        if collection_memory_policy != "existing":
            from envs.collection import CollectionMemoryPolicy

            self._collection_startup = CollectionMemoryPolicy(collection_memory_policy)
        if reset_render_capacity and "fly_vision" in self.observation_space.spaces:
            from envs.collection import SelectedResetRendering

            self.runner.reset_rendering = SelectedResetRendering(self.runner, reset_render_capacity)

    def reset(self):
        seeds = list(self._seeds)
        if self._layout_seed_base is not None:
            self._episode_counters[:] = 0
            seeds = [self._next_layout_seed(idx) for idx in range(self.num_envs)]
        observations, self.reset_infos = self._reset_worlds(np.arange(self.num_envs), seeds)
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        specs = [
            resolve_discrete_action(action) for action in np.asarray(self._actions).reshape(-1)
        ]
        self._actions = None
        observations = self.runner.step(
            np.array([spec.forces for spec in specs], dtype=np.float32),
            np.array([spec.groom for spec in specs], dtype=bool),
        )
        fly_positions = self.runner.fly_positions()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        for idx, spec in enumerate(specs):
            reward, terminated, truncated, info = self._episodes[idx].step(fly_positions[idx])
            row = {key: value[idx] for key, value in observations.items()}
            info.update(action_info(spec, row))
            row, reward, sanitized = sanitize_nonfinite_transition(
                self.train_config,
                row,
                reward,
                info,
                physics_invalid=bool(self.runner.physics_recovery.invalid[idx]),
            )
            if sanitized:
                terminated, truncated = True, False
                for key, value in row.items():
                    observations[key][idx] = value
            rewards[idx] = reward
            dones[idx] = terminated or truncated
            info["TimeLimit.truncated"] = truncated
            infos.append(info)

        done_indices = np.flatnonzero(dones)
        for idx in done_indices:
            infos[idx]["terminal_observation"] = {
                key: value[idx].copy() for key, value in observations.items()
            }
        if self._record_dir is not None and dones[0]:
            self.runner.save_recordings(
                self._record_dir / f"episode_{self._recording_episode_idx}.mp4"
            )
            self.runner.reset_recordings()
            self._recording_episode_idx += 1
        if done_indices.size:
            recovered = np.array(
                [
                    idx
                    for idx in done_indices
                    if infos[idx]["failure_reason"] == "nonfinite_physics"
                ],
                dtype=np.int32,
            )
            if recovered.size:
                self.runner.physics_recovery.clear_invalid_state(recovered)
            seeds = [None] * len(done_indices)
            if self._layout_seed_base is not None:
                seeds = [self._next_layout_seed(idx) for idx in done_indices]
            reset_observations, reset_infos = self._reset_worlds(done_indices, seeds)
            if recovered.size:
                self.runner.physics_recovery.validate_reset(recovered)
            for row, idx in enumerate(done_indices):
                for key, value in reset_observations.items():
                    observations[key][idx] = value[row]
                self.reset_infos[idx] = reset_infos[row]
        if self._collection_startup is not None:
            self._collection_startup.after_first_decision()
            self._collection_startup = None
        return observations, rewards, dones, infos

    def _next_layout_seed(self, idx: int) -> int:
        seed = layout_episode_seed(
            self._layout_seed_base, idx, self._episode_counters[idx], self.num_envs
        )
        self._episode_counters[idx] += 1
        return seed

    def _reset_worlds(self, indices: np.ndarray, seeds: list[int | None]):
        layouts = []
        for idx, seed in zip(indices, seeds, strict=True):
            if seed is not None:
                self._rngs[idx] = np.random.default_rng(seed)
            layouts.append(sample_episode_layout(self.train_config, self._rngs[idx]))
        observations = self.runner.reset_at(indices, layouts, seeds)
        fly_positions = self.runner.fly_positions()
        infos = [
            self._episodes[idx].reset(fly_positions[idx], layout)
            for idx, layout in zip(indices, layouts, strict=True)
        ]
        return observations, infos

    def reset_recordings(self) -> None:
        self.runner.reset_recordings()

    def close(self) -> None:
        self.runner.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> list[Any]:
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]


def make_flygym_warp_vec_env(
    config,
    *,
    num_envs: int,
    layout_seed_base: int | None = None,
    render_mode: str | None = None,
    record_dir: Path | None = None,
    eval_camera: str = "navigation_birdeye",
) -> FlyGymWarpVecEnv:
    return FlyGymWarpVecEnv(
        make_flygym_train_config(config),
        num_envs=num_envs,
        layout_seed_base=layout_seed_base,
        render_mode=render_mode,
        record_dir=record_dir,
        eval_camera=eval_camera,
        collection_memory_policy=str(getattr(config, "collection_memory_policy", "existing")),
        reset_render_capacity=int(getattr(config, "reset_render_capacity", 0)),
    )

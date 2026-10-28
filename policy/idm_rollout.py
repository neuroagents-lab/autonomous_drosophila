"""Connect a PPO rollout to the IDM: skill observations, features, rewards, updates.

Wrapper order on the training env (inner to outer)::

    VecMonitor -> OriginalTerminalObservations -> VecNormalize -> IDMSkillVecWrapper

The IDM reads the skill-free pre-LSTM features of the policy's extractor
(``FlyGymMultimodalExtractor.base_features``), so the skill reaches it only
through behaviour.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.vec_env import VecEnvWrapper

from policy.idm import PPOSkillSampler
from policy.idm_manager import JaxIDMManager, copy_observation, observation_row

# Observations per extractor forward pass when computing IDM features.
_ENCODE_CHUNK = 1024


class OriginalTerminalObservations(VecEnvWrapper):
    """Keep each terminal observation before VecNormalize normalizes and clips it."""

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos, strict=True):
            if done:
                info["idm_original_terminal_observation"] = copy_observation(
                    info["terminal_observation"]
                )
        return obs, rewards, dones, infos


class IDMSkillVecWrapper(VecEnvWrapper):
    """Add the sampled skill to observations (key ``skill``) and to ``info``."""

    def __init__(self, venv, sampler: PPOSkillSampler):
        spaces = dict(venv.observation_space.spaces)
        spaces["skill"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(sampler.skill_event_dim,), dtype=np.float32
        )
        super().__init__(venv, observation_space=gym.spaces.Dict(spaces))
        self.sampler = sampler

    def reset(self):
        obs = self.venv.reset()
        self.sampler.reset_skills(self.num_envs)
        return {**obs, "skill": self.sampler.current_skills(self.num_envs)}

    def step_wait(self):
        skills = self.sampler.current_skills(self.num_envs)
        obs, rewards, dones, infos = self.venv.step_wait()
        self.sampler.reset_skills(self.num_envs, dones=dones)
        boundaries = self.sampler.skill_boundaries(self.num_envs)
        for info, skill, boundary in zip(infos, skills, boundaries, strict=True):
            info["idm_skill"] = skill
            info["idm_skill_end"] = bool(boundary)
            if "terminal_observation" in info:
                info["terminal_observation"] = {**info["terminal_observation"], "skill": skill}
        return {**obs, "skill": self.sampler.current_skills(self.num_envs)}, rewards, dones, infos


class RolloutFeatures:
    """IDM features of a rollout's transitions under rollout-frozen obs normalization."""

    def begin_rollout(self, model) -> None:
        self.policy = model.policy
        self.vecnormalize = model.get_vec_normalize_env()
        norm = self.vecnormalize
        # VecNormalize keeps updating its statistics during the rollout; freeze them
        # so every feature of this rollout uses the same normalization.
        self.obs_rms = deepcopy(norm.obs_rms) if norm.norm_obs else {}
        self.current_raw = copy_observation(norm.get_original_obs())
        self._cached_next = None

    def normalize(self, raw):
        norm = self.vecnormalize
        normalized = dict(raw)
        for key, rms in self.obs_rms.items():
            normalized[key] = np.clip(
                (raw[key] - rms.mean) / np.sqrt(rms.var + norm.epsilon),
                -norm.clip_obs,
                norm.clip_obs,
            ).astype(np.float32)
        return normalized

    def encode(self, raw) -> np.ndarray:
        policy = self.policy
        obs = self.normalize(raw)
        count = len(next(iter(obs.values())))
        features = []
        for start in range(0, count, _ENCODE_CHUNK):
            part = {key: value[start : start + _ENCODE_CHUNK] for key, value in obs.items()}
            tensor = preprocess_obs(
                policy.obs_to_tensor(part)[0],
                policy.observation_space,
                normalize_images=policy.normalize_images,
            )
            with th.no_grad():
                features.append(policy.pi_features_extractor.base_features(tensor))
        return th.cat(features).cpu().numpy()

    def transition(self, locals_) -> tuple[np.ndarray, np.ndarray]:
        """Features of this step's (obs, next obs); next obs is the terminal one at dones."""
        raw_new = copy_observation(self.vecnormalize.get_original_obs())
        raw_next = copy_observation(raw_new)
        dones = np.asarray(locals_["dones"], dtype=bool)
        for i in np.flatnonzero(dones):
            terminal = locals_["infos"][i]["idm_original_terminal_observation"]
            for key in raw_next:
                raw_next[key][i] = terminal[key]
        self.last_raw, self.last_next_raw = self.current_raw, raw_next
        self.current_raw = raw_new
        # A continuing env's next features are its features at the following step.
        obs = self.encode(self.last_raw) if self._cached_next is None else self._cached_next
        next_obs = self.encode(raw_next)
        self._cached_next = None if dones.any() else next_obs
        return obs, next_obs


class IDMRolloutCallback(BaseCallback):
    """Replace PPO rewards with IDM rewards and train the IDM once per rollout."""

    def __init__(self, manager: JaxIDMManager, verbose: int = 0):
        super().__init__(verbose)
        self.manager = manager
        self.features = RolloutFeatures()

    def _on_rollout_start(self) -> None:
        self.features.begin_rollout(self.model)
        n_steps = self.model.n_steps
        self._replay_rows = self.manager.begin_rollout(
            self.model.n_envs, n_steps, self.features.encode
        )
        self._replay_at_step = defaultdict(list)
        for flat in self._replay_rows:
            env, step = divmod(int(flat), n_steps)
            self._replay_at_step[step].append((env, int(flat)))
        self._replay_raw = {}
        self._history = defaultdict(list)

    def _on_step(self) -> bool:
        manager, locals_ = self.manager, self.locals
        dones = np.asarray(locals_["dones"], dtype=bool)
        obs, next_obs = self.features.transition(locals_)
        raw_obs, raw_next = self.features.last_raw, self.features.last_next_raw
        for env, flat in self._replay_at_step[len(self._history["obs"])]:
            self._replay_raw[flat] = (observation_row(raw_obs, env), observation_row(raw_next, env))
        skills = skill_ends = None
        if manager.skill_sampler is not None:
            skills = np.stack([info["idm_skill"] for info in locals_["infos"]])
            skill_ends = np.array([info["idm_skill_end"] for info in locals_["infos"]])
            self._history["skills"].append(skills)

        rewards = locals_["rewards"]
        if manager.idm_type == "ctec":
            # C-TeC rewards need future states; they are written at rollout end.
            # Until then the buffer only receives SB3's timeout bootstrap.
            rewards[:] = 0.0
        else:
            intrinsic = manager.intrinsic_reward(
                obs,
                locals_["actions"],
                next_obs,
                episode_starts=locals_["episode_starts"].cpu().numpy().astype(bool),
                dones=dones,
                raw_obs=raw_obs,
                raw_next_obs=raw_next,
                skills=skills,
                skill_ends=skill_ends,
                action_probabilities=(
                    self._action_probabilities()
                    if manager.idm_type == "empowerment_process"
                    else None
                ),
            )
            rewards[:] = manager.ppo_reward(intrinsic, dones)
            self._history["intrinsic"].append(intrinsic)
        self._history["obs"].append(obs)
        self._history["next_obs"].append(next_obs)
        self._history["dones"].append(dones)
        return True

    def _on_rollout_end(self) -> None:
        manager, buffer = self.manager, self.model.rollout_buffer
        obs, next_obs, dones = (
            np.stack(self._history[key], axis=1) for key in ("obs", "next_obs", "dones")
        )
        actions = buffer.actions[:, :, 0].T
        episode_starts = buffer.episode_starts.T.astype(bool)
        if manager.idm_type == "ctec":
            intrinsic = manager.ctec_rollout_reward(obs, actions, next_obs, episode_starts, dones)
            for t in range(intrinsic.shape[1]):
                buffer.rewards[t] += manager.ppo_reward(intrinsic[:, t], dones[:, t])
            self._recompute_returns()
        else:
            intrinsic = np.stack(self._history["intrinsic"], axis=1)
        metrics = manager.train_step(
            obs,
            actions,
            next_obs,
            episode_starts=episode_starts,
            dones=dones,
            skills=np.stack(self._history["skills"], axis=1) if self._history["skills"] else None,
            replay_raw=[self._replay_raw[int(flat)] for flat in self._replay_rows],
        )
        metrics["intrinsic_reward_mean"] = float(intrinsic.mean())
        _record_idm_metrics(self.logger, metrics)

    def _action_probabilities(self) -> np.ndarray:
        """Behavior policy action probabilities, from the pre-step recurrent state."""
        with th.no_grad():
            distribution, _ = self.model.policy.get_distribution(
                self.locals["obs_tensor"],
                self.model._last_lstm_states.pi,
                self.locals["episode_starts"],
            )
        return distribution.distribution.probs.cpu().numpy()

    def _recompute_returns(self) -> None:
        model = self.model
        with th.no_grad():
            last_values = model.policy.predict_values(
                model.policy.obs_to_tensor(model._last_obs)[0],
                model._last_lstm_states.vf,
                th.as_tensor(model._last_episode_starts, dtype=th.float32, device=model.device),
            )
        model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=model._last_episode_starts
        )


def _record_idm_metrics(logger, metrics) -> None:
    names = {f"idm/{key}": value for key, value in metrics.items()}
    # SB3's console writer truncates long keys, which can make two keys equal.
    width = max(len(name) for name in names) + 3
    for output in logger.output_formats:
        if isinstance(output, HumanOutputFormat):
            output.max_length = max(output.max_length, width)
    for name, value in names.items():
        logger.record(name, value)

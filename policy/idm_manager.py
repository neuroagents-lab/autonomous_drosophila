"""Train an ssrl IDM on PPO rollouts and turn it into the PPO reward.

The IDM sees skill-free features of the policy's extractor (see
``policy.idm_rollout``). Every environment step it scores the transitions;
once per rollout it takes one optimizer step on a replay batch drawn from that
rollout. Features of observations kept across rollouts (NGU episodes, VIC and
VALOR skill segments, the APT/RE3 k-NN memory) are recomputed from the raw
observations at the start of each rollout, because PPO changes the extractor.
"""

from __future__ import annotations

import pathlib
import pickle
from collections import defaultdict
from os import PathLike
from typing import Any

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from ssrl.idms.chunking import chunked_rows
from ssrl.idms.entropy.common import RunningRMSNormalizer
from stable_baselines3.common.running_mean_std import RunningMeanStd

from policy.idm import SKILL_IDMS, PPOSkillSampler, build_idm

# k-NN IDMs that keep a memory of embedded past observations.
MEMORY_IDMS = frozenset({"apt", "re3"})
# IDMs scored on a segment of states: NGU on the episode so far, VIC on the
# start and final state of a skill segment, VALOR on the whole skill segment.
SEGMENT_IDMS = frozenset({"ngu", "vic", "valor"})

# Modules updated by Adam (default: the whole IDM). Slow and target copies
# follow the trained modules in post_update(); RE3's encoder stays random.
_TRAINABLE_MODULES = {
    "apt": lambda m: list(m.get_modules().values()),
    "empowerment_gain": lambda m: list(m.get_trainable_modules().values()),
    "gamma_progress": lambda m: [m.dynamics],
    "ngu": lambda m: [
        v for k, v in m.get_modules().items() if not k.endswith(("_target", "_stats"))
    ],
    "proto_rl": lambda m: [m.online_encoder, m.online_projector, m.predictor, m.prototype_bank],
    "re3": lambda m: [],
}


class _IDMOptimizer(nj.Module):
    """One Adam step on ``modules`` for an impure loss function."""

    def __init__(self, modules, opt):
        self.modules = modules
        self.opt = opt
        self.step = nj.Variable(jnp.array, 0, jnp.int32, name="step")

    def __call__(self, loss_fn, *, has_aux):
        loss, params, grads, aux = nj.grad(loss_fn, self.modules, has_aux=has_aux)()
        state = self.sub("state", nj.Tree, self.opt.init, params)
        updates, new_state = self.opt.update(grads, state.read(), params)
        nj.context().update(optax.apply_updates(params, updates))
        state.write(new_state)
        self.step.write(self.step.read() + 1)
        metrics = {
            "loss": loss,
            "updates": self.step.read(),
            "grad_norm": optax.global_norm(grads),
        }
        return {f"{self.name}/{key}": value for key, value in metrics.items()}, aux


def one_hot_actions(actions, n_actions: int) -> np.ndarray:
    """One-hot encode integer PPO actions of any shape."""
    actions = np.asarray(actions)
    if not np.all((actions == np.round(actions)) & (actions >= 0) & (actions < n_actions)):
        raise ValueError(f"PPO actions must be integers in [0, {n_actions - 1}]")
    return np.eye(n_actions, dtype=np.float32)[actions.astype(np.int64)]


def copy_observation(obs: dict) -> dict:
    """Copy a dict observation batch without the policy-only ``skill`` key."""
    return {k: np.array(v) for k, v in obs.items() if k != "skill"}


def observation_row(obs: dict, index: int) -> dict:
    return {k: np.array(v[index]) for k, v in obs.items() if k != "skill"}


def stack_observations(rows: list[dict]) -> dict:
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def sample_future_indices(episode_starts, dones, discount, rng):
    """Geometric future offset for each transition, truncated at its episode end.

    Index t selects next_obs[t], including real terminal successors. Inputs are
    [env, time]; the result holds time indices of the same shape.
    """
    starts, dones = np.asarray(episode_starts, bool), np.asarray(dones, bool)
    batch, horizon = starts.shape
    lengths = np.ones((batch, horizon), np.int32)
    for t in range(horizon - 2, -1, -1):
        boundary = dones[:, t] | starts[:, t + 1]
        lengths[:, t] = np.where(boundary, 1, lengths[:, t + 1] + 1)
    u = rng.random(lengths.shape)
    gamma = float(discount)
    if gamma == 0:
        offset = np.zeros_like(lengths)
    elif gamma == 1:
        offset = np.floor(u * lengths).astype(np.int32)
    else:
        offset = np.floor(
            np.log1p(-u * (-np.expm1(lengths * np.log(gamma)))) / np.log(gamma)
        ).astype(np.int32)
    return np.arange(horizon)[None] + np.minimum(offset, lengths - 1)


class JaxIDMManager:
    """Stateful ssrl IDM for vectorized PPO rollouts."""

    def __init__(
        self,
        config,
        *,
        obs_dim: int,
        n_actions: int,
        skill_sampler: PPOSkillSampler | None = None,
    ):
        idm = config.idm
        self.idm_type = idm.type
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.intrinsic_scale = float(idm.intrinsic_scale)
        self.replay_batch_size = int(idm.replay_batch_size)
        if (skill_sampler is not None) != (self.idm_type in SKILL_IDMS):
            raise ValueError(f"idm.type={self.idm_type!r} needs a skill sampler iff it uses skills")
        self.skill_sampler = skill_sampler
        self.module = build_idm(config, obs_dim=self.obs_dim, n_actions=self.n_actions)
        if self.idm_type in {"vic", "valor"} and not self.module.reward_at_end:
            raise ValueError(f"PPO pays {self.idm_type} rewards when a skill segment ends")
        if self.idm_type == "ctec" and self.module.reward_estimator != "single":
            raise ValueError("PPO C-TeC supports reward_estimator=single only")

        self._rng = int(config.seed)
        self._updates = 0
        self._intrinsic_rms = (
            RunningRMSNormalizer(eps=1e-8, name="ppo_idm_reward_rms")
            if idm.normalize_intrinsic
            else None
        )
        # PPO return normalization, as VecNormalize(norm_reward=True) does for
        # extrinsic runs; VecNormalize only sees the extrinsic reward.
        self._return_rms = RunningMeanStd(shape=()) if config.algo.norm_reward else None
        self._returns = np.zeros(0)
        self._gamma = float(config.algo.gamma)
        modules = _TRAINABLE_MODULES.get(self.idm_type, lambda m: [m])(self.module)
        self._optimizer = (
            _IDMOptimizer(modules, optax.adam(float(idm.lr)), name="ppo_idm_opt")
            if modules
            else None
        )
        # Per env: [(features, raw observation), ...] of the current segment.
        self._segments: list[list[tuple[np.ndarray, Any]]] = []
        # VIC/VALOR (states, skill) segments completed since the last update.
        self._completed: list[tuple[np.ndarray, np.ndarray]] = []
        # Raw observations behind the APT/RE3 memory slots.
        self._memory_rows: list[Any] = []
        self._replay_rows: np.ndarray | None = None
        self._reward_fn = jax.jit(nj.pure(self._transition_reward))
        self._state = self._init_state()

    # ----------------------------------------------------------- rollout API

    def begin_rollout(self, n_envs: int, n_steps: int, encode) -> np.ndarray:
        """Re-embed kept observations with ``encode(raw) -> features``.

        Returns the flat ``env * n_steps + t`` indices of the transitions the
        next ``train_step`` will replay (empty for VIC/VALOR, which replay
        completed segments).
        """
        rows = [raw for segment in self._segments for _, raw in segment]
        if rows:
            features = iter(encode(stack_observations(rows)))
            self._segments = [[(next(features), raw) for _, raw in seg] for seg in self._segments]
        if self._memory_rows:
            self._run(
                self._refresh_memory, jnp.asarray(encode(stack_observations(self._memory_rows)))
            )
        self._replay_rows = np.array([], np.int64)
        if self.idm_type not in {"vic", "valor"}:
            total = n_envs * n_steps
            self._replay_rows = np.random.default_rng(self._next_seed()).choice(
                total, min(self.replay_batch_size, total), replace=False
            )
        return self._replay_rows

    def intrinsic_reward(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        next_obs: np.ndarray,
        *,
        episode_starts: np.ndarray,
        dones: np.ndarray,
        raw_obs=None,
        raw_next_obs=None,
        skills: np.ndarray | None = None,
        skill_ends: np.ndarray | None = None,
        action_probabilities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Intrinsic reward of one vectorized step, shape (n_envs,).

        ``obs``/``next_obs`` are features; ``raw_obs``/``raw_next_obs`` are the
        observations they came from (kept only by segment IDMs).
        """
        if self.idm_type in SEGMENT_IDMS:
            ended = dones if skill_ends is None else dones | skill_ends
            reward = self._segment_reward(
                obs, next_obs, raw_obs, raw_next_obs, episode_starts, ended, skills
            )
        else:
            # Score each transition as a length-1 sequence: [env, 1, dim].
            inputs = [obs, one_hot_actions(actions, self.n_actions), next_obs]
            inputs += [skills, action_probabilities]
            inputs = [None if x is None else jnp.asarray(x)[:, None] for x in inputs]
            self._state, reward = self._reward_fn(self._state, *inputs, seed=self._next_seed())
            reward = reward[:, 0, 0]
        return self._normalize_intrinsic(reward)

    def ctec_rollout_reward(self, obs, actions, next_obs, episode_starts, dones) -> np.ndarray:
        """C-TeC rewards of a whole [env, time] rollout, scored against a future state."""
        indices = sample_future_indices(
            episode_starts, dones, self.module.discount, np.random.default_rng(self._next_seed())
        )
        future = next_obs[np.arange(len(obs))[:, None], indices]
        rows = tuple(
            jnp.asarray(x.reshape(-1, x.shape[-1]))
            for x in (obs, one_hot_actions(actions, self.n_actions), future)
        )

        def score(rows):
            return chunked_rows(
                lambda xs: self.module.compute_reward(xs[0], xs[1], future_obs=xs[2]), rows, 256
            )

        reward = self._run(score, rows).reshape(obs.shape[:2])
        return self._normalize_intrinsic(reward)

    def ppo_reward(self, intrinsic: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Scale the intrinsic reward and normalize it by its discounted return."""
        reward = self.intrinsic_scale * np.asarray(intrinsic, np.float32)
        if not np.all(np.isfinite(reward)):
            raise FloatingPointError(f"Nonfinite {self.idm_type} intrinsic reward")
        if self._return_rms is None:
            return reward
        if self._returns.shape != reward.shape:
            self._returns = np.zeros(reward.shape)
        self._returns = self._returns * self._gamma + reward
        self._return_rms.update(self._returns)
        reward = np.clip(reward / np.sqrt(self._return_rms.var + 1e-8), -10.0, 10.0)
        self._returns[np.asarray(dones, bool)] = 0.0
        return reward.astype(np.float32)

    def train_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        next_obs: np.ndarray,
        *,
        episode_starts: np.ndarray,
        dones: np.ndarray,
        skills: np.ndarray | None = None,
        replay_raw: list | None = None,
    ) -> dict[str, float]:
        """One IDM update from an [env, time] rollout.

        ``replay_raw`` holds the raw (obs, next_obs) of the transitions planned
        by ``begin_rollout``; APT/RE3 keep them to refresh their memory.
        """
        if self.idm_type in {"vic", "valor"}:
            if not self._completed:
                return {}
            completed, self._completed = self._completed, []
            batch = self._segment_batch(completed)
            metrics = {"completed_segments": len(completed), "replay_batch": len(batch["skill"])}
        else:
            if self.idm_type == "ctec":
                rng = np.random.default_rng(self._next_seed())
                future = sample_future_indices(episode_starts, dones, self.module.discount, rng)
                next_obs = next_obs[np.arange(len(obs))[:, None], future]
            batch = self._transition_batch(obs, actions, next_obs, skills, self._replay_rows)
            metrics = {"replay_batch": len(self._replay_rows)}
        self._replay_rows = None
        metrics.update(self._run(self._train, batch))
        if self._optimizer is not None:
            self._updates += 1
        if self.idm_type in MEMORY_IDMS:
            # APT remembers next observations, RE3 current ones.
            self._remember([pair[1] if self.idm_type == "apt" else pair[0] for pair in replay_raw])
        metrics["updates"] = self._updates
        return {key: float(value) for key, value in metrics.items()}

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict[str, Any]:
        sampler = self.skill_sampler
        return {
            "idm_type": self.idm_type,
            "state": jax.device_get(self._state),
            "rng": self._rng,
            "updates": self._updates,
            "segments": self._segments,
            "memory_rows": self._memory_rows,
            "return_rms": self._return_rms,
            "returns": self._returns,
            "skill_sampler": None if sampler is None else sampler.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload["idm_type"] != self.idm_type:
            raise ValueError(f"IDM checkpoint is {payload['idm_type']!r}, not {self.idm_type!r}")
        state = payload["state"]
        mismatched = sorted(
            key
            for key, value in self._state.items()
            if key not in state or np.shape(state[key]) != np.shape(value)
        )
        if mismatched:
            raise ValueError(f"IDM checkpoint arrays do not match this IDM: {mismatched[:5]}")
        self._state = jax.tree.map(jnp.asarray, state)
        self._rng = payload["rng"]
        self._updates = payload["updates"]
        self._segments = payload["segments"]
        self._memory_rows = payload["memory_rows"]
        self._return_rms = payload["return_rms"]
        self._returns = payload["returns"]
        if self.skill_sampler is not None:
            self.skill_sampler.load_state_dict(payload["skill_sampler"])

    def save(self, path: str | PathLike[str]) -> None:
        path = pathlib.Path(path)
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("wb") as file:
            pickle.dump(self.state_dict(), file, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    def load(self, path: str | PathLike[str]) -> None:
        with pathlib.Path(path).open("rb") as file:
            self.load_state_dict(pickle.load(file))

    # -------------------------------------------------------------- internals

    def _next_seed(self) -> int:
        self._rng += 1
        return self._rng

    def _run(self, fn, *args):
        """Run an impure ninjax function on the IDM state."""
        self._state, out = nj.pure(fn)(self._state, *args, seed=self._next_seed(), create=True)
        return out

    def _init_state(self) -> dict[str, Any]:
        """Create the IDM and optimizer variables by tracing one update on placeholders."""
        n = 4  # placeholder transitions or segments
        skill_dim = 0 if self.skill_sampler is None else self.skill_sampler.skill_event_dim
        if self.idm_type in {"vic", "valor"}:
            length = 2 if self.idm_type == "vic" else int(self.module.num_subseq)
            segment = (
                np.zeros((length, self.obs_dim), np.float32),
                np.zeros(skill_dim, np.float32),
            )
            batch = self._segment_batch([segment] * n)
        else:
            batch = self._transition_batch(
                np.zeros((1, n, self.obs_dim), np.float32),
                np.zeros((1, n), np.int64),
                np.zeros((1, n, self.obs_dim), np.float32),
                None if self.skill_sampler is None else np.zeros((1, n, skill_dim), np.float32),
                np.arange(n),
            )

        def init():
            if self._intrinsic_rms is not None:
                self._intrinsic_rms.reset()
            return self._train(batch)

        state = nj.init(init)({}, seed=self._next_seed())
        if self.idm_type in MEMORY_IDMS:
            # Drop the placeholder rows that initialization wrote to the k-NN memory.
            state, _ = nj.pure(self.module.reset)(state, seed=self._next_seed(), create=True)
        return state

    def _transition_batch(self, obs, actions, next_obs, skills, rows) -> dict[str, Any]:
        """Replay rows of [env, time] arrays as a [1, rows, ...] batch."""
        if rows is None:
            raise RuntimeError("begin_rollout() must precede train_step()")

        def take(values):
            values = np.asarray(values)
            flat = values.reshape(-1, *values.shape[2:])[rows][None]
            # Ensemble IDMs bootstrap over the leading axis, so put the rows there.
            return flat.swapaxes(0, 1) if self.idm_type in {"disagreement", "max"} else flat

        return {
            "obs": take(obs),
            "action": take(one_hot_actions(actions, self.n_actions)),
            "next_obs": take(next_obs),
            "skill": None if skills is None else take(skills),
        }

    def _segment_batch(self, segments) -> dict[str, Any]:
        """At most replay_batch_size completed (states, skill) segments."""
        if len(segments) > self.replay_batch_size:
            rng = np.random.default_rng(self._next_seed())
            keep = rng.choice(len(segments), self.replay_batch_size, replace=False)
            segments = [segments[i] for i in keep]
        return {
            "states": np.stack([states for states, _ in segments]),
            "skill": np.stack([skill for _, skill in segments]),
        }

    def _loss(self, batch):
        module, kind = self.module, self.idm_type
        if kind == "vic":
            states = batch["states"]
            return module.loss(states[:, 0], states[:, -1], batch["skill"])
        if kind == "valor":
            return module.loss(batch["states"], batch["skill"])
        obs, action, next_obs, skill = (batch[k] for k in ("obs", "action", "next_obs", "skill"))
        mask = jnp.ones(obs.shape[:2], bool)
        if kind == "rnd":
            return module.loss(obs, mask)
        if kind == "re3":
            return module.loss(obs, mask, update_reward_norm=False)
        if kind == "apt":
            return module.loss(next_obs, mask, update_reward_norm=False)
        if kind == "proto_rl":
            return module.loss(obs, next_obs, mask, update_queue=True)
        if kind == "diayn":
            return module.loss(obs, skill, mask)
        if kind in SKILL_IDMS:  # dads, rdads, csf
            return module.loss(obs, next_obs, skill, mask)
        return module.loss(obs, action, next_obs, mask)

    def _train(self, batch):
        batch = jax.tree.map(jnp.asarray, batch)
        if self._optimizer is None:
            return self._loss(batch)[1]
        metrics, aux = self._optimizer(lambda: self._loss(batch), has_aux=True)
        if hasattr(self.module, "post_update"):
            self.module.post_update()
        return {**metrics, **aux}

    def _transition_reward(self, obs, action, next_obs, skill, probabilities):
        extra = {}
        if self.idm_type == "empowerment_process":
            # Sample the behavior policy's actions for the process-MI estimator.
            indices = jax.random.categorical(
                nj.seed(),
                jnp.log(probabilities)[..., None, :],
                shape=(*probabilities.shape[:-1], int(self.module.particles)),
            )
            extra["process_action_particles"] = jax.nn.one_hot(indices, self.n_actions)
        return self.module.compute_reward(
            obs, action, next_obs, skill=skill, update_reward_norm=True, **extra
        )

    def _segment_reward(self, obs, next_obs, raw_obs, raw_next_obs, starts, ended, skills):
        if len(self._segments) != len(obs):
            self._segments = [[] for _ in obs]
        for i, segment in enumerate(self._segments):
            if starts[i] or not segment:
                segment[:] = [(obs[i], observation_row(raw_obs, i))]
            elif self.idm_type == "vic" and len(segment) > 1:
                segment.pop()  # VIC only needs the start and the latest state.
            segment.append((next_obs[i], observation_row(raw_next_obs, i)))

        reward = np.zeros(len(obs), np.float32)
        if self.idm_type == "ngu":
            reward = self._ngu_reward(next_obs, starts)
        elif ended.any():
            index = np.flatnonzero(ended)
            states = [np.stack([f for f, _ in self._segments[i]]) for i in index]
            if self.idm_type == "valor":
                samples = int(self.module.num_subseq)
                states = [
                    s[np.rint(np.linspace(0, len(s) - 1, samples)).astype(int)] for s in states
                ]
            self._completed.extend(zip(states, skills[index], strict=True))
            reward[index] = self._run(self._skill_segment_reward, np.stack(states), skills[index])
        for i in np.flatnonzero(ended):
            self._segments[i] = []
        return reward

    def _skill_segment_reward(self, states, skills):
        if self.idm_type == "vic":
            return self.module.segment_reward(states[:, 0], states[:, -1], skills)[:, 0]
        return self.module.compute_reward(states, skill=skills)[:, -1, 0]

    def _ngu_reward(self, next_obs, starts):
        if self.module.use_rnd:
            self._run(
                lambda: self.module.compute_reward(
                    jnp.asarray(next_obs[None]), is_first=jnp.asarray(starts[None])
                )
            )
        by_length = defaultdict(list)
        for i, segment in enumerate(self._segments):
            by_length[len(segment)].append(i)
        reward = np.zeros(len(next_obs), np.float32)
        for index in by_length.values():
            history = np.stack([[f for f, _ in self._segments[i]] for i in index], axis=1)
            reward[index] = self._run(self._ngu_episode_reward, jnp.asarray(history))
        return reward

    def _ngu_episode_reward(self, history):
        """Reward of the last step of each [time, env] episode history."""
        first = jnp.zeros(history.shape[:2], bool).at[0].set(True)
        reward = self.module.compute_reward(history, is_first=first, update_reward_norm=False)
        return reward[-1, :, 0]

    def _normalize_intrinsic(self, reward) -> np.ndarray:
        if self._intrinsic_rms is not None:
            reward = self._run(
                lambda: self._intrinsic_rms.normalize(jnp.asarray(reward), update=True)
            )
        return np.asarray(reward, np.float32)

    def _refresh_memory(self, features):
        memory = self.module._mem
        memory.write(memory.read().at[: len(features)].set(self.module.embed(features)))

    def _remember(self, rows) -> None:
        """Store the raw observations of the rows the last update wrote to memory."""
        size = int(self.module.memory_size)
        ptr = int(self._state[f"{self.idm_type}/mem_ptr/value"])
        count = int(self._state[f"{self.idm_type}/mem_count/value"])
        self._memory_rows.extend([None] * (count - len(self._memory_rows)))
        for i, row in enumerate(rows):
            self._memory_rows[(ptr - len(rows) + i) % size] = row

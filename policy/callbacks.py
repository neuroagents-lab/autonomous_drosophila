from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization


class SaveAtRolloutStartCallback(BaseCallback):
    """Call ``save_fn(model)`` before each rollout, i.e. after every PPO update."""

    def __init__(self, save_fn: Callable[[object], None]):
        super().__init__()
        self.save_fn = save_fn

    def _on_rollout_start(self) -> None:
        self.save_fn(self.model)

    def _on_step(self) -> bool:
        return True


_ROLLOUT_LOGGER_PREFIXES = (
    "rollout/",
    "reward/",
    "behavior/",
    "options/",
    "episode/",
    "vision/",
    "actions/",
    "bench/",
    "time/",
    "idm/",
)


def _wandb_log(metrics: dict[str, float], *, step: int) -> None:
    if metrics and wandb.run is not None:
        wandb.log({**metrics, "global_step": float(step)}, step=step)


def _logger_metrics(logger, prefixes: tuple[str, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in logger.name_to_value.items():
        if key.startswith(prefixes):
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics


class PPOFlyGymWandbCallback(BaseCallback):
    """Log the FlyGym PPO scalars to W&B at SB3 iteration boundaries."""

    def __init__(self, *, eval_callback: NavigationEvalCallback | None = None) -> None:
        super().__init__()
        self.eval_callback = eval_callback
        self._iteration = 0

    def _on_training_start(self) -> None:
        self._iteration = 0

    def _on_rollout_start(self) -> None:
        # SB3 records train/ metrics after the rollout end of the previous iteration.
        if self._iteration >= 1:
            self._log_train_metrics()
        self._iteration += 1

    def _on_rollout_end(self) -> None:
        metrics = _logger_metrics(self.model.logger, _ROLLOUT_LOGGER_PREFIXES)
        _wandb_log(metrics, step=self.model.num_timesteps)

    def _on_step(self) -> bool:
        if self.eval_callback is not None:
            _wandb_log(self.eval_callback.pop_last_eval_metrics(), step=self.model.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self._log_train_metrics()

    def _log_train_metrics(self) -> None:
        metrics = _logger_metrics(self.model.logger, ("train/",))
        elapsed = max((time.time_ns() - self.model.start_time) / 1e9, sys.float_info.epsilon)
        steps = self.model.num_timesteps - self.model._num_timesteps_at_start
        metrics["speed/train_steps_per_sec"] = float(int(steps / elapsed))
        _wandb_log(metrics, step=self.model.num_timesteps)


class NavigationEvalCallback(BaseCallback):
    """Evaluate every ``eval_every_rollouts`` PPO updates and save the best policies.

    ``save_best(model, name)`` is called with ``best_model`` on a new best mean
    eval reward and with ``best_success_model`` on a new best success rate.
    The evaluation history is kept in ``log_path`` (an ``.npz`` file).
    """

    def __init__(
        self,
        eval_env,
        *,
        save_best: Callable[[object, str], None],
        n_eval_episodes: int,
        eval_every_rollouts: int,
        log_path: Path,
        completed_rollouts: int = 0,
        resume: bool = False,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.save_best = save_best
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_every_rollouts = int(eval_every_rollouts)
        self.log_path = Path(log_path)
        self._completed_rollouts = int(completed_rollouts)
        self.history: dict[str, list] = {
            "timesteps": [],
            "results": [],
            "ep_lengths": [],
            "successes": [],
        }
        self.best_mean_reward = -np.inf
        self.best_success_rate = -np.inf
        self._pending_eval_metrics: dict[str, float] = {}
        if resume and self.log_path.is_file():
            self._restore_history()

    def _restore_history(self) -> None:
        with np.load(self.log_path, allow_pickle=True) as saved:
            for key in self.history:
                if key in saved:
                    self.history[key] = list(saved[key])
        if self.history["results"]:
            self.best_mean_reward = max(float(np.mean(r)) for r in self.history["results"])
        rates = [float(np.mean(s)) for s in self.history["successes"] if len(s)]
        if rates:
            self.best_success_rate = max(rates)

    def pop_last_eval_metrics(self) -> dict[str, float]:
        metrics, self._pending_eval_metrics = self._pending_eval_metrics, {}
        return metrics

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._completed_rollouts += 1
        if self._completed_rollouts % self.eval_every_rollouts == 0:
            self._evaluate()

    def _evaluate(self) -> None:
        sync_envs_normalization(self.training_env, self.eval_env)
        successes: list[bool] = []

        def record_success(locals_, _globals) -> None:
            if locals_["done"]:
                info = locals_["info"]
                success = info.get("is_success", info.get("success"))
                if success is not None:
                    successes.append(bool(success))

        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
            callback=record_success,
        )
        for key, value in (
            ("timesteps", self.num_timesteps),
            ("results", rewards),
            ("ep_lengths", lengths),
            ("successes", successes),
        ):
            self.history[key].append(value)
        np.savez(
            self.log_path,
            timesteps=self.history["timesteps"],
            results=self.history["results"],
            ep_lengths=self.history["ep_lengths"],
            successes=np.asarray(self.history["successes"], dtype=object),
        )

        metrics = {
            "eval/mean_reward": float(np.mean(rewards)),
            "eval/std_reward": float(np.std(rewards)),
            "eval/mean_ep_length": float(np.mean(lengths)),
            "eval/std_ep_length": float(np.std(lengths)),
        }
        if successes:
            metrics["eval/success_rate"] = float(np.mean(successes))
        for key, value in metrics.items():
            self.logger.record(key, value)
        self._pending_eval_metrics = metrics
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        if metrics["eval/mean_reward"] > self.best_mean_reward:
            self.best_mean_reward = metrics["eval/mean_reward"]
            self.save_best(self.model, "best_model")
        if successes and metrics["eval/success_rate"] > self.best_success_rate:
            self.best_success_rate = metrics["eval/success_rate"]
            self.save_best(self.model, "best_success_model")

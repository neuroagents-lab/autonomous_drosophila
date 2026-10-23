#!/usr/bin/env python3
"""Evaluate a trained FlyGym odor-navigation policy."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.training import (
    FlyBodyNavigationTrainEnv,
    make_flygym_train_config,
    resolved_env_timing_from_ssrl_envs,
)
from policy.ppo import add_skill_observations, load_policy, resolve_checkpoint_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FlyGym odor-navigation policy."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing .hydra/config.yaml and checkpoints.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-env-num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--backend",
        choices=["cpu", "warp"],
        default="warp",
        help="Physics backend for eval rollouts.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=("Directory for episode mp4s and trajectory_overview.png."),
    )
    parser.add_argument(
        "--eval-camera",
        choices=("navigation_birdeye", "follow"),
        default="navigation_birdeye",
        help="Camera used for render_eval episode videos.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for per-episode metrics JSON.",
    )
    args = parser.parse_args(argv)
    if args.episodes < 1 or (args.eval_env_num is not None and args.eval_env_num < 1):
        parser.error("episodes and eval-env-num must be positive")
    return args


def load_run_config(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    return OmegaConf.load(config_path)


def fly_xy_from_info(info: dict[str, Any]) -> np.ndarray:
    fly_position = info.get("fly_position")
    if fly_position is not None:
        return np.asarray(fly_position, dtype=np.float32)[:2].copy()
    raise KeyError("info missing fly position")


def select_trajectory_episodes(
    episode_logs: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(episode_logs) <= 4:
        return sorted(
            episode_logs,
            key=lambda episode_log: int(episode_log["episode_idx"]),
        )

    ranked_episode_logs = sorted(
        episode_logs,
        key=lambda episode_log: (
            float(episode_log["total_reward"]),
            int(episode_log["episode_idx"]),
        ),
    )
    selected_by_episode_idx = {
        int(episode_log["episode_idx"]): episode_log
        for episode_log in ranked_episode_logs[:2] + ranked_episode_logs[-2:]
    }
    return sorted(
        selected_by_episode_idx.values(),
        key=lambda episode_log: (
            float(episode_log["total_reward"]),
            int(episode_log["episode_idx"]),
        ),
    )


def save_trajectory_plot(
    record_dir: Path,
    episode_logs: list[dict[str, object]],
) -> None:
    if not episode_logs:
        return

    selected_episode_logs = select_trajectory_episodes(episode_logs)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), tight_layout=True)
    color_map = plt.get_cmap("tab10")
    x_values: list[float] = []
    y_values: list[float] = []
    max_obstacle_radius = 0.0

    for color_index, episode_log in enumerate(selected_episode_logs):
        color = color_map(color_index % 10)
        trajectory_xy = np.asarray(episode_log["trajectory_xy"], dtype=np.float32)
        odor_source = np.asarray(episode_log["odor_source"], dtype=np.float32)
        obstacle_positions = np.asarray(
            episode_log["obstacle_positions"],
            dtype=np.float32,
        )
        obstacle_radius = float(episode_log["obstacle_radius"])
        target_odor_source_index = int(episode_log["target_odor_source_index"])
        reward_odor_source_indices = {
            int(source_index)
            for source_index in episode_log.get(
                "reward_odor_source_indices",
                (target_odor_source_index,),
            )
        }

        x_values.extend(trajectory_xy[:, 0].tolist())
        y_values.extend(trajectory_xy[:, 1].tolist())
        if odor_source.size > 0:
            x_values.extend(odor_source[:, 0].tolist())
            y_values.extend(odor_source[:, 1].tolist())
        if obstacle_positions.size > 0:
            x_values.extend(obstacle_positions[:, 0].tolist())
            y_values.extend(obstacle_positions[:, 1].tolist())
            max_obstacle_radius = max(max_obstacle_radius, obstacle_radius)

        for odor_index, odor_position in enumerate(odor_source):
            rewarded = odor_index in reward_odor_source_indices
            marker = "*" if rewarded else "o"
            ax.scatter(
                [float(odor_position[0])],
                [float(odor_position[1])],
                marker=marker,
                color=color,
                s=90 if rewarded else 45,
                alpha=0.9,
                zorder=3,
            )

        for obstacle_position in obstacle_positions:
            ax.add_patch(
                Circle(
                    (float(obstacle_position[0]), float(obstacle_position[1])),
                    radius=obstacle_radius,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=1.2,
                    alpha=0.4,
                    zorder=1,
                )
            )

        ax.plot(
            trajectory_xy[:, 0],
            trajectory_xy[:, 1],
            color=color,
            linewidth=2.0,
            alpha=0.95,
            label=(
                f"episode {int(episode_log['episode_idx'])}: {float(episode_log['total_reward']):g}"
            ),
            zorder=2,
        )
        ax.scatter(
            [float(trajectory_xy[0, 0])],
            [float(trajectory_xy[0, 1])],
            color=color,
            s=24,
            zorder=4,
        )

    if x_values and y_values:
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        span = max(x_max - x_min, y_max - y_min, 1.0)
        margin = max(max_obstacle_radius + 1.0, 0.08 * span)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

    title_suffix = "Top 2 / Bottom 2 by reward" if len(episode_logs) > 4 else "All episodes"
    ax.set_title(f"Trajectory Overview ({title_suffix}; stars=rewarded odors)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(record_dir / "trajectory_overview.png", dpi=150)
    plt.close(fig)


class SaveEpisodeRecordingsWrapper(gym.Wrapper):
    """Save rollout videos before VecEnv auto-reset clears the renderer buffer."""

    def __init__(self, env: gym.Env, record_dir: Path) -> None:
        super().__init__(env)
        self.record_dir = record_dir
        self._episode_idx = 0

    def save_recordings(self, video_path, stabilization_time: float = 0.02) -> None:
        self.env.save_recordings(video_path, stabilization_time=stabilization_time)

    def reset_recordings(self) -> None:
        self.env.reset_recordings()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            video_path = self.record_dir / f"episode_{self._episode_idx}.mp4"
            self.save_recordings(video_path)
            self.reset_recordings()
            self._episode_idx += 1
        return observation, reward, terminated, truncated, info


def make_eval_vec_env(
    env_config,
    train_config,
    *,
    backend: str,
    num_envs: int,
    seed: int,
    render_mode: str | None,
    record_dir: Path | None = None,
    eval_camera: str = "navigation_birdeye",
):
    if backend == "warp":
        from envs.warp import make_flygym_warp_vec_env

        return make_flygym_warp_vec_env(
            env_config,
            num_envs=num_envs,
            layout_seed_base=seed,
            render_mode=render_mode,
            record_dir=record_dir,
            eval_camera=eval_camera,
        )

    def make_env():
        env_config_copy = deepcopy(train_config)
        env_config_copy.base_env_config.eval_camera = eval_camera
        env = FlyBodyNavigationTrainEnv(
            config=env_config_copy,
            render_mode=render_mode,
        )
        if record_dir is not None:
            env = SaveEpisodeRecordingsWrapper(env, record_dir)
        return env

    vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
    vec_env.seed(seed)
    return vec_env


def unwrap_vec_env(vec_env):
    current = vec_env
    while hasattr(current, "venv"):
        current = current.venv
    return current


def reset_infos(vec_env) -> list[dict[str, Any]]:
    current = unwrap_vec_env(vec_env)
    infos = getattr(current, "reset_infos", None)
    if infos is None:
        return [{} for _ in range(vec_env.num_envs)]
    return [dict(info) for info in infos]


def call_env_method(vec_env, method_name: str, *args, **kwargs) -> None:
    method = getattr(vec_env, "env_method", None)
    if method is None:
        return
    try:
        method(method_name, *args, **kwargs)
    except AttributeError:
        return


def active_obstacle_positions_from_info(info: dict[str, Any]) -> np.ndarray:
    obstacles = info.get("active_obstacle_positions")
    if obstacles is not None:
        return np.asarray(obstacles, dtype=np.float32)
    all_obstacles = info.get("obstacle_positions")
    active_count = int(info.get("active_obstacle_count", 0))
    if all_obstacles is None:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(all_obstacles, dtype=np.float32)[:active_count]


def make_episode_state(reset_info: dict[str, Any]) -> dict[str, object]:
    initial_xy = fly_xy_from_info(reset_info)
    return {
        "reward": 0.0,
        "trajectory_xy": [initial_xy],
    }


def run_policy_eval(
    vec_env,
    model,
    *,
    episodes: int,
    record_dir: Path | None,
    train_config,
) -> list[dict[str, object]]:
    obs = vec_env.reset()
    if record_dir is not None:
        call_env_method(vec_env, "reset_recordings")
    states = None
    episode_starts = np.ones(vec_env.num_envs, dtype=bool)
    current = [make_episode_state(info) for info in reset_infos(vec_env)]
    episode_logs: list[dict[str, object]] = []

    while len(episode_logs) < episodes:
        actions, states = model.predict(
            obs,
            state=states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, rewards, dones, infos = vec_env.step(actions)
        episode_starts = np.asarray(dones, dtype=bool)

        for env_idx, info in enumerate(infos):
            state = current[env_idx]
            state["reward"] = float(state["reward"]) + float(rewards[env_idx])
            state["trajectory_xy"].append(fly_xy_from_info(info))
            if not dones[env_idx]:
                continue

            episode_idx = len(episode_logs)
            episode_logs.append(
                {
                    "episode_idx": episode_idx,
                    "total_reward": float(state["reward"]),
                    "success": bool(info["success"]),
                    "min_distance": float(info["min_distance_to_target"]),
                    "steps": int(info["episode_step_count"]),
                    "trajectory_xy": np.stack(state["trajectory_xy"], axis=0),
                    "odor_source": np.asarray(
                        info.get("target_position", [np.nan, np.nan, np.nan]),
                        dtype=np.float32,
                    ).reshape(1, 3),
                    "obstacle_positions": active_obstacle_positions_from_info(info),
                    "obstacle_radius": float(train_config.base_env_config.obstacle_radius),
                    "target_odor_source_index": int(train_config.target_odor_source_index),
                    "reward_odor_source_indices": (train_config.reward_odor_source_indices or (0,)),
                }
            )
            print(
                f"episode={episode_idx} reward={float(state['reward']):.3f} "
                f"success={bool(info['success'])} "
                f"min_distance={float(info['min_distance_to_target']):.3f} "
                f"steps={int(info['episode_step_count'])}",
                flush=True,
            )
            if len(episode_logs) >= episodes:
                break
            reset_info = reset_infos(vec_env)[env_idx]
            current[env_idx] = make_episode_state(reset_info)

    return episode_logs


def _print_eval_header(
    run_dir: Path,
    model_path: Path,
    vecnormalize_path: Path | None,
    timing,
    train_config,
) -> None:
    base = train_config.base_env_config
    print(f"Eval run: {run_dir.name}")
    print(f"RUN_DIR={run_dir}")
    print(f"model={model_path}")
    print(f"vecnormalize={vecnormalize_path or '<disabled>'}")
    print(
        "Resolved env timing: "
        f"decision_steps={timing.decision_steps} "
        f"decision_dt={timing.decision_dt:.6f}s "
        f"max_decision_steps={timing.max_decision_steps} "
        f"episode_time={timing.episode_time_seconds:.6f}s "
        f"backend={base.backend}"
    )
    print(
        "Policy: "
        f"policy_action_space={train_config.policy_action_space} "
        f"action_set={train_config.action_set}"
    )
    print(
        f"Reward shaping: w_p={float(train_config.progress_reward_weight):g} "
        f"c={float(train_config.time_penalty):g} "
        f"success_bonus={float(train_config.success_bonus):g} "
        f"terminate_on_success={train_config.terminate_on_success} "
        f"reward_odor_source_indices={train_config.reward_odor_source_indices or (0,)}"
    )


def _write_metrics_json(
    output_json: Path,
    *,
    run_dir: Path,
    model_path: Path,
    vecnormalize_path: Path | None,
    eval_mode: str,
    backend: str,
    seed: int,
    eval_env_num: int,
    episode_logs,
) -> None:
    successes = [bool(log["success"]) for log in episode_logs]
    rewards = [float(log["total_reward"]) for log in episode_logs]
    min_distances = [float(log["min_distance"]) for log in episode_logs]
    payload = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path else None,
        "eval_mode": eval_mode,
        "backend": backend,
        "render_backend": backend if eval_mode == "render_eval" else None,
        "seed": int(seed),
        "eval_env_num": int(eval_env_num),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_min_distance": (float(np.mean(min_distances)) if min_distances else float("nan")),
        "episodes": [
            {
                "episode_idx": int(log["episode_idx"]),
                "total_reward": float(log["total_reward"]),
                "success": bool(log["success"]),
                "min_distance": float(log["min_distance"]),
                "steps": int(log["steps"]),
            }
            for log in episode_logs
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {output_json}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    run_config = load_run_config(run_dir)
    model_path, vecnormalize_path = resolve_checkpoint_paths(
        run_dir,
        args.model_path,
        vecnormalize_required=True,
    )

    env_config = deepcopy(run_config.envs)
    env_config.backend = args.backend
    train_config = make_flygym_train_config(env_config)
    timing = resolved_env_timing_from_ssrl_envs(env_config)
    _print_eval_header(run_dir, model_path, vecnormalize_path, timing, train_config)

    eval_mode = "render_eval" if args.record_dir is not None else "canonical_eval"
    eval_env_num = (
        1
        if args.backend == "cpu" or args.record_dir
        else int(args.eval_env_num or min(args.episodes, 10))
    )
    render_mode = "rgb_array" if args.record_dir else None
    if args.record_dir is not None:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        print(f"Recordings -> {args.record_dir}")
    print(f"eval_mode={eval_mode} eval_env_num={eval_env_num}")
    rollout_record_dir = args.record_dir if eval_mode == "render_eval" else None

    vec_env = make_eval_vec_env(
        env_config,
        train_config,
        backend=args.backend,
        num_envs=eval_env_num,
        seed=args.seed,
        render_mode=render_mode,
        record_dir=rollout_record_dir,
        eval_camera=args.eval_camera,
    )
    eval_vec_env = vec_env
    try:
        eval_vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False
        eval_vec_env = add_skill_observations(run_config, eval_vec_env)[0]
        model = load_policy(model_path, args.device)
        episode_logs = run_policy_eval(
            eval_vec_env,
            model,
            episodes=args.episodes,
            record_dir=rollout_record_dir,
            train_config=train_config,
        )
    finally:
        eval_vec_env.close()

    if args.record_dir is not None:
        save_trajectory_plot(args.record_dir, episode_logs)
        print(f"Wrote trajectory overview to {args.record_dir / 'trajectory_overview.png'}")

    if args.output_json is not None:
        _write_metrics_json(
            args.output_json,
            run_dir=run_dir,
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            eval_mode=eval_mode,
            backend=args.backend,
            seed=args.seed,
            eval_env_num=eval_env_num,
            episode_logs=episode_logs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

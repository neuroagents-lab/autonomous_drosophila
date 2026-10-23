#!/usr/bin/env python3
"""Evaluate blinded intrinsic-exploration policies on the tethered ball treadmill."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from envs.action_spaces import ACTION_SET, discrete_action_specs
from envs.tethered_ball.policy_runner import (
    BEHAVIOR_TRACKING_HZ,
    DEFAULT_TIMING,
    RECORDING_DURATION_SECONDS,
    BallRunner,
    PolicyObservationAdapter,
    load_frozen_vecnormalize,
    load_policy_skill_sampler,
)
from envs.tethered_ball.world import BALL_RADIUS_MM
from envs.tethered_ball.yaw_processing import (
    VELOCITY_FILTER_WINDOW_SECONDS,
    ball_angular_velocity_to_fictive_velocity,
    smooth_fictive_velocity,
)
from envs.training import resolved_env_timing_from_ssrl_envs
from policy.activations import (
    RecurrentActivationRecorder,
    write_activation_hdf5,
)
from policy.ppo import load_policy, resolve_checkpoint_paths

SACCADE_THRESHOLD_DEG_S = 200.0
STATIONARY_FORWARD_THRESHOLD_MM_S = 1.0
STATIONARY_ROTATION_THRESHOLD_DEG_S = 20.0
FLY_NEURAL_POOLING_INTERVAL_SECONDS = 0.532


def rollout_batch(
    model,
    runner,
    n_decisions: int,
    substeps: int,
    deterministic: bool,
    observation_adapter: PolicyObservationAdapter,
    activation_recorder: RecurrentActivationRecorder | None = None,
):
    num_envs = runner.num_envs
    state = None
    episode_starts = np.ones(num_envs, dtype=bool)
    actions = np.empty((num_envs, n_decisions), dtype=int)
    angular_velocity = []
    skills = np.empty(
        (num_envs, n_decisions, observation_adapter.skill_event_dim),
        dtype=np.float32,
    )
    skill_boundaries = np.zeros((num_envs, n_decisions), dtype=bool)
    record_observations = activation_recorder is not None
    proprioception = None
    observation_adapter.reset()
    if activation_recorder is not None:
        activation_recorder.reset()
    for decision in range(n_decisions):
        raw_observation, observation = observation_adapter.observation_pair(runner)
        if record_observations:
            proprio = np.asarray(raw_observation["proprioception"], dtype=np.float32)
            if proprio.ndim != 2 or proprio.shape[0] != num_envs:
                raise ValueError(
                    f"batched proprioception must have shape ({num_envs}, dim); got {proprio.shape}"
                )
            if proprioception is None:
                proprioception = np.empty(
                    (num_envs, n_decisions, int(proprio.shape[-1])),
                    dtype=np.float32,
                )
            proprioception[:, decision] = proprio
        if observation_adapter.skill_event_dim:
            skills[:, decision] = observation["skill"]
        if activation_recorder is None:
            action, state = model.predict(
                observation,
                state=state,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
        else:
            action, _ = activation_recorder.predict(
                observation,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
        action_indices = np.asarray(action).reshape(num_envs)
        actions[:, decision] = action_indices
        angular_velocity.append(runner.run_decision(action_indices, substeps))
        skill_boundaries[:, decision] = observation_adapter.advance_batch()
        episode_starts.fill(False)
    observations = None if proprioception is None else {"proprioception": proprioception}
    return (
        actions,
        np.concatenate(angular_velocity, axis=1),
        skills,
        skill_boundaries,
        observations,
    )


def process_velocity(
    angular_velocity: np.ndarray, ball_radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forward, rotational = ball_angular_velocity_to_fictive_velocity(angular_velocity, ball_radius)
    smoothed = smooth_fictive_velocity(np.column_stack((forward, rotational)))
    return forward, rotational, smoothed[:, 0], smoothed[:, 1]


def summarize(
    actions: np.ndarray,
    forward_velocity: np.ndarray,
    rotational_velocity: np.ndarray,
) -> dict:
    action_specs = discrete_action_specs()
    action_labels = {spec.flat_index: spec.name for spec in action_specs}
    walk_actions = [
        spec.flat_index
        for spec in action_specs
        if spec.coarse_label in {"left", "forward", "right"}
    ]
    positive_saccades, negative_saccades = _detect_turn_peaks(rotational_velocity)
    stationary = (np.abs(forward_velocity) < STATIONARY_FORWARD_THRESHOLD_MM_S) & (
        np.abs(rotational_velocity) < STATIONARY_ROTATION_THRESHOLD_DEG_S
    )
    return {
        "action_fraction": {action_labels[i]: float(np.mean(actions == i)) for i in action_labels},
        "walk_action_fraction": float(np.mean(np.isin(actions, walk_actions))),
        "stationary_fraction": float(np.mean(stationary)),
        "mean_forward_velocity_mm_s": float(np.mean(forward_velocity)),
        "mean_abs_forward_velocity_mm_s": float(np.mean(np.abs(forward_velocity))),
        "mean_rotational_velocity_deg_s": float(np.mean(rotational_velocity)),
        "mean_abs_rotational_velocity_deg_s": float(np.mean(np.abs(rotational_velocity))),
        "positive_saccade_count": int(positive_saccades.size),
        "negative_saccade_count": int(negative_saccades.size),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "local_data" / "ball_eval" / "ball_policy_eval",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="training run directory containing .hydra/config.yaml",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="explicit checkpoint zip (latest_model.zip or best_model.zip)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=9,
        help="rollouts per policy (default: the paper's nine animals)",
    )
    p.add_argument(
        "--backend",
        choices=("cpu", "warp"),
        default="warp",
        help="physics backend for evaluation",
    )
    p.add_argument(
        "--eval-env-num",
        type=int,
        default=None,
        help="parallel Warp worlds (default: min(episodes, 10))",
    )
    p.add_argument(
        "--duration-seconds",
        type=float,
        default=RECORDING_DURATION_SECONDS,
        help="duration per rollout (default: the paper's 30-minute recording)",
    )
    p.add_argument(
        "--decisions",
        type=int,
        default=None,
        help="policy decisions per rollout; overrides --duration-seconds",
    )
    p.add_argument(
        "--decision-steps",
        type=int,
        default=None,
        help="physics steps per decision (default: saved training configuration)",
    )
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="skip episode yaw PNG outputs",
    )
    p.add_argument(
        "--save-activations",
        action="store_true",
        help=(
            "save every policy-decision activation (plus a derived pooled view) "
            "and raw pre-decision observations (proprioception) aligned to those "
            "decisions"
        ),
    )
    p.add_argument(
        "--activation-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="storage dtype for policy activations",
    )
    p.add_argument(
        "--activation-pooling-interval-seconds",
        type=float,
        default=FLY_NEURAL_POOLING_INTERVAL_SECONDS,
        help="mean-pooling interval for fly-cadence activations",
    )
    return p.parse_args(argv)


def _detect_turn_peaks(yaw_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = find_peaks(yaw_velocity, height=SACCADE_THRESHOLD_DEG_S)[0]
    negative = find_peaks(-yaw_velocity, height=SACCADE_THRESHOLD_DEG_S)[0]
    return positive, negative


def plot_episode_yaw(
    output_stem: Path,
    *,
    method: str,
    episode: int,
    time_s: np.ndarray,
    yaw_velocity: np.ndarray,
) -> Path:
    """Plot the episode yaw velocity and detected turn peaks."""
    positive_peaks, negative_peaks = _detect_turn_peaks(yaw_velocity)

    fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
    ax.plot(time_s, yaw_velocity, label="Policy yaw")
    for peaks, sign in ((positive_peaks, 1), (negative_peaks, -1)):
        ax.axhline(sign * SACCADE_THRESHOLD_DEG_S, color="gray", linestyle="--")
        ax.scatter(time_s[peaks], yaw_velocity[peaks], s=18)
    ax.set(xlabel="Time (s)", ylabel="Yaw (°/s)", title=f"{method}, episode {episode}")
    ax.legend()
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def write_yaw_plots(
    out_dir: Path,
    method: str,
    yaw_traces: list[np.ndarray],
) -> list[Path]:
    paths: list[Path] = []
    for ep, yaw in enumerate(yaw_traces):
        time_s = (np.arange(yaw.size) + 1) / BEHAVIOR_TRACKING_HZ
        paths.append(
            plot_episode_yaw(
                out_dir / f"{method}_ep{ep:02d}_yaw",
                method=method,
                episode=ep,
                time_s=time_s,
                yaw_velocity=yaw,
            )
        )
        print(f"  wrote {paths[-1]}")
    return paths


def _aggregate(per_ep):
    action_labels = [spec.name for spec in discrete_action_specs()]
    agg = {k: float(np.mean([e[k] for e in per_ep])) for k in per_ep[0] if k != "action_fraction"}
    agg["action_fraction"] = {
        label: float(np.mean([e["action_fraction"][label] for e in per_ep]))
        for label in action_labels
    }
    return agg


def _load_policy(
    checkpoint: Path,
    run_dir: Path,
    vecnormalize_path: Path | None,
    *,
    device: str,
    num_envs: int,
):
    model = load_policy(checkpoint, device)
    if model.action_space.n != len(discrete_action_specs()):
        raise ValueError("policy action count does not match the ball action map")
    skill_sampler = load_policy_skill_sampler(run_dir, model.observation_space)
    normalizer = (
        None
        if vecnormalize_path is None
        else load_frozen_vecnormalize(
            vecnormalize_path,
            model.observation_space,
            model.action_space,
        )
    )
    observation_adapter = PolicyObservationAdapter(
        model.observation_space,
        normalizer=normalizer,
        skill_sampler=skill_sampler,
        num_envs=num_envs,
    )
    return model, observation_adapter, normalizer


def _episode_result(
    episode: int,
    actions: np.ndarray,
    skills: np.ndarray,
    skill_boundaries: np.ndarray,
    angular_velocity: np.ndarray,
    activations: dict[str, np.ndarray] | None,
    observations: dict[str, np.ndarray] | None,
    *,
    ball_radius: float,
) -> dict:
    forward, rotational, forward_smoothed, rotational_smoothed = process_velocity(
        angular_velocity,
        ball_radius,
    )
    summary = summarize(actions, forward_smoothed, rotational_smoothed)
    return {
        "ep": episode,
        "summary": summary,
        "actions": actions,
        "skills": skills,
        "skill_boundaries": skill_boundaries,
        "activations": activations,
        "observations": observations,
        "angular_velocity": angular_velocity,
        "forward": forward,
        "rotational": rotational,
        "forward_smoothed": forward_smoothed,
        "rotational_smoothed": rotational_smoothed,
        "action_set": ACTION_SET,
    }


def _print_episode(result: dict) -> None:
    summary = result["summary"]
    print(
        f"  ep{result['ep']}: actions={summary['action_fraction']} "
        f"forward={summary['mean_forward_velocity_mm_s']:+.2f}mm/s "
        f"|rotation|={summary['mean_abs_rotational_velocity_deg_s']:.0f}deg/s"
    )


def _unpack_results(results: list[dict]):
    return {key: [result[key] for result in results] for key in results[0]}


def write_observation_hdf5(
    path: str | Path,
    *,
    method: str,
    episode_observations: Sequence[Mapping[str, np.ndarray]],
    actions: Sequence[np.ndarray],
    skills: Sequence[np.ndarray],
    skill_boundaries: Sequence[np.ndarray],
    angular_velocity: Sequence[np.ndarray],
    episode_seeds: np.ndarray,
    decision_dt: float,
    physics_timestep: float,
    decision_steps: int,
    contact_preset: str,
    spawn_z: float,
    ball_radius_mm: float,
    backend: str,
    settle_steps: int,
    checkpoint: str | Path,
) -> Path:
    """Write raw pre-decision proprioception aligned to actions and ball motion."""
    output_path = Path(path)
    n_episodes = len(episode_observations)
    if n_episodes < 1:
        raise ValueError("episode_observations must be non-empty")
    proprio = np.stack(
        [np.asarray(obs["proprioception"], dtype=np.float32) for obs in episode_observations],
        axis=0,
    )
    if proprio.ndim != 3:
        raise ValueError(f"proprioception must stack to (E, T, D); got {proprio.shape}")
    n_decisions = int(proprio.shape[1])
    actions_arr = np.stack([np.asarray(a) for a in actions], axis=0)
    skills_arr = np.stack([np.asarray(s) for s in skills], axis=0)
    boundaries_arr = np.stack([np.asarray(b) for b in skill_boundaries], axis=0)
    omega_arr = np.stack([np.asarray(w, dtype=np.float32) for w in angular_velocity], axis=0)
    seeds = np.asarray(episode_seeds, dtype=np.int64)
    if actions_arr.shape != (n_episodes, n_decisions):
        raise ValueError(f"actions shape {actions_arr.shape} != ({n_episodes}, {n_decisions})")
    if seeds.shape != (n_episodes,):
        raise ValueError(f"episode_seeds must have shape {(n_episodes,)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["method"] = method
        handle.attrs["backend"] = backend
        handle.attrs["settle_steps"] = int(settle_steps)
        handle.attrs["action_set"] = ACTION_SET
        handle.attrs["contact_preset"] = contact_preset
        handle.attrs["spawn_z"] = float(spawn_z)
        handle.attrs["ball_radius_mm"] = float(ball_radius_mm)
        handle.attrs["physics_timestep"] = float(physics_timestep)
        handle.attrs["decision_steps"] = int(decision_steps)
        handle.attrs["ball_velocity_frame"] = "lab"
        handle.attrs["behavior_tracking_hz"] = float(BEHAVIOR_TRACKING_HZ)
        handle.attrs["vision_zero_by_construction"] = True
        handle.attrs["odor_zero_by_construction"] = True
        handle.attrs["observation_timing"] = "pre_decision"
        handle.attrs["checkpoint"] = str(checkpoint)
        handle.attrs["recorded_live_during_ball_eval"] = True

        handle.create_dataset("episode_seeds", data=seeds)
        handle.create_dataset("actions", data=actions_arr, compression="gzip")
        handle.create_dataset("skills", data=skills_arr, compression="gzip")
        handle.create_dataset("skill_boundaries", data=boundaries_arr, compression="gzip")
        handle.create_dataset(
            "action_time_seconds",
            data=np.arange(n_decisions, dtype=np.float64) * float(decision_dt),
        )
        handle.create_dataset(
            "proprioception",
            data=proprio,
            compression="gzip",
            chunks=(1, min(1000, n_decisions), int(proprio.shape[-1])),
        )
        handle.create_dataset(
            "ball_angular_velocity_rad_s",
            data=omega_arr,
            compression="gzip",
        )
    return output_path


def _run_episodes(
    *,
    checkpoint: Path,
    run_dir: Path,
    vecnormalize_path: Path | None,
    args: argparse.Namespace,
    n_decisions: int,
    substeps: int,
):
    from envs.tethered_ball.warp import WarpBallRunner

    num_envs = min(args.eval_env_num or 10, args.episodes) if args.backend == "warp" else 1
    model, observation_adapter, normalizer = _load_policy(
        checkpoint,
        run_dir,
        vecnormalize_path,
        device=args.device,
        num_envs=num_envs,
    )
    results = []
    activation_recorder = None
    print(f"[ball-eval] backend={args.backend}, batch size={num_envs}")
    try:
        if args.save_activations:
            activation_recorder = RecurrentActivationRecorder(model)
        runner = None
        try:
            for start in range(0, args.episodes, num_envs):
                if args.backend == "warp":
                    if runner is None:
                        runner = WarpBallRunner(num_envs)
                    runner.reset([start + row for row in range(num_envs)])
                else:
                    if runner is not None:
                        runner.close()
                    runner = BallRunner(seed=start)
                runner.settle(500)
                (
                    actions,
                    angular_velocity,
                    skills,
                    skill_boundaries,
                    batch_observations,
                ) = rollout_batch(
                    model,
                    runner,
                    n_decisions,
                    substeps,
                    args.deterministic,
                    observation_adapter,
                    activation_recorder,
                )
                batch_activations = (
                    None if activation_recorder is None else activation_recorder.take_activations()
                )
                valid_rows = min(num_envs, args.episodes - start)
                for row in range(valid_rows):
                    activations = (
                        None
                        if batch_activations is None
                        else {name: values[row] for name, values in batch_activations.items()}
                    )
                    observations = (
                        None
                        if batch_observations is None
                        else {name: values[row] for name, values in batch_observations.items()}
                    )
                    result = _episode_result(
                        start + row,
                        actions[row],
                        skills[row],
                        skill_boundaries[row],
                        angular_velocity[row],
                        activations,
                        observations,
                        ball_radius=BALL_RADIUS_MM,
                    )
                    results.append(result)
                    _print_episode(result)
        finally:
            if runner is not None:
                runner.close()
    finally:
        if activation_recorder is not None:
            activation_recorder.close()
        if normalizer is not None:
            normalizer.close()
    return _unpack_results(results)


def _validate_args(args) -> None:
    if args.episodes < 1:
        raise SystemExit("[ball-eval] --episodes must be >= 1")
    if args.eval_env_num is not None and args.eval_env_num < 1:
        raise SystemExit("[ball-eval] --eval-env-num must be >= 1")
    if args.backend == "cpu" and args.eval_env_num not in {None, 1}:
        raise SystemExit("[ball-eval] CPU evaluation requires --eval-env-num=1")


def _resolve_rollout_budget(args, substeps: int) -> tuple[float, int]:
    decision_dt = substeps * DEFAULT_TIMING.timestep
    n_decisions = (
        args.decisions if args.decisions is not None else round(args.duration_seconds / decision_dt)
    )
    if n_decisions < 1:
        raise SystemExit("[ball-eval] rollout must contain at least one decision")
    if args.save_activations and (
        not np.isfinite(args.activation_pooling_interval_seconds)
        or args.activation_pooling_interval_seconds < decision_dt
    ):
        raise SystemExit(
            "[ball-eval] --activation-pooling-interval-seconds must be finite and "
            "at least one policy decision interval"
        )
    tracking_interval_steps = round(1 / (BEHAVIOR_TRACKING_HZ * DEFAULT_TIMING.timestep))
    minimum_behavior_samples = round(BEHAVIOR_TRACKING_HZ * VELOCITY_FILTER_WINDOW_SECONDS)
    behavior_samples = n_decisions * substeps // tracking_interval_steps
    if behavior_samples < minimum_behavior_samples:
        minimum_decisions = (
            minimum_behavior_samples * tracking_interval_steps + substeps - 1
        ) // substeps
        raise SystemExit(
            "[ball-eval] rollout must produce at least "
            f"{minimum_behavior_samples} behavior samples for velocity filtering; "
            f"use --decisions >= {minimum_decisions}"
        )
    return decision_dt, n_decisions


def _save_activation_h5(
    checkpoint,
    *,
    args,
    n_decisions: int,
    decision_dt: float,
    substeps: int,
    results,
) -> list[Path]:
    episode_activations = [a for a in results["activations"] if a is not None]
    if len(episode_activations) != args.episodes:
        raise RuntimeError("activation recording did not return every episode")
    episode_observations = [o for o in results["observations"] if o is not None]
    if len(episode_observations) != args.episodes:
        raise RuntimeError("observation recording did not return every episode")
    activation_path = args.out_dir / "activations.h5"
    write_activation_hdf5(
        activation_path,
        episode_activations,
        np.arange(n_decisions) * decision_dt,
        np.arange(args.episodes),
        pool_interval_seconds=args.activation_pooling_interval_seconds,
        dtype=args.activation_dtype,
        metadata={
            "method": args.run_dir.name,
            "checkpoint": str(checkpoint),
            "decision_dt_seconds": decision_dt,
            "physics_backend": args.backend,
            "action_set": ACTION_SET,
        },
    )
    print(f"  wrote {activation_path}")
    observation_path = args.out_dir / "observations.h5"
    write_observation_hdf5(
        observation_path,
        method=args.run_dir.name,
        episode_observations=episode_observations,
        actions=results["actions"],
        skills=results["skills"],
        skill_boundaries=results["skill_boundaries"],
        angular_velocity=results["angular_velocity"],
        episode_seeds=np.arange(args.episodes),
        decision_dt=decision_dt,
        physics_timestep=DEFAULT_TIMING.timestep,
        decision_steps=substeps,
        contact_preset="tibia_tarsus_only",
        spawn_z=0.9,
        ball_radius_mm=BALL_RADIUS_MM,
        backend=args.backend,
        settle_steps=500,
        checkpoint=checkpoint,
    )
    print(
        f"  wrote {observation_path} "
        f"({observation_path.stat().st_size / 1e6:.1f} MB compressed; "
        "raw pre-decision proprioception)"
    )
    return [activation_path, observation_path]


def main(argv: list[str] | None = None):
    from omegaconf import OmegaConf

    args = parse_args(argv)
    _validate_args(args)
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint, stats = resolve_checkpoint_paths(
        run_dir, args.model_path, vecnormalize_required=True
    )
    config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    timing = resolved_env_timing_from_ssrl_envs(config.envs)
    if not np.isclose(timing.timestep, DEFAULT_TIMING.timestep):
        raise ValueError("training and ball evaluation physics timesteps must match")
    substeps = args.decision_steps if args.decision_steps is not None else timing.decision_steps
    if substeps < 1:
        raise ValueError("decision-steps must be positive")
    decision_dt, n_decisions = _resolve_rollout_budget(args, substeps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = _run_episodes(
        checkpoint=checkpoint,
        run_dir=run_dir,
        vecnormalize_path=stats,
        args=args,
        n_decisions=n_decisions,
        substeps=substeps,
    )
    summary = _aggregate(results["summary"])
    np.savez(
        args.out_dir / "rollouts.npz",
        **{
            key: np.asarray(results[key])
            for key in (
                "actions",
                "skills",
                "skill_boundaries",
                "angular_velocity",
                "forward",
                "rotational",
                "forward_smoothed",
                "rotational_smoothed",
            )
        },
        action_time_seconds=np.arange(n_decisions) * decision_dt,
        behavior_time_seconds=(np.arange(results["angular_velocity"][0].shape[0]) + 1)
        / BEHAVIOR_TRACKING_HZ,
        episode_seeds=np.arange(args.episodes),
        decision_steps=substeps,
        physics_timestep=DEFAULT_TIMING.timestep,
        physics_backend=args.backend,
        ball_radius_mm=BALL_RADIUS_MM,
        behavior_tracking_hz=BEHAVIOR_TRACKING_HZ,
    )
    if args.save_activations:
        _save_activation_h5(
            checkpoint,
            args=args,
            n_decisions=n_decisions,
            decision_dt=decision_dt,
            substeps=substeps,
            results=results,
        )
    if not args.no_plots:
        write_yaw_plots(args.out_dir, run_dir.name, results["rotational_smoothed"])
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[ball-eval] wrote {summary_path}")


if __name__ == "__main__":
    main()

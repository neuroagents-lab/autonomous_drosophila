"""Train a FlyGym policy with sb3-contrib RecurrentPPO, optionally on an IDM reward.

A run writes to ``config.logdir``: ``latest_model.zip`` and ``vecnormalize.pkl``
(plus ``idm_state.pkl`` for IDM runs) at every rollout start and when training
ends, and ``best_model/`` and ``best_success_model/`` with the best evaluated
policy (``best_model.zip``) and its normalization statistics. With
``algo.resume=true`` an existing ``latest_model.zip`` is continued.
"""

from __future__ import annotations

import pathlib

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

MODEL_FILE = "latest_model.zip"
BEST_MODEL_FILE = "best_model.zip"
VECNORMALIZE_FILE = "vecnormalize.pkl"
IDM_FILE = "idm_state.pkl"
# Observation components normalized by VecNormalize (vision is already in [0, 1]).
_NORMALIZED_OBSERVATIONS = ("odor", "proprioception")


def run(config: DictConfig) -> None:
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    resume = bool(config.algo.resume) and (logdir / MODEL_FILE).is_file()
    resume_dir = logdir if resume else None
    train_env = eval_env = wandb_run = None
    try:
        train_env, skill_sampler = make_vec_env(config, "train", resume_dir=resume_dir)
        if config.trainer.eval_episode_num > 0:
            eval_env, _ = make_vec_env(config, "eval")
        wandb_run = _init_wandb(config, logdir)
        model = _make_model(config, train_env, resume_dir)
        idm_manager = _make_idm_manager(config, model, skill_sampler, resume_dir)
        callback = _make_callbacks(config, logdir, model, eval_env, idm_manager, resume)
        model.learn(
            total_timesteps=max(int(config.trainer.steps) - model.num_timesteps, 0),
            callback=callback,
            reset_num_timesteps=not resume,
        )
        save_checkpoint(logdir, model, idm_manager)
    finally:
        for env in (train_env, eval_env):
            if env is not None:
                env.close()
        if wandb_run is not None:
            wandb_run.finish()


def make_vec_env(config, mode: str, *, resume_dir: pathlib.Path | None = None):
    """FlyGym vec env for ``mode`` "train" or "eval", wrapped as PPO trains on it.

    Returns ``(vec_env, skill_sampler)``; the sampler is None unless the IDM
    is skill-conditioned.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    envs, training = config.envs, mode == "train"
    eval_episodes = int(config.trainer.eval_episode_num)
    # Evaluation layouts use their own seed stream, shared by all evaluated policies.
    layout_seed = None if training else int(envs.eval_seed_offset) + int(config.seed)
    if envs.backend == "warp":
        from envs.warp import make_flygym_warp_vec_env

        num_envs = int(envs.env_num) if training else min(int(envs.eval_env_num), eval_episodes)
        venv = make_flygym_warp_vec_env(envs, num_envs=num_envs, layout_seed_base=layout_seed)
    else:
        from envs.training import make_flygym_navigation_env

        num_envs = int(envs.env_num) if training else 1

        def make_env(index):
            seed = int(envs.seed) + index
            return lambda: make_flygym_navigation_env(
                envs,
                seed=seed,
                worker_rank=index,
                layout_seed_base=int(envs.seed) if training else layout_seed,
                layout_num_envs=num_envs,
            )

        venv = DummyVecEnv([make_env(index) for index in range(num_envs)])
    venv = VecMonitor(venv)

    uses_idm = config.idm.type != "none"
    if uses_idm:
        from policy.idm_rollout import OriginalTerminalObservations

        venv = OriginalTerminalObservations(venv)
    if resume_dir is not None:
        venv = VecNormalize.load(str(resume_dir / VECNORMALIZE_FILE), venv)
    else:
        keys = [key for key in venv.observation_space.spaces if key in _NORMALIZED_OBSERVATIONS]
        venv = VecNormalize(venv, norm_obs=bool(keys), norm_obs_keys=keys)
    venv.training = training
    # IDM runs normalize the intrinsic reward in the IDM manager instead.
    venv.norm_reward = training and bool(config.algo.norm_reward) and not uses_idm
    return add_skill_observations(config, venv)


def add_skill_observations(config, vec_env):
    """Add the sampled IDM skill to the observations of skill-conditioned runs.

    Returns ``(vec_env, skill_sampler)``, unchanged and None for other runs.
    Evaluation scripts use this to roll out skill-conditioned policies.
    """
    if config.idm.type == "none":
        return vec_env, None
    from policy.idm import SKILL_IDMS, PPOSkillSampler

    if config.idm.type not in SKILL_IDMS:
        return vec_env, None
    from policy.idm_rollout import IDMSkillVecWrapper

    sampler = PPOSkillSampler.from_hydra_config(config)
    return IDMSkillVecWrapper(vec_env, sampler), sampler


def load_policy(path, device: str = "auto"):
    """Load a policy saved by ``run``."""
    from sb3_contrib import RecurrentPPO

    return RecurrentPPO.load(str(path), device=device)


def resolve_checkpoint_paths(
    run_dir,
    model_path=None,
    vecnormalize_path=None,
    *,
    vecnormalize_required: bool = False,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Return the model zip in ``run_dir`` (or ``model_path``) and its VecNormalize stats.

    ``run_dir`` is a run directory (``latest_model.zip``) or one of its
    ``best_model``/``best_success_model`` directories (``best_model.zip``).
    The statistics are the ``vecnormalize.pkl`` saved next to the model.
    """
    if model_path is None:
        candidates = [pathlib.Path(run_dir) / name for name in (MODEL_FILE, BEST_MODEL_FILE)]
        model_path = next((path for path in candidates if path.is_file()), candidates[0])
    model_path = pathlib.Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")
    stats = pathlib.Path(vecnormalize_path or model_path.parent / VECNORMALIZE_FILE)
    if not stats.is_file():
        if vecnormalize_required or vecnormalize_path is not None:
            raise FileNotFoundError(f"VecNormalize statistics not found: {stats}")
        stats = None
    return model_path, stats


def save_checkpoint(directory, model, idm_manager=None, *, model_file: str = MODEL_FILE) -> None:
    """Save the model, its VecNormalize statistics and the IDM state, each atomically."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for path, save in (
        (directory / model_file, model.save),
        (directory / VECNORMALIZE_FILE, model.get_vec_normalize_env().save),
    ):
        tmp_path = path.with_suffix(f".tmp{path.suffix}")
        save(str(tmp_path))
        tmp_path.replace(path)
    if idm_manager is not None:
        idm_manager.save(directory / IDM_FILE)


def _make_model(config, env, resume_dir):
    from sb3_contrib import RecurrentPPO

    from policy.multimodal import FlyGymMultimodalExtractor

    if resume_dir is not None:
        return RecurrentPPO.load(str(resume_dir / MODEL_FILE), env=env, device=config.device)
    kwargs = OmegaConf.to_container(config.algo, resolve=True)
    del kwargs["norm_reward"], kwargs["resume"]
    kwargs["policy_kwargs"].update(
        features_extractor_class=FlyGymMultimodalExtractor, normalize_images=False
    )
    return RecurrentPPO(
        "MultiInputLstmPolicy", env, seed=int(config.seed), device=config.device, **kwargs
    )


def _make_idm_manager(config, model, skill_sampler, resume_dir):
    if config.idm.type == "none":
        return None
    from policy.idm_manager import JaxIDMManager

    if not isinstance(model.action_space, gym.spaces.Discrete):
        raise ValueError(f"PPO IDMs require a Discrete action space, got {model.action_space}")
    manager = JaxIDMManager(
        config,
        obs_dim=model.policy.pi_features_extractor.base_features_dim,
        n_actions=int(model.action_space.n),
        skill_sampler=skill_sampler,
    )
    if resume_dir is not None:
        manager.load(resume_dir / IDM_FILE)
    return manager


def _make_callbacks(config, logdir, model, eval_env, idm_manager, resume):
    from stable_baselines3.common.callbacks import CallbackList

    from policy.callbacks import (
        NavigationEvalCallback,
        PPOFlyGymWandbCallback,
        SaveAtRolloutStartCallback,
    )

    # Save first, so a checkpoint never holds a half-started rollout.
    callbacks = [SaveAtRolloutStartCallback(lambda m: save_checkpoint(logdir, m, idm_manager))]
    if idm_manager is not None:
        from policy.idm_rollout import IDMRolloutCallback

        callbacks.append(IDMRolloutCallback(idm_manager))
    eval_callback = None
    if eval_env is not None:
        eval_callback = NavigationEvalCallback(
            eval_env,
            save_best=lambda m, name: save_checkpoint(logdir / name, m, model_file=BEST_MODEL_FILE),
            n_eval_episodes=int(config.trainer.eval_episode_num),
            eval_every_rollouts=max(int(config.trainer.eval_every), 1),
            log_path=logdir / "evaluations.npz",
            completed_rollouts=model.num_timesteps // (model.n_steps * model.n_envs),
            resume=resume,
        )
    if config.wandb.enabled:
        # W&B must run before eval: NavigationEvalCallback calls logger.dump().
        callbacks.append(PPOFlyGymWandbCallback(eval_callback=eval_callback))
    if eval_callback is not None:
        callbacks.append(eval_callback)
    return CallbackList(callbacks)


def _init_wandb(config, logdir):
    if not config.wandb.enabled:
        return None
    import wandb

    kwargs = OmegaConf.to_container(config.wandb, resolve=True)
    del kwargs["enabled"]
    run = wandb.init(**kwargs, dir=str(logdir), config=OmegaConf.to_container(config, resolve=True))
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    return run

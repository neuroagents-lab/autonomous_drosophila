"""Parallel training worlds sample different layouts from one seed."""

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from envs.layouts import layout_episode_seed, sample_episode_layout
from envs.training import make_flygym_train_config
from policy.ppo import make_vec_env

ROOT = Path(__file__).resolve().parents[1]


def _config():
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="configs",
            overrides=[
                "seed=0",
                "envs.backend=cpu",
                "envs.env_num=2",
                "envs.eval_episode_num=0",
                "envs.decision_dt=0.0001",
                "envs.episode_time_seconds=1",
                "envs.observation_components=[odor]",
            ],
        )


def _targets(vec_env):
    while hasattr(vec_env, "venv"):
        vec_env = vec_env.venv
    return [info["target_position"] for info in vec_env.reset_infos]


def test_two_worlds_do_not_share_a_layout():
    config = _config()
    train_config = make_flygym_train_config(config.envs)
    vec_env, _ = make_vec_env(config, "train")
    try:
        for episode in range(2):
            vec_env.reset()
            targets = _targets(vec_env)
            assert not np.allclose(targets[0], targets[1])
            for world, target in enumerate(targets):
                seed = layout_episode_seed(int(config.seed), world, episode, 2)
                layout = sample_episode_layout(train_config, np.random.default_rng(seed))
                np.testing.assert_allclose(target, layout.odor_source[0])
    finally:
        vec_env.close()

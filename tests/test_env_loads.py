"""Each configs/envs/*.yaml places the fly, and odor eval keeps a saved run's arena."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from envs.training import make_flygym_navigation_env, make_flygym_train_config
from scripts.eval_odor import load_run_config, make_eval_vec_env, reset_infos

ROOT = Path(__file__).resolve().parents[1]
ENV_NAMES = sorted(path.stem for path in (ROOT / "configs" / "envs").glob("*.yaml"))
FORWARD = 1
GROOM = 4


def hydra_config(*overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="configs", overrides=list(overrides))


def assert_fly_landed(env, info):
    """Root pose is the sampled spawn, and the thorax is above the floor."""
    layout = env.sim.layout
    root = env.sim.sim.mj_data.qpos[env.sim._root_qpos()]
    np.testing.assert_allclose(root[:3], layout.spawn_position, atol=1e-5)
    position = np.asarray(info["fly_position"], np.float32)
    assert np.isfinite(position).all()
    assert float(position[2]) > env.train_config.min_height


@pytest.mark.parametrize("name", ENV_NAMES)
def test_environment_is_selectable_and_fly_lands(name):
    # One physics step per decision. Arena and observations stay the yaml's.
    config = hydra_config(
        f"envs={name}",
        "envs.backend=cpu",
        "envs.decision_dt=0.0001",
        "envs.episode_time_seconds=1",
    )
    train_config = make_flygym_train_config(config.envs)
    env = make_flygym_navigation_env(config.envs, seed=0)
    try:
        observation, info = env.reset(seed=0)
        assert set(observation) == set(train_config.base_env_config.observation_components)
        assert env.observation_space.contains(observation)
        assert_fly_landed(env, info)
        for action in (FORWARD, GROOM):
            observation, reward, _, _, info = env.step(action)
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)
            assert info["failure_reason"] is None
            assert np.isfinite(info["fly_position"]).all()
    finally:
        env.close()


def test_odor_eval_uses_the_saved_run_config(tmp_path):
    config = hydra_config(
        "envs=flygym_odor_navigation",
        "envs.backend=cpu",
        "envs.decision_dt=0.0001",
        "envs.episode_time_seconds=1",
        "envs.progress_reward_weight=0.7",
        "envs.observation_components=[odor,proprioception]",
    )
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    OmegaConf.save(config, hydra_dir / "config.yaml", resolve=True)

    run_config = load_run_config(tmp_path)
    env_config = deepcopy(run_config.envs)
    train_config = make_flygym_train_config(env_config)
    assert train_config.progress_reward_weight == pytest.approx(0.7)
    assert train_config.base_env_config.observation_components == ("odor", "proprioception")

    vec_env = make_eval_vec_env(
        env_config,
        train_config,
        backend="cpu",
        num_envs=1,
        seed=0,
        render_mode=None,
    )
    try:
        vec_env.reset()
        env = vec_env.envs[0]
        info = reset_infos(vec_env)[0]
        assert set(env.observation_space.spaces) == {"odor", "proprioception"}
        assert_fly_landed(env, info)
    finally:
        vec_env.close()

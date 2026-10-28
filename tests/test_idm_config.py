"""Each configs/idm/*.yaml builds its ssrl module and returns one finite reward."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from policy.idm import IDM_CLASSES, SKILL_IDMS, PPOSkillSampler
from policy.idm_manager import JaxIDMManager

ROOT = Path(__file__).resolve().parents[1]
IDM_NAMES = sorted(path.stem for path in (ROOT / "configs" / "idm").glob("*.yaml"))
N_ENVS = 2
OBS_DIM = 8
N_ACTIONS = 7


def hydra_config(*overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="configs", overrides=list(overrides))


@pytest.mark.parametrize("name", IDM_NAMES)
def test_idm_config_loads_and_scores(name):
    import jax

    config = hydra_config(f"idm={name}")
    assert config.idm.type == name
    assert name in IDM_CLASSES

    sampler = PPOSkillSampler.from_hydra_config(config) if name in SKILL_IDMS else None
    if sampler is not None:
        kwargs = OmegaConf.to_container(config.idm.kwargs, resolve=True)
        skills = sampler.current_skills(N_ENVS)
        expected = kwargs["num_skills"] if kwargs["skill_dim"] == 0 else kwargs["skill_dim"]
        assert skills.shape == (N_ENVS, expected)
        assert expected == sampler.skill_event_dim

    manager = JaxIDMManager(
        config, obs_dim=OBS_DIM, n_actions=N_ACTIONS, skill_sampler=sampler
    )
    module = manager.module
    assert type(module).__name__ == IDM_CLASSES[name]
    for key, value in OmegaConf.to_container(config.idm.kwargs, resolve=True).items():
        assert getattr(module, key) == value, key

    obs = np.zeros((N_ENVS, OBS_DIM), np.float32)
    actions = np.zeros(N_ENVS, np.int64)
    raw = {"x": obs}
    skills = None if sampler is None else sampler.current_skills(N_ENVS)
    if name == "ctec":
        reward = manager.ctec_rollout_reward(
            obs[:, None],
            actions[:, None],
            obs[:, None],
            np.ones((N_ENVS, 1), dtype=bool),
            np.ones((N_ENVS, 1), dtype=bool),
        )
        assert reward.shape == (N_ENVS, 1)
    else:
        reward = manager.intrinsic_reward(
            obs,
            actions,
            obs,
            episode_starts=np.ones(N_ENVS, dtype=bool),
            dones=np.ones(N_ENVS, dtype=bool),
            raw_obs=raw,
            raw_next_obs=raw,
            skills=skills,
            skill_ends=np.ones(N_ENVS, dtype=bool),
            action_probabilities=np.full((N_ENVS, N_ACTIONS), 1 / N_ACTIONS, np.float32),
        )
        assert reward.shape == (N_ENVS,)
    assert np.isfinite(reward).all()
    jax.clear_caches()

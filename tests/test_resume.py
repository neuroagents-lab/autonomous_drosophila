"""A second launch keeps the checkpoint instead of training a new run."""

import pickle
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from policy import ppo

ROOT = Path(__file__).resolve().parents[1]


def _config(logdir: Path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="configs",
            overrides=[
                f"logdir={logdir}",
                "seed=0",
                "device=cpu",
                "wandb.enabled=false",
                "algo.resume=true",
                "envs.backend=cpu",
                "envs.env_num=1",
                "envs.eval_episode_num=0",
                "envs.steps=2",
                "envs.decision_dt=0.0001",
                "envs.episode_time_seconds=0.0002",
                "envs.observation_components=[odor,proprioception]",
                "algo.n_steps=2",
                "algo.batch_size=2",
                "algo.n_epochs=1",
                "algo.policy_kwargs.lstm_hidden_size=8",
                "algo.verbose=0",
                "idm=rnd",
                "idm.replay_batch_size=2",
            ],
        )


def _weights(path: Path):
    model = ppo.load_policy(path, device="cpu")
    return {key: value.detach().cpu() for key, value in model.policy.state_dict().items()}


def _idm_progress(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["updates"], payload["rng"]


def test_preempted_run_resumes(tmp_path):
    config = _config(tmp_path)
    ppo.run(config)
    checkpoint = tmp_path / ppo.MODEL_FILE
    assert ppo.load_policy(checkpoint, device="cpu").num_timesteps == 2
    weights = _weights(checkpoint)
    idm = _idm_progress(tmp_path / ppo.IDM_FILE)

    config.seed = 1
    ppo.run(config)
    resumed = ppo.load_policy(checkpoint, device="cpu")
    assert resumed.num_timesteps == 2
    for key, value in _weights(checkpoint).items():
        assert torch.equal(value, weights[key]), key
    assert _idm_progress(tmp_path / ppo.IDM_FILE) == idm

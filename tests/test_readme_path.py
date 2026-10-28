"""A few training steps leave a checkpoint that odor eval and ball eval can read."""

from pathlib import Path

import h5py
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from policy import ppo
from scripts import eval_ball, eval_odor

ROOT = Path(__file__).resolve().parents[1]


def test_short_train_then_odor_and_ball_eval(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(
            config_name="configs",
            overrides=[
                f"logdir={tmp_path}",
                "device=cpu",
                "wandb.enabled=false",
                "envs.backend=cpu",
                "envs.env_num=1",
                "envs.eval_env_num=1",
                "envs.eval_episode_num=0",
                "envs.steps=2",
                "envs.decision_dt=0.0001",
                "envs.episode_time_seconds=0.0002",
                "envs.observation_components=[odor,proprioception]",
                "algo.n_steps=2",
                "algo.batch_size=2",
                "algo.n_epochs=1",
                "algo.verbose=0",
                "idm=diayn",
                "idm.replay_batch_size=2",
            ],
        )
    ppo.run(config)
    assert (tmp_path / ppo.MODEL_FILE).is_file()
    assert (tmp_path / ppo.VECNORMALIZE_FILE).is_file()
    assert (tmp_path / ppo.IDM_FILE).is_file()

    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    OmegaConf.save(config, hydra_dir / "config.yaml", resolve=True)

    assert (
        eval_odor.main(
            [
                "--run-dir",
                str(tmp_path),
                "--episodes",
                "1",
                "--backend",
                "cpu",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    out_dir = tmp_path / "ball"
    hidden = int(config.algo.policy_kwargs.lstm_hidden_size)
    eval_ball.main(
        [
            "--run-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--episodes",
            "1",
            "--duration-seconds",
            "0.5",
            "--decision-steps",
            "1000",
            "--backend",
            "cpu",
            "--device",
            "cpu",
            "--save-activations",
            "--deterministic",
        ]
    )
    with (
        h5py.File(out_dir / "activations.h5") as activations,
        h5py.File(out_dir / "observations.h5") as observations,
    ):
        assert activations["raw/actor_lstm_hidden"].shape == (1, 5, hidden)
        assert np.isfinite(activations["raw/value"][:]).all()
        assert observations["proprioception"].shape[:2] == (1, 5)
        assert observations["ball_angular_velocity_rad_s"].shape == (1, 25, 3)
    assert (out_dir / "summary.json").is_file()
    assert list(out_dir.glob("*_yaw.png"))

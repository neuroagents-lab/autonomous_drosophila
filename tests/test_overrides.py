"""Command-line overrides are the experiment written into the run config."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from scripts.eval_odor import load_run_config

ROOT = Path(__file__).resolve().parents[1]


def test_command_line_overrides_are_saved_for_eval(tmp_path):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(
            config_name="configs",
            overrides=[
                "envs=flygym_odor_navigation",
                "envs.env_num=7",
                "envs.steps=11",
                "idm=rnd",
                "idm.intrinsic_scale=0.25",
                "algo.learning_rate=1.5e-4",
            ],
        )
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    OmegaConf.save(config, hydra_dir / "config.yaml", resolve=True)

    saved = load_run_config(tmp_path)
    assert int(saved.envs.env_num) == 7
    assert int(saved.envs.steps) == 11
    assert int(saved.trainer.steps) == 11
    assert float(saved.idm.intrinsic_scale) == 0.25
    assert float(saved.algo.learning_rate) == 1.5e-4
    assert saved.idm.type == "rnd"

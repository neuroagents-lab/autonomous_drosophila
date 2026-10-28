"""Train a FlyGym policy.

Hydra loads ``configs/configs.yaml``. Choose the environment and IDM with
config groups, and override any field on the command line::

    python scripts/train.py envs=flygym_odor_navigation
    python scripts/train.py envs=flygym_intrinsic_exploration idm=diayn seed=1
    python scripts/train.py envs=flygym_intrinsic_exploration idm=rnd \\
        algo.learning_rate=1e-4 envs.env_num=256 wandb.enabled=false

``idm=<name>`` loads ``configs/idm/<name>.yaml`` (hyperparameters only).
The IDM implementation is imported from the ``ssrl.idms`` package.
"""

from __future__ import annotations

import hydra

from policy.ppo import run


@hydra.main(version_base=None, config_path="../configs", config_name="configs")
def main(config) -> None:
    run(config)


if __name__ == "__main__":
    main()

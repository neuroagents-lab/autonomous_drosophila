"""PPO training of FlyGym policies, optionally on an intrinsic reward module (IDM).

- ``ppo``: training entry point, env wrapping, checkpoints, policy loading.
- ``multimodal``: the policy's feature extractor.
- ``callbacks``: checkpoint, evaluation and W&B callbacks.
- ``idm``: builds the ``ssrl.idms`` module from ``configs/idm/*.yaml``; skill sampling.
- ``idm_manager``: IDM rewards and updates on PPO rollouts.
- ``idm_rollout``: env wrappers and the SB3 callback that connect a rollout to the IDM.

Only IDM runs import ``ssrl`` (through the ``idm*`` modules).
"""

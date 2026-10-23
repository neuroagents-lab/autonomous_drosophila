"""Build the paper's intrinsic reward modules (IDMs) from ``configs/idm/*.yaml``.

The algorithms live in the ``ssrl.idms`` package. ``config.idm.kwargs`` holds
the constructor arguments of the ssrl class named in ``IDM_CLASSES``; this
module adds the runtime dimensions and samples skills for skill-conditioned IDMs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import OmegaConf

try:
    import ssrl.idms
    from ssrl.idms import SKILL_COMPETENCY_IDMS, get_skill_prior
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"SSRL IDMs are unavailable. Run `uv sync`. Missing import: {exc.name}"
    ) from exc

# idm.type -> ssrl.idms class.
IDM_CLASSES = {
    "apt": "APT",
    "csf": "CSF",
    "ctec": "CTEC",
    "dads": "DADS",
    "diayn": "DIAYN",
    "disagreement": "Disagreement",
    "empowerment_forward": "ForwardEmpowerment",
    "empowerment_gain": "EmpowermentGain",
    "empowerment_potential": "PotentialEmpowerment",
    "empowerment_process": "ProcessEmpowerment",
    "gamma_progress": "GammaLearningProgress",
    "icm_pathak": "ICM",
    "max": "MAX",
    "ngu": "NGU",
    "proto_rl": "ProtoRL",
    "rdads": "RDADS",
    "re3": "RE3",
    "rnd": "RND",
    "valor": "VALOR",
    "vic": "VIC",
}
# IDMs whose reward depends on a skill the policy is conditioned on.
SKILL_IDMS = frozenset(SKILL_COMPETENCY_IDMS)


def idm_kwargs(config) -> dict[str, Any]:
    """``config.idm.kwargs`` as plain Python values."""
    return OmegaConf.to_container(config.idm.kwargs, resolve=True)


def build_idm(config, *, obs_dim: int, n_actions: int):
    """Construct the ssrl IDM for ``obs_dim`` features and ``n_actions`` discrete actions."""
    idm_type = config.idm.type
    if idm_type not in IDM_CLASSES:
        raise ValueError(f"Unknown idm.type {idm_type!r}; expected one of {sorted(IDM_CLASSES)}.")
    cls = getattr(ssrl.idms, IDM_CLASSES[idm_type])
    kwargs = idm_kwargs(config)
    if hasattr(cls, "action_dim"):
        # Action-conditioned IDMs receive the PPO action as a one-hot vector.
        kwargs["action_dim"] = n_actions
    return cls(obs_dim=obs_dim, **kwargs, name=idm_type)


class PPOSkillSampler:
    """Sample one skill per environment and hold it for ``skill_horizon`` decisions.

    ``skill_horizon=0`` holds each skill for the whole episode. Training uses it
    through ``IDMSkillVecWrapper``; evaluation of skill-conditioned policies
    uses it directly.
    """

    def __init__(self, prior, *, skill_horizon: int, seed: int):
        self.prior = prior
        self.skill_event_dim = int(prior.event_dim)
        self.skill_horizon = int(skill_horizon)
        self._rng = np.random.default_rng(seed)
        self._skills: np.ndarray | None = None
        self._steps: np.ndarray | None = None
        self._boundaries: np.ndarray | None = None

    @classmethod
    def from_hydra_config(cls, config) -> PPOSkillSampler:
        if config.idm.type not in SKILL_IDMS:
            raise ValueError(f"idm.type={config.idm.type!r} does not use skills.")
        kwargs = idm_kwargs(config)
        prior = get_skill_prior(
            kwargs["prior_name"],
            num_skills=kwargs["num_skills"],
            skill_dim=kwargs["skill_dim"],
            **(kwargs.get("prior_kwargs") or {}),
        )
        return cls(prior, skill_horizon=config.idm.skill_horizon, seed=config.seed)

    def reset_skills(self, n_envs: int, dones: np.ndarray | None = None) -> None:
        """Advance one decision; resample after ``dones`` or ``skill_horizon`` steps.

        Without ``dones`` (an environment reset), every skill is resampled.
        """
        if self._skills is None or len(self._skills) != n_envs or dones is None:
            self._skills = self._sample(n_envs)
            self._steps = np.zeros(n_envs, dtype=np.int32)
            self._boundaries = np.ones(n_envs, dtype=bool)
            return
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        if dones.shape != (n_envs,):
            raise ValueError(f"Skill sampler dones must have shape {(n_envs,)}, got {dones.shape}.")
        self._steps += 1
        resample = dones.copy()
        if self.skill_horizon > 0:
            resample |= self._steps >= self.skill_horizon
        self._boundaries = resample
        if resample.any():
            self._skills[resample] = self._sample(int(resample.sum()))
            self._steps[resample] = 0

    def current_skills(self, n_envs: int) -> np.ndarray:
        if self._skills is None or len(self._skills) != n_envs:
            self.reset_skills(n_envs)
        return self._skills.copy()

    def skill_boundaries(self, n_envs: int) -> np.ndarray:
        """Whether each environment's skill was (re)sampled at the last step."""
        if self._boundaries is None or len(self._boundaries) != n_envs:
            return np.zeros(n_envs, dtype=bool)
        return self._boundaries.copy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self._rng.bit_generator.state,
            "skills": self._skills,
            "steps": self._steps,
            "boundaries": self._boundaries,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = state["rng"]
        self._skills, self._steps, self._boundaries = (
            state["skills"],
            state["steps"],
            state["boundaries"],
        )

    def _sample(self, count: int) -> np.ndarray:
        prior, shape = self.prior, (count, self.skill_event_dim)
        if prior.is_categorical:
            return np.eye(self.skill_event_dim, dtype=np.float32)[
                self._rng.integers(self.skill_event_dim, size=count)
            ]
        if prior.skill_space == "sphere":
            values = self._rng.normal(size=shape)
            return (values / np.linalg.norm(values, axis=-1, keepdims=True)).astype(np.float32)
        return self._rng.uniform(prior.low, prior.high, size=shape).astype(np.float32)

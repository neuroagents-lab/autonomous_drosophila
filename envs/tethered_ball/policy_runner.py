"""Policy evaluation on the tethered-ball apparatus used by Brezovec et al."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import gymnasium as gym
import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import ActuatorType
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    apply_locomotion_action,
    get_default_locomotion_dof_order,
)
from flygym_demo.complex_terrain.common import LocomotionAction

from envs.action_spaces import discrete_action_specs
from envs.observations import world_vectors_to_body
from envs.tethered_ball.world import (
    BALL_RADIUS_MM,
    TetheredBallWorld,
)
from envs.tethered_ball.yaw_processing import (
    BEHAVIOR_TRACKING_HZ,
)
from envs.training import resolved_env_timing_from_ssrl_envs
from fly_agent.odor.olfaction import add_olfaction
from fly_agent.primitives.groom_replay import (
    GroomReplay,
    GroomReplayConfig,
    build_groom_actuated_indices,
    build_head_actuated_indices,
    make_groom_fly,
)

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecNormalize

    from policy.idm import PPOSkillSampler

DEFAULT_TIMING = resolved_env_timing_from_ssrl_envs(SimpleNamespace())
RECORDING_DURATION_SECONDS = 30 * 60

DEFAULT_GROOM_REPLAY_PATH = Path(__file__).resolve().parents[2] / "data" / "grooming.npz"


class _PolicySpaceEnv(gym.Env):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if isinstance(self.observation_space, gym.spaces.Dict):
            observation = {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.spaces.items()
            }
        else:
            observation = np.zeros(
                self.observation_space.shape,
                dtype=self.observation_space.dtype,
            )
        return observation, {}

    def step(self, action):
        raise RuntimeError("_PolicySpaceEnv only supplies spaces for VecNormalize")


def load_frozen_vecnormalize(
    path: Path,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> VecNormalize:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    base_observation_space = policy_base_observation_space(observation_space)
    vec_env = DummyVecEnv([lambda: _PolicySpaceEnv(base_observation_space, action_space)])
    try:
        normalizer = VecNormalize.load(str(path), vec_env)
    except Exception:
        vec_env.close()
        raise
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def policy_base_observation_space(observation_space: gym.Space) -> gym.Space:
    if not isinstance(observation_space, gym.spaces.Dict):
        return observation_space
    if "skill" not in observation_space.spaces:
        return observation_space
    return gym.spaces.Dict(
        {key: space for key, space in observation_space.spaces.items() if key != "skill"}
    )


def load_policy_skill_sampler(
    run_dir: Path,
    observation_space: gym.Space,
) -> PPOSkillSampler | None:
    if not isinstance(observation_space, gym.spaces.Dict):
        return None
    skill_space = observation_space.spaces.get("skill")
    if skill_space is None:
        return None

    from omegaconf import OmegaConf

    from policy.idm import PPOSkillSampler

    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Skill-conditioned policy requires its saved Hydra config: {config_path}"
        )
    sampler = PPOSkillSampler.from_hydra_config(OmegaConf.load(config_path))
    expected_shape = (sampler.skill_event_dim,)
    if skill_space.shape != expected_shape:
        raise ValueError(
            "Saved skill config and policy observation space disagree: "
            f"{expected_shape} != {skill_space.shape}"
        )
    return sampler


class PolicyObservationAdapter:
    def __init__(
        self,
        observation_space: gym.Space,
        *,
        normalizer: VecNormalize | None = None,
        skill_sampler: PPOSkillSampler | None = None,
        num_envs: int = 1,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("tethered-ball policies require a Dict observation space")
        skill_space = observation_space.spaces.get("skill")
        if skill_space is not None and skill_sampler is None:
            raise ValueError("Skill-conditioned policy requires a skill sampler")
        if skill_space is None and skill_sampler is not None:
            raise ValueError("Policy observation space does not accept skills")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")

        self.base_observation_space = policy_base_observation_space(observation_space)
        self.normalizer = normalizer
        self.skill_sampler = skill_sampler
        self.num_envs = int(num_envs)
        self.skill_event_dim = 0 if skill_space is None else int(skill_space.shape[0])

    def reset(self) -> None:
        if self.skill_sampler is not None:
            self.skill_sampler.reset_skills(self.num_envs)

    def observation_pair(
        self, runner: BallRunner
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return ``(raw_obs, policy_obs)`` for one decision.

        ``raw_obs`` is what the runner emits (pre-VecNormalize). ``policy_obs`` is
        what the SB3 policy receives (normalized + optional skill channel).
        """
        raw = runner.obs(self.base_observation_space)
        policy = raw
        if self.normalizer is not None:
            policy = self.normalizer.normalize_obs(raw)
        if self.skill_sampler is not None:
            policy = dict(policy)
            policy["skill"] = self.skill_sampler.current_skills(self.num_envs)
        return raw, policy

    def advance_batch(self) -> np.ndarray:
        if self.skill_sampler is None:
            return np.zeros(self.num_envs, dtype=bool)
        self.skill_sampler.reset_skills(
            self.num_envs,
            dones=np.zeros(self.num_envs, dtype=bool),
        )
        return self.skill_sampler.skill_boundaries(self.num_envs)


def _policy_observation(
    proprioception: np.ndarray,
    observation_space: gym.spaces.Dict,
) -> dict[str, np.ndarray]:
    if not isinstance(observation_space, gym.spaces.Dict):
        raise TypeError("tethered-ball policies require a Dict observation space")
    observations = {}
    for key, space in observation_space.spaces.items():
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(f"unsupported observation space for {key!r}: {space}")
        if key == "proprioception":
            if space.shape != proprioception.shape[1:]:
                raise ValueError(
                    "proprioception shape mismatch: "
                    f"ball runner has {proprioception.shape[1:]}, "
                    f"policy expects {space.shape}"
                )
            observations[key] = proprioception.astype(space.dtype, copy=False)
        elif key in {"vision", "odor", "fly_vision"}:
            observations[key] = np.zeros(
                (proprioception.shape[0], *space.shape),
                dtype=space.dtype,
            )
        else:
            raise ValueError(f"unsupported policy observation key: {key!r}")
    return observations


class BallRunner:
    def __init__(
        self,
        *,
        ball_top_z: float = 0.15,
        ball_radius: float = BALL_RADIUS_MM,
        contact_preset: ContactBodiesPreset = ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        adhesion_gain: float | dict[str, float] = 40.0,
        joint_stiffness: float = 0.05,
        joint_damping: float = 0.06,
        actuator_gain: float = 45.0,
        actuator_force_limit: float = 65.0,
        spawn_z: float = 0.9,
        seed: int = 0,
        groom_replay_path: str | Path | None = DEFAULT_GROOM_REPLAY_PATH,
    ):
        self.num_envs = 1
        self._build_sim(
            ball_top_z=ball_top_z,
            ball_radius=ball_radius,
            contact_preset=contact_preset,
            adhesion_gain=adhesion_gain,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            actuator_gain=actuator_gain,
            actuator_force_limit=actuator_force_limit,
            spawn_z=spawn_z,
        )
        self._cache_model_addresses()
        self._setup_controller(seed)
        self._setup_groom(groom_replay_path)
        action_specs = discrete_action_specs()
        self.action_labels = {spec.flat_index: spec.name for spec in action_specs}
        default_action_forces = {spec.flat_index: spec.forces for spec in action_specs}
        self._action_forces = default_action_forces
        # The paper's flies walked in complete darkness without odor cues.
        self._zero_vision = np.zeros((1, 6, 32, 32), dtype=np.uint8)
        self._zero_odor = np.zeros((1, 2), dtype=np.float32)
        self._tracking_interval_steps = round(1 / (BEHAVIOR_TRACKING_HZ * self.dt))
        actual_tracking_hz = 1 / (self._tracking_interval_steps * self.dt)
        if not np.isclose(actual_tracking_hz, BEHAVIOR_TRACKING_HZ):
            raise ValueError(f"timestep {self.dt} cannot represent {BEHAVIOR_TRACKING_HZ} Hz")
        self._steps_since_tracking_sample = 0

    def _build_sim(
        self,
        *,
        ball_top_z: float,
        ball_radius: float,
        contact_preset: ContactBodiesPreset,
        adhesion_gain: float | dict[str, float],
        joint_stiffness: float,
        joint_damping: float,
        actuator_gain: float,
        actuator_force_limit: float,
        spawn_z: float,
    ) -> None:
        fly = make_groom_fly(
            colorize=True,
            adhesion_gain=adhesion_gain,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            actuator_gain=actuator_gain,
            actuator_forcerange=(-actuator_force_limit, actuator_force_limit),
        )
        fly.add_vision()
        add_olfaction(fly)
        world = TetheredBallWorld(
            ball_top_z=ball_top_z,
            ball_radius=ball_radius,
            contact_preset=contact_preset,
        )
        world.add_fly(
            fly,
            spawn_position=(0.0, 0.0, spawn_z),
            spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        )
        self.world = world
        self.fly = fly
        self.sim = Simulation(world)
        self.dt = float(self.sim.mj_model.opt.timestep)

    def _cache_model_addresses(self) -> None:
        m = self.sim.mj_model
        self._dof_order = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        qp, qv = [], []
        for jd in self._dof_order:
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, self.fly.jointdof_to_mjcfjoint[jd].name)
            qp.append(int(m.jnt_qposadr[jid]))
            qv.append(int(m.jnt_dofadr[jid]))
        self._qp = np.asarray(qp)
        self._qv = np.asarray(qv)

        leg_dofs = set(get_default_locomotion_dof_order())
        self._leg_indices = [i for i, jd in enumerate(self._dof_order) if jd in leg_dofs]
        pos_actuator_ids = self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][
            self.fly.name
        ]
        self._full_pose = np.asarray(
            self.sim.mj_data.ctrl[pos_actuator_ids], dtype=np.float32
        ).copy()
        root_index = self.fly.get_bodysegs_order().index(self.fly.root_segment)
        self._root_body_id = int(self.sim._internal_bodyids_by_fly[self.fly.name][root_index])

    def _setup_controller(self, seed: int) -> None:
        self._controller_seed = seed
        self.controller = HybridTurningController(
            timestep=self.dt,
            output_dof_order=get_default_locomotion_dof_order(),
            max_correction=0.0,
        )
        self.controller.reset(seed=seed)

    def _setup_groom(
        self,
        groom_replay_path: str | Path | None,
    ) -> None:
        if groom_replay_path is None:
            self._groom_replay = None
            return
        actuated_joint_names = [jd.name for jd in self._dof_order]
        self._groom_replay = GroomReplay(
            GroomReplayConfig(replay_path=Path(groom_replay_path)),
            groom_actuated_indices=build_groom_actuated_indices(
                actuated_joint_names, include_head=True
            ),
            include_head=True,
        )
        self._head_indices = build_head_actuated_indices(actuated_joint_names)

    def settle(self, steps: int) -> None:
        for _ in range(steps):
            self.sim.step()

    def proprioception(self) -> np.ndarray:
        d = self.sim.mj_data
        qpos, qvel = d.qpos, d.qvel
        joint_pos = qpos[self._qp][None, :]
        joint_vel = qvel[self._qv][None, :]
        root_height = d.xpos[self._root_body_id, 2:3][None, :]
        quat = d.xquat[self._root_body_id][None, :]
        velocity = np.empty(6)
        mj.mj_objectVelocity(
            self.sim.mj_model,
            d,
            mj.mjtObj.mjOBJ_BODY,
            self._root_body_id,
            velocity,
            0,
        )
        ang = velocity[:3][None, :]
        lin = velocity[3:][None, :]
        grav = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        feat = np.concatenate(
            [
                joint_pos,
                joint_vel,
                root_height,
                world_vectors_to_body(quat, grav),
                world_vectors_to_body(quat, lin),
                world_vectors_to_body(quat, ang),
            ],
            axis=1,
        ).astype(np.float32)
        feat[~np.isfinite(feat)] = 0.0
        return feat

    def obs(
        self,
        observation_space: gym.spaces.Dict | None = None,
    ) -> dict[str, np.ndarray]:
        if observation_space is None:
            return {
                "vision": self._zero_vision,
                "odor": self._zero_odor,
                "proprioception": self.proprioception(),
            }
        return _policy_observation(self.proprioception(), observation_space)

    def _activate_groom(self) -> None:
        self._groom_replay.activate(self._full_pose.copy(), sim_time=float(self.sim.time))

    def _deactivate_groom(self) -> None:
        support_pose = self._groom_replay.support_pose
        if support_pose is not None:
            self._full_pose[self._head_indices] = support_pose[self._head_indices]
        self._groom_replay.deactivate()
        self.controller.reset(seed=self._controller_seed)

    def _groom_locomotion_action(self) -> LocomotionAction:
        self.controller.step(
            np.zeros(2, dtype=float),
            HybridControllerObservation.from_sim(self.sim, self.fly.name),
        )
        loc = self._groom_replay.make_locomotion_action(float(self.sim.time))
        self._full_pose = np.asarray(loc.joint_angles, dtype=np.float32).copy()
        return loc

    def _walk_locomotion_action(self, sig: np.ndarray) -> LocomotionAction:
        if self._groom_replay is not None and self._groom_replay.active:
            self._deactivate_groom()
        loc = self.controller.step(
            sig, HybridControllerObservation.from_sim(self.sim, self.fly.name)
        )
        self._full_pose[self._leg_indices] = np.asarray(loc.joint_angles, dtype=np.float32)
        return LocomotionAction(
            joint_angles=self._full_pose.copy(), adhesion_onoff=loc.adhesion_onoff
        )

    def run_decision(
        self,
        action: np.ndarray,
        substeps: int,
    ) -> np.ndarray:
        action = int(np.asarray(action).reshape(1)[0])
        groom_active = self.action_labels.get(action) == "groom" and self._groom_replay is not None
        sig = np.array(self._action_forces[action], dtype=float)
        m, d = self.sim.mj_model, self.sim.mj_data
        ball_state = self.world.ball_state
        samples = []

        for _ in range(substeps):
            if groom_active:
                if not self._groom_replay.active:
                    self._activate_groom()
                loc = self._groom_locomotion_action()
            else:
                loc = self._walk_locomotion_action(sig)
            apply_locomotion_action(self.sim, self.fly.name, loc)
            self.sim.step()
            self._steps_since_tracking_sample += 1
            if self._steps_since_tracking_sample == self._tracking_interval_steps:
                samples.append(ball_state(m, d)[1])
                self._steps_since_tracking_sample = 0

        return np.asarray(samples, dtype=float).reshape((1, -1, 3))

    def close(self) -> None:
        self.sim.close()

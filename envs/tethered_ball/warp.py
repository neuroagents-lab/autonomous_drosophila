from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import warp as wp
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import ActuatorType
from flygym.utils.math import Rotation3D
from flygym.warp import GPUSimulation

from envs.action_spaces import discrete_action_specs
from envs.observations import world_vectors_to_body
from envs.tethered_ball.policy_runner import (
    BEHAVIOR_TRACKING_HZ,
    DEFAULT_GROOM_REPLAY_PATH,
    DEFAULT_TIMING,
    _policy_observation,
)
from envs.tethered_ball.world import (
    BALL_RADIUS_MM,
    TetheredBallWorld,
)
from envs.warp_runner import GPUOpenLoopTurningController, preserve_head_from_ctrl_kernel
from fly_agent.odor.olfaction import add_olfaction
from fly_agent.primitives.groom_replay import GroomReplayConfig, make_groom_fly
from fly_agent.primitives.warp_groom import GPUGroomController
from fly_agent.vision.warp_runtime import warp_graph_capture_requested

__all__ = ["WarpBallRunner"]


@wp.kernel
def _sample_ball_velocity_kernel(
    cvel: wp.array2d(dtype=wp.spatial_vectorf),
    samples: wp.array3d(dtype=wp.float32),
    ball_body_id: int,
    sample_index: int,
):
    env_index = wp.tid()
    velocity = cvel[env_index, ball_body_id]
    samples[env_index, sample_index, 0] = velocity[0]
    samples[env_index, sample_index, 1] = velocity[1]
    samples[env_index, sample_index, 2] = velocity[2]


class WarpBallRunner:
    """Batched hard-tethered ball simulation backed by MuJoCo-Warp."""

    def __init__(
        self,
        num_envs: int,
        *,
        timestep: float = DEFAULT_TIMING.timestep,
        ball_top_z: float = 0.15,
        ball_radius: float = BALL_RADIUS_MM,
        contact_preset: ContactBodiesPreset = ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        adhesion_gain: float | dict[str, float] = 40.0,
        joint_stiffness: float = 0.05,
        joint_damping: float = 0.06,
        actuator_gain: float = 45.0,
        actuator_force_limit: float = 65.0,
        spawn_z: float = 0.9,
        groom_replay_path: str | Path | None = DEFAULT_GROOM_REPLAY_PATH,
        max_constraints: int = 500,
        max_contacts: int = 500,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self._require_cuda_device()
        self.num_envs = int(num_envs)
        self.dt = float(timestep)
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
            max_constraints=max_constraints,
            max_contacts=max_contacts,
        )
        self._cache_model_addresses()
        self._setup_actions()
        self._setup_controller(groom_replay_path)
        self._seeds = [None] * self.num_envs
        self._neutral_qpos = self.sim.mjw_data.qpos.numpy()[0].copy()
        self._neutral_qvel = self.sim.mjw_data.qvel.numpy()[0].copy()
        self._neutral_ctrl = self.sim.mjw_data.ctrl.numpy()[0].copy()
        self._neutral_mocap_pos = self.sim.mjw_data.mocap_pos.numpy()[0].copy()
        self._neutral_mocap_quat = self.sim.mjw_data.mocap_quat.numpy()[0].copy()
        self._tracking_interval_steps = round(1 / (BEHAVIOR_TRACKING_HZ * self.dt))
        actual_tracking_hz = 1 / (self._tracking_interval_steps * self.dt)
        if not np.isclose(actual_tracking_hz, BEHAVIOR_TRACKING_HZ):
            raise ValueError(f"timestep {self.dt} cannot represent {BEHAVIOR_TRACKING_HZ} Hz")
        self._tracking_offset = 0
        self._decision_graphs = {}
        self._settle_graphs = {}

    @staticmethod
    def _require_cuda_device() -> None:
        if not any(device.is_cuda for device in wp.get_devices()):
            devices = ", ".join(str(device) for device in wp.get_devices()) or "none"
            raise RuntimeError(
                f"WarpBallRunner requires a CUDA Warp device; available Warp devices: {devices}"
            )

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
        max_constraints: int,
        max_contacts: int,
    ) -> None:
        self.fly = make_groom_fly(
            colorize=True,
            adhesion_gain=adhesion_gain,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            actuator_gain=actuator_gain,
            actuator_forcerange=(-actuator_force_limit, actuator_force_limit),
        )
        self.fly.add_vision()
        add_olfaction(self.fly)
        self.world = TetheredBallWorld(
            ball_top_z=ball_top_z,
            ball_radius=ball_radius,
            contact_preset=contact_preset,
        )
        self.world.add_fly(
            self.fly,
            spawn_position=(0.0, 0.0, spawn_z),
            spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        )
        self.world.mjcf_root.option.timestep = self.dt
        self.world.mjcf_root.option.noslip_iterations = 0
        self.sim = GPUSimulation(
            self.world,
            n_worlds=self.num_envs,
            timestep=self.dt,
            max_constraints=max_constraints,
            max_contacts=max_contacts,
        )
        self._wp_device = self.sim.mjw_data.qpos.device

    def _cache_model_addresses(self) -> None:
        model = self.sim.mj_model
        qpos_addresses = []
        qvel_addresses = []
        for joint_dof in self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION):
            joint = self.fly.jointdof_to_mjcfjoint[joint_dof]
            joint_id = mj.mj_name2id(
                model,
                mj.mjtObj.mjOBJ_JOINT,
                joint.name,
            )
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
        self._actuated_qpos_addresses = np.asarray(qpos_addresses, dtype=np.int32)
        self._actuated_qvel_addresses = np.asarray(qvel_addresses, dtype=np.int32)
        root_index = self.fly.get_bodysegs_order().index(self.fly.root_segment)
        self._root_body_id = int(self.sim._internal_bodyids_by_fly[self.fly.name][root_index])
        self._ball_body_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "treadmill"))

    def _setup_actions(self) -> None:
        specs = discrete_action_specs()
        self.action_labels = {spec.flat_index: spec.name for spec in specs}
        forces = {spec.flat_index: spec.forces for spec in specs}
        self._action_forces = np.asarray(
            [forces[index] for index in range(len(specs))],
            dtype=np.float32,
        )
        self._groom_action = next(
            (index for index, label in self.action_labels.items() if label == "groom"),
            None,
        )

    def _setup_controller(self, groom_replay_path: str | Path | None) -> None:
        self._actions_wp = wp.zeros(
            (self.num_envs, 2),
            dtype=wp.float32,
            device=self._wp_device,
        )
        action_limit = float(np.max(np.abs(self._action_forces)))
        self.controller = GPUOpenLoopTurningController(
            num_envs=self.num_envs,
            timestep=self.dt,
            output_dof_order=self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION),
            action_bounds=(-action_limit, action_limit),
            device=self._wp_device,
        )
        self._groom = None
        self._head_indices_wp = None
        self._head_actuator_ids_wp = None
        if groom_replay_path is not None:
            self._groom = GPUGroomController(
                num_envs=self.num_envs,
                fly=self.fly,
                sim=self.sim,
                config=GroomReplayConfig(replay_path=Path(groom_replay_path)),
                include_head=True,
                anchor_root=False,
                device=self._wp_device,
            )
            self._head_indices_wp = self._groom._head_indices_wp
            self._head_actuator_ids_wp = self._groom._position_actuator_ids_wp

    def reset(self, seeds: list[int | None] | None = None) -> dict[str, np.ndarray]:
        seeds = [None] * self.num_envs if seeds is None else list(seeds)
        if len(seeds) != self.num_envs:
            raise ValueError(f"seeds must have length {self.num_envs}")
        self._seeds = seeds
        mjw.reset_data(self.sim.mjw_model, self.sim.mjw_data)
        self.sim.mjw_data.qpos.assign(np.repeat(self._neutral_qpos[None, :], self.num_envs, axis=0))
        self.sim.mjw_data.qvel.assign(np.repeat(self._neutral_qvel[None, :], self.num_envs, axis=0))
        self.sim.mjw_data.ctrl.assign(np.repeat(self._neutral_ctrl[None, :], self.num_envs, axis=0))
        self.sim.mjw_data.mocap_pos.assign(
            np.repeat(self._neutral_mocap_pos[None, :], self.num_envs, axis=0)
        )
        self.sim.mjw_data.mocap_quat.assign(
            np.repeat(self._neutral_mocap_quat[None, :], self.num_envs, axis=0)
        )
        mjw.forward(self.sim.mjw_model, self.sim.mjw_data)
        indices = np.arange(self.num_envs, dtype=np.int32)
        self.controller.reset_at(indices, seeds)
        if self._groom is not None:
            self._groom.reset_rows(indices)
        self._tracking_offset = 0
        return self.obs()

    def settle(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return
        if not warp_graph_capture_requested():
            for _ in range(steps):
                self.sim.step()
            return
        graph = self._settle_graphs.get(steps)
        if graph is None:
            with wp.ScopedCapture() as capture:
                for _ in range(steps):
                    self.sim.step()
            graph = capture.graph
            self._settle_graphs[steps] = graph
        wp.capture_launch(graph)

    def proprioception(self) -> np.ndarray:
        indices = np.arange(self.num_envs)
        qpos = self.sim.mjw_data.qpos.numpy()
        qvel = self.sim.mjw_data.qvel.numpy()
        joint_pos = qpos[indices[:, None], self._actuated_qpos_addresses]
        joint_vel = qvel[indices[:, None], self._actuated_qvel_addresses]
        root_positions = self.sim.mjw_data.xpos.numpy()[:, self._root_body_id]
        root_quaternions = self.sim.mjw_data.xquat.numpy()[:, self._root_body_id]
        zeros = np.zeros((self.num_envs, 3), dtype=np.float32)
        gravity = np.repeat(
            np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
            self.num_envs,
            axis=0,
        )
        features = np.concatenate(
            [
                joint_pos,
                joint_vel,
                root_positions[:, 2:3],
                world_vectors_to_body(root_quaternions, gravity),
                zeros,
                zeros,
            ],
            axis=1,
        ).astype(np.float32)
        features[~np.isfinite(features)] = 0.0
        return features

    def obs(
        self,
        observation_space: gym.spaces.Dict | None = None,
    ) -> dict[str, np.ndarray]:
        if observation_space is None:
            return {
                "vision": np.zeros(
                    (self.num_envs, 6, 32, 32),
                    dtype=np.uint8,
                ),
                "odor": np.zeros((self.num_envs, 2), dtype=np.float32),
                "proprioception": self.proprioception(),
            }
        return _policy_observation(self.proprioception(), observation_space)

    def run_decision(
        self,
        actions: np.ndarray,
        substeps: int,
    ) -> np.ndarray:
        action_indices = np.asarray(actions)
        if action_indices.shape == (self.num_envs, 1):
            action_indices = action_indices[:, 0]
        if action_indices.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if not np.issubdtype(action_indices.dtype, np.integer) and not np.all(
            action_indices == np.floor(action_indices)
        ):
            raise ValueError("actions must contain integer indices")
        action_indices = action_indices.astype(np.int64)
        if np.any((action_indices < 0) | (action_indices >= len(self.action_labels))):
            raise ValueError("action index out of range")
        if substeps < 1:
            raise ValueError("substeps must be positive")

        self._actions_wp.assign(self._action_forces[action_indices])
        groom_flags = (
            np.zeros(self.num_envs, dtype=bool)
            if self._groom_action is None
            else action_indices == self._groom_action
        )
        if self._groom is not None:
            deactivated = self._groom.update_active_rows(groom_flags, 0)
            if deactivated.size:
                self.controller.reset_at(
                    deactivated,
                    [self._seeds[int(index)] for index in deactivated],
                )
        if not warp_graph_capture_requested():
            samples, end_offset = self._run_decision_eager(
                substeps,
            )
            self._tracking_offset = end_offset
            if samples is None:
                return np.empty((self.num_envs, 0, 3), dtype=np.float32)
            return samples.numpy()
        graph, samples, end_offset = self._decision_graph(
            substeps,
        )
        wp.capture_launch(graph)
        self._tracking_offset = end_offset
        if samples is None:
            return np.empty((self.num_envs, 0, 3), dtype=np.float32)
        return samples.numpy()

    def _run_decision_eager(self, substeps: int):
        samples, sample_indices, end_offset = self._prepare_decision_samples(substeps)
        self._decision_substeps(substeps, sample_indices, samples)
        return samples, end_offset

    def _prepare_decision_samples(self, substeps: int):
        sample_steps = range(
            self._tracking_interval_steps - self._tracking_offset,
            substeps + 1,
            self._tracking_interval_steps,
        )
        sample_indices = {step: index for index, step in enumerate(sample_steps)}
        samples = None
        if sample_indices:
            samples = wp.empty(
                (self.num_envs, len(sample_indices), 3),
                dtype=wp.float32,
                device=self._wp_device,
            )
        end_offset = (self._tracking_offset + substeps) % self._tracking_interval_steps
        return samples, sample_indices, end_offset

    def _decision_substeps(
        self,
        substeps: int,
        sample_indices: dict[int, int],
        samples,
    ) -> None:
        for step in range(1, substeps + 1):
            joint_angles, adhesion = self.controller.step(self._actions_wp)
            if self._head_indices_wp is not None:
                wp.launch(
                    preserve_head_from_ctrl_kernel,
                    dim=(self.num_envs, self._groom.n_head_joints),
                    inputs=[
                        joint_angles,
                        self.sim.mjw_data.ctrl,
                        self._head_indices_wp,
                        self._head_actuator_ids_wp,
                    ],
                    device=self._wp_device,
                )
            if self._groom is not None:
                joint_angles, adhesion = self._groom.step(
                    self.sim.mjw_data.time,
                    joint_angles,
                    adhesion,
                    0,
                    0,
                )
            self.sim.set_actuator_inputs(
                self.fly.name,
                ActuatorType.POSITION,
                joint_angles,
            )
            self.sim.set_leg_adhesion_states(self.fly.name, adhesion)
            self.sim.step()
            sample_index = sample_indices.get(step)
            if sample_index is not None:
                wp.launch(
                    _sample_ball_velocity_kernel,
                    dim=self.num_envs,
                    inputs=[
                        self.sim.mjw_data.cvel,
                        samples,
                        self._ball_body_id,
                        sample_index,
                    ],
                    device=self._wp_device,
                )

    def _decision_graph(self, substeps: int):
        key = (substeps, self._tracking_offset)
        cached = self._decision_graphs.get(key)
        if cached is not None:
            return cached
        samples, sample_indices, end_offset = self._prepare_decision_samples(substeps)
        with wp.ScopedCapture() as capture:
            self._decision_substeps(substeps, sample_indices, samples)
        result = (capture.graph, samples, end_offset)
        self._decision_graphs[key] = result
        return result

    def close(self) -> None:
        self.sim.close()

from __future__ import annotations

import numpy as np
import warp as wp
from flygym.compose import ActuatorType, Fly
from flygym.warp import GPUSimulation

from fly_agent.primitives.groom_replay import (
    GroomReplayConfig,
    build_groom_actuated_indices,
    build_head_actuated_indices,
)


@wp.kernel
def _groom_pose_kernel(
    groom_active: wp.array(dtype=wp.int32),
    sim_time: wp.array(dtype=wp.float32),
    phase_start_time: wp.array(dtype=wp.float32),
    support_pose: wp.array2d(dtype=wp.float32),
    anchor_pose: wp.array2d(dtype=wp.float32),
    clip: wp.array2d(dtype=wp.float32),
    groom_indices: wp.array(dtype=wp.int32),
    groom_joint_angles: wp.array2d(dtype=wp.float32),
    source_hz: float,
    startup_blend_seconds: float,
    max_index: int,
    shared_time: int,
    n_actuators: int,
    n_groom_joints: int,
):
    env_idx = wp.tid()
    for actuator_idx in range(n_actuators):
        groom_joint_angles[env_idx, actuator_idx] = support_pose[env_idx, actuator_idx]

    if groom_active[env_idx] == 0:
        return

    time_idx = wp.int32(0) if shared_time != 0 else env_idx
    elapsed = sim_time[time_idx] - phase_start_time[env_idx]
    if elapsed < 0.0:
        elapsed = 0.0

    n_frames = max_index + 1
    frame_pos = elapsed * source_hz
    wrapped_pos = frame_pos - wp.floor(frame_pos / wp.float32(n_frames)) * wp.float32(n_frames)
    start_idx = wp.int32(wrapped_pos)
    end_idx = start_idx + 1
    if end_idx >= n_frames:
        end_idx = 0
    alpha = wrapped_pos - wp.float32(start_idx)

    alpha_blend = wp.float32(1.0)
    if startup_blend_seconds > 0.0:
        alpha_blend = wp.min(1.0, elapsed / startup_blend_seconds)

    for joint_idx in range(n_groom_joints):
        actuator_idx = groom_indices[joint_idx]
        clip_val = (1.0 - alpha) * clip[start_idx, joint_idx] + alpha * clip[end_idx, joint_idx]
        anchor_val = anchor_pose[env_idx, joint_idx]
        groom_val = (1.0 - alpha_blend) * anchor_val + alpha_blend * clip_val
        groom_joint_angles[env_idx, actuator_idx] = groom_val


@wp.kernel
def _fuse_control_kernel(
    groom_active: wp.array(dtype=wp.int32),
    cpg_joint_angles: wp.array2d(dtype=wp.float32),
    groom_joint_angles: wp.array2d(dtype=wp.float32),
    cpg_adhesion: wp.array2d(dtype=wp.float32),
    groom_adhesion: wp.array2d(dtype=wp.float32),
    final_joint_angles: wp.array2d(dtype=wp.float32),
    final_adhesion: wp.array2d(dtype=wp.float32),
    n_actuators: int,
    n_legs: int,
):
    env_idx = wp.tid()
    use_groom = groom_active[env_idx] != 0
    for actuator_idx in range(n_actuators):
        if use_groom:
            final_joint_angles[env_idx, actuator_idx] = groom_joint_angles[env_idx, actuator_idx]
        else:
            final_joint_angles[env_idx, actuator_idx] = cpg_joint_angles[env_idx, actuator_idx]
    for leg_idx in range(n_legs):
        if use_groom:
            final_adhesion[env_idx, leg_idx] = groom_adhesion[env_idx, leg_idx]
        else:
            final_adhesion[env_idx, leg_idx] = cpg_adhesion[env_idx, leg_idx]


@wp.kernel
def _groom_root_anchor_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    groom_active: wp.array(dtype=wp.int32),
    anchor_valid: wp.array(dtype=wp.int32),
    root_anchor: wp.array2d(dtype=wp.float32),
    root_qposadr: int,
    root_qveladr: int,
):
    env_idx = wp.tid()
    if groom_active[env_idx] == 0 or anchor_valid[env_idx] == 0:
        return
    for i in range(7):
        qpos[env_idx, root_qposadr + i] = root_anchor[env_idx, i]
    for i in range(6):
        qvel[env_idx, root_qveladr + i] = 0.0


@wp.kernel
def _reset_groom_rows_kernel(
    rows: wp.array(dtype=wp.int32),
    groom_active: wp.array(dtype=wp.int32),
    anchor_valid: wp.array(dtype=wp.int32),
):
    row = rows[wp.tid()]
    groom_active[row] = 0
    anchor_valid[row] = 0


@wp.kernel
def _activate_groom_rows_kernel(
    rows: wp.array(dtype=wp.int32),
    ctrl: wp.array2d(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    position_actuator_ids: wp.array(dtype=wp.int32),
    groom_indices: wp.array(dtype=wp.int32),
    support_pose: wp.array2d(dtype=wp.float32),
    anchor_pose: wp.array2d(dtype=wp.float32),
    phase_start_time: wp.array(dtype=wp.float32),
    root_anchor: wp.array2d(dtype=wp.float32),
    groom_active: wp.array(dtype=wp.int32),
    anchor_valid: wp.array(dtype=wp.int32),
    root_qposadr: int,
    shared_time: int,
    anchor_root: int,
    n_actuators: int,
    n_groom_joints: int,
):
    env_idx = rows[wp.tid()]
    for actuator_idx in range(n_actuators):
        support_pose[env_idx, actuator_idx] = ctrl[env_idx, position_actuator_ids[actuator_idx]]
    for joint_idx in range(n_groom_joints):
        anchor_pose[env_idx, joint_idx] = support_pose[env_idx, groom_indices[joint_idx]]
    phase_start_time[env_idx] = sim_time[0] if shared_time != 0 else sim_time[env_idx]
    if anchor_root != 0:
        for index in range(7):
            root_anchor[env_idx, index] = qpos[env_idx, root_qposadr + index]
        anchor_valid[env_idx] = 1
    else:
        anchor_valid[env_idx] = 0
    groom_active[env_idx] = 1


@wp.kernel
def _deactivate_groom_rows_kernel(
    rows: wp.array(dtype=wp.int32),
    ctrl: wp.array2d(dtype=wp.float32),
    position_actuator_ids: wp.array(dtype=wp.int32),
    head_indices: wp.array(dtype=wp.int32),
    support_pose: wp.array2d(dtype=wp.float32),
    groom_active: wp.array(dtype=wp.int32),
    anchor_valid: wp.array(dtype=wp.int32),
    n_head_joints: int,
):
    env_idx = rows[wp.tid()]
    if groom_active[env_idx] == 0:
        return
    for head_idx in range(n_head_joints):
        actuator_idx = head_indices[head_idx]
        ctrl[env_idx, position_actuator_ids[actuator_idx]] = support_pose[env_idx, actuator_idx]
    groom_active[env_idx] = 0
    anchor_valid[env_idx] = 0


class GPUGroomController:
    """Batched groom replay for MuJoCo-Warp navigation."""

    _N_LEGS = 6

    def __init__(
        self,
        *,
        num_envs: int,
        fly: Fly,
        sim: GPUSimulation,
        config: GroomReplayConfig,
        include_head: bool,
        anchor_root: bool,
        device,
    ) -> None:
        self.num_envs = int(num_envs)
        self.sim = sim
        self.device = device
        self.anchor_root = bool(anchor_root)

        actuated_names = [
            dof.name for dof in fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        ]
        self.n_actuators = len(actuated_names)
        self.groom_actuated_indices = build_groom_actuated_indices(
            actuated_names, include_head=include_head
        )
        self.head_actuated_indices = (
            build_head_actuated_indices(actuated_names)
            if include_head
            else np.array([], dtype=np.int32)
        )
        self.n_groom_joints = int(self.groom_actuated_indices.shape[0])
        self.n_head_joints = int(self.head_actuated_indices.shape[0])

        self._position_actuator_ids = np.asarray(
            sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly.name],
            dtype=np.int32,
        )
        if self._position_actuator_ids.shape[0] != self.n_actuators:
            raise RuntimeError("Position actuator id count does not match actuated DOF count.")

        clip = config.load_clip(include_head=include_head)
        self.max_index = len(clip) - 1
        self.source_hz = float(config.source_hz)
        self.startup_blend_seconds = float(config.startup_blend_seconds)

        self._clip_wp = wp.array(clip, dtype=wp.float32, device=device)
        self._groom_indices_wp = wp.array(
            self.groom_actuated_indices, dtype=wp.int32, device=device
        )
        self._head_indices_wp = wp.array(self.head_actuated_indices, dtype=wp.int32, device=device)
        self._position_actuator_ids_wp = wp.array(
            self._position_actuator_ids, dtype=wp.int32, device=device
        )
        self._groom_adhesion_wp = wp.array(
            np.tile(config.adhesion, (self.num_envs, 1)),
            dtype=wp.float32,
            device=device,
        )

        self.groom_active_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=device)
        self.phase_start_time_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=device)
        self.support_pose_wp = wp.zeros(
            (self.num_envs, self.n_actuators), dtype=wp.float32, device=device
        )
        self.anchor_pose_wp = wp.zeros(
            (self.num_envs, self.n_groom_joints), dtype=wp.float32, device=device
        )
        self.root_anchor_wp = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=device)
        self.anchor_valid_wp = wp.zeros(self.num_envs, dtype=wp.int32, device=device)
        self.groom_joint_angles_wp = wp.zeros(
            (self.num_envs, self.n_actuators), dtype=wp.float32, device=device
        )
        self.final_joint_angles_wp = wp.zeros(
            (self.num_envs, self.n_actuators), dtype=wp.float32, device=device
        )
        self.final_adhesion_wp = wp.zeros(
            (self.num_envs, self._N_LEGS), dtype=wp.float32, device=device
        )

        self._groom_active_host = np.zeros(self.num_envs, dtype=np.int32)
        self._saw_grooming_host = np.zeros(self.num_envs, dtype=bool)

    def reset_rows(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int32)
        if indices.size == 0:
            return
        self._groom_active_host[indices] = 0
        self._saw_grooming_host[indices] = False
        rows = wp.array(indices, dtype=wp.int32, device=self.device)
        wp.launch(
            _reset_groom_rows_kernel,
            dim=indices.size,
            inputs=[rows, self.groom_active_wp, self.anchor_valid_wp],
            device=self.device,
        )

    def update_active_rows(self, groom_flags: np.ndarray, root_qposadr: int) -> np.ndarray:
        groom_flags = np.asarray(groom_flags, dtype=bool)
        active = self._groom_active_host.astype(bool, copy=False)
        if not np.any(groom_flags) and not np.any(active):
            return np.empty(0, dtype=np.int32)

        deactivate = np.flatnonzero(active & ~groom_flags).astype(np.int32)
        if deactivate.size:
            self._deactivate_rows(deactivate)
            self._groom_active_host[deactivate] = 0

        activate = np.flatnonzero(groom_flags & ~active).astype(np.int32)
        if activate.size:
            rows = wp.array(activate, dtype=wp.int32, device=self.device)
            wp.launch(
                _activate_groom_rows_kernel,
                dim=activate.size,
                inputs=[
                    rows,
                    self.sim.mjw_data.ctrl,
                    self.sim.mjw_data.qpos,
                    self.sim.mjw_data.time,
                    self._position_actuator_ids_wp,
                    self._groom_indices_wp,
                    self.support_pose_wp,
                    self.anchor_pose_wp,
                    self.phase_start_time_wp,
                    self.root_anchor_wp,
                    self.groom_active_wp,
                    self.anchor_valid_wp,
                    root_qposadr,
                    int(self.sim.mjw_data.time.shape[0] == 1),
                    int(self.anchor_root),
                    self.n_actuators,
                    self.n_groom_joints,
                ],
                device=self.device,
            )
            self._groom_active_host[activate] = 1
        return deactivate

    def finish_decision(self, groom_flags: np.ndarray) -> None:
        # Consecutive groom actions keep the bout alive so the clip can resume.
        # Walking deactivates in update_active_rows.
        groom_flags = np.asarray(groom_flags, dtype=bool)
        if not np.any(groom_flags) and not np.any(self._groom_active_host):
            self._saw_grooming_host.fill(False)
            return
        self._saw_grooming_host = groom_flags.copy()

    def saw_grooming(self, env_idx: int) -> bool:
        return bool(self._saw_grooming_host[env_idx])

    def _deactivate_rows(self, indices: np.ndarray) -> None:
        rows = wp.array(indices, dtype=wp.int32, device=self.device)
        wp.launch(
            _deactivate_groom_rows_kernel,
            dim=indices.size,
            inputs=[
                rows,
                self.sim.mjw_data.ctrl,
                self._position_actuator_ids_wp,
                self._head_indices_wp,
                self.support_pose_wp,
                self.groom_active_wp,
                self.anchor_valid_wp,
                self.n_head_joints,
            ],
            device=self.device,
        )

    def step(
        self,
        sim_time_wp: wp.array,
        cpg_joint_angles: wp.array,
        cpg_adhesion: wp.array,
        root_qposadr: int,
        root_qveladr: int,
    ) -> tuple[wp.array, wp.array]:
        wp.launch(
            _groom_pose_kernel,
            dim=self.num_envs,
            inputs=[
                self.groom_active_wp,
                sim_time_wp,
                self.phase_start_time_wp,
                self.support_pose_wp,
                self.anchor_pose_wp,
                self._clip_wp,
                self._groom_indices_wp,
                self.groom_joint_angles_wp,
                self.source_hz,
                self.startup_blend_seconds,
                self.max_index,
                int(sim_time_wp.shape[0] == 1),
                self.n_actuators,
                self.n_groom_joints,
            ],
            device=self.device,
        )
        wp.launch(
            _fuse_control_kernel,
            dim=self.num_envs,
            inputs=[
                self.groom_active_wp,
                cpg_joint_angles,
                self.groom_joint_angles_wp,
                cpg_adhesion,
                self._groom_adhesion_wp,
                self.final_joint_angles_wp,
                self.final_adhesion_wp,
                self.n_actuators,
                self._N_LEGS,
            ],
            device=self.device,
        )
        self.apply_root_anchor(root_qposadr, root_qveladr)
        return self.final_joint_angles_wp, self.final_adhesion_wp

    def apply_root_anchor(self, root_qposadr: int, root_qveladr: int) -> None:
        wp.launch(
            _groom_root_anchor_kernel,
            dim=self.num_envs,
            inputs=[
                self.sim.mjw_data.qpos,
                self.sim.mjw_data.qvel,
                self.groom_active_wp,
                self.anchor_valid_wp,
                self.root_anchor_wp,
                root_qposadr,
                root_qveladr,
            ],
            device=self.device,
        )

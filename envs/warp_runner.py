"""Batched MuJoCo-Warp simulation of ``num_envs`` flies in the odor/obstacle arena.

The GPU counterpart of ``envs.navigation.FlyBodyNavigationEnv``: same scene, same
observations, with the CPG walking controller and grooming replay ported to Warp kernels.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import warp as wp
from flygym import Renderer
from flygym.anatomy import LEGS, BodySegment
from flygym.compose import ActuatorType
from flygym.vision.retina import Retina
from flygym.warp import GPUSimulation
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
from flygym_demo.complex_terrain.cpg_controller import make_tripod_cpg_network
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

from envs.action_spaces import ACTION_FORCE_RANGE
from envs.layouts import EpisodeLayout
from envs.navigation import (
    RECORDING_CAMERA_RES,
    RECORDING_FPS,
    RECORDING_PLAYBACK_SPEED,
    FlyBodyEnvConfig,
)
from envs.observations import (
    FLY_VISION_OBSERVATION,
    ODOR_OBSERVATION,
    PROPRIOCEPTION_OBSERVATION,
    make_multimodal_observation_space,
    odor_observation,
    proprioception_observation,
)
from envs.physics_recovery import PhysicsRecovery
from envs.world import (
    actuated_joint_addresses,
    add_eval_camera,
    build_scene,
    mocap_ids,
    root_addresses,
    yaw_to_quat_wxyz,
)
from fly_agent.odor.olfaction import olfaction_site_ids
from fly_agent.primitives.warp_groom import GPUGroomController
from fly_agent.vision.constants import EYE_CAMERA_NAMES, WARP_COLORIZE_FLY
from fly_agent.vision.gpu_retina import GPURetina
from fly_agent.vision.warp_gpu_patches import create_eye_render_context
from fly_agent.vision.warp_runtime import warp_graph_capture_requested

_N_LEGS = len(LEGS)
_N_STEP_DOFS = len(PreprogrammedSteps.dofs_per_leg)
_TWO_PI = 2 * np.pi


@wp.kernel
def preserve_head_from_ctrl_kernel(
    joint_angles: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    head_actuator_indices: wp.array(dtype=wp.int32),
    position_actuator_ids: wp.array(dtype=wp.int32),
):
    """Hold the head at its current control target (the CPG only drives the legs)."""
    env_idx, head_idx = wp.tid()
    actuator_idx = head_actuator_indices[head_idx]
    joint_angles[env_idx, actuator_idx] = ctrl[env_idx, position_actuator_ids[actuator_idx]]


@wp.kernel
def _copy_row_kernel(
    dst: wp.array2d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
):
    row_idx, col_idx = wp.tid()
    dst[rows[row_idx], col_idx] = values[col_idx]


@wp.kernel
def _set_root_state_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    poses: wp.array2d(dtype=wp.float32),
    root_qposadr: int,
    root_qveladr: int,
):
    """Write position + wxyz quaternion (7 values per row) and zero the root velocity."""
    row_idx = wp.tid()
    world_idx = rows[row_idx]
    for i in range(7):
        qpos[world_idx, root_qposadr + i] = poses[row_idx, i]
    for i in range(6):
        qvel[world_idx, root_qveladr + i] = 0.0


@wp.kernel
def _set_mocap_positions_kernel(
    mocap_pos: wp.array2d(dtype=wp.vec3f),
    rows: wp.array(dtype=wp.int32),
    mocap_ids: wp.array(dtype=wp.int32),
    positions: wp.array2d(dtype=wp.vec3f),
):
    row_idx, item_idx = wp.tid()
    mocap_pos[rows[row_idx], mocap_ids[item_idx]] = positions[row_idx, item_idx]


@wp.kernel
def _cpg_update_kernel(
    actions: wp.array2d(dtype=wp.float32),
    phases: wp.array2d(dtype=wp.float32),
    magnitudes: wp.array2d(dtype=wp.float32),
    base_intrinsic_freqs: wp.array(dtype=wp.float32),
    coupling_weights: wp.array2d(dtype=wp.float32),
    phase_biases: wp.array2d(dtype=wp.float32),
    convergence_coefs: wp.array(dtype=wp.float32),
    next_phases: wp.array2d(dtype=wp.float32),
    next_magnitudes: wp.array2d(dtype=wp.float32),
    timestep: float,
    action_low: float,
    action_high: float,
):
    env_idx, leg_idx = wp.tid()
    side_idx = wp.int32(0)
    if leg_idx >= 3:
        side_idx = 1
    descending_signal = actions[env_idx, side_idx]
    if descending_signal < action_low:
        descending_signal = action_low
    if descending_signal > action_high:
        descending_signal = action_high

    intrinsic_freq = base_intrinsic_freqs[leg_idx]
    if descending_signal < 0.0:
        intrinsic_freq = -intrinsic_freq

    phase = phases[env_idx, leg_idx]
    coupling = wp.float32(0.0)
    for other_leg_idx in range(_N_LEGS):
        phase_diff = phases[env_idx, other_leg_idx] - phase
        coupling += (
            magnitudes[env_idx, other_leg_idx]
            * coupling_weights[leg_idx, other_leg_idx]
            * wp.sin(phase_diff - phase_biases[leg_idx, other_leg_idx])
        )

    dtheta_dt = wp.float32(_TWO_PI) * intrinsic_freq + coupling
    intrinsic_amp = wp.abs(descending_signal)
    dr_dt = convergence_coefs[leg_idx] * (intrinsic_amp - magnitudes[env_idx, leg_idx])

    next_phase = phase + dtheta_dt * timestep
    next_phase = next_phase - wp.floor(next_phase / wp.float32(_TWO_PI)) * wp.float32(_TWO_PI)
    next_phases[env_idx, leg_idx] = next_phase
    next_magnitudes[env_idx, leg_idx] = magnitudes[env_idx, leg_idx] + dr_dt * timestep


@wp.kernel
def _cpg_output_kernel(
    phases: wp.array2d(dtype=wp.float32),
    magnitudes: wp.array2d(dtype=wp.float32),
    next_phases: wp.array2d(dtype=wp.float32),
    next_magnitudes: wp.array2d(dtype=wp.float32),
    spline_coeffs: wp.array4d(dtype=wp.float32),
    neutral_angles: wp.array2d(dtype=wp.float32),
    dof_to_output_idx: wp.array2d(dtype=wp.int32),
    swing_periods: wp.array2d(dtype=wp.float32),
    joint_angles: wp.array2d(dtype=wp.float32),
    adhesion: wp.array2d(dtype=wp.float32),
    phase_grid_step: float,
    n_intervals: int,
):
    env_idx, leg_idx = wp.tid()
    phase = next_phases[env_idx, leg_idx]
    magnitude = next_magnitudes[env_idx, leg_idx]
    phases[env_idx, leg_idx] = phase
    magnitudes[env_idx, leg_idx] = magnitude

    adhesion_value = wp.float32(1.0)
    if phase > swing_periods[leg_idx, 0] and phase < swing_periods[leg_idx, 1]:
        adhesion_value = wp.float32(0.0)
    adhesion[env_idx, leg_idx] = adhesion_value

    interval_idx = wp.int32(wp.floor(phase / phase_grid_step))
    if interval_idx < 0:
        interval_idx = 0
    if interval_idx >= n_intervals:
        interval_idx = n_intervals - 1
    phase_offset = phase - wp.float32(interval_idx) * phase_grid_step

    for dof_idx in range(_N_STEP_DOFS):
        output_idx = dof_to_output_idx[leg_idx, dof_idx]
        if output_idx >= 0:
            value = (
                (
                    spline_coeffs[leg_idx, dof_idx, 0, interval_idx] * phase_offset
                    + spline_coeffs[leg_idx, dof_idx, 1, interval_idx]
                )
                * phase_offset
                + spline_coeffs[leg_idx, dof_idx, 2, interval_idx]
            ) * phase_offset + spline_coeffs[leg_idx, dof_idx, 3, interval_idx]
            neutral = neutral_angles[leg_idx, dof_idx]
            joint_angles[env_idx, output_idx] = neutral + magnitude * (value - neutral)


@wp.kernel
def _reset_cpg_rows_kernel(
    rows: wp.array(dtype=wp.int32),
    reset_phases: wp.array2d(dtype=wp.float32),
    phases: wp.array2d(dtype=wp.float32),
    magnitudes: wp.array2d(dtype=wp.float32),
):
    row_idx, leg_idx = wp.tid()
    env_idx = rows[row_idx]
    phases[env_idx, leg_idx] = reset_phases[row_idx, leg_idx]
    magnitudes[env_idx, leg_idx] = 0.0


class WarpNavigationTaskRunner:
    """``num_envs`` parallel worlds in one ``GPUSimulation``, with numpy-facing results.

    ``reset_at`` places each world's ``EpisodeLayout``; ``step`` runs one decision for all
    worlds. Rewards live in ``envs.warp.FlyGymWarpVecEnv``.
    """

    def __init__(self, config: FlyBodyEnvConfig, num_envs: int, *, render: bool = False) -> None:
        devices = wp.get_devices()
        if not any(device.is_cuda for device in devices):
            names = ", ".join(str(device) for device in devices) or "none"
            raise RuntimeError(
                f"WarpNavigationTaskRunner requires a CUDA Warp device; got {names}."
            )
        self.config = config
        self.num_envs = int(num_envs)
        self.fly, self.world = build_scene(config, colorize=WARP_COLORIZE_FLY)
        self._recording_model_path: Path | None = None
        if render:
            self._snapshot_recording_model()
        self.sim = GPUSimulation(
            self.world,
            n_worlds=self.num_envs,
            timestep=config.timestep,
            max_constraints=config.max_constraints,
            max_contacts=config.max_contacts,
        )
        self._wp_device = self.sim.mjw_data.qpos.device
        self._setup_vision()
        self._recording_renderer = None
        self._recording_prev_state: dict | None = None
        if render:
            self._recording_renderer = Renderer(
                self._recording_mj_model,
                self._recording_camera_name,
                camera_res=RECORDING_CAMERA_RES,
                playback_speed=RECORDING_PLAYBACK_SPEED,
                output_fps=RECORDING_FPS,
                buffer_frames=True,
            )

        model = self.sim.mj_model
        self._root_qposadr, self._root_qveladr = root_addresses(model)
        self._joint_qposadrs, self._joint_qveladrs = actuated_joint_addresses(model, self.fly)
        self.observation_space = make_multimodal_observation_space(
            config.observation_components,
            rgb_size=config.rgb_size,
            proprioception_dim=2 * len(self._joint_qposadrs) + 10,
        )
        self._odor_mocap_ids_wp = self._device_array(
            mocap_ids(model, self.world.odor_marker_body_names()), wp.int32
        )
        self._obstacle_mocap_ids_wp = self._device_array(
            mocap_ids(model, self.world.obstacle_body_names()), wp.int32
        )
        self._thorax_body_idx = self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        self._olfaction_site_ids = olfaction_site_ids(self.sim, self.fly)
        self._setup_controller()
        self._neutral_qpos = np.asarray(self.sim.mj_data.qpos, dtype=np.float32).copy()
        self._neutral_qvel = np.asarray(self.sim.mj_data.qvel, dtype=np.float32).copy()
        self._neutral_ctrl = np.asarray(self.sim.mj_data.ctrl, dtype=np.float32).copy()
        self.current_odor_source = np.repeat(config.odor_source[np.newaxis], self.num_envs, axis=0)
        self.physics_recovery = PhysicsRecovery(self.sim.mjw_data)
        # Optional ``envs.collection.SelectedResetRendering`` for partial resets.
        self.reset_rendering = None

    def _device_array(self, values, dtype):
        return wp.array(np.asarray(values), dtype=dtype, device=self._wp_device)

    def _setup_controller(self) -> None:
        self._actions_wp = wp.zeros((self.num_envs, 2), dtype=wp.float32, device=self._wp_device)
        self.controller = GPUOpenLoopTurningController(
            num_envs=self.num_envs,
            timestep=self.config.timestep,
            output_dof_order=self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION),
            action_bounds=ACTION_FORCE_RANGE,
            device=self._wp_device,
        )
        self._groom = GPUGroomController(
            num_envs=self.num_envs,
            fly=self.fly,
            sim=self.sim,
            config=self.config.groom_replay_config(),
            include_head=True,
            anchor_root=True,
            device=self._wp_device,
        )
        self._decision_graph = None
        if warp_graph_capture_requested():
            with wp.ScopedCapture() as capture:
                self._run_decision_physics()
            self._decision_graph = capture.graph

    def _setup_vision(self) -> None:
        self._vision_renderer = None
        if FLY_VISION_OBSERVATION not in self.config.observation_components:
            return
        self._retina = Retina()
        self.sim.retina = self._retina
        self._gpu_retina = GPURetina(
            self._retina, output_size=self.config.rgb_size, device=self._wp_device
        )
        eye_cameras = [self.fly.eyecameraname_to_mjcfcamera[name].name for name in EYE_CAMERA_NAMES]
        # Recompiles the model for batch rendering, so the internal id maps are refreshed.
        self._vision_renderer = self.sim.set_renderer(
            eye_cameras,
            camera_res=(self._retina.nrows, self._retina.ncols),
            buffer_frames=False,
            worlds=list(range(self.num_envs)),
            use_gpu_batch_rendering=True,
        )
        self._refresh_simulation_mappings()
        self._vision_renderer._rendering_context = create_eye_render_context(
            self._vision_renderer, self.num_envs
        )
        self._vision_readouts = self._gpu_retina.allocate_readouts(self.num_envs)
        self._vision_pixels = self._gpu_retina.allocate_pixel_observation(self.num_envs)
        self._vision_readouts_valid = False

    def _refresh_simulation_mappings(self) -> None:
        self.sim._map_internal_bodyids()
        self.sim._map_internal_geom_ids()
        self.sim._map_internal_ground_geom_ids()
        self.sim._map_internal_jointids()
        self.sim._map_internal_qposqveladrs()
        self.sim._map_internal_actuator_ids()
        self.sim._map_internal_adhesionactuator_ids()
        self.sim._map_internal_site_ids()
        self.sim._map_internal_eye_camera_ids()

    def reset_at(
        self,
        indices,
        layouts: list[EpisodeLayout],
        seeds: list[int | None],
    ) -> dict[str, np.ndarray]:
        """Reset worlds ``indices`` to their layouts; returns their observations."""
        indices = np.asarray(indices, dtype=np.int32)
        rows = self._device_array(indices, wp.int32)
        for env_idx, layout in zip(indices, layouts, strict=True):
            self.current_odor_source[env_idx] = layout.odor_source
        poses = np.array(
            [(*layout.spawn_position, *yaw_to_quat_wxyz(layout.spawn_yaw)) for layout in layouts],
            dtype=np.float32,
        )
        obstacles = np.array(
            [
                np.column_stack(
                    [
                        layout.obstacle_positions,
                        np.full(len(layout.obstacle_positions), self.config.obstacle_height / 2),
                    ]
                )
                for layout in layouts
            ],
            dtype=np.float32,
        ).reshape(len(layouts), -1, 3)
        data = self.sim.mjw_data
        for dst, values in (
            (data.qpos, self._neutral_qpos),
            (data.qvel, self._neutral_qvel),
            (data.ctrl, self._neutral_ctrl),
        ):
            wp.launch(
                _copy_row_kernel,
                dim=(len(indices), len(values)),
                inputs=[dst, rows, self._device_array(values, wp.float32)],
                device=self._wp_device,
            )
        wp.launch(
            _set_root_state_kernel,
            dim=len(indices),
            inputs=[
                data.qpos,
                data.qvel,
                rows,
                self._device_array(poses, wp.float32),
                self._root_qposadr,
                self._root_qveladr,
            ],
            device=self._wp_device,
        )
        for ids, positions in (
            (self._odor_mocap_ids_wp, self.current_odor_source[indices]),
            (self._obstacle_mocap_ids_wp, obstacles),
        ):
            if ids.shape[0]:
                wp.launch(
                    _set_mocap_positions_kernel,
                    dim=(len(indices), ids.shape[0]),
                    inputs=[data.mocap_pos, rows, ids, self._device_array(positions, wp.vec3f)],
                    device=self._wp_device,
                )
        mjw.forward(self.sim.mjw_model, data)
        self.controller.reset_at(indices, seeds)
        self._groom.reset_rows(indices)
        self._vision_readouts_valid = False
        if self.reset_rendering is not None:
            self.reset_rendering.refresh(indices)
        return self._observations(indices)

    def step(self, forces: np.ndarray, groom_flags: np.ndarray) -> dict[str, np.ndarray]:
        """Run one decision for every world; ``forces`` is ``(num_envs, 2)``."""
        groom_flags = np.asarray(groom_flags, dtype=bool)
        self._groom.update_active_rows(groom_flags, self._root_qposadr)
        self._actions_wp.assign(np.asarray(forces, dtype=np.float32))
        if self._decision_graph is None:
            self._run_decision_physics()
        else:
            wp.capture_launch(self._decision_graph)
        self.physics_recovery.check_decision()
        self._groom.finish_decision(groom_flags)
        mjw.forward(self.sim.mjw_model, self.sim.mjw_data)
        self._vision_readouts_valid = False
        if self._recording_renderer is not None:
            self._render_recording_substeps()
        return self._observations(np.arange(self.num_envs, dtype=np.int32))

    def fly_positions(self) -> np.ndarray:
        positions = self.sim.get_body_positions(self.fly.name).numpy()
        return positions[:, self._thorax_body_idx, :].astype(np.float32)

    def close(self) -> None:
        if self._recording_model_path is not None and self._recording_model_path.exists():
            self._recording_model_path.unlink()
        self.sim.close()

    def _run_decision_physics(self) -> None:
        for _ in range(self.config.decision_steps):
            joint_angles, adhesion = self.controller.step(self._actions_wp)
            if self._groom.n_head_joints > 0:
                wp.launch(
                    preserve_head_from_ctrl_kernel,
                    dim=(self.num_envs, self._groom.n_head_joints),
                    inputs=[
                        joint_angles,
                        self.sim.mjw_data.ctrl,
                        self._groom._head_indices_wp,
                        self._groom._position_actuator_ids_wp,
                    ],
                    device=self._wp_device,
                )
            joint_angles, adhesion = self._groom.step(
                self.sim.mjw_data.time,
                joint_angles,
                adhesion,
                self._root_qposadr,
                self._root_qveladr,
            )
            self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, joint_angles)
            self.sim.set_leg_adhesion_states(self.fly.name, adhesion)
            self.sim.step()

    def _observations(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        components = self.config.observation_components
        data = self.sim.mjw_data
        observations = {}
        if ODOR_OBSERVATION in components:
            sites = data.site_xpos.numpy()[indices[:, np.newaxis], self._olfaction_site_ids]
            observations[ODOR_OBSERVATION] = odor_observation(
                sites,
                self.current_odor_source[indices],
                self.config.peak_odor_intensity,
                log_transform=self.config.odor_log_transform,
            )
        if FLY_VISION_OBSERVATION in components:
            self._refresh_vision_readouts()
            pixels = self._gpu_retina.pixels_to_numpy(self._vision_pixels)[indices]
            observations[FLY_VISION_OBSERVATION] = np.clip(pixels, 0.0, 1.0)
        if PROPRIOCEPTION_OBSERVATION in components:
            observations[PROPRIOCEPTION_OBSERVATION] = proprioception_observation(
                data.qpos.numpy()[indices],
                data.qvel.numpy()[indices],
                root_qposadr=self._root_qposadr,
                root_qveladr=self._root_qveladr,
                joint_qposadrs=self._joint_qposadrs,
                joint_qveladrs=self._joint_qveladrs,
            )
        return observations

    def _refresh_vision_readouts(self) -> None:
        """Batch-render all eye cameras and turn them into ommatidia readouts and pixels."""
        if self._vision_renderer is None or self._vision_readouts_valid:
            return
        renderer = self._vision_renderer
        context = renderer._rendering_context
        mjw.refit_bvh(renderer.mjw_model, self.sim.mjw_data, context)
        mjw.render(renderer.mjw_model, self.sim.mjw_data, context)
        self._gpu_retina.accumulate_from_packed(
            context.rgb_data,
            context.rgb_adr,
            renderer._world_ids_gpu,
            renderer._enabled_cam_ids_gpu,
            int(self._retina.ncols),
            self._vision_readouts,
            num_worlds=self.num_envs,
        )
        self._gpu_retina.downsample_to_pixels(
            self._vision_readouts, self._vision_pixels, num_worlds=self.num_envs
        )
        self._vision_readouts_valid = True

    # Eval recording: world 0 is replayed on a CPU MuJoCo copy of the model (with the eval
    # camera, which is kept out of the GPU model so the batch renderer sees only the eyes).

    def _snapshot_recording_model(self) -> None:
        eval_camera = add_eval_camera(self.world, self.fly, self.config)
        self._recording_camera_name = eval_camera.name
        fd, path = tempfile.mkstemp(suffix=".mjb", prefix="warp_recording_")
        os.close(fd)
        self._recording_model_path = Path(path)
        recording_mj_model, _ = self.world.compile()
        mj.mj_saveModel(recording_mj_model, str(self._recording_model_path))
        self._recording_mj_model = mj.MjModel.from_binary_path(str(self._recording_model_path))
        self._recording_mj_data = mj.MjData(self._recording_mj_model)
        self.world.mjcf_root.delete(eval_camera)

    def reset_recordings(self) -> None:
        self._recording_prev_state = None
        if self._recording_renderer is not None:
            self._recording_renderer.reset()
            self._sync_recording_state()
            self._recording_prev_state = self._copy_recording_state()
            self._recording_renderer.render_as_needed(self._recording_mj_data)

    def save_recordings(self, video_path: str | Path) -> None:
        if self._recording_renderer is None:
            raise RuntimeError("render_mode='rgb_array' is required to save Warp recordings.")
        self._recording_renderer.save_video(Path(video_path))

    def _copy_recording_state(self) -> dict:
        data = self._recording_mj_data
        return {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(),
            "mocap_pos": data.mocap_pos.copy(),
            "time": float(data.time),
        }

    def _render_recording_substeps(self) -> None:
        """Render frames between the last two decisions by interpolating world 0's state."""
        renderer = self._recording_renderer
        data = self._recording_mj_data
        self._sync_recording_state()
        curr = self._copy_recording_state()
        prev = self._recording_prev_state
        self._recording_prev_state = curr
        if prev is None:
            renderer.render_as_needed(data)
            return
        start_time, end_time = prev["time"], curr["time"]
        if end_time <= start_time + 1e-12:
            return
        interval = renderer._secs_between_renders
        t = start_time
        while t < end_time - 1e-9:
            t = min(t + interval, end_time)
            alpha = (t - start_time) / (end_time - start_time)
            for key in ("qpos", "qvel", "ctrl", "mocap_pos"):
                getattr(data, key)[:] = (1.0 - alpha) * prev[key] + alpha * curr[key]
            data.time = t
            mj.mj_forward(self._recording_mj_model, data)
            # Writes the renderer's private clock so render_as_needed renders this frame.
            renderer._last_render_time_sec = t - interval - 1e-9
            renderer.render_as_needed(data)

    def _sync_recording_state(self) -> None:
        data = self._recording_mj_data
        mj.mj_resetData(self._recording_mj_model, data)
        mjw_data = self.sim.mjw_data
        data.qpos[:] = mjw_data.qpos.numpy()[0]
        data.qvel[:] = mjw_data.qvel.numpy()[0]
        data.ctrl[:] = mjw_data.ctrl.numpy()[0]
        if data.mocap_pos.size:
            data.mocap_pos[:] = mjw_data.mocap_pos.numpy()[0]
        data.time = float(mjw_data.time.numpy()[0])
        mj.mj_forward(self._recording_mj_model, data)


class GPUOpenLoopTurningController:
    """Warp port of FlyGym's tripod CPG + preprogrammed steps (no stumbling corrections)."""

    def __init__(
        self,
        num_envs: int,
        timestep: float,
        output_dof_order,
        action_bounds: tuple[float, float],
        device,
    ) -> None:
        self.num_envs = int(num_envs)
        self.timestep = float(timestep)
        self._device = device
        self.output_dof_order = list(output_dof_order)
        self.preprogrammed_steps = PreprogrammedSteps()
        network = make_tripod_cpg_network(timestep)

        def array(values, dtype=wp.float32):
            return wp.array(np.asarray(values), dtype=dtype, device=device)

        def zeros(columns):
            return wp.zeros((self.num_envs, columns), dtype=wp.float32, device=device)

        self.phases = zeros(_N_LEGS)
        self.magnitudes = zeros(_N_LEGS)
        self._next_phases = zeros(_N_LEGS)
        self._next_magnitudes = zeros(_N_LEGS)
        self.joint_angles = zeros(len(self.output_dof_order))
        self.adhesion = zeros(_N_LEGS)
        self._base_intrinsic_freqs = array(network.intrinsic_freqs.astype(np.float32))
        self._coupling_weights = array(network.coupling_weights.astype(np.float32))
        self._phase_biases = array(network.phase_biases.astype(np.float32))
        self._convergence_coefs = array(network.convergence_coefs.astype(np.float32))
        spline_coeffs, neutral_angles, self._phase_grid_step = (
            self._make_step_interpolation_tables()
        )
        self._n_intervals = spline_coeffs.shape[3]
        self._spline_coeffs = array(spline_coeffs)
        self._neutral_angles = array(neutral_angles)
        self._dof_to_output_idx_wp = array(self._make_dof_to_output_idx(), wp.int32)
        self._swing_periods = array(self._make_swing_periods())
        self._action_low = float(action_bounds[0])
        self._action_high = float(action_bounds[1])
        self.reset_at(np.arange(self.num_envs, dtype=np.int32), [None] * self.num_envs)

    def reset_at(self, indices: np.ndarray, seeds: list[int | None]) -> None:
        """Random initial phases per row, drawn like FlyGym's CPU controller for ``seed``."""
        indices = np.asarray(indices, dtype=np.int32)
        reset_phases = np.empty((len(indices), _N_LEGS), dtype=np.float32)
        for row, seed in enumerate(seeds):
            rng = np.random.RandomState(seed)
            reset_phases[row] = rng.random(_N_LEGS).astype(np.float32) * np.float32(_TWO_PI)
        wp.launch(
            _reset_cpg_rows_kernel,
            dim=(len(indices), _N_LEGS),
            inputs=[
                wp.array(indices, dtype=wp.int32, device=self._device),
                wp.array(reset_phases, dtype=wp.float32, device=self._device),
                self.phases,
                self.magnitudes,
            ],
            device=self._device,
        )

    def step(self, descending_signals) -> tuple[wp.array, wp.array]:
        wp.launch(
            _cpg_update_kernel,
            dim=(self.num_envs, _N_LEGS),
            inputs=[
                descending_signals,
                self.phases,
                self.magnitudes,
                self._base_intrinsic_freqs,
                self._coupling_weights,
                self._phase_biases,
                self._convergence_coefs,
                self._next_phases,
                self._next_magnitudes,
                self.timestep,
                self._action_low,
                self._action_high,
            ],
            device=self._device,
        )
        wp.launch(
            _cpg_output_kernel,
            dim=(self.num_envs, _N_LEGS),
            inputs=[
                self.phases,
                self.magnitudes,
                self._next_phases,
                self._next_magnitudes,
                self._spline_coeffs,
                self._neutral_angles,
                self._dof_to_output_idx_wp,
                self._swing_periods,
                self.joint_angles,
                self.adhesion,
                self._phase_grid_step,
                self._n_intervals,
            ],
            device=self._device,
        )
        return self.joint_angles, self.adhesion

    def _make_step_interpolation_tables(self) -> tuple[np.ndarray, np.ndarray, float]:
        n_intervals = self.preprogrammed_steps._length - 1
        coeffs = np.empty((_N_LEGS, _N_STEP_DOFS, 4, n_intervals), dtype=np.float32)
        neutral_angles = np.empty((_N_LEGS, _N_STEP_DOFS), dtype=np.float32)
        for leg_idx, leg in enumerate(LEGS):
            coeffs[leg_idx] = np.transpose(self.preprogrammed_steps._psi_funcs[leg].c, (2, 0, 1))
            neutral_angles[leg_idx] = self.preprogrammed_steps.neutral_pos[leg][:, 0]
        return coeffs, neutral_angles, float(_TWO_PI / n_intervals)

    def _make_dof_to_output_idx(self) -> np.ndarray:
        dof_to_output_idx = {dof: idx for idx, dof in enumerate(self.output_dof_order)}
        table = np.full((_N_LEGS, _N_STEP_DOFS), -1, dtype=np.int32)
        for leg_idx, leg in enumerate(LEGS):
            for dof_idx, dof_spec in enumerate(self.preprogrammed_steps.dofs_per_leg):
                output_idx = dof_to_output_idx.get(dof_spec_to_jointdof(leg, dof_spec))
                if output_idx is not None:
                    table[leg_idx, dof_idx] = output_idx
        return table

    def _make_swing_periods(self) -> np.ndarray:
        swing_periods = np.asarray(
            [self.preprogrammed_steps.swing_period[leg] for leg in LEGS], dtype=np.float32
        )
        swing_periods[:, 1] += np.float32(np.pi / 4)
        return swing_periods

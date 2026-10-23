"""Simulator config and the CPU MuJoCo simulation of one fly in the odor/obstacle arena."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import ActuatorType
from flygym.vision.retina import Retina
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    apply_locomotion_action,
    get_default_locomotion_dof_order,
)
from flygym_demo.complex_terrain.common import LocomotionAction

from envs.layouts import (
    DEFAULT_OBSTACLE_POSITIONS,
    SINGLE_ODOR_PEAK_INTENSITY,
    SINGLE_ODOR_SOURCE,
    EpisodeLayout,
)
from envs.observations import (
    DEFAULT_OBSERVATION_COMPONENTS,
    DEFAULT_RGB_SIZE,
    FLY_VISION_OBSERVATION,
    ODOR_OBSERVATION,
    PROPRIOCEPTION_OBSERVATION,
    finite_float32,
    make_multimodal_observation_space,
    normalize_observation_components,
    normalize_rgb_size,
    odor_observation,
    proprioception_observation,
)
from envs.world import (
    EVAL_CAMERAS,
    MULTICCD_MODES,
    actuated_joint_addresses,
    add_eval_camera,
    build_scene,
    mocap_ids,
    root_addresses,
    yaw_to_quat_wxyz,
)
from fly_agent.odor.olfaction import olfaction_site_ids
from fly_agent.primitives.groom_replay import (
    GROOM_STARTUP_BLEND_SECONDS,
    GroomReplay,
    GroomReplayConfig,
    build_groom_actuated_indices,
    build_head_actuated_indices,
)
from fly_agent.vision.sample_map import (
    ommatidia_readouts_to_pixel_observation,
    raw_image_to_hex_pxls,
)

DEFAULT_GROOM_REPLAY_PATH = Path(__file__).resolve().parents[1] / "data" / "grooming.npz"
# Eval video settings: (height, width) pixels, playback speed relative to real time, fps.
RECORDING_CAMERA_RES = (608, 800)
RECORDING_PLAYBACK_SPEED = 0.2
RECORDING_FPS = 24


@dataclass(slots=True)
class FlyBodyEnvConfig:
    """Physics, sensing and arena geometry shared by the CPU and Warp simulators."""

    backend: str = "cpu"
    timestep: float = 1e-4
    decision_steps: int = 500
    max_decision_steps: int = 100
    ccd_iterations: int = 300
    multiccd: str = "model"
    max_contacts: int = 500
    max_constraints: int = 500
    noslip_iterations: int = 0
    enable_controller_corrections: bool = False
    floor_contact_body_preset: str = "tibia_tarsus_only"
    arena_mode: str = "flat_odor_obstacles"
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.8)
    spawn_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    odor_source: np.ndarray = field(default_factory=SINGLE_ODOR_SOURCE.copy)
    peak_odor_intensity: np.ndarray = field(default_factory=SINGLE_ODOR_PEAK_INTENSITY.copy)
    marker_colors: tuple[tuple[float, float, float, float], ...] | None = None
    obstacle_positions: np.ndarray = field(default_factory=DEFAULT_OBSTACLE_POSITIONS.copy)
    obstacle_radius: float = 1.0
    obstacle_height: float = 4.0
    obstacle_colors: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    observation_components: tuple[str, ...] = DEFAULT_OBSERVATION_COMPONENTS
    rgb_size: tuple[int, int] = DEFAULT_RGB_SIZE
    odor_log_transform: bool = False
    eval_camera: str = "navigation_birdeye"
    groom_replay_path: Path = DEFAULT_GROOM_REPLAY_PATH
    groom_startup_blend_seconds: float = GROOM_STARTUP_BLEND_SECONDS

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "warp"}:
            raise ValueError("backend must be one of: cpu, warp")
        if self.arena_mode not in {"flat_odor", "flat_odor_obstacles"}:
            raise ValueError(f"Unsupported arena mode: {self.arena_mode}")
        if self.multiccd not in MULTICCD_MODES:
            raise ValueError(f"multiccd must be one of {MULTICCD_MODES}")
        if self.eval_camera not in EVAL_CAMERAS:
            raise ValueError(f"eval_camera must be one of {EVAL_CAMERAS}")
        if self.backend == "warp" and self.noslip_iterations != 0:
            raise ValueError("MJWarp does not support nonzero noslip_iterations")
        if self.backend == "warp" and self.enable_controller_corrections:
            raise NotImplementedError(
                "Warp controller corrections are not implemented; set "
                "enable_controller_corrections=false"
            )
        self.observation_components = normalize_observation_components(self.observation_components)
        self.rgb_size = normalize_rgb_size(self.rgb_size)
        self.odor_source = np.asarray(self.odor_source, dtype=np.float32)
        self.peak_odor_intensity = np.asarray(self.peak_odor_intensity, dtype=np.float32)
        self.obstacle_positions = np.asarray(self.obstacle_positions, dtype=np.float32)
        if self.odor_source.ndim != 2 or self.odor_source.shape[1] != 3:
            raise ValueError("odor_source must have shape (n_sources, 3)")
        if self.peak_odor_intensity.shape[0] != self.odor_source.shape[0]:
            raise ValueError("odor_source and peak_odor_intensity must have matching rows")
        if self.obstacle_positions.ndim != 2 or self.obstacle_positions.shape[1] != 2:
            raise ValueError("obstacle_positions must have shape (n_obstacles, 2)")
        self.groom_replay_path = Path(self.groom_replay_path)
        if not self.groom_replay_path.exists():
            raise FileNotFoundError(f"Groom replay source {self.groom_replay_path} does not exist")

    def groom_replay_config(self) -> GroomReplayConfig:
        return GroomReplayConfig(
            replay_path=self.groom_replay_path,
            startup_blend_seconds=self.groom_startup_blend_seconds,
        )


class FlyBodyNavigationEnv:
    """One fly walking (hybrid CPG controller) or grooming (replayed clip) on CPU MuJoCo.

    ``reset`` places the fly, odor sources and obstacles of an ``EpisodeLayout``; ``step``
    runs one decision of ``decision_steps`` physics steps. Rewards live in
    ``envs.training.FlyBodyNavigationTrainEnv``.
    """

    def __init__(self, config: FlyBodyEnvConfig, render_mode: str | None = None) -> None:
        if render_mode not in {None, "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.config = config
        self.fly, self.world = build_scene(config, colorize=render_mode is not None)
        self.camera = None if render_mode is None else add_eval_camera(self.world, self.fly, config)
        self.sim = Simulation(self.world, timestep=config.timestep)
        self.renderer = None
        if self.camera is not None:
            self.renderer = self.sim.set_renderer(
                self.camera,
                camera_res=RECORDING_CAMERA_RES,
                playback_speed=RECORDING_PLAYBACK_SPEED,
                output_fps=RECORDING_FPS,
            )

        actuated_dofs = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        leg_dofs = get_default_locomotion_dof_order()
        self._leg_indices = [idx for idx, dof in enumerate(actuated_dofs) if dof in leg_dofs]
        self.controller = HybridTurningController(
            timestep=config.timestep,
            output_dof_order=leg_dofs,
            max_correction=80.0 if config.enable_controller_corrections else 0.0,
        )
        actuated_names = [dof.name for dof in actuated_dofs]
        self._groom_replay = GroomReplay(
            config.groom_replay_config(),
            groom_actuated_indices=build_groom_actuated_indices(actuated_names),
        )
        self._head_indices = build_head_actuated_indices(actuated_names)
        self._last_joint_angles: np.ndarray | None = None
        self._groom_root_anchor: np.ndarray | None = None

        model = self.sim.mj_model
        self._root_qposadr, self._root_qveladr = root_addresses(model)
        self._joint_qposadrs, self._joint_qveladrs = actuated_joint_addresses(model, self.fly)
        self._odor_mocap_ids = mocap_ids(model, self.world.odor_marker_body_names())
        self._obstacle_mocap_ids = mocap_ids(model, self.world.obstacle_body_names())
        self._olfaction_site_ids = olfaction_site_ids(self.sim, self.fly)
        self._thorax_idx = self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        self._retina = Retina() if FLY_VISION_OBSERVATION in config.observation_components else None
        self.sim.retina = self._retina
        self.observation_space = make_multimodal_observation_space(
            config.observation_components,
            rgb_size=config.rgb_size,
            proprioception_dim=2 * len(self._joint_qposadrs) + 10,
        )
        self.layout: EpisodeLayout | None = None

    def reset(self, layout: EpisodeLayout, *, seed: int | None = None) -> dict[str, np.ndarray]:
        self.sim.reset()
        self.controller.reset(seed=seed)
        self._groom_replay.reset()
        self._last_joint_angles = None
        self._groom_root_anchor = None
        self.layout = layout
        data = self.sim.mj_data
        data.mocap_pos[self._odor_mocap_ids] = layout.odor_source
        data.mocap_pos[self._obstacle_mocap_ids, :2] = layout.obstacle_positions
        data.mocap_pos[self._obstacle_mocap_ids, 2] = self.config.obstacle_height / 2
        data.qpos[self._root_qpos()] = (*layout.spawn_position, *yaw_to_quat_wxyz(layout.spawn_yaw))
        self._zero_root_velocity()
        mj.mj_forward(self.sim.mj_model, data)
        return self._observation()

    def step(self, forces: np.ndarray, *, groom: bool = False) -> dict[str, np.ndarray]:
        forces = np.asarray(forces, dtype=np.float32)
        for _ in range(self.config.decision_steps):
            if groom:
                if not self._groom_replay.active:
                    self._activate_groom()
                action = self._groom_replay.make_locomotion_action(float(self.sim.time))
            else:
                if self._groom_replay.active:
                    self._deactivate_groom()
                controller_obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
                action = self._full_pose_action(self.controller.step(forces, controller_obs))
            self._anchor_root_while_grooming()
            apply_locomotion_action(self.sim, self.fly.name, action)
            self._anchor_root_while_grooming()
            self._last_joint_angles = np.asarray(action.joint_angles, dtype=np.float32).copy()
            self.sim.step()
            if self.renderer is not None:
                self.sim.render_as_needed()
        return self._observation()

    def fly_position(self) -> np.ndarray:
        return self.sim.get_body_positions(self.fly.name)[self._thorax_idx].astype(np.float32)

    def save_video(self, path: str | Path) -> None:
        if self.renderer is None:
            raise RuntimeError("render_mode='rgb_array' is required to save video.")
        self.renderer.save_video(Path(path))

    def reset_recordings(self) -> None:
        if self.renderer is not None:
            self.renderer.reset()

    def close(self) -> None:
        self.sim.close()

    def _observation(self) -> dict[str, np.ndarray]:
        components = self.config.observation_components
        data = self.sim.mj_data
        observation = {}
        if ODOR_OBSERVATION in components:
            observation[ODOR_OBSERVATION] = odor_observation(
                data.site_xpos[self._olfaction_site_ids][np.newaxis],
                self.layout.odor_source[np.newaxis],
                self.config.peak_odor_intensity,
                log_transform=self.config.odor_log_transform,
            )[0]
        if FLY_VISION_OBSERVATION in components:
            raw_vision = self.sim.get_raw_vision(self.fly.name)
            readouts = finite_float32(
                [raw_image_to_hex_pxls(self._retina, eye) for eye in raw_vision]
            )
            observation[FLY_VISION_OBSERVATION] = ommatidia_readouts_to_pixel_observation(
                self._retina, np.clip(readouts, 0.0, 1.0), self.config.rgb_size
            )
        if PROPRIOCEPTION_OBSERVATION in components:
            observation[PROPRIOCEPTION_OBSERVATION] = proprioception_observation(
                data.qpos[np.newaxis],
                data.qvel[np.newaxis],
                root_qposadr=self._root_qposadr,
                root_qveladr=self._root_qveladr,
                joint_qposadrs=self._joint_qposadrs,
                joint_qveladrs=self._joint_qveladrs,
            )[0]
        return observation

    def _full_pose_action(self, leg_action: LocomotionAction) -> LocomotionAction:
        """Expand the controller's leg-only pose to all actuators, holding the others."""
        if self._last_joint_angles is not None:
            pose = self._last_joint_angles.copy()
        else:
            actuator_ids = self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][
                self.fly.name
            ]
            pose = np.asarray(self.sim.mj_data.ctrl[actuator_ids], dtype=np.float32).copy()
        pose[self._leg_indices] = leg_action.joint_angles
        return LocomotionAction(joint_angles=pose, adhesion_onoff=leg_action.adhesion_onoff)

    def _root_qpos(self) -> slice:
        return slice(self._root_qposadr, self._root_qposadr + 7)

    def _zero_root_velocity(self) -> None:
        self.sim.mj_data.qvel[self._root_qveladr : self._root_qveladr + 6] = 0.0

    def _anchor_root_while_grooming(self) -> None:
        """Pin the body in place while the grooming clip plays."""
        if self._groom_root_anchor is None or not self._groom_replay.active:
            return
        self.sim.mj_data.qpos[self._root_qpos()] = self._groom_root_anchor
        self._zero_root_velocity()

    def _activate_groom(self) -> None:
        self._groom_root_anchor = self.sim.mj_data.qpos[self._root_qpos()].copy()
        if self._last_joint_angles is not None:
            support_pose = self._last_joint_angles.copy()
        else:
            controller_obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            leg_action = self.controller.step(np.zeros(2, dtype=np.float32), controller_obs)
            support_pose = self._full_pose_action(leg_action).joint_angles
        self._zero_root_velocity()
        self._groom_replay.activate(support_pose, sim_time=float(self.sim.time))

    def _deactivate_groom(self) -> None:
        """Return the head to its pre-groom pose, then hand control back to walking."""
        support_pose = self._groom_replay.support_pose
        restored = (
            support_pose.copy()
            if self._last_joint_angles is None
            else self._last_joint_angles.copy()
        )
        restored[self._head_indices] = support_pose[self._head_indices]
        self._last_joint_angles = restored
        apply_locomotion_action(
            self.sim,
            self.fly.name,
            LocomotionAction(
                joint_angles=restored, adhesion_onoff=self._groom_replay.adhesion_states
            ),
        )
        mj.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self._groom_replay.deactivate()
        self._groom_root_anchor = None

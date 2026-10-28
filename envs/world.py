"""Arena, fly model, cameras and model lookups shared by the CPU and Warp simulators."""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import ActuatorType, ContactParams
from flygym.utils.math import Rotation3D
from flygym.utils.mjcf import GEOM_TYPES

from envs.observations import FLY_VISION_OBSERVATION, ODOR_OBSERVATION
from fly_agent.odor.odor_world import OdorWorld
from fly_agent.odor.olfaction import add_olfaction
from fly_agent.primitives.groom_replay import make_groom_fly

EVAL_CAMERAS = ("navigation_birdeye", "follow")
BIRDEYE_FOVY = 45.0
BIRDEYE_HEIGHT_ZOOM = 0.55
MULTICCD_MODES = ("model", "enable", "disable")


class ObstacleOdorWorld(OdorWorld):
    """Odor world plus cylindrical obstacles on mocap bodies, moved every episode."""

    def __init__(
        self,
        *,
        obstacle_positions: np.ndarray,
        odor_source: np.ndarray | None = None,
        peak_odor_intensity: np.ndarray | None = None,
        marker_colors=None,
        obstacle_radius: float = 1.0,
        obstacle_height: float = 4.0,
        obstacle_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        super().__init__(
            name="obstacle_odor_world",
            odor_source=odor_source,
            peak_odor_intensity=peak_odor_intensity,
            marker_colors=marker_colors,
        )
        material = self.mjcf_root.add_material(name="obstacle", reflectance=0.1)
        self.obstacle_geoms = []
        for idx, (x, y) in enumerate(np.asarray(obstacle_positions, dtype=float)):
            body = self.mjcf_root.worldbody.add_body(
                name=f"obstacle_{idx}", mocap=True, pos=(x, y, obstacle_height / 2)
            )
            geom = body.add_geom(
                type=GEOM_TYPES["cylinder"],
                name=f"obstacle_{idx}",
                size=(obstacle_radius, obstacle_height / 2),
                rgba=tuple(float(v) for v in obstacle_color),
                material=material.name,
                contype=0,
                conaffinity=0,
            )
            # Treated as ground: the fly's floor-contact pairs also collide with obstacles.
            self.ground_geoms.append(geom)
            self.obstacle_geoms.append(geom)

    def odor_marker_body_names(self) -> list[str]:
        return [f"odor_source_marker_{idx}" for idx in range(self.num_odor_sources)]

    def obstacle_body_names(self) -> list[str]:
        return [f"obstacle_{idx}" for idx in range(len(self.obstacle_geoms))]


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def build_scene(config, *, colorize: bool):
    """Build the fly and the arena for ``config`` (a ``FlyBodyEnvConfig``)."""
    fly = make_groom_fly(colorize=colorize)
    if FLY_VISION_OBSERVATION in config.observation_components:
        fly.add_vision()
    if ODOR_OBSERVATION in config.observation_components:
        add_olfaction(fly)
    world = ObstacleOdorWorld(
        odor_source=config.odor_source,
        peak_odor_intensity=config.peak_odor_intensity,
        marker_colors=config.marker_colors,
        obstacle_positions=config.obstacle_positions,
        obstacle_radius=config.obstacle_radius,
        obstacle_height=config.obstacle_height,
        obstacle_color=config.obstacle_colors,
    )
    world.add_fly(
        fly,
        config.spawn_pos,
        Rotation3D("quat", yaw_to_quat_wxyz(config.spawn_orientation[2])),
        bodysegs_with_ground_contact=ContactBodiesPreset(config.floor_contact_body_preset),
        ground_contact_params=ContactParams(),
        add_ground_contact_sensors=False,
    )
    world.mjcf_root.option.timestep = config.timestep
    world.mjcf_root.option.ccd_iterations = config.ccd_iterations
    multiccd_bit = int(mj.mjtDisableBit.mjDSBL_MULTICCD)
    for option in (world.mjcf_root.option, fly.mjcf_root.option):
        option.noslip_iterations = config.noslip_iterations
        if config.multiccd == "disable":
            option.disableflags |= multiccd_bit
        elif config.multiccd == "enable":
            option.disableflags &= ~multiccd_bit
    return fly, world


def birdeye_camera_pose(
    *, spawn_position: np.ndarray, odor_source: np.ndarray, obstacle_positions: np.ndarray
) -> tuple[tuple[float, float, float], float]:
    """Top-down pose above the spawn point whose view covers the whole arena."""
    half_extent = 10.0
    for points in (odor_source[:, :2], obstacle_positions):
        if points.size:
            half_extent = max(half_extent, float(np.abs(points).max()))
    full_cover_height = half_extent / math.tan(math.radians(BIRDEYE_FOVY / 2.0))
    height = max(20.0, full_cover_height * BIRDEYE_HEIGHT_ZOOM + 6.0)
    return (float(spawn_position[0]), float(spawn_position[1]), height), BIRDEYE_FOVY


def add_eval_camera(world: ObstacleOdorWorld, fly, config):
    if config.eval_camera == "navigation_birdeye":
        pos, fovy = birdeye_camera_pose(
            spawn_position=np.asarray(config.spawn_pos),
            odor_source=config.odor_source,
            obstacle_positions=config.obstacle_positions,
        )
        return world.mjcf_root.worldbody.add_camera(
            name="navigation_birdeye",
            mode=mj.mjtCamLight.mjCAMLIGHT_FIXED,
            pos=pos,
            euler=(0.0, 0.0, 0.0),
            fovy=fovy,
        )
    # Tracking cameras must live in the fly subtree: world-body cameras with a
    # cross-subtree target do not update in MuJoCo.
    return fly.add_tracking_camera(
        name="navigation_follow",
        pos_offset=(0.0, -7.5, 6.0),
        rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
        fovy=30.0,
    )


def root_addresses(mj_model) -> tuple[int, int]:
    """qpos/qvel addresses of the fly's free joint."""
    free_joints = np.flatnonzero(mj_model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise RuntimeError("Expected exactly one free joint for the navigation fly.")
    return int(mj_model.jnt_qposadr[free_joints[0]]), int(mj_model.jnt_dofadr[free_joints[0]])


def actuated_joint_addresses(mj_model, fly) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = [
        mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_JOINT, fly.jointdof_to_mjcfjoint[jointdof].name)
        for jointdof in fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    ]
    if min(joint_ids) < 0:
        raise RuntimeError("Actuated joint missing from the compiled model.")
    return (
        np.asarray(mj_model.jnt_qposadr[joint_ids], dtype=np.int32),
        np.asarray(mj_model.jnt_dofadr[joint_ids], dtype=np.int32),
    )


def mocap_ids(mj_model, body_names: list[str]) -> np.ndarray:
    ids = []
    for name in body_names:
        body_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id < 0 or mj_model.body_mocapid[body_id] < 0:
            raise RuntimeError(f"Expected mocap body {name!r} in the compiled model.")
        ids.append(int(mj_model.body_mocapid[body_id]))
    return np.asarray(ids, dtype=np.int32)

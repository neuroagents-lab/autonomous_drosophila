"""Head-and-thorax-fixed fly on a free-spinning foam ball treadmill."""

from __future__ import annotations

import mujoco
import numpy as np
from flygym.anatomy import ContactBodiesPreset
from flygym.compose.fly import Fly
from flygym.compose.physics import ContactParams
from flygym.compose.world import TetheredWorld
from flygym.utils.math import Rotation3D, Vec3
from flygym.utils.mjcf import GEOM_TYPES

__all__ = [
    "BALL_DIAMETER_MM",
    "BALL_RADIUS_MM",
    "FR4615_DENSITY_G_MM3",
    "TetheredBallWorld",
]

BALL_DIAMETER_MM = 9.0
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2
FR4615_DENSITY_G_MM3 = 240.277e-6
BALL_RGBA = (1.0, 1.0, 1.0, 1.0)


class TetheredBallWorld(TetheredWorld):
    def __init__(
        self,
        name: str = "tethered_ball_world",
        *,
        ball_radius: float = BALL_RADIUS_MM,
        ball_top_z: float = 0.0,
        leg_grip_friction: float = 2.0,
        contact_preset: ContactBodiesPreset = ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        ball_density: float = FR4615_DENSITY_G_MM3,
        ball_damping: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self._ball_radius = float(ball_radius)
        self._ball_top_z = float(ball_top_z)
        self._leg_grip_friction = float(leg_grip_friction)
        self._contact_preset = ContactBodiesPreset(contact_preset)
        self._ball_joint_name = "treadmill_joint"
        self._ball_joint_id: int | None = None
        self._ball_body_id: int | None = None

        self._add_ball(float(ball_density), float(ball_damping))

    def _add_ball(self, ball_density: float, ball_damping: float) -> None:
        self._ball_body = self.mjcf_root.worldbody.add_body(
            name="treadmill",
            pos=(0.0, 0.0, self._ball_top_z - self._ball_radius),
        )
        self._ball_body.add_joint(
            name=self._ball_joint_name,
            type=mujoco.mjtJoint.mjJNT_BALL,
            damping=ball_damping,
        )
        self._set_ball_inertia(ball_density)
        self._ball_geom = self._ball_body.add_geom(
            name="treadmill",
            type=GEOM_TYPES["sphere"],
            size=(self._ball_radius, 0.0, 0.0),
            rgba=BALL_RGBA,
            contype=0,
            conaffinity=0,
        )
        self.ground_geoms = [self._ball_geom]

    def _set_ball_inertia(self, ball_density: float) -> None:
        mass = ball_density * (4.0 / 3.0) * np.pi * self._ball_radius**3
        inertia = 0.4 * mass * self._ball_radius**2
        self._ball_body.explicitinertial = True
        self._ball_body.ipos = (0.0, 0.0, 0.0)
        self._ball_body.mass = mass
        self._ball_body.inertia = (inertia, inertia, inertia)

    def _attach_fly_mjcf(
        self,
        fly: Fly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
    ):
        world_dofs = super()._attach_fly_mjcf(fly, spawn_position, spawn_rotation)
        root_body = fly.bodyseg_to_mjcfbody[fly.root_segment]
        root_body.pos = (0.0, 0.0, 0.0)
        root_body.quat = (1.0, 0.0, 0.0, 0.0)
        self._add_leg_contacts(fly)
        return world_dofs

    def _add_leg_contacts(self, fly: Fly) -> None:
        params = ContactParams()
        for seg in self._contact_preset.to_body_segments_list():
            for leg_geom in fly.bodyseg_to_mjcfgeom.get(seg, []):
                self.mjcf_root.add_pair(
                    geomname1=leg_geom.name,
                    geomname2=self._ball_geom.name,
                    name=f"{leg_geom.name}-treadmill",
                    friction=(
                        self._leg_grip_friction,
                        self._leg_grip_friction,
                        *params.get_friction_tuple()[2:],
                    ),
                    solref=params.get_solref_tuple(),
                    solimp=params.get_solimp_tuple(),
                    margin=params.margin,
                )

    def _ball_addr(self, model: mujoco.MjModel) -> tuple[int, int]:
        if self._ball_joint_id is None:
            self._ball_joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, self._ball_joint_name
            )
            if self._ball_joint_id < 0:
                raise RuntimeError(f"treadmill joint {self._ball_joint_name!r} not found in model.")
        qadr = int(model.jnt_qposadr[self._ball_joint_id])
        vadr = int(model.jnt_dofadr[self._ball_joint_id])
        return qadr, vadr

    def ball_state(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray]:
        qadr, _ = self._ball_addr(model)
        if self._ball_body_id is None:
            self._ball_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self._ball_body.name
            )
        quat = np.array(data.qpos[qadr : qadr + 4], copy=True)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._ball_body_id,
            velocity,
            0,
        )
        return quat, velocity[:3]

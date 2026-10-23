from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flygym.anatomy import (
    PASSIVE_TARSAL_LINKS,
    ActuatedDOFPreset,
    AxisOrder,
    BodySegment,
    JointDOF,
    JointPreset,
    RotationAxis,
    Skeleton,
)
from flygym.compose import ActuatorType, KinematicPosePreset, NeuroMechFly
from flygym_demo.complex_terrain.common import LocomotionAction

from fly_agent.primitives.grooming_kinematics import (
    DEFAULT_GROOM_WINDOW_LENGTH,
    load_groom_clip,
)

GROOM_STARTUP_BLEND_SECONDS = 0.05
# Leg order: lf, lm, lh, rf, rm, rh.
GROOM_ADHESION_STANDING = np.array([0, 1, 1, 0, 1, 1], dtype=np.float32)

# Recorded clip column -> FlyGym actuated DOF: head, RF forelimb, LF forelimb.
GROOM_JOINT_MAP = {
    "Angle_head_pitch": "c_thorax-c_head-pitch",
    "Angle_head_roll": "c_thorax-c_head-roll",
    "Angle_head_yaw": "c_thorax-c_head-yaw",
    "Angle_RF_ThC_yaw": "c_thorax-rf_coxa-yaw",
    "Angle_RF_ThC_pitch": "c_thorax-rf_coxa-pitch",
    "Angle_RF_ThC_roll": "c_thorax-rf_coxa-roll",
    "Angle_RF_CTr_pitch": "rf_coxa-rf_trochanterfemur-pitch",
    "Angle_RF_CTr_roll": "rf_coxa-rf_trochanterfemur-roll",
    "Angle_RF_FTi_pitch": "rf_trochanterfemur-rf_tibia-pitch",
    "Angle_RF_TiTa_pitch": "rf_tibia-rf_tarsus1-pitch",
    "Angle_LF_ThC_yaw": "c_thorax-lf_coxa-yaw",
    "Angle_LF_ThC_pitch": "c_thorax-lf_coxa-pitch",
    "Angle_LF_ThC_roll": "c_thorax-lf_coxa-roll",
    "Angle_LF_CTr_pitch": "lf_coxa-lf_trochanterfemur-pitch",
    "Angle_LF_CTr_roll": "lf_coxa-lf_trochanterfemur-roll",
    "Angle_LF_FTi_pitch": "lf_trochanterfemur-lf_tibia-pitch",
    "Angle_LF_TiTa_pitch": "lf_tibia-lf_tarsus1-pitch",
}
GROOM_CLIP_COLUMNS = tuple(GROOM_JOINT_MAP)
GROOM_FORELIMB_CLIP_COLUMNS = tuple(
    column for column in GROOM_CLIP_COLUMNS if not column.startswith("Angle_head")
)


def normalize_groom_adhesion(adhesion: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(adhesion, dtype=np.float32)
    if values.shape != (6,):
        raise ValueError(f"groom adhesion must have shape (6,); got {values.shape}")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("groom adhesion values must be in [0, 1]")
    return values


def build_groom_actuated_indices(
    actuated_joint_names: Sequence[str],
    *,
    include_head: bool = True,
) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(actuated_joint_names)}
    columns = GROOM_CLIP_COLUMNS if include_head else GROOM_FORELIMB_CLIP_COLUMNS
    indices = []
    for column in columns:
        target_name = GROOM_JOINT_MAP[column]
        try:
            indices.append(name_to_idx[target_name])
        except KeyError as exc:
            raise KeyError(
                f"Groom target joint {target_name!r} is missing from the fly actuators."
            ) from exc
    return np.asarray(indices, dtype=np.int32)


def _legs_and_head_skeleton() -> Skeleton:
    """Legs-only skeleton plus a single actuated thorax-head joint."""
    leg_joints = JointPreset.LEGS_ONLY.to_joint_list()
    head_joint = next(
        joint
        for joint in JointPreset._get_all_biological_joints()
        if joint.parent.name == "c_thorax" and joint.child.name == "c_head"
    )
    return Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        anatomical_joints=[*leg_joints, head_joint],
    )


def build_head_actuated_indices(
    actuated_joint_names: Sequence[str],
) -> np.ndarray:
    """Indices of thorax-head actuators within the full actuated pose vector."""
    groom_indices = build_groom_actuated_indices(actuated_joint_names, include_head=True)
    return groom_indices[: len(GROOM_CLIP_COLUMNS) - len(GROOM_FORELIMB_CLIP_COLUMNS)]


def make_groom_fly(
    name: str = "nmf",
    *,
    joint_stiffness: float = 0.05,
    joint_damping: float = 0.06,
    passive_tarsus_stiffness: float = 7.5,
    passive_tarsus_damping: float = 1e-2,
    actuator_gain: float = 45.0,
    actuator_forcerange: tuple[float, float] = (-65.0, 65.0),
    add_adhesion: bool = True,
    adhesion_gain: float | dict[str, float] = 40.0,
    colorize: bool = False,
) -> NeuroMechFly:
    """Locomotion fly plus actuated thorax-head joints for grooming replay."""
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = _legs_and_head_skeleton()
    fly = NeuroMechFly(name=name)
    joints = fly.add_joints(
        skeleton,
        neutral_pose=neutral_pose,
        stiffness=joint_stiffness,
        damping=joint_damping,
    )
    for jointdof, joint in joints.items():
        if jointdof.child.link in PASSIVE_TARSAL_LINKS:
            joint.stiffness[0] = passive_tarsus_stiffness
            joint.damping[0] = passive_tarsus_damping

    leg_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    head_dofs = [
        JointDOF(BodySegment("c_thorax"), BodySegment("c_head"), RotationAxis.PITCH),
        JointDOF(BodySegment("c_thorax"), BodySegment("c_head"), RotationAxis.ROLL),
        JointDOF(BodySegment("c_thorax"), BodySegment("c_head"), RotationAxis.YAW),
    ]
    actuated_dofs = leg_dofs + head_dofs
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=actuator_gain,
        forcerange=actuator_forcerange,
    )
    if add_adhesion:
        fly.add_leg_adhesion(gain=adhesion_gain)
    if colorize:
        fly.colorize()
    return fly


@dataclass(slots=True)
class GroomReplayConfig:
    replay_path: Path | None = None
    source_hz: float = 100.0
    start_frame: int | None = None
    end_frame: int | None = None
    window_length: int = DEFAULT_GROOM_WINDOW_LENGTH
    startup_blend_seconds: float = GROOM_STARTUP_BLEND_SECONDS
    adhesion: Sequence[float] | np.ndarray = field(
        default_factory=lambda: GROOM_ADHESION_STANDING.copy()
    )

    def __post_init__(self) -> None:
        if self.source_hz <= 0:
            raise ValueError("groom_source_hz must be positive")
        if self.startup_blend_seconds < 0:
            raise ValueError("groom_startup_blend_seconds must be non-negative")
        self.adhesion = normalize_groom_adhesion(self.adhesion)

    def load_clip(self, *, include_head: bool) -> np.ndarray:
        """Replay window of joint angles, ordered like ``build_groom_actuated_indices``."""
        return load_groom_clip(
            self.replay_path,
            GROOM_CLIP_COLUMNS if include_head else GROOM_FORELIMB_CLIP_COLUMNS,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            window_length=self.window_length,
        )


class GroomReplay:
    """CPU replay of the grooming clip, blended in from the pose at activation."""

    def __init__(
        self,
        config: GroomReplayConfig | None = None,
        *,
        groom_actuated_indices: np.ndarray | None = None,
        include_head: bool = True,
    ) -> None:
        self.config = config or GroomReplayConfig()
        self._clip = (
            None
            if self.config.replay_path is None
            else self.config.load_clip(include_head=include_head)
        )
        self._groom_actuated_indices = groom_actuated_indices
        if groom_actuated_indices is not None:
            self._groom_actuated_indices = np.asarray(groom_actuated_indices, dtype=np.int32)
            n_columns = len(GROOM_CLIP_COLUMNS if include_head else GROOM_FORELIMB_CLIP_COLUMNS)
            if self._groom_actuated_indices.shape != (n_columns,):
                raise ValueError(
                    f"groom_actuated_indices must have shape ({n_columns},); "
                    f"got {self._groom_actuated_indices.shape}."
                )
        self.reset()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def adhesion_states(self) -> np.ndarray:
        return self.config.adhesion.copy()

    @property
    def support_pose(self) -> np.ndarray | None:
        return None if self._support_pose is None else self._support_pose.copy()

    def reset(self) -> None:
        self._active = False
        self._phase_start_time = 0.0
        self._support_pose: np.ndarray | None = None
        self._anchor_pose: np.ndarray | None = None

    def activate(self, support_pose: np.ndarray, *, sim_time: float) -> None:
        if self._clip is None:
            raise RuntimeError("Grooming replay is not configured; set groom_replay_path.")
        if self._groom_actuated_indices is None:
            raise RuntimeError("Groom actuated joint indices have not been configured.")
        support_pose = np.asarray(support_pose, dtype=np.float32)
        if support_pose.ndim != 1:
            raise ValueError(
                f"support_pose must be one-dimensional; got shape {support_pose.shape}"
            )
        self._active = True
        self._phase_start_time = float(sim_time)
        self._support_pose = support_pose.copy()
        self._anchor_pose = support_pose[self._groom_actuated_indices]

    def deactivate(self) -> None:
        self._active = False
        self._support_pose = None
        self._anchor_pose = None

    def make_locomotion_action(self, sim_time: float) -> LocomotionAction:
        if not self._active:
            raise RuntimeError("Grooming action requested while grooming is inactive")
        elapsed_time = max(0.0, float(sim_time) - self._phase_start_time)
        groom_pose = _sample_clip_pose(self._clip, elapsed_time, self.config.source_hz)
        blend_seconds = self.config.startup_blend_seconds
        if blend_seconds > 0:
            alpha = min(1.0, elapsed_time / blend_seconds)
            groom_pose = (1.0 - alpha) * self._anchor_pose + alpha * groom_pose
        joint_angles = self._support_pose.copy()
        joint_angles[self._groom_actuated_indices] = groom_pose
        return LocomotionAction(joint_angles=joint_angles, adhesion_onoff=self.adhesion_states)


def _sample_clip_pose(
    clip: np.ndarray,
    elapsed_time: float,
    source_hz: float,
) -> np.ndarray:
    """Linearly interpolate clip frames, wrapping when elapsed time exceeds the clip."""
    n_frames = len(clip)
    wrapped_pos = (max(0.0, float(elapsed_time)) * source_hz) % n_frames
    start_idx = int(wrapped_pos)
    end_idx = (start_idx + 1) % n_frames
    alpha = wrapped_pos - start_idx
    return ((1.0 - alpha) * clip[start_idx] + alpha * clip[end_idx]).astype(np.float32)

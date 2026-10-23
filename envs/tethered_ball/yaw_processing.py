from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from envs.tethered_ball.world import BALL_RADIUS_MM

BEHAVIOR_TRACKING_HZ = 50.0
VELOCITY_FILTER_WINDOW_SECONDS = 0.5
VELOCITY_FILTER_POLYORDER = 3


def ball_angular_velocity_to_fictive_velocity(
    angular_velocity: np.ndarray, ball_radius: float = BALL_RADIUS_MM
) -> tuple[np.ndarray, np.ndarray]:
    angular_velocity = np.asarray(angular_velocity, dtype=float)
    forward_velocity = -float(ball_radius) * angular_velocity[..., 1]
    rotational_velocity = -np.degrees(angular_velocity[..., 2])
    return forward_velocity, rotational_velocity


def smooth_fictive_velocity(velocity: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smooth a °/s (or mm/s) fictive-velocity trace."""
    window_length = round(BEHAVIOR_TRACKING_HZ * VELOCITY_FILTER_WINDOW_SECONDS)
    return savgol_filter(
        np.asarray(velocity, dtype=float),
        window_length=window_length,
        polyorder=VELOCITY_FILTER_POLYORDER,
        axis=0,
    )

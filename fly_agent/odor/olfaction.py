from __future__ import annotations

import mujoco as mj
import numpy as np

# Sensor site -> (parent body, position relative to the parent body).
_OLFACTION_SENSORS = {
    "l_maxillary_palp": ("c_rostrum", (-0.15, 0.15, -0.15)),
    "r_maxillary_palp": ("c_rostrum", (-0.15, -0.15, -0.15)),
    "l_antenna": ("l_funiculus", (0.02, 0.0, -0.10)),
    "r_antenna": ("r_funiculus", (0.02, 0.0, -0.10)),
}


def add_olfaction(fly) -> dict:
    sensors = getattr(fly, "olfactionsensorname_to_mjcfsite", None)
    if sensors is None:
        fly.olfactionsensorname_to_mjcfsite = {}
    elif sensors:
        raise ValueError("Olfaction sensors have already been added.")

    for sensor_name, (parent_name, rel_pos) in _OLFACTION_SENSORS.items():
        parent_body = fly.mjcf_root.body(parent_name)
        if parent_body is None:
            raise ValueError(f"Could not find olfaction parent body {parent_name}.")
        sensor_body = parent_body.add_body(name=f"{sensor_name}_body", pos=rel_pos)
        site = sensor_body.add_site(name=sensor_name, pos=(0, 0, 0))
        fly.olfactionsensorname_to_mjcfsite[sensor_name] = site

    return fly.olfactionsensorname_to_mjcfsite


def olfaction_site_ids(sim, fly) -> np.ndarray:
    sites = getattr(fly, "olfactionsensorname_to_mjcfsite", {}).values()
    return np.asarray(
        [mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_SITE, site.name) for site in sites],
        dtype=np.int32,
    )


def olfaction_at_sources(
    sensor_positions: np.ndarray,
    odor_source: np.ndarray,
    peak_odor_intensity: np.ndarray,
) -> np.ndarray:
    """Inverse-square odor intensity, shape ``(odor_dimensions, n_sensors)``.

    ``sensor_positions`` is ``(n_sensors, 3)``, ``odor_source`` ``(n_sources, 3)``
    and ``peak_odor_intensity`` ``(n_sources, odor_dimensions)``; sources add up.
    """
    distance = np.linalg.norm(sensor_positions[np.newaxis] - odor_source[:, np.newaxis], axis=-1)
    return (peak_odor_intensity[:, :, np.newaxis] * distance[:, np.newaxis, :] ** -2).sum(axis=0)

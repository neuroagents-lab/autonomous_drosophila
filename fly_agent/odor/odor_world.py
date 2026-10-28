from __future__ import annotations

from typing import override

import numpy as np
from flygym.compose import FlatGroundWorld
from flygym.utils.mjcf import GEOM_TYPES

from fly_agent.odor.olfaction import olfaction_at_sources
from fly_agent.vision.constants import ODOR_MARKER_GEOM_GROUP

_DEFAULT_COLOR_CYCLE_RGB = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ],
    dtype=float,
)


class OdorWorld(FlatGroundWorld):
    def __init__(
        self,
        name: str = "odor_world",
        *,
        half_size: float = 1000,
        odor_source: np.ndarray | None = None,
        peak_odor_intensity: np.ndarray | None = None,
        marker_colors: np.ndarray | None = None,
        marker_size: float = 0.25,
    ) -> None:
        super().__init__(name=name, half_size=half_size)

        if odor_source is None:
            odor_source = np.array([[10, 0, 0]], dtype=float)
        else:
            odor_source = np.asarray(odor_source, dtype=float)
        if peak_odor_intensity is None:
            peak_odor_intensity = np.array([[1]], dtype=float)
        else:
            peak_odor_intensity = np.asarray(peak_odor_intensity, dtype=float)

        if odor_source.ndim != 2 or odor_source.shape[1] != 3:
            raise ValueError("odor_source must have shape (n_sources, 3).")
        if peak_odor_intensity.ndim != 2:
            raise ValueError("peak_odor_intensity must have shape (n_sources, odor_dimensions).")
        if odor_source.shape[0] != peak_odor_intensity.shape[0]:
            raise ValueError("Number of odor source locations and peak intensities must match.")

        self.odor_source = odor_source
        self.peak_odor_intensity = peak_odor_intensity
        self.num_odor_sources = odor_source.shape[0]

        if marker_colors is None:
            color_idx = np.arange(self.num_odor_sources) % len(_DEFAULT_COLOR_CYCLE_RGB)
            marker_colors = np.column_stack(
                (
                    _DEFAULT_COLOR_CYCLE_RGB[color_idx] / 255,
                    np.ones(self.num_odor_sources),
                )
            )
        marker_colors = np.asarray(marker_colors, dtype=float)
        if marker_colors.shape != (self.num_odor_sources, 4):
            raise ValueError("marker_colors must have shape (n_sources, 4).")

        for idx, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors, strict=True)):
            marker_body = self.mjcf_root.worldbody.add_body(
                name=f"odor_source_marker_{idx}",
                pos=pos,
                mocap=True,
            )
            marker_body.add_geom(
                type=GEOM_TYPES["capsule"],
                size=(marker_size, marker_size, 0),
                rgba=tuple(float(v) for v in rgba),
                group=ODOR_MARKER_GEOM_GROUP,
                contype=0,
                conaffinity=0,
            )

    @override
    def get_olfaction(
        self,
        sensor_positions: np.ndarray,
        *,
        time: float | None = None,
    ) -> np.ndarray:
        return olfaction_at_sources(sensor_positions, self.odor_source, self.peak_odor_intensity)

    @property
    @override
    def odor_dimensions(self) -> int:
        return self.peak_odor_intensity.shape[1]

"""Opt-in memory and reset-rendering controls for the Warp collector (``FlyGymWarpVecEnv``)."""

from __future__ import annotations

import copy
from typing import Any

import mujoco_warp as mjw
import numpy as np
import warp as wp

from fly_agent.vision.warp_gpu_patches import create_eye_render_context


class CollectionMemoryPolicy:
    """Configure the device-wide pool; startup cleanup runs once after replay."""

    def __init__(self, policy):
        if policy not in {"existing", "continuous", "startup"}:
            raise ValueError(f"Unknown collection memory policy: {policy}")
        self.policy = policy
        self.device = wp.get_device()
        self.previous_threshold = wp.get_mempool_release_threshold(self.device)
        if policy == "continuous":
            wp.set_mempool_release_threshold(self.device, 0)

    def after_first_decision(self):
        if self.policy == "startup":
            wp.set_mempool_release_threshold(self.device, 0)
            try:
                wp.synchronize_device(self.device)
            finally:
                wp.set_mempool_release_threshold(self.device, self.previous_threshold)


@wp.kernel
def gather_scene(
    source: wp.array2d(dtype=Any),
    rows: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=Any),
):
    row, column = wp.tid()
    out[row, column] = source[rows[row], column]


@wp.kernel
def scatter_vision(
    source: wp.array4d(dtype=wp.float32),
    rows: wp.array(dtype=wp.int32),
    count: int,
    out: wp.array4d(dtype=wp.float32),
):
    row, eye, channel, pixel = wp.tid()
    if row < count:
        out[rows[row], eye, channel, pixel] = source[row, eye, channel, pixel]


class RenderWorkspace:
    """Render only the reset worlds: gather their scene state into a fixed-capacity context."""

    def __init__(self, runner, capacity):
        if capacity < 1:
            raise ValueError("Render workspace capacity must be positive")
        self.runner = runner
        self.capacity = min(capacity, runner.num_envs)
        self.rows = wp.zeros(self.capacity, dtype=wp.int32)
        self.local_rows = wp.array(np.arange(self.capacity), dtype=wp.int32)
        self.context = create_eye_render_context(runner._vision_renderer, self.capacity)
        assert runner.sim.mjw_model.nflex == 0
        self.data = copy.copy(runner.sim.mjw_data)
        self.data.nworld = self.capacity
        self.fields = []
        for name in (
            "geom_xpos",
            "geom_xmat",
            "cam_xpos",
            "cam_xmat",
            "light_xpos",
            "light_xdir",
            "flexvert_xpos",
        ):
            source = getattr(runner.sim.mjw_data, name)
            target = wp.empty((self.capacity, source.shape[1]), dtype=source.dtype)
            setattr(self.data, name, target)
            if source.shape[1]:
                self.fields.append((source, target))
        self.readouts = runner._gpu_retina.allocate_readouts(self.capacity)
        self.pixels = runner._gpu_retina.allocate_pixel_observation(self.capacity)

    def render(self, indices):
        runner = self.runner
        indices = np.asarray(indices, dtype=np.int32)
        for start in range(0, len(indices), self.capacity):
            chunk = indices[start : start + self.capacity]
            padded = np.full(self.capacity, chunk[0], dtype=np.int32)
            padded[: len(chunk)] = chunk
            self.rows.assign(padded)
            for source, target in self.fields:
                wp.launch(gather_scene, dim=target.shape, inputs=[source, self.rows, target])
            mjw.refit_bvh(runner.sim.mjw_model, self.data, self.context)
            mjw.render(runner.sim.mjw_model, self.data, self.context)
            runner._gpu_retina.accumulate_from_packed(
                self.context.rgb_data,
                self.context.rgb_adr,
                self.local_rows,
                runner._vision_renderer._enabled_cam_ids_gpu,
                int(runner._retina.ncols),
                self.readouts,
                num_worlds=self.capacity,
            )
            runner._gpu_retina.downsample_to_pixels(
                self.readouts, self.pixels, num_worlds=self.capacity
            )
            wp.launch(
                scatter_vision,
                dim=self.pixels.shape,
                inputs=[self.pixels, self.rows, len(chunk), runner._vision_pixels],
            )


class SelectedResetRendering:
    def __init__(self, runner, capacity=128):
        self.runner = runner
        self.workspace = RenderWorkspace(runner, capacity)

    def refresh(self, indices):
        if len(indices) == self.runner.num_envs:
            self.runner._refresh_vision_readouts()
        else:
            self.workspace.render(indices)
            self.runner._vision_readouts_valid = True

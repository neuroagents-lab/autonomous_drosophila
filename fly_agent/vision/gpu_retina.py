"""GPU ommatidia readouts from MJWarp's packed batch-rendered eye images."""

from __future__ import annotations

import numpy as np
import warp as wp
from flygym.vision.retina import Retina

from fly_agent.vision.sample_map import build_downsample_ommatidium_map, build_fisheye_sample_map


@wp.kernel
def _accumulate_ommatidia_from_packed_kernel(
    packed: wp.array2d(dtype=wp.uint32),
    rgb_adr: wp.array(dtype=int),
    world_ids: wp.array(dtype=int),
    cam_ids: wp.array(dtype=int),
    src_rows: wp.array(dtype=wp.int32),
    src_cols: wp.array(dtype=wp.int32),
    ommatidia_ids: wp.array(dtype=wp.int32),
    pale_type_mask: wp.array(dtype=wp.int32),
    num_pixels_per_ommatidium: wp.array(dtype=wp.float32),
    width: int,
    num_ommatidia_per_eye: int,
    readouts: wp.array2d(dtype=wp.float32),
):
    world_idx, eye_idx, pixel_idx = wp.tid()
    ommatidium_idx = ommatidia_ids[pixel_idx]
    channel_idx = pale_type_mask[ommatidium_idx]
    pixel_id = src_rows[pixel_idx] * width + src_cols[pixel_idx]
    packed_val = packed[world_ids[world_idx], rgb_adr[cam_ids[eye_idx]] + pixel_id]
    g = wp.float32((packed_val >> wp.uint32(8)) & wp.uint32(0xFF))
    b = wp.float32(packed_val & wp.uint32(0xFF))
    value = g if channel_idx == 0 else b
    value = value / (num_pixels_per_ommatidium[ommatidium_idx] * wp.float32(255.0))
    readout_col = eye_idx * num_ommatidia_per_eye * 2 + ommatidium_idx * 2 + channel_idx
    wp.atomic_add(readouts, world_idx, readout_col, value)


@wp.kernel
def _downsample_ommatidia_kernel(
    readouts: wp.array2d(dtype=wp.float32),
    ommatidium_map: wp.array2d(dtype=wp.int32),
    num_ommatidia_per_eye: int,
    width: int,
    observation: wp.array4d(dtype=wp.float32),
):
    world_idx, eye_idx, pixel = wp.tid()
    row = pixel // width
    col = pixel % width
    ommatidium_idx = ommatidium_map[row, col]
    if ommatidium_idx < 0:
        observation[world_idx, eye_idx, 0, pixel] = 0.0
        observation[world_idx, eye_idx, 1, pixel] = 0.0
        return
    base = eye_idx * num_ommatidia_per_eye * 2 + ommatidium_idx * 2
    observation[world_idx, eye_idx, 0, pixel] = readouts[world_idx, base]
    observation[world_idx, eye_idx, 1, pixel] = readouts[world_idx, base + 1]


class GPURetina:
    """Batched GPU version of FlyGym's ``Retina`` readout and pixel downsampling.

    Readouts have shape ``(worlds, 2 eyes * ommatidia * 2 channels)``; pixel observations
    ``(worlds, 2 eyes, 2 channels, height * width)``.
    """

    def __init__(self, retina: Retina, *, output_size: tuple[int, int], device) -> None:
        self.retina = retina
        sample_map = build_fisheye_sample_map(retina)
        self.num_ommatidia_per_eye = int(retina.num_ommatidia_per_eye)
        self.num_samples = int(sample_map.src_rows.shape[0])
        self._device = device

        def array(values, dtype):
            return wp.array(np.asarray(values), dtype=dtype, device=device)

        self._src_rows = array(sample_map.src_rows, wp.int32)
        self._src_cols = array(sample_map.src_cols, wp.int32)
        self._ommatidia_ids = array(sample_map.ommatidia_ids, wp.int32)
        self._pale_type_mask = array(np.asarray(retina.pale_type_mask, dtype=np.int32), wp.int32)
        self._pixels_per_ommatidium = array(
            np.asarray(retina.num_pixels_per_ommatidia, dtype=np.float32), wp.float32
        )
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self._ommatidium_map = array(build_downsample_ommatidium_map(retina, output_size), wp.int32)

    def allocate_readouts(self, num_worlds: int) -> wp.array:
        return wp.zeros(
            (num_worlds, 2 * self.num_ommatidia_per_eye * 2), dtype=wp.float32, device=self._device
        )

    def allocate_pixel_observation(self, num_worlds: int) -> wp.array:
        height, width = self.output_size
        return wp.zeros((num_worlds, 2, 2, height * width), dtype=wp.float32, device=self._device)

    def accumulate_from_packed(
        self,
        packed: wp.array,
        rgb_adr: wp.array,
        world_ids: wp.array,
        cam_ids: wp.array,
        width: int,
        readouts: wp.array,
        *,
        num_worlds: int,
    ) -> None:
        """Average each ommatidium's packed-RGB pixels (green or blue channel by type)."""
        readouts.fill_(0.0)
        wp.launch(
            _accumulate_ommatidia_from_packed_kernel,
            dim=(num_worlds, 2, self.num_samples),
            inputs=[
                packed,
                rgb_adr,
                world_ids,
                cam_ids,
                self._src_rows,
                self._src_cols,
                self._ommatidia_ids,
                self._pale_type_mask,
                self._pixels_per_ommatidium,
                int(width),
                self.num_ommatidia_per_eye,
            ],
            outputs=[readouts],
            device=self._device,
        )

    def downsample_to_pixels(self, readouts: wp.array, observation: wp.array, *, num_worlds: int):
        height, width = self.output_size
        wp.launch(
            _downsample_ommatidia_kernel,
            dim=(num_worlds, 2, height * width),
            inputs=[readouts, self._ommatidium_map, self.num_ommatidia_per_eye, width],
            outputs=[observation],
            device=self._device,
        )

    def pixels_to_numpy(self, observation: wp.array) -> np.ndarray:
        height, width = self.output_size
        return observation.numpy().reshape(observation.shape[0], 2, 2, height, width)

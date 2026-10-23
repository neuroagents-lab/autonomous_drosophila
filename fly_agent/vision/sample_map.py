"""Fisheye sampling maps shared by CPU Retina and GPU readout kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from flygym.vision.retina import Retina


@dataclass(frozen=True)
class FisheyeSampleMap:
    src_rows: np.ndarray
    src_cols: np.ndarray
    ommatidia_ids: np.ndarray


def build_fisheye_sample_map(retina: Retina) -> FisheyeSampleMap:
    """Precompute inverse fisheye samples for fused GPU ommatidia readouts.

    Each sample maps one destination lattice pixel (post-fisheye space) to a
    source pixel in the raw MuJoCo camera frame. This is equivalent to applying
    ``Retina.correct_fisheye`` followed by ``Retina.raw_image_to_hex_pxls``, but
    without materializing the corrected image.
    """
    src_rows: list[int] = []
    src_cols: list[int] = []
    ommatidia_ids: list[int] = []
    nrows = int(retina.nrows)
    ncols = int(retina.ncols)
    ommatidia_id_map = np.asarray(retina.ommatidia_id_map)
    for dst_row in range(nrows):
        for dst_col in range(ncols):
            ommatidium_idx = int(ommatidia_id_map[dst_row, dst_col]) - 1
            if ommatidium_idx < 0:
                continue
            dst_row_norm = ((2 * dst_row - nrows) / nrows) / retina.zoom
            dst_col_norm = ((2 * dst_col - ncols) / ncols) / retina.zoom
            dst_radius_norm_sq = dst_col_norm**2 + dst_row_norm**2
            denom = 1 - (retina.distortion_coefficient * dst_radius_norm_sq) + 1e-6
            src_row_norm = dst_row_norm / denom
            src_col_norm = dst_col_norm / denom
            src_row = int(((src_row_norm + 1) * nrows) / 2)
            src_col = int(((src_col_norm + 1) * ncols) / 2)
            if 0 <= src_row < nrows and 0 <= src_col < ncols:
                src_rows.append(src_row)
                src_cols.append(src_col)
                ommatidia_ids.append(ommatidium_idx)
    return FisheyeSampleMap(
        src_rows=np.asarray(src_rows, dtype=np.int32),
        src_cols=np.asarray(src_cols, dtype=np.int32),
        ommatidia_ids=np.asarray(ommatidia_ids, dtype=np.int32),
    )


def raw_image_to_hex_pxls(retina: Retina, raw_img: np.ndarray) -> np.ndarray:
    """CPU ommatidia readout that supports resized Retina camera frames."""
    raw_img = np.asarray(raw_img)
    map_nrows, map_ncols = retina.ommatidia_id_map.shape
    raw_nrows, raw_ncols = raw_img.shape[:2]
    pixel_ids = np.asarray(retina.ommatidia_id_map, dtype=np.intp).ravel() - 1
    valid = pixel_ids >= 0
    pixel_ids = pixel_ids[valid]
    pale_type = np.asarray(retina.pale_type_mask, dtype=np.intp)[pixel_ids]
    if (raw_nrows, raw_ncols) == (map_nrows, map_ncols):
        pixels = raw_img.reshape((-1, 3))[valid]
    else:
        rows = np.repeat(np.arange(map_nrows) * raw_nrows // map_nrows, map_ncols)
        cols = np.tile(np.arange(map_ncols) * raw_ncols // map_ncols, map_nrows)
        pixels = raw_img[rows[valid], cols[valid]]
    weights = (
        pixels[np.arange(pixel_ids.size), pale_type + 1]
        / np.asarray(
            retina.num_pixels_per_ommatidia,
            dtype=np.float32,
        )[pixel_ids]
    )
    bins = np.bincount(
        pixel_ids * 2 + pale_type,
        weights=weights,
        minlength=retina.num_ommatidia_per_eye * 2,
    )
    return (bins.reshape(retina.num_ommatidia_per_eye, 2) / 255.0).astype(np.float32)


def ommatidia_readouts_to_pixel_observation(
    retina: Retina,
    readouts: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Map ommatidium readouts to pixel space and nearest-downsample to CHW."""
    height, width = output_size
    pixel_ids = build_downsample_ommatidium_map(retina, output_size)
    valid = pixel_ids >= 0

    readouts = np.asarray(readouts)
    channels = readouts.shape[-1]
    pixels = np.zeros(
        (*readouts.shape[:-2], height, width, channels),
        dtype=readouts.dtype,
    )
    pixels.reshape(*readouts.shape[:-2], height * width, channels)[..., valid.ravel(), :] = (
        readouts[..., pixel_ids[valid], :]
    )
    return np.moveaxis(pixels, -1, -3)


def build_downsample_ommatidium_map(
    retina: Retina,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor ommatidium ids for a ``(height, width)`` policy image.

    Off-lattice pixels are ``-1``. The mapping is constant for a Retina and
    output size, so it can be uploaded once and reused across environments.
    """
    height, width = output_size
    source_ids = np.asarray(retina.ommatidia_id_map, dtype=np.int32)
    source_height, source_width = source_ids.shape
    rows = (2 * np.arange(height) + 1) * source_height // (2 * height)
    cols = (2 * np.arange(width) + 1) * source_width // (2 * width)
    return source_ids[np.ix_(rows, cols)] - 1

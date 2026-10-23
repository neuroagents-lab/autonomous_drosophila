"""Load windows of the recorded grooming clip.

``data/grooming.npz`` holds anatomical joint angles in radians for every frame
of the recording (``joint_angles``, one column per name in ``columns``) and
``default_start_frame``, where the default replay window begins.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_GROOM_WINDOW_LENGTH = 35


def load_groom_clip(
    path: str | Path,
    columns: Sequence[str],
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    window_length: int = DEFAULT_GROOM_WINDOW_LENGTH,
) -> np.ndarray:
    """Return a ``(frames, len(columns))`` float32 window of joint angles.

    A missing bound is filled in from the other one and ``window_length``; with
    neither bound the window starts at the file's ``default_start_frame``.
    """
    with np.load(path, allow_pickle=False) as data:
        names = data["columns"].tolist()
        angles = data["joint_angles"]
        default_start = int(data["default_start_frame"])
    if start_frame is None:
        start_frame = default_start if end_frame is None else end_frame - window_length
    if end_frame is None:
        end_frame = start_frame + window_length
    if not 0 <= start_frame < end_frame <= len(angles):
        raise ValueError(
            f"Groom window {start_frame}:{end_frame} falls outside the {len(angles)}-frame clip"
        )
    missing = [column for column in columns if column not in names]
    if missing:
        raise KeyError(f"Missing grooming columns in {path}: {missing}")
    return angles[start_frame:end_frame, [names.index(c) for c in columns]].astype(np.float32)

"""Warp/MJWarp runtime environment helpers."""

from __future__ import annotations

import os


def warp_graph_capture_requested() -> bool:
    flag = os.environ.get("WARP_DISABLE_GRAPH_CAPTURE", "").strip().lower()
    return flag not in {"1", "true", "yes", "on"}

"""Headless, CPU-only pytest. Application code does not set these."""

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

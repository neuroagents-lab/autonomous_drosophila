"""The install can run the CPU path, and a GPU is reported only when one is present."""

import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SSRL_COMMIT = "8d1ac3ee36a1da18da01d296f4f4d92d98987996"


def _exact_pins() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pins = {}
    for dep in project["project"]["dependencies"]:
        if " @ " in dep or "==" not in dep:
            continue
        name, version = dep.split("==", 1)
        pins[name.split("[", 1)[0].strip()] = version.strip()
    return pins


def test_cpu_preflight():
    assert sys.version_info[:2] == (3, 12)
    for name, version in _exact_pins().items():
        assert importlib.metadata.version(name) == version, name

    direct = json.loads(importlib.metadata.distribution("ssrl").read_text("direct_url.json"))
    commit = (direct.get("vcs_info") or {}).get("commit_id")
    if commit is not None and "ssrl-agent" in direct.get("url", ""):
        assert commit == SSRL_COMMIT
    else:
        importlib.import_module("ssrl")

    with np.load(ROOT / "data" / "grooming.npz") as groom:
        start = int(groom["default_start_frame"])
        assert groom["joint_angles"].shape[0] >= start + 35
        assert groom["joint_angles"].shape[1] == len(groom["columns"])

    import jax
    import mujoco as mj
    import torch
    import warp as wp

    assert jax.devices()
    assert torch.zeros(1).shape == (1,)
    assert wp.get_devices()

    model = mj.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
    renderer = mj.Renderer(model, 32, 32)
    try:
        renderer.update_scene(mj.MjData(model))
        assert renderer.render().shape == (32, 32, 3)
    finally:
        renderer.close()


def test_gpu_preflight():
    import warp as wp

    if not any(device.is_cuda for device in wp.get_devices()):
        pytest.skip("no Warp CUDA device")
    import torch

    assert torch.cuda.is_available()
    # CPU JAX is the documented default. CUDA JAX is checked only after the plugin is installed.
    if importlib.util.find_spec("jax_cuda12_plugin") is None:
        return
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda"
    probe = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.devices())"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert probe.returncode == 0, probe.stderr
    assert "CudaDevice" in probe.stdout

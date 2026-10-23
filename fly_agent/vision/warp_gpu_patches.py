"""MJWarp eye-camera render context with the CPU renderer's visibility settings."""

from __future__ import annotations

import mujoco_warp as mjw

from fly_agent.vision.constants import (
    VISION_RENDER_GEOM_GROUPS,
    VISION_RENDER_SKYBOX,
    VISION_USE_AMBIENT_LIGHTING,
)


def create_eye_render_context(renderer, nworld: int):
    """Batch render context for ``renderer``'s eye cameras over ``nworld`` worlds.

    FlyGym's ``WarpGPUBatchRenderer`` cannot forward geom-group/skybox/lighting options to
    MJWarp, so callers replace ``renderer._rendering_context`` with this one.
    """
    return mjw.create_render_context(
        mjm=renderer.mj_model,
        nworld=nworld,
        cam_active=[
            renderer._cameras_id2name.get(cid) in renderer.enabled_cam_names
            for cid in range(renderer.mj_model.ncam)
        ],
        cam_res=renderer.camera_res[::-1],  # MJWarp expects (width, height).
        enabled_geom_groups=list(VISION_RENDER_GEOM_GROUPS),
        render_skybox=VISION_RENDER_SKYBOX,
        use_ambient_lighting=VISION_USE_AMBIENT_LIGHTING,
    )

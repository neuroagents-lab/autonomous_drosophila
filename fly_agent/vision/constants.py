"""Shared CPU/Warp eye-camera visibility settings."""

ODOR_MARKER_GEOM_GROUP = 1
# Eye cameras render MuJoCo geom groups 0-5 except odor markers and group 2.
EYE_VISIBLE_GEOM_GROUPS = tuple(g for g in range(6) if g not in (ODOR_MARKER_GEOM_GROUP, 2))
VISION_RENDER_GEOM_GROUPS = EYE_VISIBLE_GEOM_GROUPS
VISION_RENDER_SKYBOX = True
VISION_USE_AMBIENT_LIGHTING = False
# Warp train/eval share the same MJCF compile path; colorize is material-only.
WARP_COLORIZE_FLY = True
# Anatomical eye camera order: index 0 = left, 1 = right (matches vision.yaml).
EYE_CAMERA_NAMES = ("l_eye_cam", "r_eye_cam")

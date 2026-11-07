"""Shared geometry and camera constants, used by both Blender and the perception code.

Part frames:
  bracket: origin on the stop plate's front face, at table level, centred in y.
  profile: origin on the end face nearest the stop, at the profile's underside, centred in y.
Both have +x along the profile. The gap is the profile origin's x in the bracket frame.
"""

import math

import numpy as np

PROFILE_LENGTH_M = 0.25
PROFILE_WIDTH_M = 0.04
SLOT_WIDTH_M = 0.008
SLOT_DEPTH_M = 0.012
BRACKET_WIDTH_M = 0.07
BRACKET_PLATE_M = 0.004
BRACKET_BASE_LENGTH_M = 0.076
BRACKET_STOP_HEIGHT_M = 0.05
NOMINAL_STOP_FACE_X_M = -0.12

IMAGE_WIDTH_PX = 1280
IMAGE_HEIGHT_PX = 720
HFOV_DEG = 69.4  # Datasheet value quoted by Carfagni et al. 2019.
BASELINE_M = 0.055
DISPARITY_RANGE_PX = 126
# Shift 0 gives MinZ of about 0.40 m, closer than the assembly's near end at 0.45 m stand-off.
DISPARITY_SHIFT_PX = 50
FOCAL_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)


def intrinsics():
    """Pinhole matrix where integer pixel coordinates refer to pixel centres."""
    return np.array(
        [
            [FOCAL_PX, 0.0, (IMAGE_WIDTH_PX - 1) / 2.0],
            [0.0, FOCAL_PX, (IMAGE_HEIGHT_PX - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def depth_range_m():
    """MinZ and MaxZ from the disparity-shift formulas in Carfagni et al. 2019."""
    f_b = FOCAL_PX * BASELINE_M
    return f_b / (DISPARITY_SHIFT_PX + DISPARITY_RANGE_PX), f_b / DISPARITY_SHIFT_PX


def _box(x0, x1, y0, y1, z0, z1):
    return ((x0, y0, z0), (x1, y1, z1))


def profile_boxes():
    """Non-overlapping boxes forming a 40 mm profile with an 8 mm slot in each long face."""
    length, half = PROFILE_LENGTH_M, PROFILE_WIDTH_M / 2.0
    slot = SLOT_WIDTH_M / 2.0
    core = half - SLOT_DEPTH_M
    boxes = []
    for y0, y1 in ((-half, -slot), (slot, half)):
        for z0, z1 in ((-half, -slot), (slot, half)):
            boxes.append(_box(0, length, y0, y1, z0 + half, z1 + half))
    boxes.append(_box(0, length, -slot, slot, half - core, half + core))
    for y0, y1 in ((-core, -slot), (slot, core)):
        boxes.append(_box(0, length, y0, y1, half - slot, half + slot))
    return boxes


def bracket_boxes():
    half = BRACKET_WIDTH_M / 2.0
    return [
        _box(-BRACKET_PLATE_M, 0, -half, half, 0, BRACKET_STOP_HEIGHT_M),
        _box(0, BRACKET_BASE_LENGTH_M, -half, half, 0, BRACKET_PLATE_M),
    ]


def translation(x=0.0, y=0.0, z=0.0):
    transform = np.eye(4)
    transform[:3, 3] = (x, y, z)
    return transform


def rotation_z(angle_rad):
    transform = np.eye(4)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    transform[:2, :2] = ((c, -s), (s, c))
    return transform


def nominal_bracket_pose():
    return translation(NOMINAL_STOP_FACE_X_M, 0.0, 0.0)


def nominal_profile_pose():
    return translation(NOMINAL_STOP_FACE_X_M, 0.0, BRACKET_PLATE_M)


def gap_m(bracket_pose, profile_pose):
    """Profile origin's x in the bracket frame."""
    return float((np.linalg.inv(bracket_pose) @ profile_pose)[0, 3])


def yaw_rad(pose):
    return float(math.atan2(pose[1, 0], pose[0, 0]))

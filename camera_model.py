"""Turn ideal depth or a rendered IR pair into depth as a D415-like camera would report it.

Analytic models add noise and holes to ideal depth. The stereo model runs matching on IR images,
so its errors on metal come from the rendered light transport and are not specified by hand.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

import assembly_model as model

# Subpixel value Intel's tuning guide uses for its theoretical D415 depth error curve
# ("Tuning depth cameras for best performance").
SUBPIXEL_SIGMA_PX = 0.08
# Power law through Carfagni et al. 2019: plane std 0.345 mm at 0.2 m and 4.74 mm at 1 m.
MEASURED_NOISE_AT_REFERENCE_M = 0.345e-3
MEASURED_NOISE_REFERENCE_Z_M = 0.2
MEASURED_NOISE_EXPONENT = np.log(4.74 / 0.345) / np.log(1.0 / 0.2)
# BIDCD reports raw holes over 8.5% of the workspace image area, shadows included. Applying it
# to the part pixels alone is an assumption.
PART_DROPOUT_FRACTION = 0.085
DROPOUT_BLOB_SIGMA_PX = 4.0
EDGE_JUMP_M = 0.003
EDGE_BAND_PX = 2

# Typical small CMOS values and an exposure target; assumptions, not D415 specifications.
FULL_WELL_ELECTRONS = 10000.0
READ_NOISE_ELECTRONS = 5.0
EXPOSURE_TARGET_LEVEL = 0.75
EXPOSURE_PERCENTILE = 99.0
MIN_LEFT_MARGIN_PX = model.DISPARITY_SHIFT_PX
BLOCK_SIZE_PX = 7
NUM_DISPARITIES = 128  # The D415's 126 steps, rounded up to OpenCV's required multiple of 16.

FOCAL_BASELINE = model.FOCAL_PX * model.BASELINE_M


def datasheet_sigma(depth):
    return depth**2 * SUBPIXEL_SIGMA_PX / FOCAL_BASELINE


def measured_sigma(depth):
    relative = depth / MEASURED_NOISE_REFERENCE_Z_M
    return MEASURED_NOISE_AT_REFERENCE_M * relative**MEASURED_NOISE_EXPONENT


def occlusion_mask(depth):
    """Left-image pixels the right camera cannot see, because a nearer surface blocks them."""
    valid = depth > 0
    disparity = np.where(valid, FOCAL_BASELINE / np.maximum(depth, 1e-6), 0.0)
    columns = np.arange(depth.shape[1], dtype=np.float64)
    right_column = np.where(valid, columns - disparity, np.inf)
    nearest_to_right = np.minimum.accumulate(right_column[:, ::-1], axis=1)[:, ::-1]
    blocked = np.zeros_like(valid)
    blocked[:, :-1] = right_column[:, :-1] >= nearest_to_right[:, 1:]
    return blocked & valid


def edge_mask(depth):
    jump_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1])) > EDGE_JUMP_M
    jump_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :])) > EDGE_JUMP_M
    return binary_dilation(jump_x | jump_y, iterations=EDGE_BAND_PX)


def blob_dropouts(part_mask, rng):
    """Random blobs covering the target fraction of the metal part pixels."""
    if not part_mask.any():
        return np.zeros_like(part_mask)
    field = gaussian_filter(rng.standard_normal(part_mask.shape), DROPOUT_BLOB_SIGMA_PX)
    threshold = np.quantile(field[part_mask], 1.0 - PART_DROPOUT_FRACTION)
    return (field > threshold) & part_mask


def analytic_depth(depth, labels, sigma_function, rng):
    """Independent Gaussian depth noise, with holes at occlusions, edges and random part blobs."""
    noisy = depth + rng.standard_normal(depth.shape) * sigma_function(depth)
    part_mask = (labels == 1) | (labels == 2)
    holes = occlusion_mask(depth) | edge_mask(depth) | blob_dropouts(part_mask, rng)
    noisy[holes | (depth <= 0)] = 0.0
    return noisy.astype(np.float32)


def expose(raw, rng):
    """Auto-exposure, photon and read noise, and saturation at the sensor's full scale."""
    linear = raw.astype(np.float64) / 65535.0
    rendered = linear[linear > 0]
    gain = EXPOSURE_TARGET_LEVEL / max(np.percentile(rendered, EXPOSURE_PERCENTILE), 1e-4) if rendered.size else 1.0
    electrons = rng.poisson(np.clip(linear * gain, 0.0, None) * FULL_WELL_ELECTRONS).astype(np.float64)
    electrons += rng.normal(0.0, READ_NOISE_ELECTRONS, electrons.shape)
    return np.clip(electrons / FULL_WELL_ELECTRONS, 0.0, 1.0), gain


def make_matcher(kind):
    shared = {
        "minDisparity": model.DISPARITY_SHIFT_PX,
        "numDisparities": NUM_DISPARITIES,
        "blockSize": BLOCK_SIZE_PX,
    }
    if kind == "sgbm":
        return cv2.StereoSGBM_create(
            **shared,
            P1=8 * BLOCK_SIZE_PX**2,
            P2=32 * BLOCK_SIZE_PX**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM,
        )
    if kind == "bm":
        matcher = cv2.StereoBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE_PX)
        matcher.setMinDisparity(model.DISPARITY_SHIFT_PX)
        matcher.setDisp12MaxDiff(1)
        matcher.setUniquenessRatio(10)
        matcher.setSpeckleWindowSize(100)
        matcher.setSpeckleRange(8)
        matcher.setTextureThreshold(5)
        return matcher
    raise ValueError(f"Unknown matcher: {kind}")


def stereo_depth(left_raw, right_raw, kind, rng):
    """Expose the IR pair, match along rows, and convert valid disparities to planar depth."""
    left, _ = expose(left_raw, rng)
    right, _ = expose(right_raw, rng)
    left_8bit = np.rint(left * 255.0).astype(np.uint8)
    right_8bit = np.rint(right * 255.0).astype(np.uint8)
    disparity = make_matcher(kind).compute(left_8bit, right_8bit).astype(np.float32) / 16.0
    depth = np.zeros_like(disparity)
    valid = disparity >= model.DISPARITY_SHIFT_PX
    depth[valid] = FOCAL_BASELINE / disparity[valid]
    return depth


def read_ir(scene_dir, name):
    image = cv2.imread(str(scene_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(scene_dir / f"{name}.png")
    return image if image.ndim == 2 else image[:, :, 0]

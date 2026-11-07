"""Measure part poses from depth alone: RANSAC table removal, then PCA and ICP per part."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline import assembly_model as model

MODEL_SPACING_M = 0.0007
TABLE_THRESHOLD_M = 0.003
OUTLIER_MIN_NEIGHBOURS = 6
OUTLIER_RADIUS_M = 0.004
ROI_MIN = np.array([-0.15, -0.07, 0.0])
ROI_MAX = np.array([0.17, 0.07, 0.07])
ICP_RADII_M = (0.012, 0.006, 0.003, 0.002)
ICP_ITERATIONS = 30
ICP_STEP_LIMIT_M = 0.002
ICP_DAMPING = 1e-6
ASSIGNMENT_ROUNDS = 2
MAX_PCA_CORRECTION_DEG = 15.0
MIN_PART_POINTS = 200
# A fit beyond these limits has left the assembly's plausible range, so it counts as failed.
MAX_TRANSLATION_FROM_NOMINAL_M = 0.015
MAX_YAW_FROM_NOMINAL_DEG = 10.0
MIN_FITNESS = 0.3


@dataclass
class PartResult:
    pose: np.ndarray
    fitness: float = 0.0
    rmse_mm: float = math.nan
    points: int = 0
    pca_fallback: bool = False
    # Eigenvalues of J^T J for translation, weakest first, per observed point.
    translation_information: list = field(default_factory=list)


def load_scene(scene_dir):
    scene_dir = Path(scene_dir)
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    geometry = np.load(scene_dir / "geometry.npz")
    return geometry["depth_m"], geometry["labels"], metadata


def back_project(depth, K, world_from_camera):
    rows, cols = np.nonzero(depth > 0)
    z = depth[rows, cols].astype(np.float64)
    camera = np.column_stack(
        ((cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z, np.ones_like(z))
    )
    return (camera @ world_from_camera.T)[:, :3]


def sample_boxes(boxes):
    """Uniform surface samples with outward normals on every box face."""
    points, normals = [], []
    for low, high in boxes:
        low, high = np.asarray(low), np.asarray(high)
        for axis in range(3):
            u, v = [a for a in range(3) if a != axis]
            nu = max(int((high[u] - low[u]) / MODEL_SPACING_M), 1)
            nv = max(int((high[v] - low[v]) / MODEL_SPACING_M), 1)
            grid_u, grid_v = np.meshgrid(
                low[u] + (np.arange(nu) + 0.5) * (high[u] - low[u]) / nu,
                low[v] + (np.arange(nv) + 0.5) * (high[v] - low[v]) / nv,
            )
            for side, value in ((-1.0, low[axis]), (1.0, high[axis])):
                face = np.empty((grid_u.size, 3))
                face[:, u], face[:, v], face[:, axis] = grid_u.ravel(), grid_v.ravel(), value
                normal = np.zeros_like(face)
                normal[:, axis] = side
                points.append(face)
                normals.append(normal)
    return np.vstack(points), np.vstack(normals)


def boxes_mesh(boxes):
    mesh = o3d.geometry.TriangleMesh()
    for low, high in boxes:
        low, high = np.asarray(low), np.asarray(high)
        box = o3d.geometry.TriangleMesh.create_box(*(high - low))
        box.translate(low)
        mesh += box
    return mesh


class PartModel:
    def __init__(self, boxes):
        self.points, self.normals = sample_boxes(boxes)
        self.mesh = boxes_mesh(boxes)

    def visible(self, pose, camera_position, other_meshes=()):
        """Model samples the camera can see at this pose, in world coordinates (Kleppe et al.)."""
        world_points = self.points @ pose[:3, :3].T + pose[:3, 3]
        world_normals = self.normals @ pose[:3, :3].T
        rays_to_camera = camera_position - world_points
        distance = np.linalg.norm(rays_to_camera, axis=1)
        keep = np.sum(world_normals * rays_to_camera, axis=1) > 0.0
        world_points, world_normals, rays_to_camera, distance = (
            world_points[keep], world_normals[keep], rays_to_camera[keep], distance[keep]
        )

        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.geometry.TriangleMesh(self.mesh).transform(pose)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        for other in other_meshes:
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(other))
        directions = rays_to_camera / distance[:, None]
        origins = world_points + directions * 1e-4
        rays = o3d.core.Tensor(np.hstack((origins, directions)), dtype=o3d.core.Dtype.Float32)
        hits = scene.count_intersections(rays).numpy()
        return world_points[hits == 0], world_normals[hits == 0]


PROFILE_MODEL = PartModel(model.profile_boxes())
BRACKET_MODEL = PartModel(model.bracket_boxes())


def remove_table(points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, _ = cloud.segment_plane(TABLE_THRESHOLD_M, 3, 300, probability=0.9999)
    plane = np.asarray(plane)
    if plane[2] < 0:
        plane = -plane
    height = points @ plane[:3] + plane[3]
    return points[height > TABLE_THRESHOLD_M]


def crop_roi(points):
    return points[np.all((points >= ROI_MIN) & (points <= ROI_MAX), axis=1)]


def remove_outliers(points):
    """Drop isolated points such as false matches and leftover table noise (see BIDCD).

    A statistical filter is unsuitable here: it removes the sparsely sampled end face,
    which is the only surface that constrains the gap.
    """
    if len(points) == 0:
        return points
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, kept = cloud.remove_radius_outlier(OUTLIER_MIN_NEIGHBOURS, OUTLIER_RADIUS_M)
    return points[kept]


def pca_yaw_correction(points, nominal_rotation):
    """Yaw that aligns the nominal axes with the principal axes; None if implausibly large."""
    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    # The principal axis closest to a horizontal nominal axis defines the in-plane direction.
    horizontal = nominal_rotation[:, :2]
    overlap = np.abs(axes @ horizontal)
    principal, nominal_axis = np.unravel_index(np.argmax(overlap), overlap.shape)
    direction = axes[principal]
    reference = horizontal[:, nominal_axis]
    if direction @ reference < 0:
        direction = -direction
    angle = math.atan2(direction[1], direction[0]) - math.atan2(reference[1], reference[0])
    angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return angle if abs(math.degrees(angle)) <= MAX_PCA_CORRECTION_DEG else None


def direction_weights(model_normals, pose):
    """Balance surface directions, so few end-face points are not outvoted by the long faces."""
    local = np.abs(model_normals @ pose[:3, :3])
    dominant = np.argmax(local, axis=1)
    counts = np.bincount(dominant, minlength=3).astype(np.float64)
    present = np.count_nonzero(counts)
    return len(model_normals) / (present * counts[dominant])


def apply_planar_step(step, pivot, pose):
    """Translate in x, y, z and rotate about the vertical axis through the pivot."""
    tx, ty, tz, angle = step
    update = (
        model.translation(pivot[0] + tx, pivot[1] + ty, tz)
        @ model.rotation_z(angle)
        @ model.translation(-pivot[0], -pivot[1], 0.0)
    )
    return update @ pose


def icp(points, part, pose, camera_position, other_meshes=()):
    """Robust point-to-plane ICP over x, y, z and yaw; the parts rest on a level table."""
    inlier_fraction, rmse = 0.0, math.nan
    for radius in ICP_RADII_M:
        target_points, target_normals = part.visible(pose, camera_position, other_meshes)
        if len(target_points) == 0:
            break
        tree_target = cKDTree(target_points)
        # Visibility is fixed per scale; the visible samples move with the accumulated steps.
        cumulative = np.eye(4)
        for _ in range(ICP_ITERATIONS):
            moved = (points - cumulative[:3, 3]) @ cumulative[:3, :3]
            distance, index = tree_target.query(moved, distance_upper_bound=radius)
            matched = np.isfinite(distance)
            if matched.sum() < MIN_PART_POINTS:
                break
            model_points = (cumulative[:3, :3] @ target_points[index[matched]].T).T + cumulative[:3, 3]
            normals = (cumulative[:3, :3] @ target_normals[index[matched]].T).T
            residual = np.sum(normals * (points[matched] - model_points), axis=1)
            pivot = model_points.mean(axis=0)
            lever = model_points - pivot
            jacobian = np.column_stack(
                (normals, normals[:, 1] * lever[:, 0] - normals[:, 0] * lever[:, 1])
            )
            tukey_scale = radius / 2.0
            tukey = np.clip(1.0 - (residual / tukey_scale) ** 2, 0.0, None) ** 2
            weights = tukey * direction_weights(normals, pose)
            weighted = jacobian * weights[:, None]
            scale = np.array([1.0, 1.0, 1.0, 0.1])
            normal_matrix = (jacobian / scale).T @ (weighted / scale)
            normal_matrix += ICP_DAMPING * np.trace(normal_matrix) * np.eye(4)
            step = np.linalg.solve(normal_matrix, (weighted / scale).T @ residual) / scale
            largest = max(abs(step[0]), abs(step[1]), abs(step[2]))
            if largest > ICP_STEP_LIMIT_M:
                step *= ICP_STEP_LIMIT_M / largest
            cumulative = apply_planar_step(step, pivot, cumulative)
            inlier_fraction = float(matched.mean())
            rmse = float(np.sqrt(np.mean(residual[tukey > 0] ** 2))) if (tukey > 0).any() else math.nan
            if largest < 1e-6 and abs(step[3]) < 1e-6:
                break
        pose = cumulative @ pose
    return pose, inlier_fraction, rmse


def translation_information(points, part, pose, camera_position, other_meshes, radius=0.002):
    """Weakest-first eigenvalues of the translation block of J^T J, normalised per point."""
    target_points, target_normals = part.visible(pose, camera_position, other_meshes)
    if len(target_points) == 0:
        return [0.0, 0.0, 0.0]
    distance, index = cKDTree(target_points).query(points, distance_upper_bound=radius)
    matched = np.isfinite(distance)
    if not matched.any():
        return [0.0, 0.0, 0.0]
    normals = target_normals[index[matched]]
    return np.linalg.eigvalsh(normals.T @ normals / len(points)).tolist()


def assign_points(points, profile_pose, bracket_pose):
    """Give each point to the nearer part model."""
    profile_points = PROFILE_MODEL.points @ profile_pose[:3, :3].T + profile_pose[:3, 3]
    bracket_points = BRACKET_MODEL.points @ bracket_pose[:3, :3].T + bracket_pose[:3, 3]
    to_profile, _ = cKDTree(profile_points).query(points)
    to_bracket, _ = cKDTree(bracket_points).query(points)
    return to_profile <= to_bracket


def fit_part(points, part, nominal_pose, start_pose, camera_position, other_meshes, use_pca):
    result = PartResult(pose=start_pose.copy(), points=len(points))
    if len(points) < MIN_PART_POINTS:
        result.pca_fallback = use_pca
        return result
    pose = start_pose.copy()
    if use_pca:
        angle = pca_yaw_correction(points, nominal_pose[:3, :3])
        if angle is None:
            result.pca_fallback = True
        else:
            pivot = nominal_pose[:3, 3]
            pose = apply_planar_step((0.0, 0.0, 0.0, angle), pivot, pose)
    result.pose, result.fitness, rmse = icp(points, part, pose, camera_position, other_meshes)
    result.rmse_mm = rmse * 1000.0
    result.translation_information = translation_information(
        points, part, result.pose, camera_position, other_meshes
    )
    return result


def has_failed(result, nominal_pose):
    offset = result.pose[:3, 3] - nominal_pose[:3, 3]
    yaw = abs(math.degrees(model.yaw_rad(result.pose)))
    return bool(
        result.points < MIN_PART_POINTS
        or result.fitness < MIN_FITNESS
        or np.linalg.norm(offset[:2]) > MAX_TRANSLATION_FROM_NOMINAL_M
        or yaw > MAX_YAW_FROM_NOMINAL_DEG
    )


def measure(depth, metadata):
    K = np.array(metadata["camera"]["K"])
    world_from_camera = np.array(metadata["camera"]["world_from_camera"])
    camera_position = world_from_camera[:3, 3]

    cloud = back_project(depth, K, world_from_camera)
    points = remove_outliers(crop_roi(remove_table(cloud)))

    nominal_profile = model.nominal_profile_pose()
    nominal_bracket = model.nominal_bracket_pose()
    profile_pose, bracket_pose = nominal_profile.copy(), nominal_bracket.copy()
    profile, bracket = None, None
    for round_index in range(ASSIGNMENT_ROUNDS):
        is_profile = assign_points(points, profile_pose, bracket_pose)
        # PCA only seeds the first round; afterwards ICP continues from its own estimate.
        use_pca = round_index == 0
        bracket_mesh = o3d.geometry.TriangleMesh(BRACKET_MODEL.mesh).transform(bracket_pose)
        profile = fit_part(
            points[is_profile], PROFILE_MODEL, nominal_profile, profile_pose,
            camera_position, [bracket_mesh], use_pca,
        )
        profile_pose = profile.pose
        profile_mesh = o3d.geometry.TriangleMesh(PROFILE_MODEL.mesh).transform(profile_pose)
        bracket = fit_part(
            points[~is_profile], BRACKET_MODEL, nominal_bracket, bracket_pose,
            camera_position, [profile_mesh], use_pca,
        )
        bracket_pose = bracket.pose
        if use_pca:
            pca_fallback_profile = profile.pca_fallback
            pca_fallback_bracket = bracket.pca_fallback

    return {
        "gap_mm": model.gap_m(bracket_pose, profile_pose) * 1000.0,
        "bracket_yaw_deg": math.degrees(model.yaw_rad(bracket_pose)),
        "profile_yaw_deg": math.degrees(model.yaw_rad(profile_pose)),
        "failed": has_failed(profile, nominal_profile) or has_failed(bracket, nominal_bracket),
        "profile_fitness": profile.fitness,
        "bracket_fitness": bracket.fitness,
        "profile_rmse_mm": profile.rmse_mm,
        "bracket_rmse_mm": bracket.rmse_mm,
        "profile_points": profile.points,
        "bracket_points": bracket.points,
        "pca_fallback_profile": pca_fallback_profile,
        "pca_fallback_bracket": pca_fallback_bracket,
        "profile_translation_information": profile.translation_information,
        "bracket_translation_information": bracket.translation_information,
    }

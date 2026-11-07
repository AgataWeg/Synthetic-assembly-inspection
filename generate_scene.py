"""Render one labelled aluminium assembly scene with Blender Cycles."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly_model as model  # noqa: E402

LABELS = {"profile": 1, "bracket": 2, "table": 3, "fixture": 4}
CAMERA_DISTANCE_M = 0.45
ASSEMBLY_CENTRE = (0.005, 0.0, 0.025)
PLACEMENT_SIGMA_M = 0.0002
PLACEMENT_YAW_SIGMA_DEG = 0.1
SEATING_SIGMA_M = 0.00005
# The D415's dot count is not published in the sources used here, so this is an assumption.
PROJECTOR_DOTS = 30000
PROJECTOR_OFFSET_M = 0.03
PROJECTOR_HFOV_DEG = 80.0
PROJECTOR_VFOV_DEG = 55.0
PROJECTOR_ENERGY_W = 1.5
IR_AMBIENT_SCALE = 0.15
IR_CROP_MARGIN_PX = 40


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/generated/scene_000"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap-mm", type=float, default=0.0)
    parser.add_argument("--bracket-yaw-deg", type=float, default=0.0)
    parser.add_argument("--material", choices=("aluminium", "matte"), default="aluminium")
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--rgb-samples", type=int, default=32)
    parser.add_argument("--ir", action="store_true", help="Also render the stereo IR pair.")
    parser.add_argument("--ir-samples", type=int, default=32)
    parser.add_argument(
        "--camera-azimuth-deg", type=float, nargs=2, default=(-58.0, -42.0), metavar=("MIN", "MAX"),
        help="Range of the camera's direction around the assembly.",
    )
    parser.add_argument(
        "--camera-elevation-deg", type=float, nargs=2, default=(46.0, 58.0), metavar=("MIN", "MAX"),
        help="Range of the camera's elevation above the table.",
    )
    parser.add_argument("--plane-distance", type=float, help="Render a validation plane at this distance (m).")
    parser.add_argument("--plane-tilt-deg", type=float, default=0.0)
    return parser.parse_args(argv)


def make_material(name, color, metallic, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def make_materials(kind, rng):
    # Drawn for both materials, so a matte scene and its aluminium twin share every other value.
    profile_roughness = float(rng.uniform(0.25, 0.45))
    bracket_roughness = float(rng.uniform(0.30, 0.50))
    if kind == "matte":
        profile = make_material("Profile", (0.60, 0.60, 0.60), 0.0, 0.9)
        bracket = make_material("Bracket", (0.45, 0.45, 0.45), 0.0, 0.9)
        profile_roughness = bracket_roughness = 0.9
    else:
        profile = make_material("Profile", (0.91, 0.92, 0.92), 1.0, profile_roughness)
        bracket = make_material("Bracket", (0.60, 0.61, 0.62), 1.0, bracket_roughness)
    materials = {
        "profile": profile,
        "bracket": bracket,
        "table": make_material("Table", (0.15, 0.15, 0.16), 0.0, 0.85),
    }
    return materials, {"profile": profile_roughness, "bracket": bracket_roughness}


def add_part(boxes, pose, material, label):
    """Create one mesh object per box, in the part frame, placed by the part's pose."""
    world = Matrix(pose.tolist())
    for index, (low, high) in enumerate(boxes):
        centre = tuple((a + b) / 2.0 for a, b in zip(low, high))
        size = tuple(b - a for a, b in zip(low, high))
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre)
        obj = bpy.context.object
        obj.name = f"part_{label}_{index}"
        obj.scale = size
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
        obj.matrix_world = world
        obj.data.materials.append(material)
        obj.pass_index = label


def sample_part_poses(gap_mm, yaw_deg, rng):
    """Small placement variation plus the requested defect; parts never interpenetrate."""
    placement = (
        model.nominal_bracket_pose()
        @ model.translation(*rng.normal(0.0, PLACEMENT_SIGMA_M, 2))
        @ model.rotation_z(math.radians(rng.normal(0.0, PLACEMENT_YAW_SIGMA_DEG)))
    )
    seat = abs(rng.normal(0.0, SEATING_SIGMA_M))
    profile = (
        placement
        @ model.translation(gap_mm / 1000.0 + seat, rng.normal(0.0, SEATING_SIGMA_M), model.BRACKET_PLATE_M)
    )
    # A yawed stop pivots about the corner of the profile end it still touches.
    pivot_y = -math.copysign(model.PROFILE_WIDTH_M / 2.0, yaw_deg) if yaw_deg else 0.0
    bracket = (
        placement
        @ model.translation(0.0, pivot_y)
        @ model.rotation_z(math.radians(yaw_deg))
        @ model.translation(0.0, -pivot_y)
    )
    return bracket, profile


def add_camera(rng, azimuth_range_deg, elevation_range_deg):
    # One draw from any range, so scenes sharing a seed differ only in the camera direction.
    azimuth = math.radians(rng.uniform(*azimuth_range_deg))
    elevation = math.radians(rng.uniform(*elevation_range_deg))
    distance = CAMERA_DISTANCE_M + rng.uniform(-0.015, 0.015)
    target = Vector(ASSEMBLY_CENTRE) + Vector(rng.uniform(-0.005, 0.005, 3).tolist())
    direction = Vector(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
    )
    bpy.ops.object.camera_add(location=target + distance * direction)
    camera = bpy.context.object
    camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = 36.0 * model.FOCAL_PX / model.IMAGE_WIDTH_PX
    camera.data.clip_start = 0.05
    camera.data.clip_end = 3.0
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def add_lights(scene, rng):
    position = Vector((0.1, -0.15, 0.6)) + Vector(rng.uniform(-0.1, 0.1, 3).tolist())
    bpy.ops.object.light_add(type="AREA", location=position)
    light = bpy.context.object
    light.data.shape = "DISK"
    light.data.size = 0.45
    light.data.energy = float(rng.uniform(4.0, 7.0))
    light.rotation_euler = (Vector((0.0, 0.0, 0.02)) - position).to_track_quat("-Z", "Y").to_euler()

    world = bpy.data.worlds.new("Cell ambient")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    background.inputs["Strength"].default_value = float(rng.uniform(0.15, 0.3))


def render_rgb(scene, output_dir, samples):
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    scene.compositing_node_group = None
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output_dir / "rgb.png")
    bpy.ops.render.render(write_still=True)


def file_output_node(tree, directory, stem, source):
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.directory = str(directory)
    node.file_name = stem
    node.file_output_items.new("FLOAT", "")
    node.format.media_type = "IMAGE"
    node.format.file_format = "OPEN_EXR"
    node.format.color_mode = "BW"
    node.format.color_depth = "32"
    tree.links.new(source, node.inputs[0])


def read_channel(path):
    image = bpy.data.images.load(str(path))
    image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    pixels = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    channel = pixels.reshape(height, width, image.channels)[::-1, :, 0].copy()
    bpy.data.images.remove(image)
    return channel


def render_geometry(scene, output_dir):
    """One unfiltered sample per pixel gives exact pixel-centre depth and labels."""
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.seed = 0
    tree = bpy.data.node_groups.new("Geometry passes", "CompositorNodeTree")
    scene.compositing_node_group = tree
    layers = tree.nodes.new("CompositorNodeRLayers")
    scratch = output_dir / "_passes"
    file_output_node(tree, scratch, "depth", layers.outputs["Depth"])
    file_output_node(tree, scratch, "label", layers.outputs["Object Index"])
    scene.render.filepath = str(scratch / "unused")
    bpy.ops.render.render(write_still=False)

    depth = read_channel(next(scratch.glob("depth*.exr")))
    labels = read_channel(next(scratch.glob("label*.exr")))
    for path in scratch.glob("*"):
        path.unlink()
    scratch.rmdir()
    depth[depth > 100.0] = 0.0
    np.savez_compressed(
        output_dir / "geometry.npz",
        depth_m=depth.astype(np.float32),
        labels=np.rint(labels).astype(np.uint8),
    )


def camera_pose_opencv(camera):
    """World-from-camera with x right, y down, z forward."""
    blender_pose = np.array(camera.matrix_world)
    return blender_pose @ np.diag([1.0, -1.0, -1.0, 1.0])


def make_dot_pattern():
    """A fixed pseudo-random dot image; a real projector's pattern does not change either."""
    width, height = 2048, 1408
    pattern = np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(415)
    rows = rng.integers(2, height - 2, PROJECTOR_DOTS)
    cols = rng.integers(2, width - 2, PROJECTOR_DOTS)
    for row_offset, col_offset, weight in ((0, 0, 1.0), (1, 0, 0.5), (-1, 0, 0.5), (0, 1, 0.5), (0, -1, 0.5)):
        pattern[rows + row_offset, cols + col_offset] = np.maximum(
            pattern[rows + row_offset, cols + col_offset], weight
        )
    image = bpy.data.images.new("Dot pattern", width, height, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    rgba = np.repeat(pattern[:, :, None], 4, axis=2)
    rgba[:, :, 3] = 1.0
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def scaled_light_coordinate(nodes, links, component, half_tan):
    """Map one light-space ray component to a 0-1 texture coordinate, for a flat projected image."""
    negated_z = nodes.new("ShaderNodeMath")
    negated_z.operation = "MULTIPLY"
    links.new(component[1], negated_z.inputs[0])
    negated_z.inputs[1].default_value = -1.0
    ratio = nodes.new("ShaderNodeMath")
    ratio.operation = "DIVIDE"
    links.new(component[0], ratio.inputs[0])
    links.new(negated_z.outputs[0], ratio.inputs[1])
    remap = nodes.new("ShaderNodeMath")
    remap.operation = "MULTIPLY_ADD"
    links.new(ratio.outputs[0], remap.inputs[0])
    remap.inputs[1].default_value = 0.5 / half_tan
    remap.inputs[2].default_value = 0.5
    return remap.outputs[0]


def add_projector(camera):
    """A textured spot light beside the left imager, as in the DREDS simulator."""
    axes = np.array(camera.matrix_world)[:3, :3]
    bpy.ops.object.light_add(type="SPOT")
    projector = bpy.context.object
    projector.name = "ir_projector"
    projector.rotation_euler = camera.matrix_world.to_euler()
    projector.location = camera.matrix_world.translation + Vector((axes[:, 0] * PROJECTOR_OFFSET_M).tolist())
    light = projector.data
    light.spot_size = math.radians(120.0)
    light.spot_blend = 0.0
    light.shadow_soft_size = 0.0
    light.energy = PROJECTOR_ENERGY_W
    light.use_nodes = True
    nodes, links = light.node_tree.nodes, light.node_tree.links
    coordinates = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coordinates.outputs["Normal"], separate.inputs[0])
    x_ray = (separate.outputs["X"], separate.outputs["Z"])
    y_ray = (separate.outputs["Y"], separate.outputs["Z"])
    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(
        scaled_light_coordinate(nodes, links, x_ray, math.tan(math.radians(PROJECTOR_HFOV_DEG / 2.0))),
        combine.inputs[0],
    )
    links.new(
        scaled_light_coordinate(nodes, links, y_ray, math.tan(math.radians(PROJECTOR_VFOV_DEG / 2.0))),
        combine.inputs[1],
    )
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = make_dot_pattern()
    texture.interpolation = "Linear"
    texture.extension = "CLIP"
    links.new(combine.outputs[0], texture.inputs["Vector"])
    emission = next(node for node in nodes if node.type == "EMISSION")
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    return projector


def crop_border(scene, world_from_camera, points):
    """Restrict rendering to where the given world points appear in either IR view."""
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = (np.linalg.inv(world_from_camera) @ homogeneous.T).T
    K = model.intrinsics()
    rows = K[1, 1] * camera_points[:, 1] / camera_points[:, 2] + K[1, 2]
    columns = np.concatenate(
        [
            K[0, 0] * (camera_points[:, 0] + offset) / camera_points[:, 2] + K[0, 2]
            for offset in (0.0, -model.BASELINE_M)
        ]
    )
    width, height = model.IMAGE_WIDTH_PX, model.IMAGE_HEIGHT_PX
    margin = IR_CROP_MARGIN_PX
    scene.render.use_border = True
    scene.render.use_crop_to_border = False
    scene.render.border_min_x = max(0.0, (columns.min() - margin) / width)
    scene.render.border_max_x = min(1.0, (columns.max() + margin) / width)
    scene.render.border_min_y = max(0.0, 1.0 - (rows.max() + margin) / height)
    scene.render.border_max_y = min(1.0, 1.0 - (rows.min() - margin) / height)


def render_ir_pair(scene, camera, output_dir, samples, seed, crop_points):
    """Render left and right IR views: linear, undenoised, lit mainly by the projector."""
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.data.energy *= IR_AMBIENT_SCALE
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value *= IR_AMBIENT_SCALE
    add_projector(camera)

    scene.compositing_node_group = None
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    scene.view_settings.view_transform = "Raw"
    settings = scene.render.image_settings
    settings.file_format = "PNG"
    settings.color_mode = "BW"
    settings.color_depth = "16"
    if crop_points is not None:
        crop_border(scene, camera_pose_opencv(camera), crop_points)

    right = camera.copy()
    scene.collection.objects.link(right)
    right.matrix_world = camera.matrix_world @ Matrix.Translation((model.BASELINE_M, 0.0, 0.0))
    bpy.context.view_layer.update()
    for name, view, offset in (("ir_left", camera, 0), ("ir_right", right, 1)):
        scene.camera = view
        scene.cycles.seed = seed + offset
        scene.render.filepath = str(output_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    scene.camera = camera


def add_plane_target(scene, args, materials):
    """A large flat target facing the camera, for validating the camera model."""
    angle = math.radians(args.plane_tilt_deg)
    tilt = np.eye(4)
    tilt[1:3, 1:3] = ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    add_part([((-1.5, -1.5, -0.02), (1.5, 1.5, 0.0))], tilt, materials["profile"], LABELS["table"])
    bpy.ops.object.camera_add(location=(0.0, 0.0, args.plane_distance))
    camera = bpy.context.object
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = 36.0 * model.FOCAL_PX / model.IMAGE_WIDTH_PX
    camera.data.clip_end = 3.0
    scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def main():
    args = parse_args()
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.max_bounces = 4
    scene.render.resolution_x = model.IMAGE_WIDTH_PX
    scene.render.resolution_y = model.IMAGE_HEIGHT_PX
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True

    materials, roughness = make_materials(args.material, rng)
    bracket_pose, profile_pose = sample_part_poses(args.gap_mm, args.bracket_yaw_deg, rng)
    crop_points = None
    if args.plane_distance is not None:
        camera = add_plane_target(scene, args, materials)
    else:
        add_part(model.profile_boxes(), profile_pose, materials["profile"], LABELS["profile"])
        add_part(model.bracket_boxes(), bracket_pose, materials["bracket"], LABELS["bracket"])
        # The profile overhangs the bracket base, so this pad keeps it level.
        add_part(
            [((0.04, -0.025, 0.0), (0.08, 0.025, model.BRACKET_PLATE_M))],
            np.eye(4),
            materials["table"],
            LABELS["fixture"],
        )
        add_part([((-0.5, -0.4, -0.02), (0.5, 0.4, 0.0))], np.eye(4), materials["table"], LABELS["table"])
        camera = add_camera(rng, args.camera_azimuth_deg, args.camera_elevation_deg)
        crop_points = np.array(
            [[x, y, z] for x in (-0.16, 0.17) for y in (-0.08, 0.08) for z in (0.0, 0.06)]
        )
    add_lights(scene, rng)

    if not args.no_rgb:
        render_rgb(scene, output_dir, args.rgb_samples)
    render_geometry(scene, output_dir)
    if args.ir:
        render_ir_pair(scene, camera, output_dir, args.ir_samples, args.seed, crop_points)

    min_z, max_z = model.depth_range_m()
    metadata = {
        "seed": args.seed,
        "camera_reference": "Intel RealSense D415 (geometry only at this stage)",
        "stage": "ideal_and_ir" if args.ir else "ideal_depth",
        "ir": {"samples": args.ir_samples, "projector_dots": PROJECTOR_DOTS} if args.ir else None,
        "material": args.material,
        "roughness": roughness,
        "defect": {"gap_mm": args.gap_mm, "bracket_yaw_deg": args.bracket_yaw_deg},
        "truth": {
            "gap_mm": model.gap_m(bracket_pose, profile_pose) * 1000.0,
            "bracket_yaw_deg": math.degrees(model.yaw_rad(bracket_pose)),
            "world_from_bracket": bracket_pose.tolist(),
            "world_from_profile": profile_pose.tolist(),
        },
        "camera": {
            "K": model.intrinsics().tolist(),
            "image_size_px": [model.IMAGE_WIDTH_PX, model.IMAGE_HEIGHT_PX],
            "baseline_m": model.BASELINE_M,
            "disparity_shift_px": model.DISPARITY_SHIFT_PX,
            "min_z_m": min_z,
            "max_z_m": max_z,
            "world_from_camera": camera_pose_opencv(camera).tolist(),
        },
        "labels": LABELS,
        "depth_units": "metres, planar z along the camera axis, 0 means no hit",
        "render_seconds": time.perf_counter() - started,
    }
    # Written last, so its presence marks a finished scene.
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Rendered scene to {output_dir} in {metadata['render_seconds']:.1f} s")


if __name__ == "__main__":
    main()

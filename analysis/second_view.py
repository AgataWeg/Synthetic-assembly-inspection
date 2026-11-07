"""Sensitivity check: repeat a subset of the experiment from another camera direction.

Each scene reuses a main-experiment seed, so only the camera direction differs and the views can be
compared pair by pair. The method and camera parameters stay frozen. Ideal depth is measured too,
to separate errors the view causes in the method from errors the camera model causes.

The steep view turned out not to see the profile's free end. The mirror view was added after that
result, to test a different direction that does see it, with the same elevation as the main view.
"""

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from pipeline import assembly_model as model  # noqa: E402
from pipeline import camera_model as cm  # noqa: E402
from pipeline import perception  # noqa: E402
from analysis.analyze import wilson  # noqa: E402
from pipeline.run_batch import BLENDER, DATA_DIR, build_manifest  # noqa: E402

MAIN_RESULTS = Path("results/main")
RESULTS = Path("results/other_views")
VIEWS = {
    "steep": {
        "out_dir": Path("data/generated/second_view"),
        "results": "second_view",
        "azimuth_deg": (-58.0, -42.0),
        "elevation_deg": (68.0, 78.0),
    },
    "mirror": {
        "out_dir": Path("data/generated/mirror_view"),
        "results": "mirror_view",
        "azimuth_deg": (42.0, 58.0),
        "elevation_deg": (46.0, 58.0),
    },
}
GOOD_SCENES = [f"good_test_{index:03d}" for index in range(20)]
GAP_SCENES = [f"gap_2mm_{index:03d}" for index in range(15)]
GAP_TOLERANCE_MM = 1.0
YAW_TOLERANCE_DEG = 1.0
STEREO_SEED_OFFSET = 3  # The offset build_conditions.py uses for stereo depth.
IR_SAMPLES = 32


def visible_end_face_samples(metadata):
    """Free-end samples the camera can see at the true pose, by the method's own visibility test."""
    world_from_profile = np.array(metadata["truth"]["world_from_profile"])
    bracket = o3d.geometry.TriangleMesh(perception.BRACKET_MODEL.mesh).transform(
        np.array(metadata["truth"]["world_from_bracket"])
    )
    camera_position = np.array(metadata["camera"]["world_from_camera"])[:3, 3]
    points, normals = perception.PROFILE_MODEL.visible(world_from_profile, camera_position, [bracket])
    local = (points - world_from_profile[:3, 3]) @ world_from_profile[:3, :3]
    local_normals = normals @ world_from_profile[:3, :3]
    return int(((local[:, 0] > model.PROFILE_LENGTH_M - 1e-6) & (local_normals[:, 0] > 0.9)).sum())


def render(task):
    scene, view = task
    output_dir = view["out_dir"] / scene["name"]
    if (output_dir / "metadata.json").exists() and (output_dir / "ir_right.png").exists():
        return
    (output_dir / "metadata.json").unlink(missing_ok=True)
    command = [
        BLENDER, "-b", "--python", "pipeline/generate_scene.py", "--python-exit-code", "1", "--",
        "--out", str(output_dir), "--seed", str(scene["seed"]), "--material", scene["material"],
        "--gap-mm", str(scene["gap_mm"]), "--bracket-yaw-deg", str(scene["bracket_yaw_deg"]),
        "--no-rgb", "--ir", "--ir-samples", str(IR_SAMPLES),
        "--camera-azimuth-deg", *map(str, view["azimuth_deg"]),
        "--camera-elevation-deg", *map(str, view["elevation_deg"]),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")


def measure_scene(task):
    scene, view_name = task
    scene_dir = VIEWS[view_name]["out_dir"] / scene["name"]
    ideal, labels, metadata = perception.load_scene(scene_dir)
    rng = np.random.default_rng(scene["seed"] * 10 + STEREO_SEED_OFFSET)
    stereo = cm.stereo_depth(cm.read_ir(scene_dir, "ir_left"), cm.read_ir(scene_dir, "ir_right"), "sgbm", rng)
    np.savez_compressed(scene_dir / "conditions.npz", stereo=stereo)
    parts = (labels == 1) | (labels == 2)
    truth = metadata["truth"]
    world_from_camera = np.array(metadata["camera"]["world_from_camera"])
    main_metadata = json.loads((DATA_DIR / scene["name"] / "metadata.json").read_text(encoding="utf-8"))
    # The pairing is only valid if the parts sit exactly where they did in the main experiment.
    if main_metadata["truth"] != truth:
        raise RuntimeError(f"{scene['name']}: the assembly differs from the main experiment")
    end_face_samples = visible_end_face_samples(metadata)
    rows = []
    for condition, depth in (("ideal", ideal), ("stereo", stereo)):
        measured = perception.measure(depth, metadata)
        rows.append(
            {
                "name": scene["name"],
                "condition": condition,
                "view": view_name,
                "end_face_samples": end_face_samples,
                "split": scene["split"],
                "defect_gap_mm": scene["gap_mm"],
                "gap_mm": measured["gap_mm"],
                "bracket_yaw_deg": measured["bracket_yaw_deg"],
                "gap_error_mm": measured["gap_mm"] - truth["gap_mm"],
                "yaw_error_deg": measured["bracket_yaw_deg"] - truth["bracket_yaw_deg"],
                "failed": measured["failed"],
                "part_fill": float((depth[parts] > 0).mean()),
                # The optical axis points down at the table, along the camera's +z in OpenCV axes.
                "elevation_deg": float(np.degrees(np.arcsin(-world_from_camera[2, 2]))),
            }
        )
    return rows


def gap_alarm(row):
    return bool(row["failed"]) or abs(float(row["gap_mm"])) > GAP_TOLERANCE_MM


def yaw_alarm(row):
    return bool(row["failed"]) or abs(float(row["bracket_yaw_deg"])) > YAW_TOLERANCE_DEG


def read_main_view():
    with (MAIN_RESULTS / "stereo.csv").open(encoding="utf-8") as handle:
        rows = {r["name"]: r for r in csv.DictReader(handle)}
    for row in rows.values():
        row["failed"] = row["failed"] == "True"
    return rows


def describe(rows):
    good = [r for r in rows if r["split"] == "good_test"]
    gap = [r for r in rows if r["split"] == "gap"]
    errors = np.array([float(r["gap_error_mm"]) for r in rows])
    yaws = np.array([float(r["yaw_error_deg"]) for r in rows])
    any_alarms = sum(gap_alarm(r) or yaw_alarm(r) for r in good)
    return {
        "false_alarms_any": any_alarms,
        "false_alarms_any_wilson_95": wilson(any_alarms, len(good)),
        "false_alarms_gap": sum(gap_alarm(r) for r in good),
        "false_alarms_yaw": sum(yaw_alarm(r) for r in good),
        "good_scenes": len(good),
        "detected_2mm": sum(gap_alarm(r) for r in gap),
        "detected_2mm_accurate": sum(
            gap_alarm(r) and not r["failed"] and abs(float(r["gap_error_mm"])) <= GAP_TOLERANCE_MM for r in gap
        ),
        "gap_scenes": len(gap),
        "failed": sum(bool(r["failed"]) for r in rows),
        "scenes": len(rows),
        "gap_error_mean_mm": float(errors.mean()),
        "gap_error_sd_mm": float(errors.std(ddof=1)),
        "gap_error_abs_median_mm": float(np.median(np.abs(errors))),
        "yaw_error_mean_deg": float(yaws.mean()),
        "yaw_error_sd_deg": float(yaws.std(ddof=1)),
        "yaw_error_abs_median_deg": float(np.median(np.abs(yaws))),
        "part_fill_mean": float(np.mean([float(r["part_fill"]) for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=tuple(VIEWS), default="mirror")
    args = parser.parse_args()
    view = VIEWS[args.view]

    manifest = {scene["name"]: scene for scene in build_manifest()}
    scenes = [manifest[name] for name in GOOD_SCENES + GAP_SCENES]
    for position, scene in enumerate(scenes, start=1):
        render((scene, view))
        print(f"[{position}/{len(scenes)}] {scene['name']}", flush=True)

    with ProcessPoolExecutor(max_workers=5) as pool:
        rows = [
            row
            for scene_rows in pool.map(measure_scene, [(scene, args.view) for scene in scenes])
            for row in scene_rows
        ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"{view['results']}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    main_view = read_main_view()
    view_stereo = [r for r in rows if r["condition"] == "stereo"]
    same_scenes_main = [main_view[r["name"]] for r in view_stereo]
    good_names = [r["name"] for r in view_stereo if r["split"] == "good_test"]
    view_by_name = {r["name"]: r for r in view_stereo}
    only_this_view = sum(
        (gap_alarm(view_by_name[n]) or yaw_alarm(view_by_name[n]))
        and not (gap_alarm(main_view[n]) or yaw_alarm(main_view[n]))
        for n in good_names
    )
    only_main = sum(
        (gap_alarm(main_view[n]) or yaw_alarm(main_view[n]))
        and not (gap_alarm(view_by_name[n]) or yaw_alarm(view_by_name[n]))
        for n in good_names
    )
    discordant = only_this_view + only_main
    summary = {
        "view": args.view,
        "azimuth_range_deg": view["azimuth_deg"],
        "elevation_range_deg": view["elevation_deg"],
        "elevation_measured_deg": [
            min(r["elevation_deg"] for r in view_stereo),
            max(r["elevation_deg"] for r in view_stereo),
        ],
        "end_face_samples_min_max": [
            min(r["end_face_samples"] for r in view_stereo),
            max(r["end_face_samples"] for r in view_stereo),
        ],
        "view_ideal": describe([r for r in rows if r["condition"] == "ideal"]),
        "view_stereo": describe(view_stereo),
        "main_view_stereo_same_scenes": describe(same_scenes_main),
        "paired_false_alarms": {
            "only_this_view": only_this_view,
            "only_main": only_main,
            "mcnemar_exact_p": binomtest(only_this_view, discordant, 0.5).pvalue if discordant else 1.0,
        },
    }
    (RESULTS / f"{view['results']}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

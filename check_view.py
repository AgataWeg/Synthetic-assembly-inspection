"""Is the profile's free end visible from each view? Uses true poses and the method's own visibility test."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import open3d as o3d

import assembly_model as model
import perception

END_LOCAL_X = model.PROFILE_LENGTH_M


def end_face(directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    world_from_profile = np.array(metadata["truth"]["world_from_profile"])
    world_from_bracket = np.array(metadata["truth"]["world_from_bracket"])
    world_from_camera = np.array(metadata["camera"]["world_from_camera"])
    camera = world_from_camera[:3, 3]
    bracket_mesh = o3d.geometry.TriangleMesh(perception.BRACKET_MODEL.mesh).transform(world_from_bracket)
    points, normals = perception.PROFILE_MODEL.visible(world_from_profile, camera, [bracket_mesh])
    local = (np.linalg.inv(world_from_profile) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    local_normals = normals @ world_from_profile[:3, :3]
    samples = int(((local[:, 0] > END_LOCAL_X - 1e-6) & (local_normals[:, 0] > 0.9)).sum())
    camera_local_x = (np.linalg.inv(world_from_profile) @ np.append(camera, 1.0))[0]
    return samples, camera_local_x - END_LOCAL_X, float(np.degrees(np.arcsin(-world_from_camera[2, 2])))


names = [f"good_test_{i:03d}" for i in range(20)] + [f"gap_2mm_{i:03d}" for i in range(15)]
for label, root in (("main", Path("data/generated/experiment")), ("steep", Path("data/generated/second_view"))):
    values = [end_face(root / n) for n in names]
    samples = np.array([v[0] for v in values])
    ahead = np.array([v[1] for v in values]) * 1000
    elevation = np.array([v[2] for v in values])
    print(label, "elevation", elevation.min().round(1), elevation.max().round(1),
          "| visible end-face samples min/median/max", samples.min(), np.median(samples), samples.max(),
          "| scenes with none", int((samples == 0).sum()), "of", len(samples),
          "| camera past the end plane by mm, min/median/max", ahead.min().round(0), np.median(ahead).round(0), ahead.max().round(0))

with open("results/other_views/second_view.csv", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
for condition in ("ideal", "stereo"):
    for split in ("good_test", "gap"):
        chosen = [float(r["gap_mm"]) for r in rows if r["condition"] == condition and r["split"] == split]
        print(condition, split, "gap reading mean", round(np.mean(chosen), 3), "sd", round(np.std(chosen, ddof=1), 3), "min", round(min(chosen), 3), "max", round(max(chosen), 3))
for r in rows:
    if r["condition"] == "stereo" and r["failed"] == "True":
        print("failed", r["name"])

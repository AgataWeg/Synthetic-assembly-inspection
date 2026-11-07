"""Render one scene in colour from the main, mirrored and steep views, and draw them side by side."""

import json
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pipeline import assembly_model as model  # noqa: E402
from pipeline.run_batch import BLENDER, DATA_DIR, build_manifest  # noqa: E402
from analysis.second_view import VIEWS  # noqa: E402

SCENE = "gap_2mm_000"
OUT_DIR = Path("data/generated/examples")
FIGURE = Path("results/figures/fig_views.png")
RGB_SAMPLES = 64
CROP_MARGIN_PX = 45
TEXT, MUTED, SURFACE, ACCENT = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
PANELS = (
    ("main", "Main view", DATA_DIR),
    ("mirror", "Mirrored view", VIEWS["mirror"]["out_dir"]),
    ("steep", "Steep view", VIEWS["steep"]["out_dir"]),
)


def render(scene, view_name):
    """Colour render of the scene; the main view's render already exists from render_examples.py."""
    output_dir = OUT_DIR / (scene["name"] if view_name == "main" else f"{view_name}_{scene['name']}")
    if (output_dir / "rgb.png").exists():
        return output_dir
    command = [
        BLENDER, "-b", "--python", "pipeline/generate_scene.py", "--python-exit-code", "1", "--",
        "--out", str(output_dir), "--seed", str(scene["seed"]), "--material", scene["material"],
        "--gap-mm", str(scene["gap_mm"]), "--bracket-yaw-deg", str(scene["bracket_yaw_deg"]),
        "--rgb-samples", str(RGB_SAMPLES),
    ]
    if view_name != "main":
        command += [
            "--camera-azimuth-deg", *map(str, VIEWS[view_name]["azimuth_deg"]),
            "--camera-elevation-deg", *map(str, VIEWS[view_name]["elevation_deg"]),
        ]
    subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")
    return output_dir


def free_end_pixel(metadata):
    """Image position of the centre of the profile's free end face."""
    end = np.array([model.PROFILE_LENGTH_M, 0.0, model.PROFILE_WIDTH_M / 2.0, 1.0])
    world = np.array(metadata["truth"]["world_from_profile"]) @ end
    camera = np.linalg.inv(np.array(metadata["camera"]["world_from_camera"])) @ world
    K = np.array(metadata["camera"]["K"])
    return K[0, 0] * camera[0] / camera[2] + K[0, 2], K[1, 1] * camera[1] / camera[2] + K[1, 2]


def camera_angles(metadata):
    world_from_camera = np.array(metadata["camera"]["world_from_camera"])
    position = world_from_camera[:3, 3]
    elevation = np.degrees(np.arcsin(-world_from_camera[2, 2]))
    azimuth = np.degrees(np.arctan2(position[1], position[0]))
    return azimuth, elevation


def main():
    scene = {s["name"]: s for s in build_manifest()}[SCENE]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    for axis, (view_name, title, measured_dir) in zip(axes, PANELS):
        render_dir = render(scene, view_name)
        measured = np.load(measured_dir / SCENE / "geometry.npz")["labels"]
        labels = np.load(render_dir / "geometry.npz")["labels"]
        # The colour render must show exactly the camera and scene that were measured.
        if not np.array_equal(labels, measured):
            raise RuntimeError(f"{view_name}: the colour render differs from the measured scene")
        metadata = json.loads((render_dir / "metadata.json").read_text(encoding="utf-8"))

        rows, columns = np.nonzero((labels == 1) | (labels == 2))
        top, bottom = max(rows.min() - CROP_MARGIN_PX, 0), min(rows.max() + CROP_MARGIN_PX, labels.shape[0])
        left, right = max(columns.min() - CROP_MARGIN_PX, 0), min(columns.max() + CROP_MARGIN_PX, labels.shape[1])
        colour = cv2.cvtColor(cv2.imread(str(render_dir / "rgb.png")), cv2.COLOR_BGR2RGB)
        axis.imshow(colour[top:bottom, left:right])

        # The free end is visible if the camera sits beyond the plane of the end face.
        camera_in_profile = np.linalg.inv(np.array(metadata["truth"]["world_from_profile"])) @ np.append(
            np.array(metadata["camera"]["world_from_camera"])[:3, 3], 1.0
        )
        visible = camera_in_profile[0] > model.PROFILE_LENGTH_M
        u, v = free_end_pixel(metadata)
        u, v = u - left, v - top
        label = "free end visible" if visible else "free end hidden"
        width, height = right - left, bottom - top
        # The label sits inside the frame, on the side away from the end it points to.
        text_x = 0.05 * width if u > width / 2 else 0.55 * width
        colour_mark = ACCENT if visible else TEXT
        axis.annotate(
            label, xy=(u, v), xytext=(text_x, 0.9 * height), color=colour_mark, fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "-|>", "color": colour_mark, "lw": 1.3},
        )
        azimuth, elevation = camera_angles(metadata)
        axis.set_title(
            f"{title}\nazimuth {azimuth:+.0f}°, elevation {elevation:.0f}°", color=TEXT, fontsize=10, loc="left"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=160)
    plt.close(figure)
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()

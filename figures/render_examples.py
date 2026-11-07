"""Render colour images for one aluminium scene and its matte twin, and draw the example figure."""

import json
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.run_batch import BLENDER, DATA_DIR, build_manifest  # noqa: E402

OUT_DIR = Path("data/generated/examples")
FIGURE = Path("results/figures/fig_scene.png")
SCENES = ("gap_2mm_000", "matte_gap_2mm_000")
CROP_MARGIN_PX = 45
RGB_SAMPLES = 64
LABEL_COLOURS = {1: (0.16, 0.47, 0.84), 2: (0.92, 0.41, 0.20), 3: (0.88, 0.88, 0.85), 4: (0.11, 0.69, 0.48)}
LABEL_NAMES = {1: "profile", 2: "bracket", 3: "table", 4: "pad"}
TEXT, SURFACE = "#0b0b0b", "#fcfcfb"


def render(scene):
    output_dir = OUT_DIR / scene["name"]
    if (output_dir / "metadata.json").exists():
        return
    command = [
        BLENDER, "-b", "--python", "pipeline/generate_scene.py", "--python-exit-code", "1", "--",
        "--out", str(output_dir), "--seed", str(scene["seed"]), "--material", scene["material"],
        "--gap-mm", str(scene["gap_mm"]), "--bracket-yaw-deg", str(scene["bracket_yaw_deg"]),
        "--rgb-samples", str(RGB_SAMPLES),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")


def crop_box(labels):
    rows, columns = np.nonzero((labels == 1) | (labels == 2))
    return (
        max(rows.min() - CROP_MARGIN_PX, 0), min(rows.max() + CROP_MARGIN_PX, labels.shape[0]),
        max(columns.min() - CROP_MARGIN_PX, 0), min(columns.max() + CROP_MARGIN_PX, labels.shape[1]),
    )


def main():
    manifest = {scene["name"]: scene for scene in build_manifest()}
    for name in SCENES:
        render(manifest[name])

    figure, axes = plt.subplots(2, 4, figsize=(12, 4.8), facecolor=SURFACE)
    titles = ("Colour image", "Labels", "Left IR image", "Stereo depth (mm); grey = none")
    for row, name in enumerate(SCENES):
        render_dir, stereo_dir = OUT_DIR / name, DATA_DIR / name
        geometry = np.load(render_dir / "geometry.npz")
        experiment_geometry = np.load(stereo_dir / "geometry.npz")
        # The colour render must show exactly the scene that was measured.
        if not np.array_equal(geometry["labels"], experiment_geometry["labels"]):
            raise RuntimeError(f"{name}: the example render differs from the experiment scene")
        labels = geometry["labels"]
        top, bottom, left, right = crop_box(labels)
        crop = (slice(top, bottom), slice(left, right))

        colour = cv2.cvtColor(cv2.imread(str(render_dir / "rgb.png")), cv2.COLOR_BGR2RGB)
        painted = np.ones((*labels.shape, 3))
        for label, value in LABEL_COLOURS.items():
            painted[labels == label] = value
        infrared = cv2.imread(str(stereo_dir / "ir_left.png"), cv2.IMREAD_UNCHANGED).astype(np.float64)
        infrared = infrared[crop]
        infrared = np.clip(infrared / max(np.percentile(infrared, 99), 1.0), 0.0, 1.0) ** (1.0 / 2.2)
        stereo = np.load(stereo_dir / "conditions.npz")["stereo"][crop] * 1000.0
        parts = ((labels == 1) | (labels == 2))[crop]
        ideal = experiment_geometry["depth_m"][crop] * 1000.0
        low, high = np.percentile(ideal[parts], [1, 99])
        colour_map = matplotlib.colormaps["Blues_r"].copy()
        colour_map.set_bad("#b9b8b2")

        axes[row, 0].imshow(colour[crop])
        axes[row, 1].imshow(painted[crop])
        axes[row, 2].imshow(infrared, cmap="gray", vmin=0, vmax=1)
        image = axes[row, 3].imshow(
            np.where(stereo > 0, stereo, np.nan), cmap=colour_map, vmin=low - 15, vmax=high + 15
        )
        figure.colorbar(image, ax=axes[row, 3], fraction=0.03)
        for column, axis in enumerate(axes[row]):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(titles[column], color=TEXT, fontsize=10, loc="left")
        axes[row, 0].set_ylabel("Aluminium" if row == 0 else "Matte twin", color=TEXT, fontsize=10)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=LABEL_COLOURS[k], label=LABEL_NAMES[k])
        for k in (1, 2, 3, 4)
    ]
    figure.legend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=9, labelcolor=TEXT)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=160)
    plt.close(figure)
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()

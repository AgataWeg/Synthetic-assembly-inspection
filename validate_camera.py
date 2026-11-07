"""Check the stereo model on flat targets against published D415 values, before any experiment."""

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

import camera_model as cm
import perception
from run_batch import BLENDER

OUT_DIR = Path("data/generated/validation")
MATTE_DISTANCES_M = (0.3, 0.45, 0.6, 0.8, 1.0)
METAL_TILTS_DEG = (0.0, 20.0, 40.0)
METAL_DISTANCE_M = 0.45
# Keeps tilted planes and frame borders out of the statistics.
CENTRE_ROWS = slice(200, 520)
CENTRE_COLUMNS = slice(400, 900)


def render(name, distance, tilt, material, samples):
    output_dir = OUT_DIR / name
    if not (output_dir / "metadata.json").exists():
        command = [
            BLENDER, "-b", "--python", "generate_scene.py", "--python-exit-code", "1", "--",
            "--out", str(output_dir), "--seed", "7", "--material", material, "--no-rgb", "--ir",
            "--ir-samples", str(samples), "--plane-distance", str(distance), "--plane-tilt-deg", str(tilt),
        ]
        subprocess.run(command, check=True, capture_output=True)
    return output_dir


def measure_plane(output_dir, kind):
    ideal, _, _ = perception.load_scene(output_dir)
    rng = np.random.default_rng(1)
    depth = cm.stereo_depth(cm.read_ir(output_dir, "ir_left"), cm.read_ir(output_dir, "ir_right"), kind, rng)
    ideal, depth = ideal[CENTRE_ROWS, CENTRE_COLUMNS], depth[CENTRE_ROWS, CENTRE_COLUMNS]
    valid = depth > 0
    error = depth[valid] - ideal[valid]
    if error.size == 0:
        return {"fill": 0.0, "mean_mm": np.nan, "std_mm": np.nan, "p99_abs_mm": np.nan}
    return {
        "fill": float(valid.mean()),
        "mean_mm": float(error.mean() * 1e3),
        "std_mm": float(error.std() * 1e3),
        "p99_abs_mm": float(np.percentile(np.abs(error), 99) * 1e3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("results/camera_check/camera_validation.csv"))
    args = parser.parse_args()

    targets = [(f"matte_{d:g}m", d, 0.0, "matte") for d in MATTE_DISTANCES_M]
    targets += [(f"metal_{t:g}deg", METAL_DISTANCE_M, t, "aluminium") for t in METAL_TILTS_DEG]
    rows = []
    for name, distance, tilt, material in targets:
        output_dir = render(name, distance, tilt, material, args.samples)
        for kind in ("sgbm", "bm"):
            rows.append(
                {
                    "target": name,
                    "matcher": kind,
                    "distance_m": distance,
                    "datasheet_mm": float(cm.datasheet_sigma(distance) * 1e3),
                    "carfagni_mm": float(cm.measured_sigma(distance) * 1e3),
                    **measure_plane(output_dir, kind),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'target':<14}{'matcher':<7}{'fill':>6}{'mean':>9}{'std':>8}{'p99':>8}{'datasheet':>11}{'Carfagni':>10}")
    for row in rows:
        print(
            f"{row['target']:<14}{row['matcher']:<7}{row['fill']:>6.2f}{row['mean_mm']:>9.3f}"
            f"{row['std_mm']:>8.3f}{row['p99_abs_mm']:>8.2f}{row['datasheet_mm']:>11.3f}{row['carfagni_mm']:>10.3f}"
        )


if __name__ == "__main__":
    main()

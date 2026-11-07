"""Create each scene's camera-model depth maps from its ideal depth and IR pair."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import camera_model as cm
from run_batch import DATA_DIR

STEREO_MATCHER = "sgbm"  # Frozen from the flat-target validation in validate_camera.py.
SEED_OFFSETS = {"datasheet": 1, "measured": 2, "stereo": 3}


def build_scene(scene):
    scene_dir = DATA_DIR / scene["name"]
    output = scene_dir / "conditions.npz"
    if output.exists() or not (scene_dir / "ir_right.png").exists():
        return scene["name"], output.exists()
    geometry = np.load(scene_dir / "geometry.npz")
    depth, labels = geometry["depth_m"].astype(np.float64), geometry["labels"]
    maps = {}
    if scene["material"] == "aluminium":
        for name, sigma in (("datasheet", cm.datasheet_sigma), ("measured", cm.measured_sigma)):
            rng = np.random.default_rng(scene["seed"] * 10 + SEED_OFFSETS[name])
            maps[name] = cm.analytic_depth(depth, labels, sigma, rng)
    rng = np.random.default_rng(scene["seed"] * 10 + SEED_OFFSETS["stereo"])
    maps["stereo"] = cm.stereo_depth(
        cm.read_ir(scene_dir, "ir_left"), cm.read_ir(scene_dir, "ir_right"), STEREO_MATCHER, rng
    )
    np.savez_compressed(output, **maps)
    return scene["name"], True


def main():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(build_scene, manifest, chunksize=4))
    print(f"Conditions available for {sum(done for _, done in results)} of {len(results)} scenes.")


if __name__ == "__main__":
    main()

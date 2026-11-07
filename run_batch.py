"""Render the experiment's scenes with Blender; finished scenes are skipped, so it can resume."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
DATA_DIR = Path("data/generated/experiment")
GAP_LEVELS_MM = (0.5, 1.0, 2.0, 3.0, 5.0)
YAW_LEVELS_DEG = (1.0, 2.0, 4.0)
GOOD_CALIBRATION_SCENES = 20
GOOD_TEST_SCENES = 40
SCENES_PER_LEVEL = 15
MATTE_GOOD_SCENES = 20
MATTE_GAP_LEVELS_MM = (1.0, 2.0)
MATTE_SCENES_PER_LEVEL = 10
DEV_GAP_LEVELS_MM = (0.0, 1.0, 2.0, 5.0)
DEV_YAW_LEVELS_DEG = (2.0,)
DEV_SCENES_PER_LEVEL = 4


def build_manifest():
    """Every scene's definition, in a fixed order, with a seed unique to its assembly."""
    scenes = []

    def add(name, split, gap_mm=0.0, yaw_deg=0.0):
        scenes.append(
            {
                "name": name,
                "split": split,
                "material": "aluminium",
                "seed": 10_000 + len(scenes),
                "gap_mm": gap_mm,
                "bracket_yaw_deg": yaw_deg,
            }
        )

    for index in range(GOOD_CALIBRATION_SCENES):
        add(f"good_cal_{index:03d}", "good_calibration")
    for index in range(GOOD_TEST_SCENES):
        add(f"good_test_{index:03d}", "good_test")
    for level in GAP_LEVELS_MM:
        for index in range(SCENES_PER_LEVEL):
            add(f"gap_{level:g}mm_{index:03d}", "gap", gap_mm=level)
    for level in YAW_LEVELS_DEG:
        for index in range(SCENES_PER_LEVEL):
            # Alternating signs, so both pivot corners are covered at every level.
            sign = 1.0 if index % 2 == 0 else -1.0
            add(f"yaw_{level:g}deg_{index:03d}", "yaw", yaw_deg=sign * level)

    # Matte twins reuse an aluminium scene's seed, so only the material differs.
    by_name = {scene["name"]: scene for scene in scenes}
    twins = [f"good_test_{index:03d}" for index in range(MATTE_GOOD_SCENES)]
    for level in MATTE_GAP_LEVELS_MM:
        twins += [f"gap_{level:g}mm_{index:03d}" for index in range(MATTE_SCENES_PER_LEVEL)]
    for name in twins:
        scenes.append({**by_name[name], "name": f"matte_{name}", "material": "matte"})

    # Method tuning uses only these scenes, whose seeds appear nowhere in the experiment.
    dev_scenes = [(gap, 0.0) for gap in DEV_GAP_LEVELS_MM for _ in range(DEV_SCENES_PER_LEVEL)]
    dev_scenes += [(0.0, yaw) for yaw in DEV_YAW_LEVELS_DEG for _ in range(DEV_SCENES_PER_LEVEL)]
    for index, (gap, yaw) in enumerate(dev_scenes):
        scenes.append(
            {
                "name": f"dev_{index:03d}",
                "split": "dev",
                "material": "aluminium",
                "seed": 90_000 + index,
                "gap_mm": gap,
                "bracket_yaw_deg": yaw if index % 2 == 0 else -yaw,
            }
        )
    return scenes


def select_scenes(scenes, per_level, good, include_matte, only_dev=False):
    """The dev scenes alone, or a stratified experiment subset, or the whole experiment."""
    selected = []
    counts = {}
    for scene in scenes:
        if only_dev or scene["split"] == "dev":
            if only_dev and scene["split"] == "dev":
                selected.append(scene)
            continue
        if scene["material"] == "matte":
            if include_matte:
                selected.append(scene)
            continue
        is_good = scene["split"].startswith("good")
        limit = good if is_good else per_level
        key = scene["split"] if is_good else (scene["gap_mm"], abs(scene["bracket_yaw_deg"]))
        counts[key] = counts.get(key, 0) + 1
        if limit is None or counts[key] <= limit:
            selected.append(scene)
    return selected


def is_finished(output_dir, needs_ir):
    if not (output_dir / "metadata.json").exists():
        return False
    return not needs_ir or (output_dir / "ir_right.png").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-level", type=int, help="Scenes per defect level (default: all).")
    parser.add_argument("--good", type=int, help="Scenes per good split (default: all).")
    parser.add_argument("--no-matte", action="store_true", help="Leave out the matte twins.")
    parser.add_argument("--rgb", action="store_true", help="Also render the colour images.")
    parser.add_argument("--ir", action="store_true", help="Also render the stereo IR pairs.")
    parser.add_argument("--ir-samples", type=int, default=32)
    parser.add_argument("--dev", action="store_true", help="Render only the method-tuning scenes.")
    args = parser.parse_args()

    manifest = build_manifest()
    scenes = select_scenes(manifest, args.per_level, args.good, not args.no_matte, args.dev)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    failed = []
    started = time.perf_counter()
    for position, scene in enumerate(scenes, start=1):
        output_dir = DATA_DIR / scene["name"]
        if is_finished(output_dir, args.ir):
            continue
        # A partial scene must not look finished if this render is interrupted.
        (output_dir / "metadata.json").unlink(missing_ok=True)
        command = [
            BLENDER, "-b", "--python", "generate_scene.py", "--python-exit-code", "1", "--",
            "--out", str(output_dir), "--seed", str(scene["seed"]), "--material", scene["material"],
            "--gap-mm", str(scene["gap_mm"]), "--bracket-yaw-deg", str(scene["bracket_yaw_deg"]),
        ]
        if not args.rgb:
            command.append("--no-rgb")
        if args.ir:
            command += ["--ir", "--ir-samples", str(args.ir_samples)]
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        if result.returncode != 0 or not is_finished(output_dir, args.ir):
            failed.append(scene["name"])
            print(f"FAILED {scene['name']}:\n{result.stdout[-1500:]}\n{result.stderr[-500:]}", flush=True)
        print(f"[{position}/{len(scenes)}] {scene['name']} {time.perf_counter() - started:.0f} s", flush=True)

    print(f"Finished: {len(scenes) - len(failed)} of {len(scenes)} scenes available.")
    if failed:
        print("Failed scenes:", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

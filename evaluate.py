"""Run the measurement on every available scene for one depth condition and save a CSV."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

import perception  # noqa: E402
from run_batch import DATA_DIR  # noqa: E402

CONDITIONS = ("ideal", "datasheet", "measured", "stereo")


def scene_is_available(scene, condition):
    scene_dir = DATA_DIR / scene["name"]
    if not (scene_dir / "metadata.json").exists():
        return False
    if condition == "ideal":
        return scene["material"] == "aluminium"
    if condition != "stereo" and scene["material"] != "aluminium":
        return False
    return (scene_dir / "conditions.npz").exists()


def evaluate_scene(task):
    scene, condition = task
    scene_dir = DATA_DIR / scene["name"]
    depth, labels, metadata = perception.load_scene(scene_dir)
    if condition != "ideal":
        depth = np.load(scene_dir / "conditions.npz")[condition]
    measured = perception.measure(depth, metadata)
    truth = metadata["truth"]
    profile_info = measured.pop("profile_translation_information") or [np.nan] * 3
    bracket_info = measured.pop("bracket_translation_information") or [np.nan] * 3
    parts = (labels == 1) | (labels == 2)
    return {
        "name": scene["name"],
        "condition": condition,
        "material": scene["material"],
        "split": scene["split"],
        "defect_gap_mm": scene["gap_mm"],
        "defect_yaw_deg": scene["bracket_yaw_deg"],
        "true_gap_mm": truth["gap_mm"],
        "true_yaw_deg": truth["bracket_yaw_deg"],
        "gap_error_mm": measured["gap_mm"] - truth["gap_mm"],
        "yaw_error_deg": measured["bracket_yaw_deg"] - truth["bracket_yaw_deg"],
        "part_fill": float((depth[parts] > 0).mean()),
        "profile_info_weakest": profile_info[0],
        "profile_info_next": profile_info[1],
        "bracket_info_weakest": bracket_info[0],
        **measured,
    }


def summarise(rows):
    print(f"{'group':<14}{'n':>4}{'gap err mean':>14}{'gap err sd':>12}{'yaw err mean':>14}{'yaw err sd':>12}{'fill':>7}")
    groups = {}
    for row in rows:
        if row["defect_gap_mm"]:
            key = f"gap {row['defect_gap_mm']:g} mm"
        elif row["defect_yaw_deg"]:
            key = f"yaw {abs(row['defect_yaw_deg']):g} deg"
        else:
            key = "good"
        if row["material"] == "matte":
            key = "matte " + key
        groups.setdefault(key, []).append(row)
    for key, members in groups.items():
        gap = np.array([m["gap_error_mm"] for m in members])
        yaw = np.array([m["yaw_error_deg"] for m in members])
        fill = np.mean([m["part_fill"] for m in members])
        print(
            f"{key:<14}{len(members):>4}{gap.mean():>11.3f} mm{gap.std():>9.3f} mm"
            f"{yaw.mean():>10.3f} deg{yaw.std():>8.3f} deg{fill:>7.2f}"
        )
    failed = sum(r["failed"] for r in rows)
    print(f"Failed measurements: {failed} of {len(rows)} scenes")
    fallbacks = sum(r["pca_fallback_profile"] or r["pca_fallback_bracket"] for r in rows)
    ratios = [
        r["profile_info_next"] / r["profile_info_weakest"]
        for r in rows
        if r["profile_info_weakest"] and np.isfinite(r["profile_info_weakest"])
    ]
    print(f"PCA fallbacks: {fallbacks} of {len(rows)} scenes")
    if ratios:
        print(f"Profile information ratio, next/weakest direction: median {np.median(ratios):.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, default="ideal")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dev", action="store_true", help="Measure the method-tuning scenes.")
    args = parser.parse_args()
    out = args.out or Path("results") / ("dev" if args.dev else "main") / f"{args.condition}.csv"

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    scenes = [s for s in manifest if scene_is_available(s, args.condition)]
    # Tuning runs see only the dev scenes, and experiment runs never include them.
    scenes = [s for s in scenes if (s["split"] == "dev") == args.dev]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate_scene, [(s, args.condition) for s in scenes]))
    print(f"Measured {len(rows)} scenes in {time.perf_counter() - started:.0f} s")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarise(rows)


if __name__ == "__main__":
    main()

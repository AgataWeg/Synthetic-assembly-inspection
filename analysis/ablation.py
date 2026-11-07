"""Post-hoc diagnosis: which analytic effect causes the errors, on the 40 good test scenes.

Nothing here feeds back into the method or the camera parameters; they were frozen before the
test scenes were analysed. Also reports matte part errors and table residuals near the parts.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

from pipeline import assembly_model as model  # noqa: E402
from pipeline import camera_model  # noqa: E402
from pipeline import perception  # noqa: E402
from pipeline.run_batch import DATA_DIR  # noqa: E402

OUT = Path("results/diagnostics/ablation.json")
GAP_TOLERANCE_MM = 1.0
YAW_TOLERANCE_DEG = 1.0
LOW_CLUSTER_MM = -0.5
TABLE_MARGIN_PX = 60
VARIANTS = {
    # name: (noise, occlusion, edge band, blob dropout)
    "all effects (Carfagni)": (True, True, True, True),
    "noise only": (True, False, False, False),
    "no edge band": (True, True, False, True),
    "noise and edge band": (True, False, True, False),
    "holes only": (False, True, True, True),
}


def variant_depth(depth, labels, switches, rng):
    noise, occlusion, edge, blobs = switches
    result = depth.astype(np.float64).copy()
    if noise:
        result += rng.standard_normal(depth.shape) * camera_model.measured_sigma(depth)
    holes = depth <= 0
    if occlusion:
        holes |= camera_model.occlusion_mask(depth)
    if edge:
        holes |= camera_model.edge_mask(depth)
    if blobs:
        holes |= camera_model.blob_dropouts((labels == 1) | (labels == 2), rng)
    result[holes] = 0.0
    return result.astype(np.float32)


def run_variant(task):
    name, variant = task
    depth, labels, metadata = perception.load_scene(DATA_DIR / name)
    rng = np.random.default_rng(metadata["seed"])
    measured = perception.measure(variant_depth(depth, labels, VARIANTS[variant], rng), metadata)
    truth = metadata["truth"]
    return {
        "scene": name,
        "variant": variant,
        "gap_error_mm": measured["gap_mm"] - truth["gap_mm"],
        "yaw_error_deg": measured["bracket_yaw_deg"] - truth["bracket_yaw_deg"],
        "gap_alarm": bool(measured["failed"] or abs(measured["gap_mm"]) > GAP_TOLERANCE_MM),
        "yaw_alarm": bool(measured["failed"] or abs(measured["bracket_yaw_deg"]) > YAW_TOLERANCE_DEG),
    }


def part_offsets(name):
    """Pose error of each part along the world x axis, from the stored stereo depth."""
    _, _, metadata = perception.load_scene(DATA_DIR / name)
    depth = np.load(DATA_DIR / name / "conditions.npz")["stereo"]
    captured = {}
    original = model.gap_m

    def capture(bracket_pose, profile_pose):
        captured["bracket"], captured["profile"] = bracket_pose.copy(), profile_pose.copy()
        return original(bracket_pose, profile_pose)

    model.gap_m = capture
    perception.measure(depth, metadata)
    model.gap_m = original
    truth = metadata["truth"]
    return {
        "profile_dx_mm": float(captured["profile"][0, 3] - truth["world_from_profile"][0][3]) * 1000.0,
        "bracket_dx_mm": float(captured["bracket"][0, 3] - truth["world_from_bracket"][0][3]) * 1000.0,
    }


def table_residual(name):
    """Stereo minus ideal on the table, near the parts and across the whole image."""
    ideal, labels, _ = perception.load_scene(DATA_DIR / name)
    stereo = np.load(DATA_DIR / name / "conditions.npz")["stereo"]
    parts = (labels == 1) | (labels == 2)
    rows, columns = np.nonzero(parts)
    near = np.zeros_like(parts)
    near[
        max(rows.min() - TABLE_MARGIN_PX, 0) : rows.max() + TABLE_MARGIN_PX,
        max(columns.min() - TABLE_MARGIN_PX, 0) : columns.max() + TABLE_MARGIN_PX,
    ] = True
    near &= ~binary_dilation(parts, iterations=10)
    result = {}
    for key, mask in (("near_parts", near & (labels == 3)), ("whole_image", labels == 3)):
        selected = mask & (stereo > 0)
        difference = (stereo[selected] - ideal[selected]) * 1000.0
        result[key] = {
            "median_mm": float(np.median(difference)),
            "over_20mm": float((np.abs(difference) > 20.0).mean()),
            "fill": float(selected.sum() / mask.sum()),
        }
    return result


def main():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    good = [s["name"] for s in manifest if s["split"] == "good_test" and s["material"] == "aluminium"]
    twins = [s["name"] for s in manifest if s["split"] == "good_test" and s["material"] == "matte"]
    if not twins:
        twins = [s["name"] for s in manifest if s["material"] == "matte" and not s["gap_mm"]]

    with ProcessPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(run_variant, [(name, variant) for variant in VARIANTS for name in good]))
        aluminium_offsets = list(pool.map(part_offsets, [n.removeprefix("matte_") for n in twins]))
        matte_offsets = list(pool.map(part_offsets, twins))
        tables = list(pool.map(table_residual, good))

    summary = {"scenes": len(good), "variants": {}}
    for variant in VARIANTS:
        chosen = [r for r in rows if r["variant"] == variant]
        gap = np.array([r["gap_error_mm"] for r in chosen])
        yaw = np.array([r["yaw_error_deg"] for r in chosen])
        summary["variants"][variant] = {
            "gap_mean_mm": float(gap.mean()),
            "gap_sd_mm": float(gap.std(ddof=1)),
            "low_cluster_scenes": int((gap < LOW_CLUSTER_MM).sum()),
            "yaw_mean_deg": float(yaw.mean()),
            "yaw_sd_deg": float(yaw.std(ddof=1)),
            "gap_alarms": int(sum(r["gap_alarm"] for r in chosen)),
            "yaw_alarms": int(sum(r["yaw_alarm"] for r in chosen)),
        }
    for label, offsets in (("aluminium", aluminium_offsets), ("matte", matte_offsets)):
        summary[f"{label}_good_twins_stereo_part_offsets"] = {
            key: {
                "mean_mm": float(np.mean([o[key] for o in offsets])),
                "sd_mm": float(np.std([o[key] for o in offsets], ddof=1)),
            }
            for key in ("profile_dx_mm", "bracket_dx_mm")
        }
    summary["stereo_table_residual"] = {
        area: {key: float(np.mean([t[area][key] for t in tables])) for key in ("median_mm", "over_20mm", "fill")}
        for area in ("near_parts", "whole_image")
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

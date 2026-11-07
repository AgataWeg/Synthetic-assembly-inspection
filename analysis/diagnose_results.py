"""Explain the test results without changing the method: everything here only reads and re-measures."""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import binomtest, fisher_exact

from pipeline import assembly_model as model
from pipeline import perception
from pipeline.run_batch import DATA_DIR

GAP, YAW = 1.0, 1.0


def load(condition):
    with open(f"results/main/{condition}.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        for k in ("gap_mm", "bracket_yaw_deg", "defect_gap_mm", "defect_yaw_deg", "gap_error_mm", "yaw_error_deg",
                  "true_gap_mm", "true_yaw_deg", "part_fill", "profile_points", "bracket_points"):
            r[k] = float(r[k])
        r["failed"] = r["failed"] == "True"
        r["pca_b"] = r["pca_fallback_bracket"] == "True"
    return {r["name"]: r for r in rows}


def gap_alarm(r):
    return r["failed"] or abs(r["gap_mm"]) > GAP


def yaw_alarm(r):
    return r["failed"] or abs(r["bracket_yaw_deg"]) > YAW


def part_detail(task):
    """Per-part pose errors along the profile axis, and how many end-face pixels survive."""
    name, condition = task
    scene_dir = DATA_DIR / name
    depth_ideal, labels, metadata = perception.load_scene(scene_dir)
    depth = depth_ideal if condition == "ideal" else np.load(scene_dir / "conditions.npz")[condition]
    captured = {}
    original = model.gap_m

    def capture(bracket_pose, profile_pose):
        captured["bracket"], captured["profile"] = bracket_pose.copy(), profile_pose.copy()
        return original(bracket_pose, profile_pose)

    model.gap_m = capture
    perception.measure(depth, metadata)
    model.gap_m = original

    truth = metadata["truth"]
    true_profile = np.array(truth["world_from_profile"])
    true_bracket = np.array(truth["world_from_bracket"])
    K = np.array(metadata["camera"]["K"])
    world_from_camera = np.array(metadata["camera"]["world_from_camera"])
    rows, cols = np.nonzero(labels == 1)
    z = depth_ideal[rows, cols].astype(np.float64)
    camera = np.column_stack(((cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)))
    local = (np.linalg.inv(true_profile) @ (world_from_camera @ camera.T)).T[:, :3]
    return {
        "name": name,
        "camera": world_from_camera[:3, 3].tolist(),
        "local_x_range": [float(local[:, 0].min()), float(local[:, 0].max())],
        "local_x_all": None,
        "end_low": int((np.abs(local[:, 0] - local[:, 0].min()) < 0.0007).sum()),
        "end_high": int((np.abs(local[:, 0] - local[:, 0].max()) < 0.0007).sum()),
        "end_high_valid": int(((np.abs(local[:, 0] - local[:, 0].max()) < 0.0007) & (depth[rows, cols] > 0)).sum()),
        "end_low_valid": int(((np.abs(local[:, 0] - local[:, 0].min()) < 0.0007) & (depth[rows, cols] > 0)).sum()),
        "profile_dx_mm": float((captured["profile"][:3, 3] - true_profile[:3, 3])[0] * 1000),
        "profile_dy_mm": float((captured["profile"][:3, 3] - true_profile[:3, 3])[1] * 1000),
        "bracket_dx_mm": float((captured["bracket"][:3, 3] - true_bracket[:3, 3])[0] * 1000),
        "bracket_dy_mm": float((captured["bracket"][:3, 3] - true_bracket[:3, 3])[1] * 1000),
    }


def residuals(name):
    """Depth error against ideal, per surface, in millimetres."""
    scene_dir = DATA_DIR / name
    ideal, labels, _ = perception.load_scene(scene_dir)
    stereo = np.load(scene_dir / "conditions.npz")["stereo"]
    out = {}
    for key, mask in (("profile", labels == 1), ("bracket", labels == 2), ("table", labels == 3)):
        selected = mask & (stereo > 0) & (ideal > 0)
        difference = (stereo[selected] - ideal[selected]) * 1000.0
        out[key] = {
            "fill": float(selected.sum() / max(mask.sum(), 1)),
            "median": float(np.median(difference)),
            "mad_sd": float(1.4826 * np.median(np.abs(difference - np.median(difference)))),
            "over5": float((np.abs(difference) > 5).mean()),
            "over20": float((np.abs(difference) > 20).mean()),
        }
    return out


if __name__ == "__main__":
    data = {c: load(c) for c in ("ideal", "datasheet", "measured", "stereo")}
    alu = [n for n, r in data["ideal"].items() if r["split"] != "good_calibration"]
    good = [n for n, r in data["ideal"].items() if r["split"] == "good_test"]

    print("--- 1. analytic gap errors: same scene, same cluster?")
    ds = np.array([data["datasheet"][n]["gap_error_mm"] for n in alu if data["ideal"][n]["split"] != "yaw"])
    me = np.array([data["measured"][n]["gap_error_mm"] for n in alu if data["ideal"][n]["split"] != "yaw"])
    st = np.array([data["stereo"][n]["gap_error_mm"] for n in alu if data["ideal"][n]["split"] != "yaw"])
    print("corr datasheet-measured", np.corrcoef(ds, me)[0, 1], "datasheet-stereo", np.corrcoef(ds, st)[0, 1])
    print("datasheet histogram edges -1.5..0.75 step .25:", np.histogram(ds, np.arange(-1.5, 1.0, 0.25))[0].tolist())
    print("measured  histogram:", np.histogram(me, np.arange(-1.5, 1.0, 0.25))[0].tolist())
    print("scenes in low cluster: datasheet", int((ds < -0.3).sum()), "measured", int((me < -0.6).sum()), "both", int(((ds < -0.3) & (me < -0.6)).sum()), "of", len(ds))

    with ProcessPoolExecutor(5) as pool:
        details = {c: {d["name"]: d for d in pool.map(part_detail, [(n, c) for n in good])} for c in ("ideal", "datasheet", "measured", "stereo")}
    print("--- 2. which part moves along the profile axis (good test scenes)")
    for condition in ("ideal", "datasheet", "measured", "stereo"):
        d = list(details[condition].values())
        err = np.array([data[condition][x["name"]]["gap_error_mm"] for x in d])
        pdx = np.array([x["profile_dx_mm"] for x in d])
        bdx = np.array([x["bracket_dx_mm"] for x in d])
        print(condition, "gap err vs profile dx corr", round(np.corrcoef(err, pdx)[0, 1], 2), "vs bracket dx", round(np.corrcoef(err, bdx)[0, 1], 2),
              "| profile dx mean/sd", round(pdx.mean(), 2), round(pdx.std(), 2), "| bracket dx mean/sd", round(bdx.mean(), 2), round(bdx.std(), 2))
    print("--- 3. end-face pixels vs cluster, measured condition")
    d = list(details["measured"].values())
    err = np.array([data["measured"][x["name"]]["gap_error_mm"] for x in d])
    for key in ("end_low", "end_high", "end_low_valid", "end_high_valid"):
        values = np.array([x[key] for x in d], dtype=float)
        print(key, "mean", round(values.mean(), 1), "min", values.min(), "corr with gap error", round(np.corrcoef(values, err)[0, 1], 2) if values.std() else None)
    azimuth = np.degrees(np.arctan2([x["camera"][1] for x in d], [x["camera"][0] for x in d]))
    elevation = np.degrees(np.arctan2([x["camera"][2] for x in d], np.hypot([x["camera"][0] for x in d], [x["camera"][1] for x in d])))
    print("camera position sample", d[0]["camera"], "azimuth range", azimuth.min().round(1), azimuth.max().round(1))
    print("corr gap err vs azimuth", round(np.corrcoef(azimuth, err)[0, 1], 2), "vs elevation", round(np.corrcoef(elevation, err)[0, 1], 2))
    low = err < -0.6
    print("low cluster n", int(low.sum()), "end_low_valid low/high cluster", np.mean([x["end_low_valid"] for x, l in zip(d, low) if l]).round(1), np.mean([x["end_low_valid"] for x, l in zip(d, low) if not l]).round(1),
          "end_high_valid", np.mean([x["end_high_valid"] for x, l in zip(d, low) if l]).round(1), np.mean([x["end_high_valid"] for x, l in zip(d, low) if not l]).round(1))
    print("elevation low/high cluster", elevation[low].mean().round(1), elevation[~low].mean().round(1), "azimuth", azimuth[low].mean().round(1), azimuth[~low].mean().round(1))

    print("--- 4. stereo: right and wrong alarms")
    stereo = data["stereo"]
    for level in (0.5, 1.0, 2.0, 3.0, 5.0):
        members = [r for r in stereo.values() if r["defect_gap_mm"] == level and r["material"] == "aluminium" and r["split"] == "gap"]
        alarms = [r for r in members if gap_alarm(r)]
        right = [r for r in alarms if not r["failed"] and abs(r["gap_error_mm"]) <= GAP]
        print(f"gap {level}: alarms {len(alarms)}/{len(members)}, failed {sum(r['failed'] for r in alarms)}, measured within 1 mm of truth {len(right)}; true gap min {min(r['true_gap_mm'] for r in members):.3f}")
    for level in (1.0, 2.0, 4.0):
        members = [r for r in stereo.values() if abs(r["defect_yaw_deg"]) == level and r["material"] == "aluminium" and r["split"] == "yaw"]
        alarms = [r for r in members if yaw_alarm(r)]
        right = [r for r in alarms if not r["failed"] and abs(r["yaw_error_deg"]) <= YAW]
        missed = [(r["name"], round(r["bracket_yaw_deg"], 2), round(r["true_yaw_deg"], 2)) for r in members if not yaw_alarm(r)]
        print(f"yaw {level}: alarms {len(alarms)}/{len(members)}, failed {sum(r['failed'] for r in alarms)}, within 1 deg {len(right)}, misses at >=2 deg {missed if level >= 2 else '-'}")
    good_rows = [stereo[n] for n in good]
    print("good scenes: failed", sum(r["failed"] for r in good_rows), "gap-only", sum(gap_alarm(r) and not r["failed"] for r in good_rows), "yaw-only", sum(yaw_alarm(r) and not r["failed"] for r in good_rows), "any", sum(gap_alarm(r) or yaw_alarm(r) for r in good_rows))
    measured_alarms = {n for n in good if gap_alarm(data["measured"][n]) or yaw_alarm(data["measured"][n])}
    stereo_alarms = {n for n in good if gap_alarm(stereo[n]) or yaw_alarm(stereo[n])}
    print("good scenes alarming: Carfagni", len(measured_alarms), "stereo", len(stereo_alarms), "both", len(measured_alarms & stereo_alarms))
    print("in-tolerance false rejects, gap 0.5 vs good, Fisher p:", fisher_exact([[9, 6], [sum(gap_alarm(r) for r in good_rows), 40 - sum(gap_alarm(r) for r in good_rows)]])[1])

    print("--- 5. stereo: bracket PCA seed used vs not (aluminium test scenes)")
    rows = [stereo[n] for n in alu]
    for label, chosen in (("seed rejected", [r for r in rows if r["pca_b"]]), ("seed accepted", [r for r in rows if not r["pca_b"]])):
        yaw = np.abs([r["yaw_error_deg"] for r in chosen])
        gap = np.abs([r["gap_error_mm"] for r in chosen])
        print(label, len(chosen), "yaw |err| median", round(np.median(yaw), 2), "p90", round(np.percentile(yaw, 90), 2), "gap median", round(np.median(gap), 2), "p90", round(np.percentile(gap, 90), 2), "failed", sum(r["failed"] for r in chosen))
    failed = [r for r in rows if r["failed"]]
    print("failed: seed accepted", sum(not r["pca_b"] for r in failed), "of", len(failed), "fill", round(np.mean([r["part_fill"] for r in failed]), 3), "others fill", round(np.mean([r["part_fill"] for r in rows if not r["failed"]]), 3))
    print("error vs fill corr (gap, yaw):", round(np.corrcoef([r["part_fill"] for r in rows], np.abs([r["gap_error_mm"] for r in rows]))[0, 1], 2), round(np.corrcoef([r["part_fill"] for r in rows], np.abs([r["yaw_error_deg"] for r in rows]))[0, 1], 2))

    print("--- 6. twins: matte vs aluminium on good scenes")
    twins = [n.removeprefix("matte_") for n, r in stereo.items() if r["material"] == "matte" and r["defect_gap_mm"] == 0]
    gap_pair = [(gap_alarm(stereo["matte_" + n]), gap_alarm(stereo[n])) for n in twins]
    any_pair = [(gap_alarm(stereo["matte_" + n]) or yaw_alarm(stereo["matte_" + n]), gap_alarm(stereo[n]) or yaw_alarm(stereo[n])) for n in twins]
    for label, pair in (("gap", gap_pair), ("any", any_pair)):
        only_alu = sum(not m and a for m, a in pair)
        only_matte = sum(m and not a for m, a in pair)
        p = binomtest(only_alu, only_alu + only_matte, 0.5).pvalue if only_alu + only_matte else 1.0
        print(label, "alarm only aluminium", only_alu, "only matte", only_matte, "both", sum(m and a for m, a in pair), "McNemar exact p", round(p, 5))
    matte_rows = [r for r in stereo.values() if r["material"] == "matte"]
    print("matte gap error mean/sd", round(np.mean([r["gap_error_mm"] for r in matte_rows]), 3), round(np.std([r["gap_error_mm"] for r in matte_rows], ddof=1), 3), "yaw mean/sd", round(np.mean([r["yaw_error_deg"] for r in matte_rows]), 3), round(np.std([r["yaw_error_deg"] for r in matte_rows], ddof=1), 3))
    print("matte any alarm on good:", sum(gap_alarm(stereo["matte_" + n]) or yaw_alarm(stereo["matte_" + n]) for n in twins), "of", len(twins))

    print("--- 7. depth residuals, stereo minus ideal, mm")
    names = {"aluminium": good[:20], "matte": ["matte_" + n for n in good[:20]]}
    for material, scenes in names.items():
        with ProcessPoolExecutor(5) as pool:
            residual = list(pool.map(residuals, scenes))
        for surface in ("profile", "bracket", "table"):
            line = {k: round(float(np.mean([r[surface][k] for r in residual])), 3) for k in residual[0][surface]}
            print(material, surface, line)

    print("--- 8. ideal-tuned threshold check: gap detection at stereo-tuned threshold")
    calibration = [r for r in stereo.values() if r["split"] == "good_calibration"]
    threshold = float(np.percentile([abs(r["gap_mm"]) for r in calibration], 95))
    for level in (0.5, 1.0, 2.0, 3.0, 5.0):
        members = [r for r in stereo.values() if r["defect_gap_mm"] == level and r["material"] == "aluminium" and r["split"] == "gap"]
        print(level, sum(r["failed"] or abs(r["gap_mm"]) > threshold for r in members), "of", len(members))
    print("threshold", threshold)

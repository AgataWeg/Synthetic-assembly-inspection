"""Compare depth conditions at the fixed tolerance: detection curves, a90 and false alarms.

The tolerance comes from the assembly requirement in Kleppe et al. 2017. A failed measurement
counts as an alarm, because an inspection that cannot measure must not pass the part.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from pipeline.run_batch import DATA_DIR, GAP_LEVELS_MM, YAW_LEVELS_DEG  # noqa: E402

MAIN_RESULTS = Path("results/main")
FIGURES = Path("results/figures")
GAP_TOLERANCE_MM = 1.0
YAW_TOLERANCE_DEG = 1.0
TARGET_DETECTION = 0.9
BOOTSTRAP_ROUNDS = 2000
SEED = 2026
CONDITIONS = ("ideal", "datasheet", "measured", "stereo")
LABELS = {
    "ideal": "Ideal depth",
    "datasheet": "Analytic, theoretical noise",
    "measured": "Analytic, Carfagni noise",
    "stereo": "Stereo simulation",
}
COLOURS = {"ideal": "#2a78d6", "datasheet": "#eb6834", "measured": "#1baf7a", "stereo": "#eda100"}
MARKERS = {"ideal": "o", "datasheet": "s", "measured": "^", "stereo": "D"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(condition):
    with (MAIN_RESULTS / f"{condition}.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("gap_mm", "bracket_yaw_deg", "defect_gap_mm", "defect_yaw_deg", "gap_error_mm", "yaw_error_deg", "part_fill"):
            row[key] = float(row[key])
        row["failed"] = row["failed"] == "True"
        row["pca_fallback_profile"] = row["pca_fallback_profile"] == "True"
        row["pca_fallback_bracket"] = row["pca_fallback_bracket"] == "True"
    return rows


def gap_alarm(row):
    return row["failed"] or abs(row["gap_mm"]) > GAP_TOLERANCE_MM


def yaw_alarm(row):
    return row["failed"] or abs(row["bracket_yaw_deg"]) > YAW_TOLERANCE_DEG


def wilson(successes, total, confidence=0.95):
    if total == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(0.5 + confidence / 2.0)
    rate = successes / total
    denominator = 1.0 + z**2 / total
    centre = (rate + z**2 / (2.0 * total)) / denominator
    spread = z * np.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) / denominator
    # Exact end values avoid rounding such as a 0.0000001 lower bound at zero successes.
    low = 0.0 if successes == 0 else centre - spread
    high = 1.0 if successes == total else centre + spread
    return (float(low), float(high))


def smallest_detected(levels, rates):
    """Interpolated defect size where detection first reaches the target; inf if it never does."""
    previous_level, previous_rate = None, None
    for level, rate in zip(levels, rates):
        if rate >= TARGET_DETECTION:
            if previous_level is None:
                return float(level)
            fraction = (TARGET_DETECTION - previous_rate) / (rate - previous_rate)
            return float(previous_level + fraction * (level - previous_level))
        previous_level, previous_rate = level, rate
    return float("inf")


def alarms_by_level(rows, key, alarm, levels):
    """Alarm flags per defect level, ordered by scene name so conditions stay paired."""
    grouped = {}
    for level in levels:
        members = sorted(
            (r for r in rows if r["material"] == "aluminium" and abs(r[key]) == level and r["split"] in ("gap", "yaw")),
            key=lambda r: r["name"],
        )
        grouped[level] = np.array([alarm(r) for r in members], dtype=bool)
    return grouped


def detection_curve(grouped, levels):
    rates = [float(grouped[level].mean()) if grouped[level].size else float("nan") for level in levels]
    return rates


def bootstrap_a90(all_grouped, levels, rng):
    """Resample scenes within each level, using the same scenes for every condition."""
    samples = {condition: [] for condition in all_grouped}
    for _ in range(BOOTSTRAP_ROUNDS):
        picks = {level: rng.integers(0, len(next(iter(all_grouped.values()))[level]), len(next(iter(all_grouped.values()))[level])) for level in levels}
        for condition, grouped in all_grouped.items():
            rates = [grouped[level][picks[level]].mean() for level in levels]
            samples[condition].append(smallest_detected(levels, rates))
    return {condition: np.array(values) for condition, values in samples.items()}


def interval(values):
    return (
        float(np.percentile(values, 2.5, method="lower")),
        float(np.percentile(values, 97.5, method="higher")),
    )


def paired_difference(first, second):
    """Percentile interval of a90 differences; rounds with undefined a90 fall at the extremes."""
    difference = np.where(np.isinf(first) & np.isinf(second), 0.0, first - second)
    return interval(difference)


def false_alarms(rows):
    good = [r for r in rows if r["split"] == "good_test" and r["material"] == "aluminium"]
    result = {}
    for name, alarm in (("gap", gap_alarm), ("yaw", yaw_alarm), ("any", lambda r: gap_alarm(r) or yaw_alarm(r))):
        count = sum(alarm(r) for r in good)
        result[name] = {"alarms": count, "scenes": len(good), "wilson_95": wilson(count, len(good))}
    return result


def error_statistics(rows):
    keep = [r for r in rows if r["material"] == "aluminium" and r["split"] != "good_calibration"]
    gap_rows = [r for r in keep if r["split"] != "yaw"]
    gap = np.array([r["gap_error_mm"] for r in gap_rows])
    yaw = np.array([r["yaw_error_deg"] for r in keep])
    return {
        "gap_error_mean_mm": float(gap.mean()),
        "gap_error_sd_mm": float(gap.std(ddof=1)),
        "gap_error_abs_median_mm": float(np.median(np.abs(gap))),
        "gap_error_abs_p95_mm": float(np.percentile(np.abs(gap), 95)),
        "yaw_error_mean_deg": float(yaw.mean()),
        "yaw_error_sd_deg": float(yaw.std(ddof=1)),
        "yaw_error_abs_median_deg": float(np.median(np.abs(yaw))),
        "yaw_error_abs_p95_deg": float(np.percentile(np.abs(yaw), 95)),
        "failed": int(sum(r["failed"] for r in keep)),
        "scenes": len(keep),
        "pca_fallback_profile": int(sum(r["pca_fallback_profile"] for r in keep)),
        "pca_fallback_bracket": int(sum(r["pca_fallback_bracket"] for r in keep)),
        "part_fill_mean": float(np.mean([r["part_fill"] for r in keep])),
    }


def threshold_side_note(all_rows):
    """A threshold tuned on ideal calibration scenes, applied to stereo, against one tuned on stereo."""
    def calibration(condition):
        rows = [r for r in all_rows[condition] if r["split"] == "good_calibration"]
        return float(np.percentile([abs(r["gap_mm"]) for r in rows], 95))

    stereo_good = [r for r in all_rows["stereo"] if r["split"] == "good_test" and r["material"] == "aluminium"]
    result = {}
    for source in ("ideal", "stereo"):
        threshold = calibration(source)
        alarms = sum(r["failed"] or abs(r["gap_mm"]) > threshold for r in stereo_good)
        result[source] = {"threshold_mm": threshold, "stereo_false_alarms": alarms, "scenes": len(stereo_good)}
    return result


def matte_comparison(stereo_rows):
    """Aluminium and matte twins share everything but the material."""
    by_name = {r["name"]: r for r in stereo_rows}
    twins = [(r, by_name[r["name"].removeprefix("matte_")]) for r in stereo_rows if r["material"] == "matte"]
    result = {}
    for label, side in (("matte", 0), ("aluminium", 1)):
        rows = [pair[side] for pair in twins]
        good = [r for r in rows if r["defect_gap_mm"] == 0.0]
        gap = np.array([r["gap_error_mm"] for r in rows])
        result[label] = {
            "scenes": len(rows),
            "gap_error_abs_median_mm": float(np.median(np.abs(gap))),
            "gap_error_sd_mm": float(gap.std(ddof=1)),
            "part_fill_mean": float(np.mean([r["part_fill"] for r in rows])),
            "failed": int(sum(r["failed"] for r in rows)),
            "false_alarms_gap": int(sum(gap_alarm(r) for r in good)),
            "good_scenes": len(good),
            "detected_1mm": int(sum(gap_alarm(r) for r in rows if r["defect_gap_mm"] == 1.0)),
            "detected_2mm": int(sum(gap_alarm(r) for r in rows if r["defect_gap_mm"] == 2.0)),
            "scenes_per_level": sum(r["defect_gap_mm"] == 1.0 for r in rows),
        }
    return result


def style_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, length=0)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def plot_detection(curves, far, out):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor="#fcfcfb")
    panels = (
        (axes[0], "gap", "Profile gap (mm)", GAP_LEVELS_MM, GAP_TOLERANCE_MM),
        (axes[1], "yaw", "Bracket yaw (degrees)", YAW_LEVELS_DEG, YAW_TOLERANCE_DEG),
    )
    for axis, name, xlabel, levels, tolerance in panels:
        axis.set_facecolor("#fcfcfb")
        style_axes(axis)
        axis.axvline(tolerance, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        axis.axhline(TARGET_DETECTION, color=GRID, linewidth=1.5)
        for condition in CONDITIONS:
            x = [0.0, *levels]
            y = [far[condition][name]["alarms"] / far[condition][name]["scenes"], *curves[condition][name]]
            axis.plot(
                x, y, color=COLOURS[condition], linewidth=2, marker=MARKERS[condition], markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=LABELS[condition], clip_on=False,
            )
        axis.set_xlabel(xlabel, color=MUTED)
        axis.set_ylim(0, 1.03)
        axis.set_xlim(0, max(levels))
        axis.text(tolerance, 1.05, "tolerance", color=MUTED, fontsize=9, ha="center", va="bottom")
        axis.text(max(levels), TARGET_DETECTION - 0.015, "90% target", color=MUTED, fontsize=8, ha="right", va="top")
    axes[0].set_ylabel("Alarm rate (0 = good assemblies)", color=MUTED)
    axes[0].legend(frameon=False, loc="lower right", labelcolor=TEXT, fontsize=9)
    figure.tight_layout()
    figure.savefig(out, dpi=160)
    plt.close(figure)


def plot_errors(all_rows, out):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    rng = np.random.default_rng(SEED)
    for axis, key, xlabel, tolerance, clip in (
        (axes[0], "gap_error_mm", "Gap error (mm), clipped at ±6", GAP_TOLERANCE_MM, 6.0),
        (axes[1], "yaw_error_deg", "Bracket yaw error (degrees), clipped at ±6", YAW_TOLERANCE_DEG, 6.0),
    ):
        axis.set_facecolor("#fcfcfb")
        style_axes(axis)
        axis.grid(False, axis="y")
        for bound in (-tolerance, tolerance):
            axis.axvline(bound, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        for position, condition in enumerate(reversed(CONDITIONS)):
            rows = [r for r in all_rows[condition] if r["material"] == "aluminium" and r["split"] != "good_calibration"]
            if key == "gap_error_mm":
                rows = [r for r in rows if r["split"] != "yaw"]
            values = np.clip([r[key] for r in rows], -clip, clip)
            axis.scatter(
                values, position + rng.uniform(-0.22, 0.22, len(values)), s=16, color=COLOURS[condition],
                edgecolors="#fcfcfb", linewidths=0.5,
            )
        axis.set_yticks(range(len(CONDITIONS)), [LABELS[c] for c in reversed(CONDITIONS)], color=TEXT)
        axis.set_xlabel(xlabel, color=MUTED)
        axis.set_xlim(-clip * 1.05, clip * 1.05)
    figure.tight_layout()
    figure.savefig(out, dpi=160)
    plt.close(figure)


def plot_depth_example(scene_name, out):
    scene_dir = DATA_DIR / scene_name
    geometry = np.load(scene_dir / "geometry.npz")
    ideal, labels = geometry["depth_m"] * 1000.0, geometry["labels"]
    stereo = np.load(scene_dir / "conditions.npz")["stereo"] * 1000.0
    rows, columns = np.nonzero((labels == 1) | (labels == 2))
    top, bottom = max(rows.min() - 40, 0), min(rows.max() + 40, ideal.shape[0])
    left, right = max(columns.min() - 40, 0), min(columns.max() + 40, ideal.shape[1])
    ideal_crop, stereo_crop = ideal[top:bottom, left:right], stereo[top:bottom, left:right]
    error = np.where(stereo_crop > 0, stereo_crop - ideal_crop, np.nan)
    limits = np.percentile(ideal_crop[ideal_crop > 0], [1, 99])

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor="#fcfcfb")
    sequential = matplotlib.colormaps["Blues_r"].copy()
    sequential.set_bad("#b9b8b2")
    panels = (
        (axes[0], ideal_crop, sequential, limits, "Ideal depth (mm)"),
        (axes[1], np.where(stereo_crop > 0, stereo_crop, np.nan), sequential, limits, "Stereo simulation (mm); grey = no depth"),
    )
    for axis, data, colour_map, (low, high), title in panels:
        image = axis.imshow(data, cmap=colour_map, vmin=low, vmax=high)
        axis.set_title(title, color=TEXT, fontsize=10, loc="left")
        figure.colorbar(image, ax=axis, fraction=0.035)
    diverging = matplotlib.colormaps["RdBu_r"].copy()
    diverging.set_bad("#b9b8b2")
    image = axes[2].imshow(error, cmap=diverging, vmin=-5, vmax=5)
    axes[2].set_title("Stereo minus ideal (mm), clipped at ±5", color=TEXT, fontsize=10, loc="left")
    figure.colorbar(image, ax=axes[2], fraction=0.035)
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(out, dpi=160)
    plt.close(figure)


def main():
    all_rows = {condition: load(condition) for condition in CONDITIONS}
    rng = np.random.default_rng(SEED)
    summary = {
        "tolerance": {"gap_mm": GAP_TOLERANCE_MM, "yaw_deg": YAW_TOLERANCE_DEG},
        "conditions": {},
        "a90_differences_vs_stereo": {},
    }
    curves = {condition: {} for condition in CONDITIONS}
    far = {condition: false_alarms(all_rows[condition]) for condition in CONDITIONS}

    for name, key, alarm, levels in (
        ("gap", "defect_gap_mm", gap_alarm, GAP_LEVELS_MM),
        ("yaw", "defect_yaw_deg", yaw_alarm, YAW_LEVELS_DEG),
    ):
        grouped = {c: alarms_by_level(all_rows[c], key, alarm, levels) for c in CONDITIONS}
        bootstrap = bootstrap_a90(grouped, levels, rng)
        for condition in CONDITIONS:
            rates = detection_curve(grouped[condition], levels)
            curves[condition][name] = rates
            entry = summary["conditions"].setdefault(condition, {})
            entry[f"{name}_detection"] = dict(zip(map(str, levels), rates))
            entry[f"{name}_a90"] = smallest_detected(levels, rates)
            entry[f"{name}_a90_ci95"] = interval(bootstrap[condition])
        summary["a90_differences_vs_stereo"][name] = {
            condition: paired_difference(bootstrap["stereo"], bootstrap[condition])
            for condition in CONDITIONS if condition != "stereo"
        }

    for condition in CONDITIONS:
        summary["conditions"][condition]["false_alarms"] = far[condition]
        summary["conditions"][condition]["errors"] = error_statistics(all_rows[condition])
    summary["threshold_side_note"] = threshold_side_note(all_rows)
    summary["matte_vs_aluminium_twins"] = matte_comparison(all_rows["stereo"])

    (MAIN_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_detection(curves, far, FIGURES / "fig_detection.png")
    plot_errors(all_rows, FIGURES / "fig_errors.png")
    plot_depth_example("gap_2mm_000", FIGURES / "fig_depth.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

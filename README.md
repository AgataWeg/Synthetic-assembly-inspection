# Synthetic RGB-D assembly experiment

This project asks how small an assembly defect a low-cost depth camera can detect on metal, and whether a realistic camera model changes that answer. The same measurement (PCA + ICP per part) runs on the same scenes with ideal depth and with depth from a camera model. The reference camera is the Intel RealSense D415.

Current state: the experiment is complete, and its results are in `results/` (see Results below). All camera behaviour is simulated; nothing was checked against a real D415. The measurement method and camera parameters are frozen.

## Setup

Requires Python 3.12 and Blender 5.2. The scripts look for Blender at its default Windows location; elsewhere, set the `BLENDER` environment variable to the Blender executable.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below assume the environment is active and are run from the project folder.

## Run

Run these in order to reproduce the results. Finished scenes are skipped, so an interrupted render can be restarted. On a 12-thread laptop CPU, rendering takes about 65 minutes and measuring about 30.

```bash
python validate_camera.py                    # flat-target check against published noise
python run_batch.py --ir --dev               # 20 scenes used only for method tuning
python run_batch.py --ir                     # 220 experiment scenes
python build_conditions.py                   # camera-model depth for each scene
python evaluate.py --condition ideal         # measure each depth condition
python evaluate.py --condition datasheet
python evaluate.py --condition measured
python evaluate.py --condition stereo
python analyze.py                            # summary.json and figures
```

The follow-up checks are run after those, and do not feed back into the method:

```bash
python diagnose_results.py > results/diagnostics/diagnostics.log   # alarm breakdowns, part errors, matte twins
python ablation.py                           # switches analytic camera effects on and off
python second_view.py --view steep           # same seeds, steep camera
python second_view.py --view mirror          # same seeds, mirrored camera
python check_view.py > results/diagnostics/view_check.log           # whether each view sees the profile's free end
python render_examples.py                    # colour examples and fig_scene.png
python render_views.py                       # one scene from the main, mirrored and steep views, fig_views.png
```

## Results

| Folder | Contents |
| --- | --- |
| `results/main/` | Per-scene measurements for each depth condition (`ideal`, `datasheet`, `measured`, `stereo`) and `summary.json` |
| `results/figures/` | Figures used in the report |
| `results/camera_check/` | Flat-target noise compared with published values |
| `results/diagnostics/` | Post-hoc explanations: effect switching, alarm breakdowns, view visibility |
| `results/other_views/` | The steep and mirrored camera checks |
| `results/dev/` | Measurements on the 20 tuning scenes |
| `results/logs/` | Console output of the runs |

To try a quick subset, use `python run_batch.py --ir --per-level 3 --good 3 --no-matte`.

To render a single scene (use the full path to Blender if it is not on your `PATH`):

```bash
blender -b --python generate_scene.py -- --out data/generated/one_scene --seed 1 --gap-mm 2
```

`generate_scene.py` also takes `--bracket-yaw-deg`, `--material matte`, `--no-rgb`, `--ir`, `--camera-azimuth-deg MIN MAX` and `--camera-elevation-deg MIN MAX`.

## Files

- `assembly_model.py`: part geometry, camera constants and pose helpers, shared by Blender and the measurement code
- `generate_scene.py`: renders one scene with Cycles
- `run_batch.py`: defines the experiment's scenes and renders them
- `camera_model.py`: analytic noise and holes, and stereo matching on the IR pair
- `validate_camera.py`: renders flat targets and compares their noise with published values
- `build_conditions.py`: writes each scene's camera-model depth maps
- `perception.py`: table removal, part separation, PCA and ICP
- `evaluate.py`: measures every scene for one depth condition and writes a CSV
- `analyze.py`: detection curves, a90, false alarms and figures
- `diagnose_results.py`, `ablation.py`, `check_view.py`: post-hoc explanations of the results
- `second_view.py`: repeats 35 scenes from another camera direction
- `render_examples.py`: colour renders and the example figure
- `render_views.py`: colour renders of one scene from the three camera directions

Each scene folder contains:

- `rgb.png`: colour image, unless `--no-rgb` was used
- `geometry.npz`: `depth_m` (planar depth in metres, 0 where nothing was hit) and `labels` (per-pixel part ID)
- `ir_left.png`, `ir_right.png`: 16-bit linear stereo pair, if `--ir` was used
- `conditions.npz`: camera-model depth maps from `build_conditions.py`
- `metadata.json`: true part poses and defects, camera intrinsics and extrinsics, materials and render time

The experiment's scenes are in `data/generated/experiment/`. The other views are in `second_view/` (steep) and `mirror_view/`, and the colour examples are in `examples/`, all under `data/generated/`.

## Conventions

- `world_from_camera` uses x right, y down, z forward.
- Integer pixel coordinates refer to pixel centres, so the principal point is at (639.5, 359.5).
- The gap is the profile origin's x position in the bracket frame.
- Label IDs are profile 1, bracket 2, table 3 and support pad 4. Labels are used only for evaluation, never by the measurement.

## Scene assumptions

- The camera is about 0.45 m from the assembly centre, using the D415's horizontal field of view of 69.4°, as reported by Carfagni et al.
- A disparity shift of 50 gives a 0.29–1.02 m depth range. The default shift of 0 would put the near end of the profile, at about 0.36 m, inside the minimum range.
- Good assemblies include small placement variation, recorded as ground truth.
- Profile length has no tolerance, and the measurement uses the exact CAD model. Real length variation would look like a gap error.
- The gap measurement needs a view of the profile's free end. Elevations of 68–78° at this azimuth do not see it, and the gap then can't be measured, even with ideal depth.
- The projector's dot count, offset and power, and the sensor's noise constants and exposure, are assumptions and do not come from sources.

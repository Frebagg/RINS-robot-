# Ring Detector

Detects colored rings (red, green, blue, black) on the TurtleBot 4 using the Orbbec Gemini 335L depth camera. Publishes confirmed ring positions to `/ring_coords` for downstream nav/planning.

## What this node does, in one paragraph

It synchronizes RGB and depth frames, builds a per-frame binary mask of "this pixel is a ring-colored surface in the right depth range and isn't a boundary," fits ellipses to the resulting contours, screens those ellipses through several geometric and depth-based filters, classifies each surviving ellipse's rim color, back-projects its center into the camera frame, transforms to `odom`, and runs a per-color nearest-neighbor tracker that requires multiple consistent detections before a ring is "confirmed" and published.

## Topics

### Subscribes

| Topic | Type | Notes |
|---|---|---|
| `/gemini/color/image_raw` | `sensor_msgs/Image` (bgr8) | RGB stream |
| `/gemini/depth/image_raw` | `sensor_msgs/Image` (uint16 mm) | depth aligned to color |
| `/gemini/depth/camera_info` | `sensor_msgs/CameraInfo` | intrinsics for back-projection |

QoS: `qos_profile_sensor_data` (best-effort, depth=5). Required because the camera driver publishes best-effort; a reliable subscriber would silently never receive anything.

RGB and depth are paired through a `message_filters.ApproximateTimeSynchronizer` with `slop = 0.08 s`. The detector only runs when a matched pair arrives.

### Publishes

| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/ring_coords` | `rins_robot/RingCoords` | 5 Hz timer | Confirmed rings only. Each message lists currently-tracked rings with `ids`, `points` (geometry_msgs/Point in `odom` frame), and `colors` (string). |

A ring must be hit at least `PUBLISH_MIN_HITS` times before it shows up in `/ring_coords`. This is for the downstream consumer's benefit — they get stable detections, not flickers.

### TF

Reads transforms from `gemini_color_optical_frame` (or whatever frame the depth `CameraInfo` carries) to `odom`. We use `odom` rather than `map` because the robot doesn't always have SLAM running; `odom` is always published by the base.

## Pipeline

Each synchronized (RGB, depth) pair goes through these stages in order:

### 1. Depth pre-processing
- Convert uint16 millimeters → float32 meters.
- Median-blur (5×5) the raw uint16 depth before conversion. Removes salt-and-pepper noise the Gemini emits at edges and on dark surfaces.
- Clip out-of-range pixels to 0.
- **Blank out the lower half of the depth image.** Rings are hung on arms above the floor; the lower half is floor and robot base. Eliminating it upfront removes most clutter for free.

### 2. Binary segmentation mask
Build a binary mask of pixels that pass *all* of:
- Depth in `[BINARY_DEPTH_MIN_M, BINARY_DEPTH_MAX_M]`.
- Neighborhood color+depth consistency (see below).
- Color falls within one of the four ring-color HSV ranges.

Then morphologically open + close to clean up; sparse-neighborhood cleanup to kill speckle; optional hollow-out to thin solid blobs into outlines.

#### Neighborhood color+depth consistency
For each pixel, count its K×K neighbors that agree on *both* color and depth:
- Color similarity: Lab Euclidean distance below `NEIGHBOURHOOD_DE_THR` (default 15).
- Depth similarity: relative distance below `NEIGHBOURHOOD_DZ_REL_THR * z_center` (default 3% of depth — handles the camera's noise scaling with distance).
- Neighbor must have valid depth to count at all.

A pixel survives only if at least `NEIGHBOURHOOD_MIN_FRAC` (default 60%) of its *valid* neighbors agree.

The kernel size varies by depth: K=9 for [0, 0.5) m, K=5 for [0.5, 1.0) m, K=3 for [1.0, ∞) m. This is because a ring's rim has roughly constant physical thickness but variable pixel thickness — a fixed K would either over-include background pixels for far rings or under-sample for near rings.

This is the most expensive single operation in the pipeline (3 passes of K×K shifts on a full-resolution image). See "Performance" below.

#### Ring-color filter
HSV mask matching any of red, green, blue, black. The HSV ranges are in `COLOR_RANGES_HSV` at the top of the file. Red wraps the hue circle and uses two ranges; black is low-V with low-S to avoid catching saturated dark colors.

#### Sparse-neighborhood cleanup
A box filter computes the local on-fraction in a K×K window; any "on" pixel whose neighborhood is less than `BINARY_CLEANUP_MIN_FRAC` (50%) on gets removed. Kills isolated speckle and thin tendrils. Softer than morphological erosion (which removes pixels with *any* off neighbor).

#### Hollow-out (optional, `HOLLOW_ENABLE`)
Inverse of sparse cleanup: removes "on" pixels whose neighborhood is *more* than `HOLLOW_MAX_FRAC` (85%) on. Converts solid filled blobs into their outlines. Useful because solid blobs of ring color (a person's clothing, a poster) can otherwise produce ring-like contours; converting them to outlines lets the residual check (below) catch the bad fit.

### 3. Contour finding and ellipse fitting
`cv2.findContours` with `RETR_TREE` (returns outer + inner contour hierarchy). For each contour:

1. **Min contour points** (15) — `fitEllipse` needs ≥5 to run, 15+ to be stable.
2. **Circularity** — `4πA/P²`. Real circles score 1.0; tilted rings around 0.4; clutter usually below `MIN_CIRCULARITY = 0.25`.
3. **`fitEllipse`** — produces (cx, cy), (axis_a, axis_b), angle.
4. **Axis validity** — both axes > 0.
5. **Ellipse fit residual** — see below.
6. **Hole check** — see below.

#### Ellipse fit residual
`fitEllipse` will fit *something* to any 5+ points. To check whether the resulting ellipse actually matches the input, transform the contour points into the ellipse's local frame (translate to center, rotate to align axes, scale by 1/(semi-axes)). In that frame the ellipse becomes the unit circle, and the residual is the mean of `||p|| - 1` over all points. Real rings score ~0.05; L-shapes and partial arcs score 0.2+. Reject above `MAX_ELLIPSE_RESIDUAL = 0.15`.

Catches what circularity misses: contours that *summarize* as roundish but have local non-roundness (a square with a curved corner, a half-arc).

#### Hole check (`_ellipse_has_real_hole`)
A real ring has a hole; a flat painted disc doesn't. We measure depth in two regions:
- **Rim depth**: median of valid depths sampled around the ellipse perimeter.
- **Hole depth**: median of valid depths in the inner ellipse region (scaled by `INNER_SCALE = 0.45`).

The candidate is "real" if either:
- ≥60% of hole pixels have invalid depth — i.e., the IR pattern saw through into far space, OR
- Hole depth ≥ rim depth + `HOLE_DEPTH_MARGIN_M` (3 cm) — the hole sees a background measurably behind the ring.

Margin of 3 cm sits above sensor noise (~2 cm) but well below any actual gap. Only fails on genuinely flat surfaces.

**This check only works when rings have at least some space behind them.** If rings are mounted directly against a wall, the hole sees the wall at rim depth, and real rings get rejected. In that scenario, disable this check.

### 4. Rim color classification
For each surviving ellipse, build a rim annulus (outer ellipse minus an inner one scaled by `INNER_SCALE`). Count how many rim pixels fall within each color's HSV range. The color with the highest count wins, provided it exceeds `COLOR_DOMINANCE_THR = 0.30` of total rim pixels. Otherwise `'unknown'`, which rejects the candidate.

This is a redundancy with the pixel-level ring color filter, but a useful one: the pixel filter says "this pixel could be red," the rim classifier says "the rim is *coherently* red." Catches cases where a candidate's rim is a mixture of all four colors at low fractions each — usually a sign of a noisy blob, not a ring.

### 5. 3D back-projection
For each candidate:
- Sample valid depths on the ring's rim (perimeter samples).
- Sample valid depths in a small patch at the ellipse center.
- Use the perimeter median as the ring's depth (the center is the hole, so its depth would be wrong).
- Back-project the image-space center through the camera intrinsics to get a 3D point in the camera frame.

### 6. TF to `odom`
Look up the transform from the camera optical frame to `odom` at the image timestamp; apply to the 3D point. 20 ms timeout — if TF isn't ready, drop the detection rather than blocking the executor.

### 7. Tracking
Each detection gets associated with a per-color track:
- **Pending tracks**: new detections start here. Need `PENDING_MINHITS = 3` consistent hits within 4 s to graduate.
- **Confirmed tracks**: published. Update position via running average; pruned if not seen for too long.
- **Association** uses separate XY and Z distance thresholds — XY because of pose drift, Z because of depth noise. Both must be within thresholds.

Confirmed tracks get published to `/ring_coords` once they have at least `PUBLISH_MIN_HITS = 6` hits.

## Visualisation

Three OpenCV windows open while running:

- **RGB**: input frame with detected ellipses drawn, plus a HUD showing accepted/pending/confirmed counts.
- **Binary**: the segmentation mask after all stages (what `findContours` sees).
- **Depth**: depth visualized as 8-bit gray, useful for spot-checking that the right things are in range.

Esc in any window shuts the node down cleanly.

If you're SSH'd in and don't have X forwarding, the windows won't render. To run headless, set `SHOW_WINDOWS = False` near the top of the file. A debug image is also published as a ROS topic (`/ring_detector/debug_image` etc.) and can be viewed with `rqt_image_view` from anywhere.

## Diagnostics: the reject summary log

Once per second the node prints a reject summary like:

```
Reject summary (1s): frames=3 contours=42 candidates=2 accepted=2
  contour_pts=25 circ=7 axis=0 resid=4 ratio=4 size=1 hole=4
  color=0 depth=0 tf=0 nonfinite=0
```

- `frames`: synchronized (RGB, depth) pairs processed.
- `contours`: total contours considered.
- `candidates`: contours that survived all geometry filters.
- `accepted`: candidates that reached tracking.
- Everything else counts contours rejected at that specific filter.

The biggest "rej_X" counter is the dominant filter. If a real ring isn't being detected, find which filter is killing it and loosen that threshold.

## Tunable parameters

All at the top of `detect_rings.py`. Names should be self-explanatory; the README sections above explain what each does. A few worth flagging:

| Param | Default | Effect of raising | Effect of lowering |
|---|---|---|---|
| `DEPTH_MAX_M` | 1.50 m | Detect farther rings | Reduce far-clutter noise |
| `NEIGHBOURHOOD_DE_THR` | 15 | More forgiving of color variation on the rim | Stricter, may fragment rim under uneven lighting |
| `COLOR_DOMINANCE_THR` | 0.30 | More strict rim color check | More permissive, may accept mixed-color blobs |
| `HOLE_DEPTH_MARGIN_M` | 0.03 m | Reject more "ring against wall" cases | Accept those but risk flat printed discs |
| `MAX_ELLIPSE_RESIDUAL` | 0.15 | Stricter fit, may miss occluded rings | More permissive, more clutter |
| `PUBLISH_MIN_HITS` | 6 | Slower to confirm but fewer false positives | Faster confirmation but more spurious rings |

## Performance

On the Pi the dominant cost is the neighborhood consistency filter — three passes of K×K shifts on a full-resolution image. Total per-frame time is in the range of 60-150 ms depending on contour count.

If frame rate is too low:

1. **Disable the neighborhood consistency filter.** The ring-color filter does most of the work; the consistency filter is robustness, not load-bearing. Comment out the `cc_mask` AND in the binary build.
2. **Drop the K=9 pass.** Change the bucket for [0, 0.5) m to use K=5 instead. ~50% saving on the consistency filter.
3. **Run the consistency filter at half resolution.** Downsample first, upsample the result mask. ~4× saving on that stage.

The camera also publishes at 8 Hz on this hardware. Even with a 0-cost detector, you can't go faster than the camera.

## Running

```bash
# Build
cd ~/Desktop/rins
colcon build --packages-select rins_robot --symlink-install
source install/setup.bash

# Run
python3 src/rins_robot/src/detect_rings.py

# Or after install:
ros2 run rins_robot detect_rings
```

To run headless: set `SHOW_WINDOWS = False` near the top of `detect_rings.py`. Debug images still go to ROS topics, viewable with `rqt_image_view` from any machine in the same `ROS_DOMAIN_ID`.

## Known limitations

- **Rings must have space behind them.** The hole check fails for rings flush against a wall. Disable with `_ellipse_has_real_hole = lambda *a: True` if needed.
- **Lighting-dependent color thresholds.** `COLOR_RANGES_HSV` is tuned for a typical classroom. Strong colored ambient light (sunset through a window, colored stage lights) will shift hues and may require retuning.
- **Depth/RGB parallax.** The two cameras have a small physical baseline. At close range (~30 cm) the rim color check can sample slightly off the actual rim. Acceptable in practice but worth knowing.
- **Single-threaded executor.** Long-running callbacks (~150 ms when the detector lags) block TF and other callbacks. Not a problem at current detection cost, but if you add expensive checks, consider a `MultiThreadedExecutor`.
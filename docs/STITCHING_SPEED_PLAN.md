# Stitching speed optimization proposal

Branch: `fix/IY-speed-optimization`, based on refreshed `origin/main` (`44d2bdc`).
Scope: reconstruction speed; background-removal implementation belongs to the other engineer.

## Measured baseline

Reviewed the previous conversation, HANDOFF.md, reconstruction code, and completed
session `248d19d41557`. Its original outputs remain the baseline.

| Stage | Seconds | Minutes |
| --- | ---: | ---: |
| Feature extraction (CPU) | 126.34 | 2.11 |
| Sequential matching (CPU) | 382.58 | 6.38 |
| Fixed-pose triangulation | 26.17 | 0.44 |
| Undistortion | 0.97 | 0.02 |
| Dense stereo (CUDA) | 4943.68 | 82.39 |
| Fusion | 143.21 | 2.39 |
| Crop/background removal | 2.84 | 0.05 |
| Meshing | 17.36 | 0.29 |
| Total pipeline | 5679.46 | 94.66 |

Stage totals leave approximately 33 seconds for other work. This measures the
desktop pipeline, excluding capture, ZIP transfer and extraction.

139 images at 1920 x 1440; 139 registered cameras; 66,347 sparse points;
mean sparse reprojection error 1.022 px. Dense fusion produced 354,712 points,
268,799 after foreground processing. These counts do not establish shape accuracy.

Dense stereo is 87% of runtime; even eliminating all feature extraction and
matching would leave over 86 minutes. Background removal is not this run's bottleneck.

## Findings in current code

- `arkit_pipeline.py` already uses fixed ARKit poses. There is no incremental
  camera mapper to remove on this path. Feature matching supplies triangulation
  and view-overlap evidence; skipping it requires a separate validated design.
- Both native extraction and matching explicitly select `pc.Device.cpu`.
  The launcher fixes CPU threads at four.
- `colmap_pipeline.py` passes geometric consistency but no resolution or work
  limits to PatchMatch. The installed CUDA COLMAP 4.2.1 help confirms defaults:
  unlimited image size, five iterations, fifteen samples, window radius five.
- The completed workspace's `stereo/patch-match.cfg` uses `__auto__, 20`.
  Dense view count is separate from sequential matching's overlap of fifteen.
- The scan box is applied after fusion. It currently saves no dense image work.
- `colmap_common.py::run_sfm_and_undistort` deletes the entire working directory
  at the start of a rerun. Completed expensive stages cannot be reused.

## Proposed implementation order

### 1. Benchmark harness and explicit reconstruction profiles

Record effective settings, executable/version, hardware, input fingerprint,
selected images and per-stage times together. Retain the current settings as
the reference profile. Run candidates in isolated sessions/workspaces; never
delete or overwrite the completed baseline. Benchmark stages independently so
dense tuning reuses the same sparse model rather than repeating matching.

### 2. Bound dense image resolution (first performance change)

Compare PatchMatch `max_image_size` values 1280 and 960 against 1920, holding
all other settings fixed. At the same aspect ratio these contain about 44% and
25% of the original pixels. Those are pixel-work ratios, not promised runtime
speedups. Keep geometric consistency enabled and ensure stereo fusion uses
compatible settings and map dimensions.

Use COLMAP's calibrated resizing support; do not resize raw images without
updating camera intrinsics. Do not hand-edit metric ARKit poses to accommodate
image changes. Inspect thin edges and small food features before picking a default.

### 3. Reduce dense source views

Compare auto source-view caps of 10 and 6 against 20, initially at the winning
resolution. Keep reference views unchanged for this experiment. Rank overlap
and useful triangulation baseline rather than choosing only adjacent frames;
near-identical views add work with little depth evidence. Preserve overlap at
the orbit seam. Separate the quality impact from resolution changes.

### 4. Select useful keyframes using ARKit poses and sharpness

Try approximately 60-90 well-distributed frames against all 139. Reject blurred
or nearly duplicate views, but retain changes in elevation, occluded surfaces,
and full orbit coverage. Use motion and view direction, not uniform time sampling
alone. Keep imported originals immutable; export a derived model/manifest whose
image set, IDs, intrinsics and fixed poses agree. Update the native image-set
validation deliberately rather than bypassing it.

This reduces extraction, matching, reference depth maps and fusion together.
Treat frame-count targets as experiments, not hard universal limits.

### 5. Reuse completed work on retries and foreground-only changes

Replace unconditional work deletion with stage checkpoints and dependency
fingerprints. Include image content, calibration/poses, selection, software
versions and effective stage options. Dense changes invalidate dense stages;
foreground/meshing changes reuse original fused geometry. Keep the raw fused
cloud separately: today's foreground step overwrites `fused.ply`. Publish stage
completion atomically and ignore incomplete artifacts after interruption.

This primarily accelerates iteration/recovery; it does not shorten a cold scan.

### 6. Accelerate the remaining feature/matching work

Probe GPU support in the actual pycolmap build before selecting CUDA; the CUDA
CLI and Python package have independent capabilities. Use a compatible CLI path
if needed, preserving calibrated camera identities and model-format conversion.
Keep a CPU fallback. Benchmark CPU thread counts on this host rather than
globally raising the container-oriented cap. Consider pose-informed pair lists
after dense improvements; preserve useful baselines and orbit closure.

### Later experiments

- Three versus five PatchMatch iterations, after resolution/source-view tests.
- Project the scan volume into each camera for calibrated image crops and
  conservative depth bounds. Handle partial visibility, margins and intrinsics
  correctly; a point-cloud crop or black background mask alone is not an MVS ROI.
- A photometric-only preview can be a separate explicitly lower-quality mode.
  Do not disable geometric consistency in the normal dense profile without
  evaluating outliers and changing fusion inputs/progress expectations together.
- Direct depth fusion/TSDF is a separate future path: this capture explicitly
  reports `depth_available: false`, so it cannot replace RGB MVS for this ZIP.

## Benchmark and acceptance

Use the same real ZIP and workstation with one GPU run at a time. Preserve the
existing full scan and create new runs with a reproducible settings manifest.
Start with resolution only, then source views, then keyframes; compare combined
winners against baseline and repeat the final candidate to check variability.

Report total and stage runtime, selected/registered cameras, map completion,
point/triangle counts, reprojection error and failures. Inspect identical views
for missing surfaces, edge loss, holes and floating geometry. Measure object
dimensions against a physical reference where available. Compare the object
region before background removal to isolate this work from segmentation changes.
Baseline-mesh agreement alone is not ground-truth accuracy.

An initial engineering target is under 20 minutes on this workstation while
preserving useful geometry; this is a target, not a measured result. A 15-minute
total needs more than a 5.5x dense-stage reduction if other work stays unchanged.
Publish speed and quality tradeoffs together before changing the default profile.

## First implementation slice

Add recorded, configurable dense resolution and source-view limits, plus an
isolated benchmark entry point. Keep current defaults until comparisons support
a faster profile. Add focused checks for option wiring, calibrated dimensions,
profile metadata and preservation of baseline artifacts. Validate the selected
profile with an actual GPU reconstruction; unit tests cannot establish speed.

Implementation status and usage are recorded in `STITCHING_SPEED_USAGE.md`. No expensive reconstruction has been launched for this optimization work.

# MealScan pipeline core

Pure-Swift implementation of the meal-scan pipeline in `tare_impl.md` (Draft v2).
No ARKit: every module takes plain value types, so the whole core builds and tests
on macOS. The iOS pod compiles these same sources (see `TareCaptureNative.podspec`);
`../KeyframeCapture.swift` is the capture-layer bridge from `ARFrame`.

```sh
cd mobile/modules/tare-capture/ios/MealScan && swift test
```

## Conventions (tare_impl.md §13)

- Meters internally; unit suffixes on every quantity (`radiusM`, `heightM`, `rmsResidualPx`).
- World = ARKit world, +Y up. Poses are `worldFromCamera` in ARKit camera axes
  (+X right, +Y up, camera looks along −Z). Images are native sensor orientation,
  u right / v down, integer pixel centers (ARKit intrinsics).
- `Plane` frames: +Y along the normal; `coordinates2D` gives in-plane (X, Z).
- Geometry runs in `Double`; ARKit `Float` matrices convert at the capture boundary.
- All tunables live in `PipelineConfig`, each citing a source (mostly "initial guess").

## What exists

| Module | Spec | Status |
| --- | --- | --- |
| `Keyframe`, `DepthFrame`, `CameraIntrinsics`, `Plane`, `Ray` | §5, §13 | Done. `KeyframeCapture.swift` builds them from `ARFrame` / `ARPlaneAnchor`. |
| `KeyframeSelector` | §6.1 | Tracking, sharpness, tilt ≤ 30° for A, baseline ≥ 10 cm for B. **Missing:** "dish fully in frame" (needs the segmenter). |
| `RimTriangulator` | §6.5 | Two-view LM on (center, radius, height), weighted per keypoint, no cross-view correspondences needed. Covariance-based uncertainties. 0.1 mm on synthetic rims. |
| `ContourRimFitter` | §6.4 fallback, §6.6 | Single view: rim pixels → plane at a given height → RANSAC circle with adaptive inlier refinement. Height must come from `RimHeightRegressor` (not built); nil uses the table and is down-weighted. |
| `CircleFit` | — | Algebraic, geometric, RANSAC. Plane-space circles replace image ellipses: a round rim projected onto any parallel plane stays a circle. |
| `ContainerProfileProvider`, `ItemHeightProvider` | §6.8, §6.15 | Protocols with the guide's signatures plus `No…` defaults returning nil. |
| `LidarItemHeightProvider` | §6.15 | Top-surface height (p90) inside the item mask, eroded at edges, vs the rim plane (default) or table. |
| `LidarContainerProfileProvider` | §6.8 | Radial interior profile + inner depth. Measures the visible surface, so food counts. |
| `LidarRimReference` | §4 validation | Ground-truth rim from depth to score the camera-only pipeline (`compare(_:)`). Eval only. |

### How rim sizing works

`RimTriangulator`'s residual for each keypoint:
1. Intersect its ray with the candidate rim plane.
2. Snap the hit to the nearest point on the candidate circle.
3. Reproject that point into the same view; the residual is its pixel offset from the keypoint.

This is zero at the truth and needs no correspondences. A grid search over height seeds it: at the true height, all views' rays land on one circle. A single view cannot see height, so the triangulator refuses one view (`insufficientViews`); use `ContourRimFitter` with a height there.

## Decisions that differ from tare_impl.md

LiDAR covers the hooks and ground truth (product decision, 2026-10-24):

- §13 bans `sceneDepth` / `smoothedSceneDepth`. They are used **only** in the capture
  layer (`KeyframeCapture.swift`, and the live point cloud in `TareCaptureModule.swift`
  from `feat/on-device-depth`). Pipeline code sees `Keyframe.depth`, which feeds only
  the LiDAR hook providers and `LidarRimReference`.
- **Rim/container sizing never reads depth.** LiDAR phones run the same geometric
  pipeline as everyone else (§4). Keep it that way so LiDAR captures stay a fair test
  of the camera-only path.
- §4 says LiDAR is internal-only. The hook providers make it product-facing on LiDAR
  phones. `tare_impl.md` should be amended to record this.

## Not built yet (non-LiDAR work starting next)

- `CaptureSession` UX (§6.1) and `CaptureRecorder` / `session.json` (§6.2). The current
  recorder is still the v1 ARKit RGB manifest from the original prototype.
- Learned models and `ModelRegistry`: ContainerSegmenter, RimDetector, RimHeightRegressor,
  SizeCorrector, ItemSegmenter, ItemClassifier.
- MaskPostprocessor, LabelMerger, FootprintMeasurer, ResultAssembler / `MealScanResult` (§7).
- Eval CLI and `Fixtures/Captures/`. Tests currently use the synthetic scenes in
  `Tests/MealScanTests/SyntheticScene.swift` (analytic rims plus a ray-marched depth renderer).
- Sharpness scoring in the capture layer (`Keyframe.sharpness` is an input).

## Test notes

- Geometry gates are 0.1 mm (§13).
- LiDAR tests use millimetre tolerances: a depth pixel covers ~2 mm at 40 cm.
- The noisy-keypoint test checks the M2 median bar (3%) and that errors fall within the reported uncertainty.
- The LiDAR suite takes ~25 s in debug, mostly ray-marching the synthetic depth maps.

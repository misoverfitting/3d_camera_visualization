# Stitching speed controls and benchmark workflow

Implemented on `fix/IY-speed-optimization`. Performance/quality validation on the
real recording is pending. Existing full-resolution reconstruction defaults are
preserved until measurements justify changing them.

## Controls

`run_accurate_pipeline(..., options=ReconstructionOptions(...))` accepts explicit
settings. The web/backend path reads these environment variables when options
are omitted; restart the backend to apply environment changes:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| RECON_MAX_IMAGE_SIZE | -1 | Dense longest edge; -1 keeps original resolution |
| RECON_SOURCE_VIEWS | 20 | Auto-ranked source views per dense reference |
| RECON_ITERATIONS | 5 | PatchMatch iterations |
| RECON_MAX_FRAMES | 0 | Native pose/sharpness selection cap; 0 keeps all |
| RECON_FEATURE_DEVICE | cpu | cpu, auto, or cuda |
| RECON_REUSE | 1 | Reuse valid checkpoints; 0 forces fresh attempts |
| COLMAP_NUM_THREADS | 4 | CPU thread limit |

Geometric consistency remains enabled. Fusion explicitly consumes geometric
maps at the same configured resolution. COLMAP handles calibrated dense resizing;
original image dimensions, intrinsics and metric poses are never edited.

Native keyframes use position and viewing-direction diversity weighted by
Laplacian sharpness, keeping first/last frames and original temporal order.
This is an experimental coverage heuristic, not a guarantee of complete geometry.
Source views remain COLMAP auto-ranked using reconstructed overlap evidence.
Keyframe selection is rejected for unposed RGB uploads.

For CUDA features, native capture uses pycolmap CUDA if available, otherwise the
configured CUDA CLI. The CLI uses a separate camera per image before recorded
intrinsics are applied. Explicit cuda fails if neither backend supports it;
auto falls back to CPU. Metadata records the actual native feature device.
This host's pycolmap build is CPU-only, so CUDA features would use the CLI.

## Isolated benchmarks (not started yet)

From the repository root in PowerShell, prepare the verified local runtime:

```powershell
$env:COLMAP_BIN = (Resolve-Path .tools/colmap-4.2.1-cuda/bin/colmap.exe).Path
$env:QT_PLUGIN_PATH = (Resolve-Path .tools/colmap-4.2.1-cuda/plugins).Path
.venv/Scripts/python.exe scripts/benchmark_stitching.py --source-session 248d19d41557 --max-image-size 1280 --source-views 20
```

This is a plan-only command: no COLMAP invocation, image copies, new session, or
reconstruction. Only adding `--run` starts reconstruction. Run one GPU benchmark
at a time on an idle workstation. Execution copies the source recording into a
new session and preserves its output and diagnostics independently.

Suggested sequence: 1280/20, 960/20, then winning resolution with 10 and 6 source
views, then 80 selected frames. Hold remaining settings fixed at each comparison.
Use `--reuse-sfm-from PREVIOUS_BENCHMARK_SESSION_ID` to copy the earlier sparse
cache into a new benchmark session for dense-only comparisons. Fingerprints
reject it if input, calibration, frame selection, device, thread count, or code
changes. Benchmark manifests record platform/GPU, source and effective options.
`reconstruction.json` records software/input provenance, settings, stage times,
cache hits, sparse diagnostics and selected native image names.

## Checkpoints and integrity

Work lives under `work/cache/sfm/` and `work/cache/dense/`, in separate attempts.
Successful stage markers are published atomically and validate artifact hashes.
Sparse reuse verifies calibrated models and images. Dense reuse verifies maps,
source-view plan and images; damaged maps create a fresh dense attempt. A failed
fusion can reuse completed stereo. Partial stereo attempts are retained for
inspection but are not resumed map by map. No automatic cache eviction is added;
these retained workspaces consume disk space.

Changing only crop/foreground/meshing settings reuses expensive geometry stages.
The raw fused cloud (`fused-raw.ply`) is preserved separately from
`fused-foreground.ply`. Foreground processing and meshing run again. Historical
workspaces without checkpoints remain intact but are not automatically trusted
or adopted. The progress API follows `work/active-dense.json` to the active
workspace and retains its existing photometric/geometric counters.

Sparse and dense attempts never remove original session work. The existing
Gaussian splat path continues to use its original SFM behavior. No background
removal algorithms or mobile capture code were changed.

## Validation boundary

Lightweight tests cover settings, calibrated command wiring, CUDA camera
separation, keyframe selection, cache reuse/invalidation, corruption recovery,
raw-cloud preservation, progress, and existing API behavior. The benchmark's
plan-only path was exercised. Expensive real/GPU reconstruction is intentionally
pending user review. No speedup or geometry-quality improvement is claimed yet.

Validation on 24 October 2026: 52 tests passed, 2 skipped, 4 deselected (the
real reconstruction tests). Python compilation and diff whitespace checks passed.

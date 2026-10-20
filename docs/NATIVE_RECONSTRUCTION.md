# Native iPhone recording to model

Implementation breakdown for `codex/ios-arkit-capture`, 20 October 2026.

## Outcome and scope

The exported iPhone recording is now connected to the server. Import its ZIP in
the web app, choose reconstruction mode, monitor the job, and view or download
the resulting model. Native images use their recorded ARKit calibration and
poses rather than estimating all cameras through the original incremental mapper.

The first connected implementation holds camera poses and intrinsics fixed while
triangulating/refining 3D points. It intentionally does not optimize the recorded
camera trajectory. This preserves the ARKit world frame and its approximate meter
scale, so the scan box can crop geometry meaningfully. The approach is a useful
baseline, not proof of improved quality on actual phone captures.

No iPhone app update is necessary for importing existing version-one ZIP exports.
The changes run in the Python server and web UI. The native app still exports
through the share sheet; automatic in-app upload is not included.

## How to use it

1. Update/install server dependencies, including the pinned `pycolmap==4.2.1`.
   Start the updated server with its existing uvicorn or Docker workflow.
2. In the iPhone app, record a stationary textured object with normal tracking.
   Optionally place a scan box; use Skip box if placement is unreliable.
3. Stop and export the ZIP to Files or cloud storage accessible from the computer.
4. Open the updated web app and use **Import iPhone recording ZIP / Import scan**.
   Import creates a new session and opens mode selection.
5. Choose Accurate for a mesh. Native meshes require successful dense processing
   by default. A working CUDA-enabled COLMAP executable and GPU are required for
   that stage. If unavailable, explicitly enable the coarse sparse-preview option
   before selecting the mode. A sparse preview is not a detailed reconstruction.
6. Inspect the result and quality message. Download the mesh or inspect diagnostics
   at `/api/sessions/SESSION_ID/diagnostics`.
7. To move the session to a GPU host, download the native recording with poses
   from the result screen and import it there. A photos-only ZIP loses the poses
   and scan box. The existing result-upload workflow can bring the model back.

Local setup from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r server/requirements-dev.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir server --host 0.0.0.0 --port 8000
```

Use `server/requirements.txt` instead for runtime-only dependencies. Existing GPU
requirements still apply to Gaussian splatting. This integration has not been
deployed to an existing hosted service; update that service before using its URL.

## Step 1 Capture calibrated images

**Input:** a real object, viewed by the standard iPhone 16 rear camera.

The existing Swift module records unrotated JPEGs, camera intrinsics, a
camera-to-world transform, monotonic timestamp, resolution, and tracking state
from each ARFrame. Sampling is limited to two frames per second, 120 seconds,
and 240 frames. Only normally tracked frames are accepted. The app also records
optional scan bounds and interruption/write diagnostics.

**Output:** `manifest.json` plus an `images/` directory, packaged as a ZIP. This
contains RGB and tracking estimates, not LiDAR depth. ARKit positions are not an
accuracy guarantee and can drift.

## Step 2 Import and validate the recording

**Implementation:** `server/app/native_import.py`, shared
`server/app/recording_format.py`, and `POST /api/recordings` in `main.py`.

The importer accepts either a ZIP containing a recording folder or a manifest at
its root. It bounds compressed size to 1 GiB, expanded size to 1.5 GiB, and member
count to 500. It rejects traversal, absolute/Windows-style paths, symlinks, and
duplicate member names. Extraction takes place in temporary staging storage.

The shared validator checks the version and coordinate conventions, readable
images, declared resolution, safe and unique paths, increasing finite timestamps,
normal tracking, positive focal lengths, and finite rigid right-handed transforms.
Images are limited to 24 megapixels. Import requires at least eight frames and
validates any supplied box origin, size, and convention.

After validation it creates a new session, copies images into a flat chronological
sequence, and saves normalized metadata as `arkit.json`. Failed validation does
not create a session. Failed installation cleans up the newly created session.
Native photo/video uploads and individual photo deletion return HTTP 409 so the
pose/image association cannot be changed through those API operations.

**Output:** an ordinary server session identified as `source=arkit_rgb`.

## Step 3 Extract and match image features

**Implementation:** `pipeline/arkit_pipeline.py`, selected automatically by
`colmap_common.run_sfm_and_undistort` when the session contains `arkit.json`.

The native route uses the pinned pycolmap Python API. CPU SIFT features are
extracted with one PINHOLE camera per image. The importer then replaces inferred
calibration in the database with that frame's recorded `fx`, `fy`, `cx`, and `cy`.
Camera and image IDs come from the actual database; filesystem order is not used
as an identity assumption.

Sequential matching searches overlapping frames with a 15-frame overlap setting.
Vocabulary-tree loop detection is disabled. This retains the original workflow's
capture-order assumption. Images must overlap and contain stable visual texture;
recorded poses do not eliminate the need for correspondences.

**Output:** a feature database, matched observations, and a seeded sparse model
whose camera/pose records agree with database IDs.

## Step 4 Triangulate and refine sparse geometry

ARKit uses camera-to-world poses with right/up/back camera axes. COLMAP expects
world-to-camera poses with right/down/forward axes. The shared exporter inverts
the recorded transform and changes the camera-axis convention. Translation is
retained in the ARKit world frame. Matrices are decoded as column-major arrays.

`pycolmap.triangulate_points` intersects matched viewing rays to estimate 3D
landmarks and refines their locations with cameras/intrinsics held fixed. Fewer
than 50 reconstructed landmarks is treated as an error instead of producing a
mesh from an empty or degenerate scene. The route prepares undistorted images
and a sparse workspace for downstream backends.

**Output:** sparse points, calibrated images, and `sfm.json` with pose policy,
registered-image count, point count, mean reprojection error, library version,
coordinate frame, and stage timings.

**Important limitation:** this is fixed-pose reconstruction, not pose-prior bundle
adjustment. Large tracking errors can prevent triangulation or distort geometry.
Small reprojection error does not establish physical measurement accuracy. A
future pose-refinement mode must preserve or explicitly recover scale and world
alignment before applying the recorded box.

## Step 5 Estimate dense depth

**Implementation:** the existing `pipeline/colmap_pipeline.py` backend.

For Accurate mode, the server probes the configured COLMAP executable. If it is
missing or explicitly lacks CUDA support, default native mesh requests fail early
with a clear requirement for dense processing. A CUDA-capable build is not proof
that a working GPU is present; runtime failures are preserved in diagnostics.

Before dense processing, the server re-encodes the sparse model through the
installed CLI from text. This avoids passing a newer pycolmap binary model
directly to a potentially older COLMAP executable. `patch_match_stereo` then
estimates per-view depth from neighboring RGB images. These are computed depth
maps, not depth measured by the phone.

**Output:** multi-view depth/normal estimates in the dense workspace. GPU execution
and the CLI compatibility handoff still require validation on the deployment GPU.

## Step 6 Fuse geometry and isolate the object

`stereo_fusion` combines consistent depth observations into a dense point cloud.
If a scan box exists, its bottom-center origin and size define a world-aligned
cube: X/Z extend half the size in each direction; Y extends upward by the full
size. Cropping happens after triangulation/fusion, preserving background features
for camera matching. It does not currently reduce the dense-stereo image workload.

Fewer than 50 surviving box points is an explicit error. With Skip box, this crop
is omitted. Existing foreground filtering then removes a dominant plane and keeps
the largest remaining cluster. These heuristics can remove legitimate flat or
disconnected object parts and need review on real scans.

**Output:** a filtered point cloud in the recorded world frame.

## Step 7 Build and export the mesh

Dense geometry feeds COLMAP Poisson meshing, then the existing PLY and best-effort
OBJ exports. If the user explicitly allows sparse preview, a failed/unavailable
dense stage may instead export triangulated points through pycolmap, crop/filter
them, estimate normals, and use Open3D Poisson reconstruction.

The fallback is a coarse inspection aid. Native requests no longer silently
substitute it for dense reconstruction. Browser photo/video sessions keep their
previous fallback behavior. Actual dense-stage failures are retained rather than
always reported as missing CUDA.

**Output:** `model.ply`, usually `model.obj`, and `reconstruction.json`. Metadata
reports source, pose policy, dense versus sparse quality or failure, original dense
error, whether cropping occurred, photo count, background-filter results, total
elapsed time and available stage durations. CPU-preview meshing does not have a
separate duration field yet.

The alternate Compelling mode reuses the native sparse/calibrated workspace for
the existing CUDA Gaussian-splat trainer. It creates a radiance representation,
not an editable triangle mesh; box cropping is currently mesh-path only.

## Step 8 Inspect appearance and results

Existing job polling and Three.js/Gaussian-splat viewers display the artifacts.
The web UI now labels dense meshes versus coarse sparse previews. Native sessions
can export a complete recording ZIP with poses through
`GET /api/sessions/{id}/recording.zip`; the server streams the generated ZIP from
spooled storage rather than retaining an arbitrarily large archive in RAM.

PLY may carry available vertex colors. No new texture-atlas baking, textured
glTF/USDZ export, automatic topology repair, or print-readiness verification was
added. A visible surface and valid file are not sufficient evidence of accuracy.

## Step 9 Measure quality and speed

The following verification has run locally:

- API import, ZIP round-trip, malformed/archive traversal rejection, no-session-on-
  failure behavior, immutable inputs, and dense-quality gating.
- COLMAP empty observation-line parsing and scan-box coordinate tests.
- Real CPU SIFT, matching, triangulation and undistortion of a synthetic moving-
  camera planar scene. Tests verify expected depth, preserved camera positions,
  and reprojection error below one pixel.
- A rendered 3D object through real native-pose sparse reconstruction, foreground
  removal, CPU Poisson meshing, and PLY/OBJ export.
- Full suite result: 25 passed and 4 skipped, plus 6 recording-format tests passed.
  The skipped tests depend on absent COLMAP/ffmpeg executables. The final
  regression run returned the same result after the metadata/UI changes.

No real iPhone scan, dense GPU run, or Gaussian-splat training was performed for
this integration. Real performance and quality gains remain unproven. Stage
timings are implemented; peak-memory measurement and a benchmark dashboard are
not. Tests use synthetic ground truth, not claimed phone accuracy.

For the next comparison, use one real exported recording and several physical
measurements of its object. Reconstruct once through native fixed poses and once
through the original mapper using the same images (photos-only ZIP uploaded to a
separate RGB session). Keep the GPU, image count, and settings consistent. Compare
total/stage time, dimensional error, completeness, artifacts, and repeatability.

## Files and interfaces

- `recording_format.py`: shared validation, axes conversion, and ID-aware export;
  the standalone validation script is now a thin wrapper around it.
- `native_import.py`: staged bounded ZIP import and new-session installation.
- `pipeline/arkit_pipeline.py`: native feature/calibration/pose pipeline and crop.
- `pipeline/colmap_common.py`: native-versus-original routing and capability probe.
- `pipeline/colmap_pipeline.py`, `mesh_fallback.py`: dense gate, crop, diagnostics,
  explicit sparse preview and exports.
- `pipeline/colmap_io.py`: preserve blank observation lines in known-pose models.
- `main.py`, `models.py`, `sessions.py`: API routes, immutable native inputs,
  source field and `allow_sparse_fallback` reconstruction request option.
- `web/index.html`, `web/app.js`, `web/api.js`: ZIP import, mode option, quality
  messaging and portable native recording download.
- `server/tests/test_native_recording.py`: integration and geometry tests;
  `requirements-dev.txt` adds the synthetic generator/test dependencies.

Reproduce the checks from the root:

```powershell
.venv\Scripts\python.exe -m pytest server/tests -q
.venv\Scripts\python.exe scripts/test_arkit_recording.py
```

## Remaining engineering path

First validate actual phone images against their calibration and poses. Then run
the dense backend on a working GPU and compare it with the original pipeline.
If pose error is limiting quality, implement controlled pose refinement with
world/scale alignment and confidence diagnostics. Add texture baking and product
integration according to the resulting evidence. TSDF and a Rust/on-device core
remain separate options once a reliable depth source and measured need exist.

The implementation follows COLMAP's known-camera reconstruction workflow:
https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses

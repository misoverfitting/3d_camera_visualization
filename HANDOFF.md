# Tare engineering handoff

Prepared 22 October 2026, approximately 6:56 PM America/New_York.
Operational update: scan completion and safe server restart verified at about
7:12 PM on 22 October 2026.
Repository: https://github.com/misoverfitting/3d_camera_visualization
Working branch: `codex/ios-arkit-capture`, created from `main`.
Implementation baseline at preparation: `311065e`.

## 1. Start here

We have connected native iPhone RGB/ARKit capture to a desktop GPU reconstruction
pipeline. The owner is testing with a **normal iPhone 16**, without LiDAR, and a
Windows desktop with an NVIDIA RTX 4060. EAS cloud builds and TestFlight deliver
the native app because the owner's Mac cannot reliably connect to the phone.

The intended direction is reliable food selection, tracking and reconstruction,
with measured improvements in geometry and turnaround. This is still a prototype:
successful capture, a working GPU pipeline and a nonempty mesh are not evidence
of accurate food geometry. Food recognition and segmentation are proposed work,
not implemented capabilities.

The shortest useful onboarding path is:

1. Read this handoff and `docs/WINDOWS_GPU.md`.
2. Check the active job before starting, restarting or testing anything heavy.
3. Start the configured GPU server when the workstation is idle.
4. Import a native recording ZIP and use **Accurate**, with sparse preview off.
5. Inspect both the model and diagnostics, then read `docs/FOOD_VISION_PLAN.md`.

Some older README text is historical: the project now includes a native app;
native Accurate mode does not silently use CPU preview; TestFlight builds have
been executed; and surface overlays now use the actual ARKit plane mesh. Use the
current code and this handoff to resolve those older statements.

## 2. Operational snapshot: check again before acting

**Latest verified outcome:** session `248d19d41557` completed successfully using
dense MVS, with no sparse fallback. Total pipeline time was 5,679.46 seconds
(about 95 minutes), including 4,943.68 seconds in dense stereo. Both `model.ply`
and `model.obj` were produced. Their hashes were preserved across the subsequent
safe server restart, and HTTP GET access to both exports was verified. Visual
quality and dimensional accuracy still need inspection.

The measured-progress backend and frontend are now active locally. The follow-up
automation described below completed its work and is **PAUSED**; no further
scheduled checks for this run are expected. Refresh the web page to load current
frontend assets. The earlier running-state measurements below are historical.

At preparation, real session **`248d19d41557`** was still running locally:

- 139 native images, with 1920 x 1440 dimensions observed in extraction logs.
- Stage: CUDA dense stereo. Photometric outputs: 139/139; geometric outputs:
  120/139. That is 259/278 expected depth-map outputs, or 93.2% of those outputs.
- Fusion, foreground cleanup, meshing and export still follow. No real-scan
  success or quality result is claimed here. These counts will become stale.
- The old server reports a fixed `percent: 65` during dense stereo. That value
  is a milestone, not measured progress.
- Local URL: `http://127.0.0.1:8000/`. The current checkout is at
  `C:\Users\ilyay\OneDrive\Documents\ChatGPT\Tare\3d_camera_visualization`.

**Do not stop this server, enable reload, or launch a second GPU workload while
the reconstruction is active.** Jobs run in background threads inside uvicorn;
restarting the process terminates them. There is no durable task queue or
automatic checkpoint/resume workflow.

A local Codex follow-up named **Apply reconstruction progress after scan
finishes** checked every **10 minutes**. Its ID is
`apply-reconstruction-progress-after-scan-finishes`. It waited until all jobs
were terminal, verified and restarted only the local server, preserved outputs
and prior logs, verified the new API and exports, and paused itself. It did not
rerun the scan or deploy to Railway. This automation is local to the owner's
computer/app; it is not part of the repository or transferred by a clone.

Read-only checks from the repo root:

```powershell
Get-Content data/sessions/248d19d41557/job.json
Invoke-RestMethod http://127.0.0.1:8000/api/sessions
Invoke-RestMethod http://127.0.0.1:8000/api/sessions/248d19d41557/job
nvidia-smi
```

The measured counter can inspect the existing run without restarting it:

```powershell
$env:PYTHONPATH = 'server'
.venv/Scripts/python.exe -c "from pathlib import Path; from app.job_progress import dense_progress; print(dense_progress(Path('data/sessions/248d19d41557/work')))"
```

The hosted backend's current deployment has **not** been independently verified.
The owner reported a Railway outage; the connector also required reauthentication
when checked. A Git push timestamp alone does not establish the deployed commit,
runtime capabilities or which backend received a recording. Localhost refers to
the computer running the browser, not the Windows desktop when opened on a phone.

## 3. What has been built and why

| Area | Implemented behavior | Reason / remaining boundary |
| --- | --- | --- |
| Native capture | Expo/RN app with a local Swift ARKit module | Access synchronized calibration and poses unavailable in the existing browser flow. |
| Placement | Horizontal support-plane raycasts, adjustable world-aligned cube, optional Skip box | Coarse region selection; no food segmentation or measured object depth. |
| Surface feedback | Cyan ARKit plane meshes and boundary outlines, feature dots, coaching, tracking reset and counters | Build 5 finally produced visible cyan material in owner testing; full placement/UX acceptance remains open. |
| Native ZIP import | Bounded archive validation, normalized images and immutable inputs | Preserve calibration, world coordinates and scan bounds through reconstruction. |
| Native reconstruction | Fixed ARKit poses/calibration, matched features and triangulated points, then dense MVS | Avoid full incremental camera estimation and preserve the world frame; pose errors are not refined away. |
| Dense runtime | Verified Windows CUDA COLMAP on RTX 4060 | Actual dense geometry rather than an implicit low-quality sparse fallback. |
| Meshing | Explicit Poisson depth/trim/thread settings and degenerate-mesh rejection | Default trimming erased a synthetic result; nonempty output still needs inspection. |
| Progress | Measured depth-map counts, elapsed times, honest indeterminate stages | Code is pushed and active locally after the completed scan's safe server restart. |
| Food vision | Research and implementation plan only | No food model, mask export, target identity or recognition inference is shipped. |

Relevant commits, oldest to newest:

| Commit | Change |
| --- | --- |
| `029d8b6` | Native ARKit capture, recording format and desktop validation. |
| `46a2bd3` | Fix native pod/app Xcode scheme collision. |
| `6019b28` | Initial plane feedback and Skip box. |
| `f1cbfe1` | Connect native ZIPs to reconstruction and viewers. |
| `e9b0788` | Shared recording-module formatting. |
| `9f8f9cf` | Windows GPU runtime, meshing fixes and empty-output rejection. |
| `4d66758` | Live-anchor reconciliation, coaching and reset. |
| `06e26ec` | Explicit materials and anchored ARKit plane-mesh rendering. |
| `8667c14` | Existing App Store app ID for noninteractive submission. |
| `311065e` | Measured reconstruction progress and elapsed times. |

## 4. Architecture and modeling pipeline

### Native route: intended baseline for this work

1. **Capture:** Swift receives ARFrames and saves unrotated RGB JPEGs with their
   own intrinsics, camera-to-world matrix, monotonic timestamp, image size and
   normal tracking status. Up to 2 saved frames/second, 120 seconds, 240 frames.
2. **Select optional bounds:** a support-plane tap sets a cube's bottom-center
   origin in the ARKit world frame. The cube extends upward. Skip box saves a
   null region. The box crops reconstructed geometry, not source images.
3. **Export:** app Documents/TareScans stores recordings; the share sheet creates
   a ZIP for Files/cloud transfer. There is no automatic in-app server upload.
4. **Import:** the web form posts the ZIP to `/api/recordings`. The importer
   stages/validates it, creates a new session, renames images chronologically,
   and writes normalized `arkit.json`. Native input mutation APIs return 409.
5. **Features/calibration:** pinned `pycolmap==4.2.1` extracts CPU SIFT features
   with per-image cameras. Recorded intrinsics replace estimates. Database IDs,
   rather than assumed sequential IDs, associate images and cameras.
6. **Known poses:** convert ARKit camera conventions to COLMAP; sequentially
   match overlapping frames with overlap 15 and loop detection off; triangulate
   points with poses/intrinsics fixed. Fewer than 50 sparse points is an error.
7. **Dense geometry:** undistort/prepare images, convert the sparse model into
   the CLI's compatible representation, run GPU PatchMatch stereo and fuse depth
   maps into a cloud. Geometric consistency is explicitly enabled in current code.
8. **Foreground:** apply the optional metric cube, then dominant-plane removal
   and largest-cluster selection. A box containing fewer than 50 points fails.
   These are generic heuristics, not semantic food selection.
9. **Mesh/export:** Poisson surface reconstruction, geometry validation and
   PLY/OBJ export. The web viewer supports inspection/download and scale tools.
   Texture baking and a new glTF pipeline have not been added.

CPU feature work and GPU dense stereo are separate. Installing CUDA does not
move all pycolmap operations onto the GPU. The current native mesh default
requires dense reconstruction; **Allow a coarse sparse preview** is an explicit
lower-fidelity fallback, not equivalent output.

### Original browser route

Browser MediaRecorder video -> ffmpeg/ffprobe frame extraction and selection ->
COLMAP feature extraction/matching/incremental mapping -> undistortion -> shared
dense/mesh or sparse fallback path. Browser RGB input lacks the native poses and
box. Downloading only photos from a native session and re-uploading them takes
this route and loses native metadata.

The **Compelling** route initializes/trains Gaussian splats and has a browser
viewer, but CUDA PyTorch/gsplat were not installed or validated in the Windows
mesh setup. Treat that mode as separate, unverified integration work.

## 5. Startup requirements and fresh checkout

### Accounts, hardware and software

- Git access to the repository and this branch. Work from this branch until its
  changes are reviewed/merged; a fresh checkout of `main` lacks this work.
- Python **3.11**, including `venv`/pip. Tested here: Python 3.11.6.
- Node/npm for mobile development and JS tests. Tested here: Node 22.16.0.
  Mobile dependencies are locked in `mobile/package-lock.json`.
- Python requirements mostly use lower bounds rather than a complete lockfile;
  record the installed versions with `pip freeze` for reproducible benchmarks.
- Dense reconstruction: NVIDIA CUDA-capable GPU and compatible driver, plus a
  CUDA-enabled COLMAP executable. Tested here: RTX 4060, 8 GiB VRAM, driver
  610.88 at setup, portable COLMAP 4.2.1. Other GPU/driver combinations need a
  real runtime test. Apple GPU/Metal is not a substitute for this COLMAP CUDA path.
- Writable local data/scratch storage with room for expanded recordings and
  large dense intermediates. Neither recordings nor installed runtimes ship in Git.
- Native distribution: authorized access to the existing Expo project and Apple
  App Store Connect app/signing workflow; TestFlight on the test iPhone. Local
  Xcode and a USB connection are not needed for the EAS cloud-build route.
- Browser video ingestion additionally needs ffmpeg **and ffprobe** on PATH.
  The native-ZIP GPU setup does not install them.
- Optional splatting additionally needs the CUDA-compatible Torch/gsplat stack
  from `server/requirements-gpu.txt`, tested separately.

### Clone

```powershell
git clone --branch codex/ios-arkit-capture https://github.com/misoverfitting/3d_camera_visualization.git
Set-Location 3d_camera_visualization
```

For an existing checkout, inspect `git status` before switching branches or
pulling. Preserve unrelated local edits. Do not reset the owner's working tree.

### Recommended path: Windows native-ZIP GPU backend

Run these from the repository root, on an idle workstation:

```powershell
.\scripts\setup_gpu.ps1
.\scripts\start_gpu.ps1
```

Setup uses `py -3.11` to create `.venv`, installs development/runtime requirements,
downloads the official CUDA COLMAP 4.2.1 ZIP and verifies its SHA-256. No Docker,
WSL or separate CUDA toolkit was needed for this tested portable runtime.
`-SkipPython` is only for an already prepared Python environment.

Open `http://127.0.0.1:8000/` on that desktop. This foreground server stays alive
while its terminal is open. If PowerShell blocks local scripts, use an invocation
such as `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_gpu.ps1`
for that process rather than changing machine-wide policy.

The launcher sets these process environment variables:

| Setting | GPU launcher value / use |
| --- | --- |
| `COLMAP_BIN` | `<repo>/.tools/colmap-4.2.1-cuda/bin/colmap.exe` |
| `QT_PLUGIN_PATH` | `<repo>/.tools/colmap-4.2.1-cuda/plugins` |
| `DATA_DIR` | `<repo>/data` |
| `COLMAP_NUM_THREADS` | `4`, to bound CPU concurrency |
| `POISSON_DEPTH` | Default `10`; can be set before launching |
| `POISSON_TRIM` | Default `0`; can be set before launching |

Starting plain uvicorn without the launcher does not automatically find this
portable COLMAP installation. These values are read from the process environment;
there is no automatic `.env` loader in the app. `CUDA_EXPECTED=1` alone does not
install CUDA or prove that dense stereo can run.

For a different idle local instance/port:

```powershell
.\scripts\start_gpu.ps1 -Port 8001
```

Do not point a second server at the same live data directory while jobs are
running: job locks are in-process and are not shared across servers.

### CPU development only

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r server/requirements-dev.txt
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir server --host 127.0.0.1 --port 8000
```

This starts the API/UI for development. Dense native reconstruction still needs
the GPU configuration above. Native coarse preview must be explicitly selected.
The original browser-photo pipeline also requires the COLMAP CLI. On non-Windows
hosts use the corresponding venv Python path and provision dependencies for that
host; the Windows setup script is not a cross-platform installer.

### Docker / hosted deployment

`docker compose up app` uses the CPU Dockerfile and a named `/data` volume.
It does not supply dense native reconstruction. `docker compose --profile gpu up
app-gpu` is a separate path requiring Docker/NVIDIA GPU-container support; that
GPU image was not validated by the Windows test and currently uses an unpinned
`colmap/colmap:latest` base. Pin/test the target image before treating it as a
reproducible deployment. Both compose services use port 8000; run one at a time.

Hosted data needs persistent storage at `DATA_DIR` (the Dockerfiles use `/data`).
The web UI and API are served by the same FastAPI app. Browser camera capture on
an actual phone needs HTTPS; desktop localhost is a different exception. Keep
the local GPU backend bound to localhost unless deliberately configuring access.
The current prototype has permissive CORS and no implemented user authentication
or tenant isolation. A hosted service does not borrow the desktop GPU: explicit
recording transfer or a future GPU worker is required.

## 6. Native app setup and TestFlight

The app uses Expo SDK 55, React 19.2, RN 0.83.10 and a Swift 5.9 module. The podspec
declares iOS 15.1; future Vision features need their own availability checks.
**Expo Go cannot load the custom native module.** Production builds bundle their
JS and do not require a Metro server on the owner's computer.

Important existing project identifiers (not login credentials):

| Identifier | Value |
| --- | --- |
| Expo owner / slug | `misoverfitting` / `tare-capture` |
| EAS project ID | `0101e44e-39da-4dbe-ab87-80eb00587ff9` |
| iOS bundle identifier | `com.tare.capture` |
| App Store Connect app ID | `6824637632` |
| Application scheme | `TareCapture` |
| Native pod / package | `TareCaptureNative` / `tare-capture-native` |

**Fresh-clone blocker:** at preparation, the Expo owner/project association was
only in the owner's uncommitted `mobile/app.json`; committed `eas.json` already
contains the App Store app ID. Merge the following into the `expo` object in
`mobile/app.json` while retaining its other fields, or link the authorized
checkout to this existing project through EAS. Do not accidentally create a new
Expo project or change the existing app's bundle identifier.

```json
{
  "owner": "misoverfitting",
  "extra": {
    "eas": { "projectId": "0101e44e-39da-4dbe-ab87-80eb00587ff9" }
  }
}
```

From `mobile/`:

```powershell
npm ci
npm run typecheck
npx expo install --check
npx expo-modules-autolinking resolve --platform apple
npx eas-cli@latest login
npx eas-cli@latest build --platform ios --profile production --auto-submit
```

Autolinking should identify `tare-capture-native`, `TareCaptureNative` and
`TareCaptureModule`. Keep the pod and application scheme names distinct: their
original collision caused an archive of the library instead of the iOS app.
Production uses remote build numbering with auto-increment. EAS handles signing
through the authorized account/remote credentials. Never copy passwords or
signing secrets into this document or the repository.

To submit an already completed build, select its exact ID:

```powershell
npx eas-cli@latest submit --platform ios --profile production --id BUILD_ID
```

Wait for Apple processing and tester-group availability, then install through
TestFlight. An upload being accepted is distinct from the build becoming
available to a tester. A native Swift change requires a new installed build;
restarting the app or updating the backend does not install it. No OTA update
workflow has been established here.

Latest build verified in this work: **0.1.0 (5)**, successfully compiled and
submitted; the owner confirmed installation and visible cyan material.
Build ID: `905d326e-d397-46de-97ad-e435abda277c`.
[EAS build](https://expo.dev/accounts/misoverfitting/projects/tare-capture/builds/905d326e-d397-46de-97ad-e435abda277c).
It includes the native code at `06e26ec`; the later web progress update does not
require an iPhone rebuild. Recheck EAS before assuming build 5 is still latest.

## 7. Recording contract and integration invariants

- Version 1: `source=arkit_rgb`, meter units, column-major matrices and
  `pose=camera_to_world`. Frames contain 9 intrinsics values and 16 pose values.
- ARKit camera axes are +X right, +Y up and -Z forward. The exporter inverts the
  pose and converts to COLMAP's right/down/forward camera convention. Do not
  change pose conversion without geometry tests.
- JPEGs are native sensor orientation, not portrait screenshots. Calibration
  corresponds to those full-resolution pixels. The cyan overlay/box is not baked
  into JPEGs. ARFrame timestamps are monotonic seconds, not wall-clock times.
- Region origin is the world-aligned cube's **bottom center**. Its size is in
  meters; it is not a rotated object bounding box. No depth map is exported.
- Import requires 8–240 frames; the shared standalone validator accepts 1–240.
  ZIP limits: 1 GiB compressed, 1.5 GiB expanded, 500 members. Individual decoded
  images are bounded to 24 MP. Unsafe paths, symlinks, duplicates, bad transforms,
  non-increasing timestamps and inconsistent resolutions are rejected.
- Re-export using `recording.zip` preserves native poses/bounds. `photos.zip`
  alone does not. Standalone COLMAP pose export has empty observations/points;
  it is not a reconstructed model.
- Rerunning reconstruction clears the session's scratch `work/` directory.
  Use separate imported sessions and retain diagnostics when comparing settings.

## 8. Progress update: implementation and activation

`311065e` adds `server/app/job_progress.py`, job timing metadata, response fields,
`web/progress.js`, and the progress UI. It counts only complete depth-map files
whose binary header/payload size agrees, against references in the current
`patch-match.cfg`. Both photometric and geometric passes count. The first pass
alone is therefore 50% of expected outputs, not completion of reconstruction.

The API retains legacy `percent` for compatibility. The new UI uses
`stage_percent`, `completed_units`, `total_units`, `unit`, `work_detail`,
`elapsed_seconds`, `stage_elapsed_seconds` and `last_output_seconds_ago`.
Unknown stages use an activity indicator, not fabricated percentages or an ETA.
Timing fields are optional for old jobs. Read-side measurement does not mutate
the reconstruction's files. Polling retries connection failures and stops when
leaving the progress screen.

The backend was safely restarted after the real scan completed, and the new
schema/frontend were verified. For future updates, static frontend edits can
appear after refresh while Python still uses its previously loaded code; in that
mixed state, progress is deliberately indeterminate. Activating backend changes
requires a restart **after all jobs finish**, then refreshing the page. No iPhone
build is involved. This older completed job has null new timing fields because
it started before timing metadata was added; its runtime remains available in
`output/reconstruction.json`.

Background logs are `.tools/server.stdout.log` and `.tools/server.stderr.log`;
the saved launcher PID is `.tools/server.pid`. Validate the executable, parent/
child relationships and uvicorn command before stopping a process; Windows PID
values can be reused. Preserve existing logs during the eventual restart.

## 9. Data, APIs and code map

```text
data/sessions/<id>/
  session.json       metadata
  job.json           latest status (file-backed, atomically replaced)
  arkit.json         native metadata, when imported
  photos/            immutable native input images
  work/              disposable reconstruction intermediates
  output/            model files and diagnostics
```

`data/`, `.tools/`, `.venv/`, `node_modules/`, generated mobile native projects,
and test scratch directories are ignored. A clone does not include recordings,
models, the COLMAP runtime, Python packages, logs or machine-specific PID files.

Useful endpoints:

| API | Use |
| --- | --- |
| `POST /api/recordings` | Multipart `file`: native recording ZIP -> new session. |
| `GET /api/sessions` | Session list/status. |
| `POST /api/sessions/{id}/reconstruct` | JSON `{"mode":"accurate","allow_sparse_fallback":false}`. |
| `GET /api/sessions/{id}/job` | Stage/progress/error/result filenames. |
| `GET /api/sessions/{id}/diagnostics` | Available `sfm.json` and `reconstruction.json`. |
| `GET /api/sessions/{id}/recording.zip` | Portable native recording with metadata. |
| `GET /api/sessions/{id}/photos.zip` | Photos only. |
| `GET /api/sessions/{id}/result/model.ply` | Mesh download; `model.obj` when generated. |
| `POST /api/sessions/{id}/result` | Upload an externally reconstructed allowed result file. |
| `/docs` and `/openapi.json` | FastAPI API documentation/schema. |

`sfm.json` reports fixed-pose policy, image/point counts, reprojection error and
SfM timings. `reconstruction.json` reports dense use/quality, dense failure,
region/background statistics, Poisson settings and stage/total timings. Inspect
`used_dense_mvs`, not merely whether a PLY file exists.

| File / area | Responsibility |
| --- | --- |
| `mobile/App.tsx` | Capture controls, state and visible diagnostics. |
| `mobile/modules/tare-capture/ios/TareCaptureModule.swift` | AR session, plane rendering, raycasts, recording and export. |
| `server/app/recording_format.py` | Shared validation and ID-aware camera/pose export. |
| `server/app/native_import.py` | Bounded ZIP import and normalized native sessions. |
| `server/app/pipeline/arkit_pipeline.py` | Native features, calibration, matching, triangulation and crop. |
| `server/app/pipeline/colmap_common.py` | Native/RGB routing and subprocess helpers. |
| `server/app/pipeline/colmap_pipeline.py` | Dense gate, MVS/fusion, foreground and mesh exports. |
| `server/app/pipeline/background_removal.py` | Dominant-plane / largest-cluster heuristics. |
| `server/app/pipeline/mesh_fallback.py` | Sparse preview meshing and OBJ conversion. |
| `server/app/pipeline/colmap_io.py` | Model readers, including empty observation-line handling. |
| `server/app/jobs.py`, `job_progress.py`, `models.py` | Job lifecycle, measured counters and API schema. |
| `server/app/main.py`, `sessions.py`, `config.py` | Routes, filesystem storage and environment settings. |
| `web/app.js`, `api.js`, `progress.js`, `viewer.js` | Plain ES-module frontend; no web bundling step. |

## 10. Verification completed and what remains unproven

Confirmed in this work:

- Latest regression run: **38 backend tests passed, 4 skipped**; **5 frontend
  progress tests passed**. JS syntax and Git whitespace checks passed.
- Native TypeScript checks passed, and EAS compiled build 5 successfully.
- The owner installed build 5 and saw cyan material after prior builds did not.
- A real synthetic CUDA mesh test on 20 October completed in **213.9 seconds**
  (208.3 seconds dense stereo), producing **65,385 vertices / 130,722 triangles**,
  PLY/OBJ and `used_dense_mvs=true`, with no sparse fallback. Retained successful
  session: `bab0ee307c74`. Earlier failed synthetic session: `53bf9b21a926`.
- The synthetic test used 32 images at 480 x 360. It is not a runtime/quality
  prediction for 139 full-resolution real phone frames. The current real run
  has already demonstrated substantially longer runtime.
- The 139-frame real scan completed in about 95 minutes with dense output and
  successful PLY/OBJ downloads. This establishes execution, not visual accuracy.

Not yet established: real-food dimensional accuracy/completeness, robust pose
alignment under difficult captures, a reliable fast preset, food segmentation/
classification, a validated splat runtime, or the visual/dimensional quality of
the completed real scan.
No controlled real-capture comparison has established that the new route is
faster or better than the old route at matched quality.

Reproduce lightweight tests from the root:

```powershell
.venv/Scripts/python.exe -m pytest server/tests -q
node --experimental-default-type=module --test web/tests/progress.test.js
.venv/Scripts/python.exe scripts/test_arkit_recording.py
```

If this Windows account's default pytest temp directory is inaccessible, use a
**new unused** workspace `--basetemp .pytest-handoff-UNIQUE_SUFFIX`; pytest may
clear an existing basetemp, so do not point it at source or recordings. Some
external-tool integration tests skip when required binaries are unavailable;
unit-test success does not replace real GPU validation.

Run the GPU smoke test only on an idle workstation:

```powershell
.\scripts\test_gpu.ps1
```

It creates its own retained synthetic session. Do not substitute a real session.
For a new phone ZIP, extract it and validate the directory containing its
manifest using `scripts/validate_arkit_recording.py`; use the actual server ZIP
import to exercise feature/matching/triangulation integration.

## 11. Known limitations and first debugging checks

| Symptom / boundary | First check |
| --- | --- |
| CUDA-enabled COLMAP unavailable | Verify which backend URL received the scan, `COLMAP_BIN`, NVIDIA driver and CUDA build. Restart with the GPU launcher only after jobs finish. |
| Many dots but no surface | Dots are image features; check normal tracking and horizontal plane count. Use a textured, lit table and lateral phone motion. |
| Surface detected but no cyan | Verify installed build and the separate overlay/in-view counts. Build 5 uses explicit materials and anchor-attached mesh geometry. Frustum counts do not prove pixels were drawn. |
| Cannot place box | Check tracking, horizontal raycast hit and tap guidance; Skip box is a supported fallback. |
| Dense stage appears frozen | Check actual map writes, process/GPU activity and measured counters. COLMAP subprocess output is captured until completion, so the server log alone can be quiet. |
| Dense mesh empty | Inspect fused cloud, crop and Poisson settings. Trim 10 previously erased a synthetic mesh; trim 0 can retain extrapolated surfaces. |
| Food/plate missing or merged | Inspect region and generic plane/cluster removal before assuming the mesher is solely responsible. |
| Model malformed despite successful run | Check calibrated image orientation, transforms, pose drift, texture/coverage and foreground selection. Fixed poses cannot correct all capture errors. |
| New code pushed but behavior unchanged | Separate Git state, deployed backend process and installed iPhone build. They update independently. |

There is no cancellation/resume UI, durable queue, shared multi-process job
locking, food-aware foreground model, or automatic GPU-worker dispatch. Keep
food/plate stationary while orbiting. Reflective, translucent, textureless and
occluded food surfaces remain difficult for image-based reconstruction.

## 12. Food-vision plan highlights and recommended next work

The complete proposal and primary-source links are in
[`docs/FOOD_VISION_PLAN.md`](docs/FOOD_VISION_PLAN.md). It is research, not shipped code.

Separate these responsibilities:

1. **Segmentation:** which pixels belong to the selected food/plate?
2. **Tracking:** is this the same target across successive frames?
3. **Identification:** what food category does a trained model suggest?
4. **AR placement:** where is its support plane / tentative world anchor?
5. **Geometry:** what multiview evidence supports its actual shape?

Keep ARKit's camera poses/metric frame. Begin with Swift Apple Vision/Core ML
for on-device masks/inference and Python OpenCV for backend experiments. Apple's
generic foreground masks may select a whole plate and are not guaranteed
individual-food segmentation. Evaluate a food-specific model where required.
OpenCV GrabCut can be an interactive baseline; morphology/contours can clean
masks; Lucas-Kanade optical flow can help between segmentation requests; DNN
executes compatible trained models. The linked Rust crate wraps the C++ library;
it does not itself add recognition or automatic speed gains. Defer a Rust/iOS
bridge until a measured need justifies its build/FFI complexity.

Recommended milestones:

1. **Selection/tracking:** tap -> correctable mask outline -> persistent target
   ID through an orbit -> explicit loss/reacquisition. Preserve manual box/Skip
   box. Prototype bounded off-main-thread inference with one request in flight;
   1–3 Hz segmentation is a test starting point, not measured throughput.
2. **Dataset contract:** save image-aligned per-frame label masks and target/
   quality metadata. Update manifest versioning, validation, ZIP limits, import,
   rename handling and re-export together. Keep older RGB/pose-only scans usable.
3. **Food-aware reconstruction:** use mask/visibility evidence to select geometry,
   preserving disconnected food components instead of keeping only one largest
   cluster. Compare missing food and retained plate/background quantitatively.
4. **Identification:** benchmark trained food labels and Unknown/uncertain cases;
   provide user confirmation. Semantic certainty and tracking confidence are
   separate values. Mixed dishes need explicit evaluation.
5. **Speed:** select sharp, nonredundant, well-distributed views using image quality
   plus ARKit movement/coverage. Test bounded resolution and calibrated cropping.
   Masks applied after MVS do not save the full-image MVS computation. COLMAP's
   documented feature-extraction masks are not automatically dense ROI processing.

Coordinate alignment is critical: portrait/aspect-fill preview, sensor-oriented
RGB, Vision/OpenCV conventions, model cropping/letterboxing and masks must map
consistently. Associate every mask with its exact source frame/pose. Cropping or
resizing reconstruction inputs requires corresponding camera-intrinsic updates.
Reject stale inference results after a new selection, reset or AR interruption.

Raycasting onto a confirmed support plane does not measure the top of the food.
A projected silhouette is only a provisional plane projection, not its true
elevated footprint, height or volume. Do not encode it as measured depth. Use
multiview geometry or an explicitly evaluated depth model for shape. Optional
known-size ArUco markers can help benchmark support-plane pose/scale without
becoming a requirement of normal capture.

Build a held-out set with white food/white plates, glossy food, occlusion,
hands/utensils, multiple pieces and target switching. Measure mask overlap,
background leakage, identity switches, reacquisition, inference latency, capture
drops/thermal behavior, geometry completeness, dimensional error and stage time.
Do not claim a speed improvement by accepting worse geometry without reporting it.

## 13. Working-tree and sharing notes

At preparation, unrelated local state included modified `mobile/app.json` and
untracked `docs/DESIGN_REVIEW.md`; these were preserved. `FOOD_VISION_PLAN.md`
is included with this handoff as a proposed roadmap. Machine-specific data/runtimes must be transferred or
recreated separately. Do not expect the owner's Codex automation or old PDFs to
arrive via Git.

Earlier PDFs under the owner's parent workspace `output/pdf/` are historical
snapshots: they predate later GPU/progress changes. The engineer's original Word
proposal was a reference for analysis, not an executable instruction set.

Useful supporting docs:

- [Native recording / reconstruction implementation](docs/NATIVE_RECONSTRUCTION.md)
- [Windows GPU setup and runtime details](docs/WINDOWS_GPU.md)
- [iOS surface diagnostics and acceptance checks](docs/IOS_SURFACE_TESTING.md)
- [Food vision proposal and references](docs/FOOD_VISION_PLAN.md)

## 14. Suggested first engineering session

1. Confirm branch/worktree and preserve local edits; establish Expo/Apple access
   and reproduce the mobile project association in the engineer's checkout.
2. Review the completed real scan. Retain its original ZIP, output and diagnostics;
   assess whether the dense model is usable before changing reconstruction settings.
3. Refresh the web page to load the active progress update. The original follow-up
   is paused; check for any new active jobs before future restarts or ownership changes.
4. Reproduce setup/tests and the synthetic GPU smoke test on an idle machine.
5. Create a separate real-capture benchmark session, establish geometry/runtime
   baselines, and inspect pose/image alignment and foreground removal.
6. Implement the selection/tracking milestone and mask round-trip contract before
   replacing reconstruction with unbenchmarked recognition/depth assumptions.

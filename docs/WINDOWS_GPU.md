# Windows GPU mesh runtime

This workstation uses the native Windows CUDA build of COLMAP 4.2.1 with the
project's Python environment. No Docker, WSL, global PATH changes, or separate
CUDA toolkit installation is needed for the verified portable executable.

## Start the local server

From the repository root in PowerShell:

```powershell
.\scripts\start_gpu.ps1
```

Open http://127.0.0.1:8000 on this desktop. Import the ZIP exported by the iPhone
app and select Accurate. Leave the sparse-preview checkbox unchecked to require
actual dense geometry. The default binding is desktop-local only. Use Files or
cloud storage to move the recording ZIP from the phone to this computer.

Closing a foreground server terminal stops that server. The server started during
setup runs in the background; its logs are `.tools/server.stdout.log` and
`.tools/server.stderr.log`. Stop that existing instance before starting another
on port 8000. If another service already uses 8000, choose a different local port:

```powershell
.\scripts\start_gpu.ps1 -Port 8001
```

The startup script explicitly sets COLMAP_BIN, QT_PLUGIN_PATH, DATA_DIR and the
COLMAP thread count for that process. Starting uvicorn without this configuration
does not automatically discover the workspace's portable COLMAP installation.

## Reconstruction progress

The web progress bar measures the current stage's completed work, not elapsed
time or an estimated fraction of the entire reconstruction. During dense stereo
it counts complete photometric and geometric depth-map files against the
reference images in `stereo/patch-match.cfg`. The pipeline explicitly enables
geometric consistency, so both passes are included: completing the first pass
alone is 50% of the expected depth-map outputs. Partial/truncated files and
outputs outside the current plan are not counted. Normal-map files are not
counted separately. A completed depth-map count does not guarantee model quality
or that COLMAP's final cleanup has finished.

The page shows both pass counts, total/stage elapsed time for new jobs, and time
since the last completed depth map. Stages without measurable work counts use an
activity indicator rather than the old fixed 65%/88% milestones. Fusion, cleanup,
meshing and export can continue after depth maps reach 100%. There is no claimed
whole-job ETA. Polling retries connection errors and stops when leaving the
progress screen. The API's legacy `percent` field remains for compatibility;
the new UI uses `stage_percent` instead.

Do not restart the backend during reconstruction: jobs run inside its process.
Apply server-code changes after all active jobs finish, then refresh the web
page. A frontend refresh alone cannot reload backend Python code. Older jobs
without start timestamps have no invented elapsed duration.

## Reproduce setup or GPU verification

```powershell
.\scripts\setup_gpu.ps1
.\scripts\test_gpu.ps1
```

Setup checks the official download against its published SHA-256:

```text
COLMAP version: 4.2.1
Asset: colmap-x64-windows-cuda.zip
SHA-256: e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881
```

It retains the portable runtime in `.tools/`, which is excluded from Git. The
test renders a 32-image, 480 x 360 synthetic object with known camera poses,
then runs real feature matching, triangulation, CUDA dense stereo, fusion,
foreground removal, Poisson meshing, and PLY/OBJ export. Sparse fallback is
disabled. It requires a nondegenerate mesh and `used_dense_mvs=true`, then marks
the synthetic session complete for viewing. Results are retained under
`data/sessions/<id>/output/`, including `gpu_smoke.json`.

The actual GPU is an NVIDIA GeForce RTX 4060 with approximately 8 GiB VRAM;
the driver checked during setup was 610.88. Runtime compatibility is established
by the actual smoke test, not merely by the presence of CUDA in the help text.
Real phone captures are larger and take longer than the small synthetic test.

On 20 October 2026, the corrected end-to-end run completed in 213.9 seconds,
including 208.3 seconds of CUDA dense stereo. It produced 65,385 vertices and
130,722 triangles, exported PLY and OBJ, and recorded `used_dense_mvs=true` without
sparse fallback. The retained synthetic session is `bab0ee307c74`. The regression
suite also passed 27 tests; four external-tool tests were skipped in that separate
test invocation. The live GPU smoke test above ran the dense stages explicitly.

## Meshing and version compatibility

COLMAP's default absolute density trim of 10 erased the entire surface in the
first GPU smoke test, despite successful depth estimation and fusion. Meshing
now explicitly uses depth 10 and trim 0, with a bounded thread count. Empty or
degenerate dense meshes are rejected instead of being reported as successful.
Both settings are configurable through POISSON_DEPTH and POISSON_TRIM before
starting the server. Disabling absolute trimming can retain unsupported surface
extrapolation, so inspect the result; a non-empty mesh is not proof of accuracy.

The browser-photo pipeline detects modern FeatureExtraction/FeatureMatching
command options and retains support for older SiftExtraction/SiftMatching names.
The native recording route uses the pinned pycolmap API directly.

## What this setup does and does not enable

- Accurate mesh mode uses the RTX 4060 for dense depth estimation. Other stages
  still run on the CPU. Model output is available in the local web viewer.
- The existing iPhone app exports usable recordings without a new TestFlight
  release. Import the ZIP through the website; there is no automatic phone upload.
- A remote hosted deployment does not borrow this desktop's GPU. To use this
  GPU, send the recording to this local server, or explicitly set up a remote
  worker/transfer workflow later.
- Browser video ingestion additionally needs ffmpeg. That executable was not
  installed by this native-ZIP setup.
- Gaussian splat mode additionally needs a CUDA PyTorch/gsplat stack. It was not
  installed or validated here; the current setup targets dense mesh production.
- This setup makes the GPU route executable. It does not establish improved
  reconstruction quality or dimensional accuracy on your actual iPhone scans.

Official runtime source: https://github.com/colmap/colmap/releases/tag/4.2.1

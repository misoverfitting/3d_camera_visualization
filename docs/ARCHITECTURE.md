# Architecture

## Why two pipelines

The 2026 phone-capture landscape has forked into two goals that use
different technology (see the report this app is built from):

- **Compelling** (photorealistic, AR, sharing) &rarr; **Gaussian splatting**.
  Renders photorealistically, handles glass/hair/foliage/reflections that
  break mesh pipelines. Not a measurable/editable mesh. Needs a CUDA GPU.
- **Accurate** (measurable geometry, 3D printing, CAD) &rarr; **photogrammetry
  mesh**. Editable polygon mesh via structure-from-motion + multi-view
  stereo. Works on a CPU-only host (at reduced fidelity); best with a GPU.

The app exposes both as `mode=accurate|compelling` on the same capture
session, matching how the commercial apps surveyed in the report (Polycam,
Scaniverse, KIRI Engine) offer both from one photo set.

## Request flow

```
Phone browser (web/)
  -> POST /api/sessions              create a capture session
  -> POST /api/sessions/{id}/photos  upload each captured photo
  -> POST /api/sessions/{id}/reconstruct   {mode: accurate|compelling}
  -> GET  /api/sessions/{id}/job     poll progress (2s interval)
  -> GET  /api/sessions/{id}/result/{file} download/view the model
  -> POST /api/sessions/{id}/scale   rescale mesh from a picked reference distance
```

`server/app/jobs.py` runs reconstruction in a background thread per
session, writing progress to `job.json` on disk so status survives a
server restart and needs no separate task queue (Redis, Celery, ...) for
what is meant to stay a *simple* app.

## The accurate pipeline (`server/app/pipeline/colmap_pipeline.py`)

Shells out to the `colmap` CLI:

1. `feature_extractor` - SIFT features per photo
2. `exhaustive_matcher` - cross-photo feature matching
3. `mapper` - incremental structure-from-motion (camera poses + sparse points)
4. `image_undistorter` - clean PINHOLE camera model for the next stages
5. `patch_match_stereo` + `stereo_fusion` - dense multi-view stereo
   (**requires CUDA**)
6. `poisson_mesher` - dense point cloud &rarr; watertight mesh

If step 5 isn't available (no CUDA), `mesh_fallback.py` builds a mesh
directly from the sparse point cloud via Open3D's Poisson reconstruction
instead - lower fidelity, but keeps the app fully functional on a CPU-only
host, matching how the commercial apps use cloud GPU workers for this stage
and only degrade gracefully without one.

## The compelling pipeline (`server/app/pipeline/splat_pipeline.py`)

Reuses the same SfM/undistortion stage, then trains a 3D Gaussian Splat
using the open-source [gsplat](https://github.com/nerfstudio-project/gsplat)
CUDA rasterizer: gaussians are initialized from the sparse point cloud and
optimized against the actual captured photos (photometric L1 loss) via
gradient descent. Exports the standard splat `.ply` schema so it's viewable
in any compatible open-source viewer. This stage hard-requires a CUDA GPU
and is not exercised by the CPU-only dev/test environment - see
`requirements-gpu.txt` and `Dockerfile.gpu`.

## Scale calibration (`server/app/pipeline/scale.py`)

Photogrammetry has no inherent sense of scale. The viewer lets you click
two points on the reconstructed mesh (raycast against the raw, untransformed
geometry, so client-picked coordinates line up exactly with the server-side
`.ply` vertex data) and enter their real-world distance; the server computes
a uniform scale factor and rewrites the mesh in place.

## Frontend (`web/`)

No build step: plain ES modules loaded via an import map, so `three.js`,
its `OrbitControls`/`PLYLoader`/`OBJLoader` addons, and
[`GaussianSplats3D`](https://github.com/mkkellogg/GaussianSplats3D) are
vendored under `web/vendor/` (see `scripts/vendor-libs.sh`) rather than
pulled from a CDN at runtime - keeps the app fully self-hosted and working
offline/behind a firewall.

- `capture.js` - camera access + shot-by-shot coaching (lighting/coverage
  reminders, live shot counter)
- `viewer.js` - `MeshViewer` (plain three.js scene) and `SplatViewer`
  (wraps `GaussianSplats3D.Viewer`); independent, only one is mounted per
  result
- `app.js` - screen navigation and wiring; no framework, no build step

## Synthetic test dataset (`scripts/generate_synthetic_dataset.py`)

A small pure-numpy software rasterizer (z-buffer, Gouraud shading, no
OpenGL/GPU) that renders a procedurally-textured object on a textured
"tabletop" from an orbiting, jittered phone-like camera path. Used by
`server/tests/test_pipeline_synthetic.py` to exercise the real `colmap`
pipeline end to end without needing a phone or a GPU, and doubles as a demo
dataset if you want to try the app without scanning a physical object.

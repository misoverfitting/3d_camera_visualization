# Phone 3D Capture

Scan a real object into a 3D model using nothing but your phone's browser
camera. Built on the open-source photogrammetry/Gaussian-splatting stack
surveyed in *Phone-Based 3D Capture: The 2026 Landscape* (Aug 2026), which
found phone-based 3D capture has forked into two goals with two different
best tools:

- **Compelling** (photorealistic, AR, sharing) &rarr; **Gaussian splatting**
- **Accurate** (measurable geometry, 3D printing, CAD) &rarr; **photogrammetry mesh**

This app offers both, from one guided phone-camera capture session.

## Quick start

```bash
pip install -r server/requirements.txt
# COLMAP is required for reconstruction - on Debian/Ubuntu:
sudo apt-get install colmap

uvicorn app.main:app --app-dir server --host 0.0.0.0 --port 8000
```

Open `http://<this-machine's-LAN-IP>:8000` **on your phone** (needs to be on
the same network) - or `http://localhost:8000` on the same machine to try it
with a webcam. Camera access requires either `localhost` or HTTPS, so for
real phone use you'll want to put this behind a reverse proxy with TLS (or
use a tool like `ngrok`/Tailscale Funnel for quick testing).

Or with Docker:

```bash
docker compose up app          # CPU: guided capture + accurate/mesh pipeline
docker compose --profile gpu up app-gpu   # + GPU: compelling/Gaussian-splat pipeline
```

## What you get

1. **Guided capture** - live camera view with a shot counter and coaching
   (lighting, orbit height, overlap, scale-reference reminders) based on the
   report's practical capture tips (`docs/CAPTURE_TIPS.md`).
2. **Choose your result:**
   - *Compelling*: trains a 3D Gaussian Splat (via the open-source
     [gsplat](https://github.com/nerfstudio-project/gsplat)) for
     photorealistic, glass/hair/foliage-capable rendering. **Requires a CUDA
     GPU.**
   - *Accurate*: runs full [COLMAP](https://colmap.github.io/)
     photogrammetry (structure-from-motion + dense multi-view stereo) into
     an exportable OBJ/PLY mesh. Works on CPU (falls back to meshing the
     sparse point cloud if no GPU is available, at reduced fidelity).
3. **In-browser 3D viewer** with orbit controls, for both meshes and splats.
4. **Real-world scale calibration** - click two points on a reference
   object of known size in the viewer, enter its length, and the mesh gets
   rescaled to true metric units.

See `docs/ARCHITECTURE.md` for how the pieces fit together.

## Requirements

- Python 3.11+
- [COLMAP](https://colmap.github.io/) on `PATH` (`apt install colmap`, or
  build with CUDA for full-quality dense stereo - the Ubuntu package is
  CPU-only)
- For splat mode: a CUDA GPU + `pip install -r server/requirements-gpu.txt`
- A modern phone browser (Chrome/Safari) with camera access, served over
  HTTPS or `localhost`

## Development

```bash
pip install -r server/requirements.txt pytest httpx
cd server && python -m pytest tests/ -v
```

`tests/test_api.py` covers the HTTP API without needing COLMAP.
`tests/test_pipeline_synthetic.py` runs the real `colmap` pipeline
end-to-end against a procedurally-rendered synthetic photo set (see
`scripts/generate_synthetic_dataset.py` - no phone, camera, or GPU needed)
and is skipped automatically if `colmap` isn't installed.

No frontend build step: `web/` is plain ES modules loaded via an import
map. Vendored dependencies (three.js, GaussianSplats3D) live in
`web/vendor/` - re-run `scripts/vendor-libs.sh` to update them.

## Limitations

- Dense multi-view stereo and Gaussian splatting both need a CUDA GPU for
  good results; without one, mesh quality is noticeably lower (sparse-cloud
  fallback) and splat mode is unavailable.
- This is an object/room-scale tool, per the report: phones aren't the
  right instrument for jewelry-scale, sub-millimeter accuracy - use a
  dedicated structured-light scanner for that.
- Single-server, filesystem-backed job storage - fine for personal/small-
  team use, not designed for high concurrency.

## License

MIT - see `LICENSE`. Third-party components are listed in
`THIRD_PARTY_NOTICES.md`.

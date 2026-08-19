# Third-party open-source components

This app is built on top of the following open-source projects.

| Component | License | Used for |
| --- | --- | --- |
| [COLMAP](https://colmap.github.io/) | BSD-3-Clause | Structure-from-motion + multi-view stereo (accurate/mesh pipeline) |
| [Open3D](https://www.open3d.org/) | MIT | CPU meshing fallback, scale calibration |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 | Gaussian splat training (compelling pipeline, GPU) |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Gaussian splat training (compelling pipeline, GPU) |
| [FastAPI](https://fastapi.tiangolo.com/) | MIT | Backend HTTP API |
| [three.js](https://threejs.org/) | MIT | In-browser mesh viewer, vendored under `web/vendor/three/` |
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | MIT | In-browser Gaussian splat viewer, vendored under `web/vendor/gaussian-splats-3d/` |

None of these are modified beyond configuration; see each project's own
repository for its full license text. Re-vendor the two frontend libraries
with `scripts/vendor-libs.sh` if you need to inspect or update the exact
bundled source.

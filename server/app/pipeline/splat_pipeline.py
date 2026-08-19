"""'Compelling' path: train a 3D Gaussian Splat radiance field from the
phone photos, using the same COLMAP camera poses as the mesh pipeline as
initialization/supervision. This is the technique behind the visual quality
of apps like Scaniverse/Polycam splat mode - it renders photorealistically,
handling glass/hair/foliage that break mesh reconstruction, at the cost of
not being a measurable/editable mesh.

Training 3D Gaussian Splats is GPU-bound: it relies on `gsplat`
(https://github.com/nerfstudio-project/gsplat, Apache-2.0), an open-source
CUDA rasterizer for differentiable Gaussian rendering. This module raises a
clear, actionable error on hosts without a CUDA GPU rather than silently
producing a bad result - run the backend on a GPU host (see README/Dockerfile.gpu)
to use this mode. It has not been exercised on real capture data in a
CPU-only development environment; validate against a small capture before
relying on it in production.

Output is a standard 3D Gaussian Splat .ply (the schema introduced by
Kerbl et al. 2023 / the INRIA reference implementation), directly loadable
by common open-source splat viewers (e.g. mkkellogg/GaussianSplats3D, used
in this project's web viewer).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..jobs import ProgressReporter
from ..sessions import Session
from . import colmap_common as cc
from . import colmap_io

SH_C0 = 0.28209479177387814  # degree-0 spherical harmonic normalization constant
DEFAULT_ITERATIONS = 3000


class SplatUnavailable(RuntimeError):
    """Raised when CUDA / gsplat is not available on this host."""


def _require_gpu_stack():
    try:
        import torch
    except ImportError as exc:
        raise SplatUnavailable(
            "PyTorch is not installed. Install requirements-gpu.txt on a "
            "CUDA-capable host to enable Gaussian splatting ('compelling' mode)."
        ) from exc

    if not torch.cuda.is_available():
        raise SplatUnavailable(
            "Gaussian splatting requires a CUDA GPU, and none was detected on this "
            "host. Use mode='accurate' for the CPU-friendly mesh pipeline, or run "
            "this backend on a GPU machine (see Dockerfile.gpu)."
        )

    try:
        import gsplat  # noqa: F401
    except ImportError as exc:
        raise SplatUnavailable(
            "The 'gsplat' package is not installed. Install requirements-gpu.txt "
            "on a CUDA-capable host to enable Gaussian splatting ('compelling' mode)."
        ) from exc

    return torch


def run_compelling_pipeline(
    session: Session, reporter: ProgressReporter, iterations: int = DEFAULT_ITERATIONS
) -> list[str]:
    torch = _require_gpu_stack()
    import torch.nn.functional as F
    from gsplat import rasterization
    from PIL import Image

    output = session.output_dir
    output.mkdir(exist_ok=True)

    sfm = cc.run_sfm_and_undistort(session, reporter)
    model_dir = sfm.undistorted_model_dir

    reporter.update("splat_init", "Loading camera poses and sparse point cloud", 55)
    cameras = colmap_io.read_cameras_txt(model_dir / "cameras.txt")
    images_meta = colmap_io.read_images_txt(model_dir / "images.txt")
    xyz, rgb = colmap_io.read_points3d_txt(model_dir / "points3D.txt")
    if len(xyz) < 20:
        raise RuntimeError(
            f"Only {len(xyz)} sparse points reconstructed - too few to initialize "
            "a splat. Capture more overlapping, well-lit photos."
        )

    device = torch.device("cuda")

    view_list, K_list, image_list = [], [], []
    for img in images_meta.values():
        img_path = sfm.undistorted_images_dir / img.name
        if not img_path.exists():
            continue
        cam = cameras[img.camera_id]
        view_list.append(img.world_to_camera())
        K_list.append(cam.intrinsics())
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        image_list.append(image)

    if not view_list:
        raise RuntimeError("No undistorted images matched the COLMAP model; aborting.")

    width, height = image_list[0].shape[1], image_list[0].shape[0]
    viewmats = torch.tensor(np.stack(view_list), dtype=torch.float32, device=device)
    Ks = torch.tensor(np.stack(K_list), dtype=torch.float32, device=device)
    gt_images = torch.tensor(np.stack(image_list), dtype=torch.float32, device=device)
    num_cameras = viewmats.shape[0]

    # --- initialize gaussians from the sparse point cloud ---
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    nn_dist, _ = tree.query(xyz, k=min(4, len(xyz)))
    avg_nn_dist = np.mean(nn_dist[:, 1:], axis=1) if nn_dist.ndim == 2 and nn_dist.shape[1] > 1 else np.full(len(xyz), 0.01)
    avg_nn_dist = np.clip(avg_nn_dist, 1e-4, None)

    means = torch.tensor(xyz, dtype=torch.float32, device=device, requires_grad=True)
    log_scales = torch.tensor(
        np.log(np.repeat(avg_nn_dist[:, None], 3, axis=1)), dtype=torch.float32, device=device
    ).requires_grad_(True)
    quats = torch.zeros((len(xyz), 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    quats = quats.requires_grad_(True)
    opacity_logits = torch.full((len(xyz),), 2.0, dtype=torch.float32, device=device).requires_grad_(True)
    colors = torch.tensor(rgb, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4, "name": "means"},
        {"params": [log_scales], "lr": 5e-3, "name": "scales"},
        {"params": [quats], "lr": 1e-3, "name": "quats"},
        {"params": [opacity_logits], "lr": 5e-2, "name": "opacities"},
        {"params": [colors], "lr": 2.5e-3, "name": "colors"},
    ])

    reporter.update("splat_training", f"Training Gaussian splat ({iterations} iterations)", 60)
    for step in range(iterations):
        cam_idx = int(np.random.randint(0, num_cameras))
        renders, _, _ = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(log_scales),
            opacities=torch.sigmoid(opacity_logits),
            colors=torch.sigmoid(colors),
            viewmats=viewmats[cam_idx : cam_idx + 1],
            Ks=Ks[cam_idx : cam_idx + 1],
            width=width,
            height=height,
        )
        pred = renders[0, ..., :3]
        loss = F.l1_loss(pred, gt_images[cam_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % max(1, iterations // 40) == 0 or step == iterations - 1:
            percent = 60 + int(35 * (step + 1) / iterations)
            reporter.update(
                "splat_training",
                f"Training Gaussian splat: step {step + 1}/{iterations}, loss={loss.item():.4f}",
                min(percent, 95),
            )

    reporter.update("export", "Exporting splat .ply", 97)
    out_path = output / "splat.ply"
    _save_splat_ply(
        out_path,
        means.detach().cpu().numpy(),
        (torch.exp(log_scales)).detach().cpu().numpy(),
        (quats / quats.norm(dim=-1, keepdim=True)).detach().cpu().numpy(),
        torch.sigmoid(opacity_logits).detach().cpu().numpy(),
        torch.sigmoid(colors).detach().cpu().numpy(),
    )

    (output / "reconstruction.json").write_text(json.dumps({
        "mode": "compelling",
        "gaussian_count": int(means.shape[0]),
        "iterations": iterations,
        "photo_count": session.photo_count(),
    }, indent=2))
    return ["splat.ply"]


def _save_splat_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    colors_rgb: np.ndarray,
) -> None:
    """Writes the standard 3D Gaussian Splat PLY schema (INRIA reference
    implementation), consumed directly by open-source splat viewers."""
    from plyfile import PlyData, PlyElement

    n = means.shape[0]
    f_dc = (colors_rgb - 0.5) / SH_C0
    opacity_logit = np.log(np.clip(opacities, 1e-6, 1 - 1e-6) / np.clip(1 - opacities, 1e-6, 1))
    log_scales = np.log(np.clip(scales, 1e-8, None))

    dtype = (
        [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        + [(f"f_dc_{i}", "f4") for i in range(3)]
        + [(f"f_rest_{i}", "f4") for i in range(45)]
        + [("opacity", "f4")]
        + [(f"scale_{i}", "f4") for i in range(3)]
        + [(f"rot_{i}", "f4") for i in range(4)]
    )
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = means[:, 0], means[:, 1], means[:, 2]
    for i in range(3):
        data[f"f_dc_{i}"] = f_dc[:, i]
    data["opacity"] = opacity_logit
    for i in range(3):
        data[f"scale_{i}"] = log_scales[:, i]
    for i in range(4):
        data[f"rot_{i}"] = quats[:, i]

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(path))

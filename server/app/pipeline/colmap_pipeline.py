"""'Accurate' path: classic photogrammetry via COLMAP structure-from-motion +
multi-view stereo, producing a measurable, printable polygon mesh.

This mirrors the pipeline described for apps like RealityScan/Polycam:
    feature extraction -> matching -> sparse SfM -> dense MVS -> meshing

Dense multi-view stereo (COLMAP's patch_match_stereo) requires CUDA. Real
deployments should run this stage on a GPU worker (as the commercial apps do
via cloud GPU processing). When no CUDA device is available, we fall back to
building a coarser mesh directly from the sparse SfM point cloud via
Poisson surface reconstruction (see mesh_fallback.py) so the pipeline still
produces a usable model on CPU-only hosts, at reduced geometric fidelity.
"""
from __future__ import annotations

import json
import shutil

import open3d as o3d

from ..jobs import ProgressReporter
from ..sessions import Session
from . import background_removal
from . import colmap_common as cc
from . import mesh_fallback


def run_accurate_pipeline(session: Session, reporter: ProgressReporter) -> list[str]:
    output = session.output_dir
    output.mkdir(exist_ok=True)

    sfm = cc.run_sfm_and_undistort(session, reporter)
    dense_dir = sfm.dense_workspace_dir
    mesh_ply = session.work_dir / "mesh.ply"
    used_dense = False
    bg_stats: dict = {}
    colmap_bin = cc.config.COLMAP_BIN

    if cc.dense_stereo_available():
        try:
            reporter.update("dense_stereo", "Running dense multi-view stereo (GPU)", 65)
            cc.run([
                colmap_bin, "patch_match_stereo",
                "--workspace_path", str(dense_dir),
            ], "Dense stereo")

            fused_ply = dense_dir / "fused.ply"
            reporter.update("fusion", "Fusing depth maps into a dense point cloud", 78)
            cc.run([
                colmap_bin, "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(fused_ply),
            ], "Stereo fusion")

            reporter.update("background_removal", "Removing background (floor/tabletop)", 84)
            fused_pcd = o3d.io.read_point_cloud(str(fused_ply))
            fused_pcd, bg_stats = background_removal.isolate_foreground(fused_pcd)
            o3d.io.write_point_cloud(str(fused_ply), fused_pcd)

            reporter.update("meshing", "Building surface mesh (Poisson)", 88)
            cc.run([
                colmap_bin, "poisson_mesher",
                "--input_path", str(fused_ply),
                "--output_path", str(mesh_ply),
            ], "Poisson meshing")
            used_dense = True
        except cc.PipelineError:
            used_dense = False  # fall through to CPU fallback below

    if not used_dense:
        reporter.update(
            "meshing_fallback",
            "No CUDA GPU available: building mesh from the sparse point cloud "
            "(lower fidelity than dense MVS)",
            70,
        )
        sparse_ply = session.work_dir / "sparse_points.ply"
        cc.run([
            cc.config.COLMAP_BIN, "model_converter",
            "--input_path", str(sfm.undistorted_model_dir),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY",
        ], "Sparse model export")
        _, bg_stats = mesh_fallback.mesh_from_point_cloud(sparse_ply, mesh_ply)

    reporter.update("export", "Exporting final model files", 95)
    final_ply = output / "model.ply"
    shutil.copy(mesh_ply, final_ply)
    obj_path = mesh_fallback.convert_to_obj(final_ply, output / "model.obj")

    result_files = ["model.ply"]
    if obj_path is not None:
        result_files.append("model.obj")

    (output / "reconstruction.json").write_text(json.dumps({
        "mode": "accurate",
        "used_dense_mvs": used_dense,
        "photo_count": session.photo_count(),
        "background_removal": bg_stats,
    }, indent=2))
    return result_files

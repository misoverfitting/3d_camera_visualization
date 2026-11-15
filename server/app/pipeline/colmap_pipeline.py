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
import time
import uuid

import open3d as o3d

from ..jobs import ProgressReporter
from ..sessions import Session
from . import background_removal
from . import colmap_common as cc
from . import mesh_fallback


def run_accurate_pipeline(session: Session, reporter: ProgressReporter, allow_sparse_fallback=True, options=None) -> list[str]:
    from .reconstruction_options import ReconstructionOptions, input_fingerprint, software_identity, cached_stage, atomic_json, file_digest
    from .reconstruction_cache import cached_sfm, dense_workspace
    options = options or ReconstructionOptions.from_env()
    started = time.perf_counter()
    native_path = session.root / 'arkit.json'
    native = json.loads(native_path.read_text()) if native_path.exists() else None
    region = native.get('region') if native else None
    output = session.output_dir
    output.mkdir(exist_ok=True)

    mesh_ply = session.work_dir / "mesh.ply"
    used_dense = False
    preview_produced = False
    crop_applied = False
    bg_stats: dict = {}
    colmap_bin = cc.config.COLMAP_BIN
    dense_error = 'CUDA-enabled COLMAP is unavailable'
    timings = {}
    cache_stats = {}
    identity = None
    def write_metadata():
        (output / 'reconstruction.json').write_text(json.dumps({
            'mode': 'accurate', 'source': 'arkit_rgb' if native else 'rgb',
            'pose_policy': 'fixed' if native else 'estimated',
            'used_dense_mvs': used_dense, 'quality': 'dense' if used_dense else ('sparse_preview' if preview_produced else 'failed'),
            'dense_failure': None if used_dense else dense_error,
            'photo_count': session.photo_count(), 'background_removal': bg_stats,
            'region_applied': crop_applied, 'elapsed_seconds': time.perf_counter() - started,
            'stage_seconds': timings, 'settings': options.to_dict(),
            'cache': cache_stats, 'provenance': identity,
            'sfm': json.loads((output / 'sfm.json').read_text()) if (output / 'sfm.json').exists() else None,
            'poisson_depth': cc.config.POISSON_DEPTH, 'poisson_trim': cc.config.POISSON_TRIM,
        }, indent=2))

    dense_available = cc.dense_stereo_available()
    if native and not allow_sparse_fallback and not dense_available:
        write_metadata()
        raise cc.PipelineError('Native scans require dense geometry by default. CUDA-enabled COLMAP is unavailable; enable sparse preview explicitly for a coarse CPU result.')
    identity = {'inputs': input_fingerprint(session), 'software': software_identity(colmap_bin),
                'threads': cc.config.COLMAP_NUM_THREADS, 'max_frames': options.max_frames,
                'feature_device': options.feature_device}
    sfm_started = time.perf_counter()
    sfm, cache_stats['sfm_reused'] = cached_sfm(session, reporter, options, identity, cc.run_sfm_and_undistort)
    timings['sfm'] = time.perf_counter() - sfm_started
    dense_identity = {'sfm': identity, 'max_image_size': options.max_image_size,
                      'source_views': options.source_views, 'iterations': options.iterations,
                      'sparse_model': {p.name: file_digest(p) for p in sfm.undistorted_model_dir.glob('*.txt')}}
    if dense_available:
        dense_work, dense_pointer = dense_workspace(session, sfm, dense_identity, options)
    else:
        dense_work = session.work_dir
    dense_dir = dense_work / 'dense'
    mesh_ply = dense_work / ('mesh-' + uuid.uuid4().hex + '.ply')
    if dense_available:
        try:
            if native:
                # The CLI may be older than pycolmap. Re-encode its sparse input
                # with that CLI rather than handing it a newer binary model.
                for binary in (dense_dir / 'sparse').glob('*.bin'):
                    binary.unlink()
                cc.run([colmap_bin, 'model_converter', '--input_path', str(sfm.undistorted_model_dir),
                        '--output_path', str(dense_dir / 'sparse'), '--output_type', 'BIN'], 'Dense model compatibility conversion')
            stage_started = time.perf_counter()
            reporter.update("dense_stereo", "Running dense multi-view stereo (GPU)", 65)
            def stereo():
                cc.run([
                    colmap_bin, "patch_match_stereo",
                    "--workspace_path", str(dense_dir),
                    *options.stereo_args(),
                ], "Dense stereo")
            plan = [line.strip() for line in (dense_dir / 'stereo/patch-match.cfg').read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
            artifacts = [dense_dir / 'stereo' / folder / f'{name}.{kind}.bin'
                         for name in plan[::2] for kind in ('photometric', 'geometric')
                         for folder in ('depth_maps', 'normal_maps')]
            artifacts += [dense_dir / 'stereo/patch-match.cfg']
            artifacts += [p for p in (dense_dir / 'images').iterdir() if p.is_file()]
            cache_stats['dense_stereo'] = cached_stage(dense_dir / 'stereo-complete.json', artifacts, stereo, options.reuse)
            atomic_json(dense_pointer, {'attempt': dense_work.name})
            timings['dense_stereo'] = time.perf_counter() - stage_started

            raw_fused_ply = dense_dir / "fused-raw.ply"
            fused_ply = dense_dir / "fused-foreground.ply"
            stage_started = time.perf_counter()
            reporter.update("fusion", "Fusing depth maps into a dense point cloud", 78)
            def fusion():
                cc.run([
                    colmap_bin, "stereo_fusion",
                    "--workspace_path", str(dense_dir),
                    "--output_path", str(raw_fused_ply),
                    "--input_type", "geometric",
                    "--StereoFusion.max_image_size", str(options.max_image_size),
                    "--StereoFusion.num_threads", str(cc.config.COLMAP_NUM_THREADS),
                ], "Stereo fusion")
            cache_stats['fusion'] = cached_stage(dense_dir / "fusion-complete.json", [raw_fused_ply], fusion, options.reuse)
            timings['fusion'] = time.perf_counter() - stage_started

            reporter.update("background_removal", "Removing background (floor/tabletop)", 84)
            stage_started = time.perf_counter()
            fused_pcd = o3d.io.read_point_cloud(str(raw_fused_ply))
            if region:
                from .arkit_pipeline import crop_cloud
                fused_pcd = crop_cloud(fused_pcd, region)
                crop_applied = True
            fused_pcd, bg_stats = background_removal.isolate_foreground(fused_pcd)
            o3d.io.write_point_cloud(str(fused_ply), fused_pcd)
            timings['crop_and_background_removal'] = time.perf_counter() - stage_started

            reporter.update("meshing", "Building surface mesh (Poisson)", 88)
            stage_started = time.perf_counter()
            cc.run([
                colmap_bin, "poisson_mesher",
                "--input_path", str(fused_ply),
                "--output_path", str(mesh_ply),
                '--PoissonMeshing.depth', str(cc.config.POISSON_DEPTH),
                '--PoissonMeshing.trim', str(cc.config.POISSON_TRIM),
                '--PoissonMeshing.num_threads', str(cc.config.COLMAP_NUM_THREADS),
            ], "Poisson meshing")
            dense_mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
            if len(dense_mesh.vertices) < 50 or len(dense_mesh.triangles) < 50:
                raise cc.PipelineError('Dense meshing produced empty or degenerate geometry; inspect meshing settings and capture coverage')
            timings['meshing'] = time.perf_counter() - stage_started
            used_dense = True
        except cc.PipelineError as exc:
            dense_error = str(exc)
            used_dense = False  # fall through to CPU fallback below

    if not used_dense:
        if native and not allow_sparse_fallback:
            write_metadata()
            raise cc.PipelineError(f'Dense reconstruction failed: {dense_error}. Native scans require dense geometry by default; enable sparse preview explicitly to inspect a coarse result.')
        reporter.update(
            "meshing_fallback",
            "Dense reconstruction unavailable or failed: building a sparse preview "
            "(lower fidelity than dense MVS)",
            70,
        )
        sparse_ply = session.work_dir / "sparse_points.ply"
        if native:
            import pycolmap
            pycolmap.Reconstruction(sfm.undistorted_model_dir).export_PLY(sparse_ply)
        else:
            cc.run([
                cc.config.COLMAP_BIN, "model_converter",
                "--input_path", str(sfm.undistorted_model_dir),
                "--output_path", str(sparse_ply),
                "--output_type", "PLY",
            ], "Sparse model export")
        _, bg_stats = mesh_fallback.mesh_from_point_cloud(sparse_ply, mesh_ply, region=region)
        preview_produced = True
        crop_applied = bool(region)

    reporter.update("export", "Exporting final model files", 95)
    final_ply = output / "model.ply"
    shutil.copy(mesh_ply, final_ply)
    obj_path = mesh_fallback.convert_to_obj(final_ply, output / "model.obj")

    result_files = ["model.ply"]
    if obj_path is not None:
        result_files.append("model.obj")

    write_metadata()
    return result_files

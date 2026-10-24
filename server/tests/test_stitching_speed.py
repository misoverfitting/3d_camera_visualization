import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.pipeline.reconstruction_options import (
    ReconstructionOptions, cached_stage, input_fingerprint, set_source_views,
    software_identity, atomic_json,
)
from app.pipeline.reconstruction_cache import cached_sfm, dense_workspace
from app.pipeline.keyframes import select_frames


@pytest.mark.parametrize('kwargs', [dict(max_image_size=0), dict(source_views=1),
    dict(iterations=0), dict(max_frames=7), dict(feature_device='magic')])
def test_reject_invalid_profiles(kwargs):
    with pytest.raises(ValueError):
        ReconstructionOptions(**kwargs)


def test_explicit_dense_options_preserve_consistency():
    args = ReconstructionOptions(max_image_size=960, iterations=3).stereo_args()
    assert args[args.index('--PatchMatchStereo.max_image_size') + 1] == '960'
    assert args[args.index('--PatchMatchStereo.geom_consistency') + 1] == 'true'
    assert args[args.index('--PatchMatchStereo.num_iterations') + 1] == '3'


def test_source_view_plan_preserves_reference_order(tmp_path):
    path = tmp_path / 'patch-match.cfg'
    path.write_text('# generated\na.jpg\n__auto__, 20\nb.jpg\n__auto__, 20\n')
    set_source_views(path, 6)
    assert path.read_text() == '# generated\na.jpg\n__auto__, 6\nb.jpg\n__auto__, 6\n'


def test_stage_cache_verifies_artifacts_and_never_publishes_failure(tmp_path):
    marker, artifact = tmp_path / 'done.json', tmp_path / 'depth.bin'
    calls = []
    def compute():
        calls.append(1)
        artifact.write_bytes(b'complete')
    assert not cached_stage(marker, [artifact], compute)['reused']
    assert cached_stage(marker, [artifact], compute)['reused']
    artifact.write_bytes(b'truncated')
    assert not cached_stage(marker, [artifact], compute)['reused']
    assert len(calls) == 2
    def fail():
        artifact.write_bytes(b'partial')
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        cached_stage(marker, [artifact], fail, reuse=False)
    assert not marker.exists()
    marker.write_text('{broken')
    assert not cached_stage(marker, [artifact], compute)['reused']


def make_session(tmp_path):
    photos = tmp_path / 'photos'
    photos.mkdir()
    (photos / 'a.jpg').write_bytes(b'original image')
    root = tmp_path
    (root / 'arkit.json').write_text(json.dumps(dict(region={'size_m': 1}, frames=[{'image': 'a.jpg', 'pose': 1}])))
    output = root / 'output'
    output.mkdir()
    return SimpleNamespace(root=root, work_dir=root / 'work', photos_dir=photos, output_dir=output)


def test_fingerprints_ignore_crop_but_invalidate_calibration_and_pixels(tmp_path):
    session = make_session(tmp_path)
    initial = input_fingerprint(session)
    path = session.root / 'arkit.json'
    data = json.loads(path.read_text())
    data['region'] = None
    path.write_text(json.dumps(data))
    assert input_fingerprint(session) == initial
    data['frames'][0]['pose'] = 2
    path.write_text(json.dumps(data))
    assert input_fingerprint(session) != initial
    path.write_text(json.dumps(dict(region=None, frames=[{'image': 'a.jpg', 'pose': 1}])))
    (session.photos_dir / 'a.jpg').write_bytes(b'changed image')
    assert input_fingerprint(session) != initial


def test_software_identity_ignores_cli_log_timestamp(monkeypatch):
    import app.pipeline.reconstruction_options as module
    responses = iter(['I20261024 12:00:00 123 file.cc:1] COLMAP 4.2.1 with CUDA',
                      'I20261024 12:00:02 456 file.cc:1] COLMAP 4.2.1 with CUDA'])
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=next(responses), stderr=''))
    assert software_identity('fake') == software_identity('fake')


def test_sfm_reuse_and_dense_profile_isolation(tmp_path):
    from app.pipeline.colmap_common import SfmResult
    session = make_session(tmp_path)
    reporter = SimpleNamespace(update=lambda *a: None)
    calls = []
    def fake_sfm(proxy, reporter, options):
        calls.append(1)
        dense = proxy.work_dir / 'dense'
        (dense / 'images').mkdir(parents=True)
        (dense / 'images/a.jpg').write_bytes(b'calibrated')
        (dense / 'sparse_txt').mkdir()
        for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
            (dense / 'sparse_txt' / name).write_text('model')
        (dense / 'stereo').mkdir()
        (dense / 'stereo/patch-match.cfg').write_text('a.jpg\n__auto__, 20\n')
        return SfmResult(dense / 'images', dense / 'sparse_txt', dense)
    options = ReconstructionOptions()
    sfm, reused = cached_sfm(session, reporter, options, {'inputs': 1}, fake_sfm)
    assert not reused
    _, reused = cached_sfm(session, reporter, options, {'inputs': 1}, fake_sfm)
    assert reused and len(calls) == 1
    first, _ = dense_workspace(session, sfm, {'size': 1280}, ReconstructionOptions(source_views=6))
    second, _ = dense_workspace(session, sfm, {'size': 960}, options)
    assert first != second
    assert (first / 'dense/stereo/patch-match.cfg').read_text().endswith('__auto__, 6\n')
    assert (sfm.dense_workspace_dir / 'stereo/patch-match.cfg').read_text().endswith('__auto__, 20\n')
    assert (session.photos_dir / 'a.jpg').read_bytes() == b'original image'
    (sfm.undistorted_images_dir / 'a.jpg').write_bytes(b'corrupt')
    _, reused = cached_sfm(session, reporter, options, {'inputs': 1}, fake_sfm)
    assert not reused and len(calls) == 2


def test_keyframes_keep_pose_coverage_and_original_calibration(tmp_path):
    from PIL import Image
    frames = []
    rng = np.random.default_rng(12)
    for i in range(24):
        pose = np.eye(4)
        angle = i / 23 * 2 * np.pi
        pose[:3, 3] = [np.cos(angle), np.sin(angle), 0.2 * np.sin(angle * 2)]
        frame = dict(image=f'{i:02d}.jpg', camera_to_world=pose.flatten(order='F').tolist(), intrinsics=[1, 2, 3])
        frames.append(frame)
        Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8)).save(tmp_path / frame['image'])
    data = {'frames': frames}
    snapshot = json.dumps(data)
    selected = select_frames(data, tmp_path, 8)
    assert len(selected) == 8 and selected[0] is frames[0] and selected[-1] is frames[-1]
    assert json.dumps(data) == snapshot
    assert [f['image'] for f in selected] == sorted(f['image'] for f in selected)
    assert select_frames(data, tmp_path, 0) is frames


def test_progress_follows_cache_workspace(tmp_path):
    from app.job_progress import dense_progress
    dense = tmp_path / 'cache/dense/test/dense'
    stereo = dense / 'stereo'
    (stereo / 'depth_maps').mkdir(parents=True)
    (stereo / 'patch-match.cfg').write_text('a.jpg\n__auto__, 6\n')
    (stereo / 'depth_maps/a.jpg.photometric.bin').write_bytes(b'1&1&1&' + b'\0' * 4)
    atomic_json(tmp_path / 'active-dense.json', {'path': str(dense)})
    assert dense_progress(tmp_path)['completed_units'] == 1

def test_pipeline_wires_profiles_and_reuses_expensive_stages(tmp_path, monkeypatch):
    from app.pipeline import colmap_pipeline as pipeline
    from app.pipeline.colmap_common import SfmResult
    session = make_session(tmp_path)
    # No native crop in this fixture; the integration checks command/data flow.
    (session.root / 'arkit.json').write_text(json.dumps({'region': None, 'frames': []}))
    session.photo_count = lambda: 1
    reporter = SimpleNamespace(update=lambda *a: None)
    calls = []
    monkeypatch.setattr(pipeline.cc, 'dense_stereo_available', lambda: True)
    import app.pipeline.reconstruction_options as settings
    monkeypatch.setattr(settings, 'software_identity', lambda binary: {'version': 'test'})
    def sfm(proxy, reporter, options):
        dense = proxy.work_dir / 'dense'
        for folder in ('images', 'sparse', 'sparse_txt', 'stereo/depth_maps', 'stereo/normal_maps'):
            (dense / folder).mkdir(parents=True, exist_ok=True)
        (dense / 'images/a.jpg').write_bytes(b'image')
        for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
            (dense / 'sparse_txt' / name).write_text('calibration')
        (dense / 'stereo/patch-match.cfg').write_text('a.jpg\n__auto__, 20\n')
        return SfmResult(dense / 'images', dense / 'sparse_txt', dense)
    monkeypatch.setattr(pipeline.cc, 'run_sfm_and_undistort', sfm)
    def command(args, step):
        calls.append(args)
        if args[1] == 'patch_match_stereo':
            dense = Path(args[args.index('--workspace_path') + 1])
            for folder in ('depth_maps', 'normal_maps'):
                for kind in ('photometric', 'geometric'):
                    (dense / 'stereo' / folder / f'a.jpg.{kind}.bin').write_bytes(b'1&1&1&' + b'\0' * 4)
        if '--output_path' in args and args[1] in ('stereo_fusion', 'poisson_mesher'):
            Path(args[args.index('--output_path') + 1]).write_bytes(b'geometry')
    monkeypatch.setattr(pipeline.cc, 'run', command)
    cloud = SimpleNamespace()
    monkeypatch.setattr(pipeline.o3d.io, 'read_point_cloud', lambda *a: cloud)
    monkeypatch.setattr(pipeline.o3d.io, 'write_point_cloud', lambda path, value: Path(path).write_bytes(b'foreground'))
    monkeypatch.setattr(pipeline.o3d.io, 'read_triangle_mesh', lambda *a: SimpleNamespace(vertices=range(60), triangles=range(60)))
    monkeypatch.setattr(pipeline.background_removal, 'isolate_foreground', lambda value: (value, {}))
    monkeypatch.setattr(pipeline.mesh_fallback, 'convert_to_obj', lambda *a: None)
    options = ReconstructionOptions(max_image_size=960, source_views=6)
    assert pipeline.run_accurate_pipeline(session, reporter, False, options) == ['model.ply']
    stereo = next(a for a in calls if a[1] == 'patch_match_stereo')
    fusion = next(a for a in calls if a[1] == 'stereo_fusion')
    assert stereo[stereo.index('--PatchMatchStereo.max_image_size') + 1] == '960'
    assert fusion[fusion.index('--StereoFusion.max_image_size') + 1] == '960'
    dense = Path(stereo[stereo.index('--workspace_path') + 1])
    assert (dense / 'stereo/patch-match.cfg').read_text().endswith('__auto__, 6\n')
    assert (dense / 'fused-raw.ply').read_bytes() == b'geometry'
    assert (dense / 'fused-foreground.ply').read_bytes() == b'foreground'
    pipeline.run_accurate_pipeline(session, reporter, False, options)
    assert sum(a[1] == 'patch_match_stereo' for a in calls) == 1
    assert sum(a[1] == 'stereo_fusion' for a in calls) == 1
    metadata = json.loads((session.output_dir / 'reconstruction.json').read_text())
    assert metadata['cache']['sfm_reused']
    assert metadata['cache']['dense_stereo']['reused']
    # Truncated maps must create a new attempt, never let COLMAP skip corrupt files.
    (dense / 'stereo/depth_maps/a.jpg.geometric.bin').write_bytes(b'partial')
    pipeline.run_accurate_pipeline(session, reporter, False, options)
    assert sum(a[1] == 'patch_match_stereo' for a in calls) == 2

def test_runtime_settings_are_opt_in(monkeypatch):
    monkeypatch.setenv('RECON_MAX_IMAGE_SIZE', '1280')
    monkeypatch.setenv('RECON_SOURCE_VIEWS', '10')
    monkeypatch.setenv('RECON_MAX_FRAMES', '80')
    settings = ReconstructionOptions.from_env()
    assert (settings.max_image_size, settings.source_views, settings.max_frames) == (1280, 10, 80)


def test_native_cuda_cli_preserves_per_image_cameras(tmp_path, monkeypatch):
    from app.pipeline import arkit_pipeline, colmap_common
    import pycolmap
    session = make_session(tmp_path)
    session.work_dir.mkdir()
    monkeypatch.setattr(pycolmap, 'has_cuda', False)
    monkeypatch.setattr(colmap_common, 'dense_stereo_available', lambda: True)
    monkeypatch.setattr(colmap_common, 'feature_option_prefixes', lambda binary: ('FeatureExtraction', 'FeatureMatching'))
    commands = []
    def intercept(args, step):
        commands.append(args)
        raise colmap_common.PipelineError('stop before feature work')
    monkeypatch.setattr(colmap_common, 'run', intercept)
    with pytest.raises(colmap_common.PipelineError, match='stop before feature work'):
        arkit_pipeline.run_known_poses(session, SimpleNamespace(update=lambda *a: None), ReconstructionOptions(feature_device='cuda'))
    args = commands[0]
    assert args[1] == 'feature_extractor'
    assert args[args.index('--ImageReader.single_camera_per_image') + 1] == '1'
    assert args[args.index('--FeatureExtraction.use_gpu') + 1] == '1'
    assert (session.work_dir / 'image-list.txt').read_text() == 'a.jpg\n'

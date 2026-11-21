import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter
from test_api import client


def make_zip(frames=8, mutate=None):
    image = io.BytesIO()
    Image.new('RGB', (64, 48), (180, 100, 40)).save(image, format='JPEG')
    data = dict(schema_version=1, source='arkit_rgb', matrix_layout='column_major',
                pose='camera_to_world', world_units='meters', region=None, frames=[])
    for i in range(frames):
        data['frames'].append(dict(image=f'images/{i:05d}.jpg', width=64, height=48,
            timestamp_seconds=i + 1, tracking='normal', intrinsics=[60,0,0,0,60,0,32,24,1],
            camera_to_world=[1,0,0,0,0,1,0,0,0,0,1,0,i*.05,0,0,1]))
    if mutate: mutate(data)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('scan/manifest.json', json.dumps(data))
        for i in range(frames): archive.writestr(f'scan/images/{i:05d}.jpg', image.getvalue())
    return buffer.getvalue()


def test_import_native_archive_and_readonly_inputs(client):
    response = client.post('/api/recordings', files={'file': ('scan.zip', make_zip())})
    assert response.status_code == 200, response.text
    session_id = response.json()['id']
    assert response.json()['photo_count'] == 8
    assert len(client.get(f'/api/sessions/{session_id}/photos').json()['photos']) == 8
    for suffix in ['photos', 'video']:
        response = client.post(f'/api/sessions/{session_id}/{suffix}', files={'file': ('a.jpg', b'x')})
        assert response.status_code == 409
    assert client.delete(f'/api/sessions/{session_id}/photos/00000.jpg').status_code == 409
    assert client.get(f'/api/sessions/{session_id}/diagnostics').json() == {}
    exported = client.get(f'/api/sessions/{session_id}/recording.zip')
    assert exported.status_code == 200
    imported_again = client.post('/api/recordings', files={'file': ('copy.zip', exported.content)})
    assert imported_again.status_code == 200
    assert imported_again.json()['photo_count'] == 8


@pytest.mark.parametrize('mutate', [
    lambda d: d['frames'][0].update(image='../outside.jpg'),
    lambda d: d['frames'][0].update(timestamp_seconds=float('nan')),
    lambda d: d.update(region={'origin':[0,0,0], 'size_m':-1}),
    lambda d: d.update(schema_version=2),
])
def test_reject_bad_manifest_without_creating_session(client, mutate):
    assert client.post('/api/recordings', files={'file': ('scan.zip', make_zip(mutate=mutate))}).status_code == 400
    assert client.get('/api/sessions').json() == []


def test_zip_traversal_and_insufficient_frames(client):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive: archive.writestr('../escape.txt', 'bad')
    for contents in [stream.getvalue(), make_zip(frames=2), b'not a zip']:
        assert client.post('/api/recordings', files={'file': ('scan.zip', contents)}).status_code == 400
    assert client.get('/api/sessions').json() == []


def test_native_dense_required_by_default(client, monkeypatch):
    from app import sessions
    from app.pipeline import colmap_pipeline, colmap_common
    response = client.post('/api/recordings', files={'file': ('scan.zip', make_zip())})
    session = sessions.get_session(response.json()['id'])
    monkeypatch.setattr(colmap_common, 'run_sfm_and_undistort', lambda *args: None)
    monkeypatch.setattr(colmap_common, 'dense_stereo_available', lambda: False)
    # A minimal result is enough to reach the dense-capability gate.
    from types import SimpleNamespace
    monkeypatch.setattr(colmap_common, 'run_sfm_and_undistort', lambda *args: SimpleNamespace(dense_workspace_dir=session.work_dir))
    with pytest.raises(colmap_common.PipelineError, match='require dense geometry'):
        colmap_pipeline.run_accurate_pipeline(session, SimpleNamespace(update=lambda *args: None), allow_sparse_fallback=False)
    metadata = client.get(f'/api/sessions/{session.id}/diagnostics').json()['reconstruction.json']
    assert not metadata['used_dense_mvs']
    assert metadata['dense_failure']


def test_empty_colmap_observation_lines_preserved(tmp_path):
    from app.pipeline.colmap_io import read_images_txt
    model = tmp_path / 'images.txt'
    model.write_text('# poses\n1 1 0 0 0 0 0 0 1 a.jpg\n\n2 1 0 0 0 1 0 0 2 b.jpg\n\n')
    assert list(read_images_txt(model)) == [1, 2]


def test_empty_dense_mesh_is_not_reported_as_success(client, monkeypatch):
    import open3d as o3d
    from types import SimpleNamespace
    from app import sessions
    from app.pipeline import colmap_common as cc, colmap_pipeline as pipeline
    response = client.post('/api/recordings', files={'file': ('scan.zip', make_zip())})
    session = sessions.get_session(response.json()['id'])
    dense = session.work_dir / 'dense'
    (dense / 'sparse').mkdir(parents=True)
    monkeypatch.setattr(cc, 'dense_stereo_available', lambda: True)
    from app.pipeline import reconstruction_options
    monkeypatch.setattr(reconstruction_options, 'software_identity', lambda *a: {'version': 'test'})
    (dense / 'images').mkdir()
    (dense / 'images/a.jpg').write_bytes(b'image')
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        (dense / 'sparse' / name).write_text('model')
    (dense / 'stereo/depth_maps').mkdir(parents=True)
    (dense / 'stereo/normal_maps').mkdir()
    (dense / 'stereo/patch-match.cfg').write_text('a.jpg\n__auto__, 20\n')
    monkeypatch.setattr(cc, 'run_sfm_and_undistort', lambda *a: SimpleNamespace(dense_workspace_dir=dense, undistorted_model_dir=dense / 'sparse', undistorted_images_dir=dense / 'images'))
    monkeypatch.setattr(pipeline.background_removal, 'isolate_foreground', lambda cloud: (cloud, {}))
    def command(args, step):
        if args[1] == 'patch_match_stereo':
            workspace = Path(args[args.index('--workspace_path') + 1])
            for folder in ('depth_maps', 'normal_maps'):
                for kind in ('photometric', 'geometric'):
                    (workspace / 'stereo' / folder / f'a.jpg.{kind}.bin').write_bytes(b'1&1&1&' + b'\0' * 4)
        if args[1] == 'stereo_fusion':
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np.random.default_rng(9).normal(size=(100,3)))
            o3d.io.write_point_cloud(args[args.index('--output_path') + 1], cloud)
        if args[1] == 'poisson_mesher':
            assert '--PoissonMeshing.trim' in args
            Path(args[args.index('--output_path') + 1]).write_text('ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n')
    monkeypatch.setattr(cc, 'run', command)
    with pytest.raises(cc.PipelineError, match='degenerate'):
        pipeline.run_accurate_pipeline(session, SimpleNamespace(update=lambda *a: None), allow_sparse_fallback=False)
    metadata = json.loads((session.output_dir / 'reconstruction.json').read_text())
    assert metadata['quality'] == 'failed' and not metadata['used_dense_mvs']
    assert not (session.output_dir / 'model.ply').exists()


def test_crop_is_in_arkit_world_coordinates():
    import open3d as o3d
    from app.pipeline.arkit_pipeline import crop_cloud
    cloud = o3d.geometry.PointCloud()
    inside = np.tile([10., 2.5, 20.], (60, 1))
    cloud.points = o3d.utility.Vector3dVector(np.vstack([inside, [[0,0,0], [10,1.9,20]]]))
    cropped = crop_cloud(cloud, {'origin': [10,2,20], 'size_m': 1})
    assert len(cropped.points) == 60
    with pytest.raises(ValueError): crop_cloud(cloud, {'origin':[0,0,0], 'size_m':1})


def test_real_known_pose_triangulation(tmp_path, monkeypatch):
    """Actual CPU features/matches/triangulation/undistortion, not mocked calls."""
    from app import config, sessions
    from app.pipeline.colmap_common import run_sfm_and_undistort
    import pycolmap
    monkeypatch.setattr(config, 'SESSIONS_DIR', tmp_path / 'sessions')
    config.SESSIONS_DIR.mkdir()
    session = sessions.create_session('known plane')
    texture = Image.fromarray(np.random.default_rng(3).integers(0, 256, (400, 700, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(.6))
    data = json.loads(zipfile.ZipFile(io.BytesIO(make_zip())).read('scan/manifest.json'))
    for i, frame in enumerate(data['frames']):
        texture.crop((i*8, 0, i*8+400, 300)).save(session.photos_dir / f'{i:05d}.jpg', quality=98)
        frame.update(image=f'{i:05d}.jpg', width=400, height=300, intrinsics=[320,0,0,0,320,0,200,150,1])
    (session.root / 'arkit.json').write_text(json.dumps(data))
    from types import SimpleNamespace
    result = run_sfm_and_undistort(session, SimpleNamespace(update=lambda *args: None))
    reconstructed = pycolmap.Reconstruction(result.undistorted_model_dir)
    assert reconstructed.num_reg_images() == 8
    assert reconstructed.num_points3D() > 50
    xyz = np.array([p.xyz for p in reconstructed.points3D.values()])
    assert abs(np.median(xyz[:,2]) + 2) < .05
    for image in reconstructed.images.values():
        index = int(Path(image.name).stem)
        np.testing.assert_allclose(image.projection_center(), [index*.05,0,0], atol=1e-6)
    metrics = json.loads((session.output_dir / 'sfm.json').read_text())
    assert metrics['mean_reprojection_error_px'] < 1
    assert metrics['timings_seconds']['triangulation'] > 0


def test_native_capture_to_cpu_mesh(tmp_path, monkeypatch):
    """Real rendered 3D object -> known poses -> sparse preview mesh/OBJ."""
    import importlib.util
    import math
    import open3d as o3d
    from types import SimpleNamespace
    from app import config, sessions
    from app.pipeline import colmap_pipeline, colmap_common
    spec = importlib.util.spec_from_file_location('scene_generator', Path(__file__).resolve().parents[2] / 'scripts/generate_synthetic_dataset.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    monkeypatch.setattr(config, 'SESSIONS_DIR', tmp_path / 'sessions')
    config.SESSIONS_DIR.mkdir()
    session = sessions.create_session('rendered object')
    generator.generate(session.photos_dir, 2, 16, 480, 360)
    poses = generator.orbit_cameras(2.2, [5., 60.], 16)
    data = json.loads(zipfile.ZipFile(io.BytesIO(make_zip())).read('scan/manifest.json'))
    data['frames'] = []
    focal = 480 / (2 * math.tan(math.radians(55) / 2))
    for i, pose in enumerate(poses):
        data['frames'].append(dict(image=f'synthetic_{i:04d}.jpg', width=480, height=360,
            timestamp_seconds=float(i+1), tracking='normal', intrinsics=[focal,0,0,0,focal,0,240,180,1],
            camera_to_world=pose.flatten(order='F').tolist()))
    (session.root / 'arkit.json').write_text(json.dumps(data))
    monkeypatch.setattr(colmap_common, 'dense_stereo_available', lambda: False)
    files = colmap_pipeline.run_accurate_pipeline(session, SimpleNamespace(update=lambda *args: None), allow_sparse_fallback=True)
    assert 'model.ply' in files and 'model.obj' in files
    mesh = o3d.io.read_triangle_mesh(str(session.output_dir / 'model.ply'))
    assert len(mesh.vertices) > 100 and len(mesh.triangles) > 100
    metadata = json.loads((session.output_dir / 'reconstruction.json').read_text())
    assert metadata['quality'] == 'sparse_preview' and metadata['pose_policy'] == 'fixed'

"""Known-pose reconstruction; preserve the recorded world frame for cropping."""
from __future__ import annotations
import json
import time
import numpy as np
from .. import config
from ..recording_format import export_colmap


def run_known_poses(session, reporter, options=None):
    import pycolmap as pc
    from .colmap_common import PipelineError, SfmResult
    data = json.loads((session.root / 'arkit.json').read_text(encoding='utf-8'))
    expected = {f['image'] for f in data['frames']}
    if expected != {p.name for p in session.photos_dir.iterdir() if p.is_file()}:
        raise PipelineError('Native images changed; import the original ZIP into a new session')
    from .reconstruction_options import ReconstructionOptions
    from .keyframes import select_frames
    options = options or ReconstructionOptions()
    data['frames'] = select_frames(data, session.photos_dir, options.max_frames)
    expected = {f['image'] for f in data['frames']}
    has_cuda = getattr(pc, 'has_cuda', False)
    has_cuda = has_cuda() if callable(has_cuda) else has_cuda
    from . import colmap_common as cc
    cli_cuda = options.feature_device != 'cpu' and not has_cuda and cc.dense_stereo_available()
    if options.feature_device == 'cuda' and not has_cuda and not cli_cuda:
        raise PipelineError('Neither pycolmap nor the configured COLMAP CLI supports CUDA')
    device = pc.Device.cuda if has_cuda and options.feature_device != 'cpu' else pc.Device.cpu
    work, database = session.work_dir, session.work_dir / 'database.db'
    timings = {}
    stats = {'source': 'arkit_rgb', 'pose_policy': 'fixed', 'coordinate_frame': 'arkit_world_meters',
             'selected_images': [f['image'] for f in data['frames']], 'feature_device': 'cli_cuda' if cli_cuda else str(device),
             'pycolmap_version': pc.__version__, 'timings_seconds': timings}
    def stage(name, message, percent, operation):
        reporter.update(name, message, percent)
        started = time.perf_counter()
        try:
            return operation()
        finally:
            timings[name] = time.perf_counter() - started
            (session.output_dir / 'sfm.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    def extract():
        if cli_cuda:
            extraction, _ = cc.feature_option_prefixes(config.COLMAP_BIN)
            names = work / 'image-list.txt'
            names.write_text('\n'.join(f['image'] for f in data['frames']) + '\n')
            cc.run([config.COLMAP_BIN, 'feature_extractor', '--database_path', str(database),
                    '--image_path', str(session.photos_dir), '--image_list_path', str(names),
                    '--ImageReader.camera_model', 'PINHOLE', '--ImageReader.single_camera_per_image', '1',
                    f'--{extraction}.use_gpu', '1',
                    f'--{extraction}.num_threads', str(config.COLMAP_NUM_THREADS)], 'CUDA features')
        else:
            pc.extract_features(database, session.photos_dir, image_names=[f['image'] for f in data['frames']],
                camera_mode=pc.CameraMode.PER_IMAGE, reader_options={'camera_model': 'PINHOLE'},
                extraction_options={'num_threads': config.COLMAP_NUM_THREADS}, device=device)
    stage('feature_extraction', 'Extracting native image features', 5, extract)
    with pc.Database.open(database) as db:
        images = {image.name: image for image in db.read_all_images()}
        if set(images) != expected:
            raise PipelineError('Feature extraction did not register every native image')
        identities = {}
        for frame in data['frames']:
            image = images[frame['image']]
            camera = db.read_camera(image.camera_id)
            k = frame['intrinsics']
            camera.params = np.array([k[0], k[4], k[6], k[7]])
            db.update_camera(camera)
            identities[image.name] = (image.image_id, image.camera_id)
    seed = work / 'known_poses'
    export_colmap(data, seed, identities)
    def match():
        if cli_cuda:
            _, matching = cc.feature_option_prefixes(config.COLMAP_BIN)
            cc.run([config.COLMAP_BIN, 'sequential_matcher', '--database_path', str(database),
                    f'--{matching}.use_gpu', '1',
                    f'--{matching}.num_threads', str(config.COLMAP_NUM_THREADS),
                    '--SequentialMatching.overlap', '15',
                    '--SequentialMatching.loop_detection', '0'], 'CUDA matching')
        else:
            pc.match_sequential(database, pairing_options={'overlap': 15, 'loop_detection': False},
                matching_options={'num_threads': config.COLMAP_NUM_THREADS}, device=device)
    stage('matching', 'Matching overlapping native images', 20, match)
    sparse = work / 'triangulated'
    sparse.mkdir()
    model = stage('triangulation', 'Triangulating geometry with fixed ARKit poses', 35,
        lambda: pc.triangulate_points(pc.Reconstruction(seed), database, session.photos_dir, sparse,
            refine_intrinsics=False, options={'num_threads': config.COLMAP_NUM_THREADS}))
    points = model.num_points3D()
    stats.update({'registered_images': model.num_reg_images(), 'sparse_points': points,
                  'mean_reprojection_error_px': model.compute_mean_reprojection_error() if points else None})
    if points < 50:
        raise PipelineError(f'Only {points} points triangulated. Check pose alignment, overlap and texture; no mesh produced.')
    dense = work / 'dense'
    stage('undistort', 'Preparing calibrated images for dense reconstruction', 50,
          lambda: pc.undistort_images(dense, sparse, session.photos_dir, num_threads=config.COLMAP_NUM_THREADS))
    text = dense / 'sparse_txt'
    text.mkdir()
    pc.Reconstruction(dense / 'sparse').write_text(text)
    (session.output_dir / 'sfm.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return SfmResult(dense / 'images', text, dense)


def crop_cloud(cloud, region):
    if not region:
        return cloud
    origin, size = np.asarray(region['origin']), region['size_m']
    lower = origin + np.array([-size / 2, 0, -size / 2])
    upper = origin + np.array([size / 2, size, size / 2])
    xyz = np.asarray(cloud.points)
    keep = np.flatnonzero(np.all((xyz >= lower) & (xyz <= upper), axis=1))
    if len(keep) < 50:
        raise ValueError('Fewer than 50 points inside scan box. Reposition the box or record with Skip box.')
    return cloud.select_by_index(keep.tolist())

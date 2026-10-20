"""Real CUDA dense-mesh smoke test with a rendered native-format recording.

Run through scripts/test_gpu.ps1. Requires the development dependencies.
Creates a retained synthetic session under data/; never uses user recordings.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from app import sessions
from app.jobs import ProgressReporter
from app.pipeline.colmap_pipeline import run_accurate_pipeline
from generate_synthetic_dataset import generate, orbit_cameras

session = sessions.create_session('GPU smoke test - synthetic object')
generate(session.photos_dir, 2, 16, 480, 360)
focal = 480 / (2 * math.tan(math.radians(55) / 2))
data = dict(schema_version=1, source='arkit_rgb', matrix_layout='column_major',
            pose='camera_to_world', world_units='meters', region=None, frames=[])
for i, pose in enumerate(orbit_cameras(2.2, [5., 60.], 16)):
    data['frames'].append(dict(image=f'synthetic_{i:04d}.jpg', width=480, height=360,
        timestamp_seconds=float(i+1), tracking='normal', intrinsics=[focal,0,0,0,focal,0,240,180,1],
        camera_to_world=pose.flatten(order='F').tolist()))
(session.root / 'arkit.json').write_text(json.dumps(data))
class Reporter(ProgressReporter):
    def __init__(self):
        super().__init__(session.job_path)
    def update(self, stage, message, percent):
        super().update(stage, message, percent)
        print(f'{percent:3}% [{stage}] {message}', flush=True)
started = time.perf_counter()
try:
    result = run_accurate_pipeline(session, Reporter(), allow_sparse_fallback=False)
    metrics = json.loads((session.output_dir / 'reconstruction.json').read_text())
    mesh = o3d.io.read_triangle_mesh(str(session.output_dir / 'model.ply'))
    assert metrics['used_dense_mvs'], 'Sparse fallback is not a successful GPU test'
    assert len(mesh.vertices) > 100 and len(mesh.triangles) > 100, 'Degenerate mesh'
    report = dict(session_id=session.id, elapsed_seconds=time.perf_counter()-started,
        vertices=len(mesh.vertices), triangles=len(mesh.triangles), files=result, reconstruction=metrics)
    (session.output_dir / 'gpu_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f'View in the web app after marking session done: {session.id}', flush=True)
    from app.jobs import mark_done
    mark_done(session, result, message='Synthetic CUDA dense reconstruction verified')
except Exception:
    session.job_path.write_text(json.dumps(dict(status='error', stage='failed',
        message='Synthetic GPU verification failed; inspect the smoke-test console log',
        percent=0, result_files=[], error='GPU smoke test failed')))
    print(f'GPU test failed; inspect session {session.id} at {session.root}', flush=True)
    raise

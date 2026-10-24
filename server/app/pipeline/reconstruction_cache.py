"""Immutable cache attempts: failed stages never replace a successful checkpoint."""
import json
from pathlib import Path
import shutil
import uuid
from .reconstruction_options import WorkSession, atomic_json, digest, set_source_views, read_record, valid_artifacts


def attempt(parent):
    path = parent / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    return path


def cached_sfm(session, reporter, options, identity, run_sfm):
    from .colmap_common import SfmResult
    parent = session.work_dir / 'cache' / 'sfm' / digest(identity)[:24]
    pointer = parent / 'complete.json'
    if options.reuse and pointer.exists():
        state = read_record(pointer)
        work = parent / state.get('attempt', 'incomplete')
        artifacts = [work / 'dense/sparse_txt' / name for name in ('cameras.txt', 'images.txt', 'points3D.txt')]
        from .reconstruction_options import file_digest
        if (all(p.exists() and state.get('artifacts', {}).get(p.name) == file_digest(p) for p in artifacts)
                and all((work / 'dense/images' / name).exists() and
                        file_digest(work / 'dense/images' / name) == fingerprint
                        for name, fingerprint in state.get('images', {}).items())
                and state.get('images')):
            if (work / 'sfm.json').exists():
                shutil.copy2(work / 'sfm.json', session.output_dir / 'sfm.json')
            reporter.update('undistort', 'Reusing calibrated sparse reconstruction', 50)
            return SfmResult(work / 'dense/images', work / 'dense/sparse_txt', work / 'dense'), True
    work = attempt(parent)
    # Existing SFM code owns this fresh scratch directory; user work is never removed.
    result = run_sfm(WorkSession(session, work), reporter, options)
    from .reconstruction_options import file_digest
    artifacts = {p.name: file_digest(p) for p in result.undistorted_model_dir.glob('*.txt')}
    if (session.output_dir / 'sfm.json').exists():
        shutil.copy2(session.output_dir / 'sfm.json', work / 'sfm.json')
    images = {p.name: file_digest(p) for p in result.undistorted_images_dir.iterdir() if p.is_file()}
    atomic_json(pointer, dict(attempt=work.name, artifacts=artifacts, images=images))
    return result, False


def dense_workspace(session, sfm, identity, options):
    parent = session.work_dir / 'cache' / 'dense' / digest(identity)[:24]
    pointer = parent / 'complete.json'
    if options.reuse and pointer.exists():
        state = read_record(pointer)
        work = parent / state.get('attempt', 'incomplete')
        record = read_record(work / 'dense/stereo-complete.json')
        artifacts = [Path(p) for p in record.get('artifacts', {})]
        if valid_artifacts(record, artifacts):
            atomic_json(session.work_dir / 'active-dense.json', {'path': str(work / 'dense')})
            return work, pointer
    work = attempt(parent)
    # Copies avoid shared mutable COLMAP models/maps across profiles. Images retain
    # their original calibrated pixels. No hard links: tools may modify inputs.
    shutil.copytree(sfm.dense_workspace_dir, work / 'dense')
    set_source_views(work / 'dense/stereo/patch-match.cfg', options.source_views)
    atomic_json(session.work_dir / 'active-dense.json', {'path': str(work / 'dense')})
    return work, pointer

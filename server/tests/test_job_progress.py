import json
import os
from types import SimpleNamespace

import pytest

from app.job_progress import dense_progress, enrich_job


def plan(tmp_path, names=('a.jpg', 'b.jpg')):
    stereo = tmp_path / 'dense' / 'stereo'
    (stereo / 'depth_maps').mkdir(parents=True)
    (stereo / 'patch-match.cfg').write_text('# references\n' + ''.join(f'{name}\n__auto__, 20\n' for name in names))
    return stereo / 'depth_maps'


def depth(folder, name, *, complete=True):
    path = folder / name
    path.write_bytes(b'2&2&1&' + bytes(16 if complete else 8))
    os.utime(path, (90, 90))
    return path


def test_both_passes_and_incomplete_files(tmp_path):
    folder = plan(tmp_path)
    depth(folder, 'a.jpg.photometric.bin')
    partial = depth(folder, 'b.jpg.photometric.bin', complete=False)
    depth(folder, 'unplanned.jpg.geometric.bin')
    result = dense_progress(tmp_path)
    assert (result['completed_units'], result['total_units'], result['stage_percent']) == (1, 4, 25)
    assert 'Geometric consistency: 0/2' in result['work_detail']
    partial.write_bytes(b'2&2&1&' + bytes(16))
    assert dense_progress(tmp_path)['stage_percent'] == 50  # first pass alone is not 100%
    depth(folder, 'a.jpg.geometric.bin')
    depth(folder, 'b.jpg.geometric.bin')
    assert dense_progress(tmp_path)['stage_percent'] == 100


@pytest.mark.parametrize('contents', [b'', b'no header', b'-1&2&1&xxxx', b'0&2&1&', b'2&2&1&' + bytes(17)])
def test_invalid_map_not_counted(tmp_path, contents):
    folder = plan(tmp_path, ('a.jpg',))
    (folder / 'a.jpg.photometric.bin').write_bytes(contents)
    assert dense_progress(tmp_path)['completed_units'] == 0


def test_missing_or_incomplete_plan_is_indeterminate(tmp_path):
    assert dense_progress(tmp_path) == {}
    plan(tmp_path)
    (tmp_path / 'dense/stereo/patch-match.cfg').write_text('a.jpg\n')
    assert dense_progress(tmp_path) == {}


def test_elapsed_output_age_and_no_job_mutation(tmp_path):
    folder = plan(tmp_path)
    depth(folder, 'a.jpg.photometric.bin')
    data = {'status': 'running', 'stage': 'dense_stereo', 'percent': 65,
            'started_at': 10, 'stage_started_at': 50}
    before = dict(data)
    result = enrich_job(SimpleNamespace(work_dir=tmp_path), data, now=100)
    assert result['elapsed_seconds'] == 90
    assert result['stage_elapsed_seconds'] == 50
    assert result['last_output_seconds_ago'] == 10
    assert result['stage_percent'] == 25
    assert data == before
    assert enrich_job(SimpleNamespace(work_dir=tmp_path), dict(data, status='done', finished_at=80), now=100)['elapsed_seconds'] == 70


def test_unknown_stage_and_legacy_job_do_not_invent_percent_or_elapsed(tmp_path):
    data = {'status': 'running', 'stage': 'meshing', 'percent': 88}
    result = enrich_job(SimpleNamespace(work_dir=tmp_path), data, now=100)
    assert result.get('stage_percent') is None
    assert result['elapsed_seconds'] is None


def test_reporter_preserves_stage_clock_and_restarts_it_for_new_stage(tmp_path, monkeypatch):
    from app import jobs
    clock = iter([10, 20, 30])
    monkeypatch.setattr(jobs.time, 'time', lambda: next(clock))
    reporter = jobs.ProgressReporter(tmp_path / 'job.json')
    reporter.update('feature_extraction', 'Features', 5)
    reporter.update('feature_extraction', 'Still features', 5)
    data = json.loads((tmp_path / 'job.json').read_text())
    assert data['stage_started_at'] == 20
    reporter.update('matching', 'Matches', 20)
    assert json.loads((tmp_path / 'job.json').read_text())['stage_started_at'] == 30


def test_job_api_preserves_measured_fields(client):
    from app import sessions
    session_id = client.post('/api/sessions', json={'name': 'progress'}).json()['id']
    session = sessions.get_session(session_id)
    folder = plan(session.work_dir)
    depth(folder, 'a.jpg.photometric.bin')
    session.job_path.write_text(json.dumps({'status': 'running', 'stage': 'dense_stereo', 'message': 'Dense stereo',
                                          'percent': 65, 'started_at': 10, 'stage_started_at': 20}))
    result = client.get(f'/api/sessions/{session_id}/job').json()
    assert result['stage_percent'] == 25
    assert result['total_units'] == 4
    assert result['last_output_seconds_ago'] >= 0


from test_api import client

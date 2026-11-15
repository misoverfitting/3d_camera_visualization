"""Read-only measurements of completed work; never infer a whole-job ETA."""
from __future__ import annotations

import time
import json
from pathlib import Path


def _complete_depth_map(path: Path) -> float | None:
    """COLMAP writes a width&height&channels& header then float32 pixels.

    A file can exist while it is still being written. Count only complete maps,
    without reading the (potentially multi-megabyte) pixel buffer.
    """
    try:
        with path.open('rb') as stream:
            header = bytearray()
            while header.count(b'&') < 3 and len(header) < 80:
                byte = stream.read(1)
                if not byte:
                    return None
                header.extend(byte)
            if header.count(b'&') != 3:
                return None
            width, height, channels = (int(v) for v in header[:-1].split(b'&'))
            stat = path.stat()
            if min(width, height, channels) <= 0 or stat.st_size != len(header) + width * height * channels * 4:
                return None
            return stat.st_mtime
    except (OSError, ValueError):
        return None


def dense_progress(work_dir: Path) -> dict:
    stereo = work_dir / 'dense' / 'stereo'
    pointer = work_dir / 'active-dense.json'
    if pointer.exists():
        try:
            dense = Path(json.loads(pointer.read_text())['path']).resolve()
            if dense.is_relative_to(work_dir.resolve()):
                stereo = dense / 'stereo'
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        # Alternating reference-image and source-selection lines. Use the actual
        # dense plan, not the upload count: not every uploaded image must register.
        lines = [line.strip() for line in (stereo / 'patch-match.cfg').read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
        if not lines or len(lines) % 2:
            return {}
        references = set(lines[::2])
        completed = {'photometric': 0, 'geometric': 0}
        last_output = None
        for path in (stereo / 'depth_maps').glob('*.bin'):
            for kind in completed:
                suffix = f'.{kind}.bin'
                if path.name.endswith(suffix) and path.name[:-len(suffix)] in references:
                    modified = _complete_depth_map(path)
                    if modified is not None:
                        completed[kind] += 1
                        last_output = max(last_output or modified, modified)
                    break
        # This pipeline runs COLMAP with geometric consistency enabled: both
        # photometric and geometric depth maps are required for every reference.
        total = len(references) * 2
        count = sum(completed.values())
        return {
            'stage_percent': round(100 * count / total, 1),
            'completed_units': count, 'total_units': total, 'unit': 'depth maps',
            'work_detail': (f"Photometric: {completed['photometric']}/{len(references)} · "
                            f"Geometric consistency: {completed['geometric']}/{len(references)}"),
            'last_output_at': last_output,
        }
    except OSError:
        return {}


def enrich_job(session, data: dict, now: float | None = None) -> dict:
    """Enrich a snapshot without changing job.json or touching pipeline state."""
    result = dict(data)
    now = time.time() if now is None else now
    end = data.get('finished_at', now)
    started = data.get('started_at')
    stage_started = data.get('stage_started_at')
    result['elapsed_seconds'] = max(0, end - started) if started is not None else None
    result['stage_elapsed_seconds'] = max(0, end - stage_started) if stage_started is not None else None
    if data.get('status') == 'running' and data.get('stage') == 'dense_stereo':
        result.update(dense_progress(session.work_dir))
    last_output = result.pop('last_output_at', None)
    if last_output is not None:
        result['last_output_seconds_ago'] = max(0, now - last_output)
    return result

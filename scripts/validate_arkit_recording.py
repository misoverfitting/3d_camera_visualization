"""Validate an extracted Tare recording and optionally export COLMAP known poses.

Usage: python scripts/validate_arkit_recording.py PATH [--colmap-output PATH]
The optional sparse TXT model is input for a future known-pose reconstruction
path; the existing server does not consume it automatically.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate(root: Path) -> dict:
    from PIL import Image

    root = root.resolve()
    data = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (data.get('schema_version'), data.get('source'), data.get('matrix_layout'),
        data.get('pose'), data.get('world_units')) != (1, 'arkit_rgb', 'column_major', 'camera_to_world', 'meters'):
        raise ValueError('Unsupported recording schema or coordinate conventions')
    frames = data.get('frames', [])
    if not frames or len(frames) > 240:
        raise ValueError('Recording must contain 1–240 frames')
    previous = -math.inf
    names = set()
    for frame in frames:
        name = frame['image']
        image = (root / name).resolve()
        if not image.is_relative_to(root / 'images') or name in names:
            raise ValueError('Unsafe or duplicate image path')
        names.add(name)
        timestamp = frame['timestamp_seconds']
        if not math.isfinite(timestamp) or timestamp <= previous:
            raise ValueError('Frame timestamps must be finite and increasing')
        previous = timestamp
        if frame['tracking'] != 'normal':
            raise ValueError('Only normally tracked frames may be integrated')
        for key, length in [('intrinsics', 9), ('camera_to_world', 16)]:
            values = frame[key]
            if len(values) != length or not all(math.isfinite(x) for x in values):
                raise ValueError(f'Invalid {key}')
        k, m = frame['intrinsics'], frame['camera_to_world']
        if k[0] <= 0 or k[4] <= 0 or abs(k[8] - 1) > 1e-5:
            raise ValueError('Invalid camera intrinsics')
        if any(abs(m[i] - v) > 1e-4 for i, v in [(3, 0), (7, 0), (11, 0), (15, 1)]):
            raise ValueError('Invalid homogeneous transform')
        cols = [m[c * 4:c * 4 + 3] for c in range(3)]
        for a in range(3):
            for b in range(3):
                if abs(sum(x*y for x, y in zip(cols[a], cols[b])) - int(a == b)) > 1e-3:
                    raise ValueError('Camera rotation must be orthonormal')
        cross = [cols[0][1]*cols[1][2]-cols[0][2]*cols[1][1],
                 cols[0][2]*cols[1][0]-cols[0][0]*cols[1][2],
                 cols[0][0]*cols[1][1]-cols[0][1]*cols[1][0]]
        if sum(a*b for a, b in zip(cross, cols[2])) < 0.999:
            raise ValueError('Camera rotation must be right handed')
        with Image.open(image) as decoded:
            if decoded.size != (frame['width'], frame['height']):
                raise ValueError('Image resolution differs from calibrated resolution')
            decoded.verify()
    return data


def export_colmap(data: dict, output: Path) -> None:
    import numpy as np

    output.mkdir(parents=True, exist_ok=False)
    cameras, images = [], []
    for index, frame in enumerate(data['frames'], 1):
        k = frame['intrinsics']
        cameras.append(f"{index} PINHOLE {frame['width']} {frame['height']} {k[0]} {k[4]} {k[6]} {k[7]}")
        c2w = np.array(frame['camera_to_world']).reshape((4, 4), order='F')
        # ARKit camera: right/up/back. COLMAP camera: right/down/forward.
        w2c = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(c2w)
        r = w2c[:3, :3]
        trace = float(np.trace(r))
        if trace > 0:
            s = math.sqrt(trace + 1) * 2
            w, x, y, z = s/4, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s
        else:
            i = int(np.argmax(np.diag(r)))
            j, k_index = (i+1) % 3, (i+2) % 3
            s = math.sqrt(1 + r[i,i] - r[j,j] - r[k_index,k_index]) * 2
            vector = [0., 0., 0.]
            vector[i] = s/4
            vector[j] = (r[j,i]+r[i,j])/s
            vector[k_index] = (r[k_index,i]+r[i,k_index])/s
            w = (r[k_index,j]-r[j,k_index])/s
            x, y, z = vector
        pose = [w, x, y, z, *w2c[:3, 3]]
        images.append(f"{index} {' '.join(str(float(v)) for v in pose)} {index} {frame['image']}\n")
    (output / 'cameras.txt').write_text('\n'.join(cameras) + '\n', encoding='utf-8')
    # Blank POINTS2D line per pose is required by COLMAP's text format.
    (output / 'images.txt').write_text('\n'.join(images) + '\n', encoding='utf-8')
    (output / 'points3D.txt').write_text('', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--colmap-output', type=Path)
    args = parser.parse_args()
    recording = validate(args.recording)
    print(f"Validated {len(recording['frames'])} RGB frames; no measured depth.")
    if args.colmap_output:
        export_colmap(recording, args.colmap_output)
        print(f'Exported known camera poses to {args.colmap_output}')

"""Plan an isolated reconstruction; execution requires the explicit --run flag.

Examples (default is a cheap plan, no images copied and no COLMAP invoked):
  python scripts/benchmark_stitching.py --source-session 248d19d41557 --max-image-size 1280 --source-views 10
Add --run only after reviewing the plan. Reuse a previous benchmark's sparse
checkpoint with --reuse-sfm-from SESSION_ID to compare dense settings alone.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session', required=True)
    parser.add_argument('--max-image-size', type=int, default=-1)
    parser.add_argument('--source-views', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--feature-device', choices=['cpu', 'auto', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reuse-sfm-from')
    parser.add_argument('--run', action='store_true', help='Start potentially expensive reconstruction')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    os.environ['COLMAP_NUM_THREADS'] = str(args.threads)
    from app.pipeline.reconstruction_options import ReconstructionOptions
    options = ReconstructionOptions(args.max_image_size, args.source_views, args.iterations,
                                    args.max_frames, args.feature_device)
    plan = dict(source_session=args.source_session, settings=options.to_dict(),
                threads=args.threads, reuse_sfm_from=args.reuse_sfm_from,
                execute=args.run, platform=platform.platform(), processor=platform.processor())
    print(json.dumps(plan, indent=2))
    if not args.run:
        print('Plan only. No reconstruction started. Add --run to execute explicitly.')
        return
    from app import sessions
    from app.pipeline.reconstruction_options import atomic_json
    from app.jobs import ProgressReporter, mark_done
    from app.pipeline.colmap_pipeline import run_accurate_pipeline
    source = sessions.get_session(args.source_session)
    previous = sessions.get_session(args.reuse_sfm_from) if args.reuse_sfm_from else None
    target = sessions.create_session('Stitching benchmark: ' + args.source_session)
    shutil.copytree(source.photos_dir, target.photos_dir, dirs_exist_ok=True)
    if (source.root / 'arkit.json').exists():
        shutil.copy2(source.root / 'arkit.json', target.root / 'arkit.json')
    if previous and (previous.work_dir / 'cache/sfm').exists():
        shutil.copytree(previous.work_dir / 'cache/sfm', target.work_dir / 'cache/sfm')
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        plan['gpu'] = gpu.stdout.strip() if gpu.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired):
        plan['gpu'] = 'unavailable'
    atomic_json(target.root / 'benchmark.json', plan)
    print(f'Isolated benchmark session: {target.id}', flush=True)
    try:
        files = run_accurate_pipeline(target, ProgressReporter(target.job_path),
                                     allow_sparse_fallback=False, options=options)
        mark_done(target, files, message='Stitching benchmark complete')
    except Exception as exc:
        atomic_json(target.job_path, dict(status='error', stage='failed', percent=0,
                    message=str(exc), error=str(exc), result_files=[]))
        raise


if __name__ == '__main__':
    main()

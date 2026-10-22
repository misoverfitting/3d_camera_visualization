from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
from app.recording_format import validate, export_colmap
import argparse

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

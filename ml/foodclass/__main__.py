"""On-device food item classifier. See ml/README.md ("On-device classifier").

    python -m foodclass prepare        # Food-101 + FoodSeg103 crops (+ --extra autolabel classify/ dirs)
    python -m foodclass train          # MobileNetV3-Large from ImageNet weights, resumable
    python -m foodclass export         # Core ML, compiled into the app's Assets
"""
from __future__ import annotations

import argparse
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
WORK = ML / "data" / "foodclass"
ASSETS = ML.parent / "mobile" / "modules" / "tare-capture" / "ios" / "Assets"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m foodclass")
    parser.add_argument("--work", type=Path, default=WORK)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--food101", type=Path, default=ML / "data" / "food101")
    p.add_argument("--foodseg", type=Path, default=ML / "data" / "FoodSeg103")
    p.add_argument("--extra", type=Path, nargs="*", default=[], help="autolabel export classify/ folders")
    p.add_argument("--limit", type=int)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--size", type=int, default=192, help="training crop size; eval and export use 224")
    t.add_argument("--freeze", type=int, default=7, help="frozen MobileNetV3 feature blocks")
    t.add_argument("--batch", type=int, default=96)
    t.add_argument("--lr", type=float, default=1.5e-3)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--limit", type=int)
    e = sub.add_parser("export")
    e.add_argument("--assets", type=Path, default=ASSETS)
    args = parser.parse_args()

    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.work, args.food101 if args.food101.exists() else None,
                args.foodseg if args.foodseg.exists() else None, args.extra, args.limit)
    elif args.command == "train":
        from .train import train
        train(args.work, args.epochs, args.batch, args.lr, args.workers, args.limit, args.size, args.freeze)
    else:
        from .export import export
        export(args.work, args.assets)


if __name__ == "__main__":
    main()

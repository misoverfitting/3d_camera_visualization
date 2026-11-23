"""On-device food item classifier. See ml/README.md ("On-device classifier").

    python -m foodclass prepare        # Food-101 + FoodSeg103 crops (+ --extra autolabel classify/ dirs)
    python -m foodclass train          # two heads (dish / item) on an ImageNet backbone, resumable
    python -m foodclass nutrition5k    # pseudo-label Nutrition5k crops with that model, then train again
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
    p.add_argument("--workers", type=int, default=1, help="parallel processes (one per parquet file)")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--size", type=int, default=224)
    t.add_argument("--arch", default="mobilenet_v3_large", help="mobilenet_v3_large, efficientnet_v2_s or convnext_tiny")
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--limit", type=int)
    t.add_argument("--no-fresh-crops", action="store_true", help="use FoodSeg103's fixed crops instead of regions.csv")
    t.add_argument("--out", default="checkpoint.pt")
    n = sub.add_parser("nutrition5k")
    n.add_argument("--root", type=Path, default=ML / "data" / "nutrition5k")
    n.add_argument("--cli", type=Path, default=ML / "plateitems-cli")
    e = sub.add_parser("export")
    e.add_argument("--assets", type=Path, default=ASSETS)
    e.add_argument("--checkpoint", default="checkpoint.pt")
    args = parser.parse_args()

    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.work, args.food101 if args.food101.exists() else None,
                args.foodseg if args.foodseg.exists() else None, args.extra, args.limit, args.workers)
    elif args.command == "train":
        from .train import train
        train(args.work, args.epochs, args.batch, args.lr, args.workers, args.limit, args.size, args.arch,
              not args.no_fresh_crops, args.out)
    elif args.command == "nutrition5k":
        from .nutrition5k import pseudo_label
        pseudo_label(args.root, args.work, args.cli)
    else:
        from .export import export
        export(args.work, args.assets, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()

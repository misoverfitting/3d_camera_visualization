"""On-device food segmenter: which pixels are food, and which food. See ml/README.md.

    python -m foodseg train     # FoodSeg103 semantic segmentation, COCO-pretrained MobileNetV3 head
    python -m foodseg eval      # mIoU, food-mask IoU and item recall on FoodSeg103 val
    python -m foodseg export    # Core ML, compiled into the app's Assets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
WORK = ML / "data" / "foodseg"
ASSETS = ML.parent / "mobile" / "modules" / "tare-capture" / "ios" / "Assets"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m foodseg")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--data", type=Path, default=ML / "data" / "FoodSeg103")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arch", default="lraspp", help="lraspp or deeplabv3")
    t.add_argument("--size", type=int, default=384)
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--limit", type=int)
    t.add_argument("--out", default="checkpoint.pt")
    v = sub.add_parser("eval")
    v.add_argument("--checkpoint", default="checkpoint.pt")
    v.add_argument("--limit", type=int)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", default="checkpoint.pt")
    e.add_argument("--assets", type=Path, default=ASSETS)
    args = parser.parse_args()

    if args.command == "train":
        from .train import train
        train(args.data, args.work, args.arch, args.size, args.epochs, args.batch, args.lr, args.workers, args.limit, args.out)
    elif args.command == "eval":
        from .train import evaluate_checkpoint
        print(json.dumps(evaluate_checkpoint(args.data, args.work / args.checkpoint, args.limit), indent=1))
    else:
        from .export import export
        export(args.work / args.checkpoint, args.assets)


if __name__ == "__main__":
    main()

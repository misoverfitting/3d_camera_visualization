"""End-to-end check on FoodSeg103 val photos, closer to what the app does than per-crop
accuracy on clean ground-truth crops.

For every ground-truth item (>= 1% of the photo, not "other ingredients"):
- **found:** the segmenter proposes a region with IoU >= 0.5 against it (proposals as in
  foodseg.train.proposals);
- **named:** the classifier, shown the app-style crop of *that proposed region*, names it
  correctly (top-1 item label == the item's normalized FoodSeg103 name).

Also scores each classifier on crops made from the ground-truth masks roughened the way
the old color segmenter gets them wrong (crops.perturb, fixed seed), to see robustness to
noisy masks separately from the segmenter.

    python -m foodclass.endtoend --seg data/gpu-run-1/checkpoint.pt \\
        --cls v1=data/foodclass/checkpoint.pt mnv3=data/gpu-run-1/foodclass-checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from foodseg.train import IGNORE, MIN_ITEM_FRACTION, OTHER_INGREDIENTS, decode, food_logit, iou, load_split, proposals, to_tensor
from foodseg.train import load as load_segmenter

from .crops import box_of, crop, perturb
from .labels import normalize
from .train import MEAN, STD, TwoHead, build_model, device


class Classifier:
    """v1 (one softmax over all labels) or two-head (item head) checkpoint."""

    def __init__(self, path: Path, dev: torch.device):
        state = torch.load(path, map_location="cpu")
        if "config" in state:
            config = state["config"]
            self.model = TwoHead(config["arch"], len(config["dishes"]), len(config["items"]), pretrained=False)
            self.labels, self.size, self.two_head = config["items"], config["size"], True
        else:
            self.labels, self.size, self.two_head = state["labels"], 224, False
            self.model = build_model(len(self.labels), pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model = self.model.to(dev).eval()
        self.dev = dev
        self.mean, self.std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)

    @torch.no_grad()
    def top1(self, crops: list[Image.Image]) -> list[str]:
        if not crops:
            return []
        batch = torch.stack([(torch.from_numpy(np.asarray(c.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR),
                                                          dtype=np.float32) / 255).permute(2, 0, 1) - self.mean) / self.std for c in crops])
        out = self.model(batch.to(self.dev))
        logits = out[1] if self.two_head else out
        return [self.labels[i] for i in logits.argmax(1).cpu().tolist()]


@torch.no_grad()
def run(seg_path: Path, classifiers: dict[str, Path], data: Path, limit: int | None) -> dict:
    dev = device()
    seg, config, _ = load_segmenter(seg_path)
    seg = seg.to(dev).eval()
    size = config["size"]
    names = {int(k): normalize(v) for k, v in json.loads((data / "id2label.json").read_text()).items()}
    models = {k: Classifier(p, dev) for k, p in classifiers.items()}
    rng = random.Random(0)
    totals = {"items": 0, "found": 0}
    named = {k: 0 for k in models}
    noisy = {k: 0 for k in models}
    clean = {k: 0 for k in models}
    for pair in load_split(data, "val", limit):
        photo, mask = decode(pair)
        scale = min(1, 512 / max(photo.size))  # Like the app's crops; keeps this fast.
        if scale < 1:
            small = (round(photo.width * scale), round(photo.height * scale))
            photo, mask = photo.resize(small, Image.Resampling.BILINEAR), mask.resize(small, Image.Resampling.NEAREST)
        truth = np.asarray(mask)
        h, w = truth.shape
        x = to_tensor(photo.resize((size, size), Image.Resampling.BILINEAR))[None].to(dev)
        logits = seg(x).float()
        food = (torch.sigmoid(food_logit(logits)) > 0.5)[0].cpu().numpy()
        cls = (logits[:, 1:].argmax(1) + 1)[0].cpu().numpy()
        # Proposals at segmenter resolution, scaled back to the photo for crops and IoU.
        props = [np.asarray(Image.fromarray(p.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
                 for p in proposals(food, cls)]
        found_crops, noisy_crops, clean_crops, labels = [], [], [], []
        for c in np.unique(truth):
            if c in (0, IGNORE, OTHER_INGREDIENTS):
                continue
            item = truth == c
            if item.mean() < MIN_ITEM_FRACTION:
                continue
            totals["items"] += 1
            label = names[int(c)]
            labels.append(label)
            best = max(((iou(item, p), p) for p in props), default=(0.0, None), key=lambda t: t[0])
            if best[0] >= 0.5:
                totals["found"] += 1
                found_crops.append((len(labels) - 1, crop(photo, box_of(best[1]), best[1])))
            others = (truth != 0) & (truth != IGNORE) & ~item
            rough = perturb(item, others, rng)
            noisy_crops.append(crop(photo, box_of(rough), rough))
            clean_crops.append(crop(photo, box_of(item), item))
        for key, model in models.items():
            got = model.top1([c for _, c in found_crops])
            named[key] += sum(g == labels[i] for (i, _), g in zip(found_crops, got))
            noisy[key] += sum(g == l for g, l in zip(model.top1(noisy_crops), labels))
            clean[key] += sum(g == l for g, l in zip(model.top1(clean_crops), labels))
    n = max(1, totals["items"])
    return {"items": totals["items"], "segmenterFound": totals["found"] / n,
            "classifiers": {k: {"endToEnd": named[k] / n, "namedWhenFound": named[k] / max(1, totals["found"]),
                                "noisyMaskTop1": noisy[k] / n, "cleanMaskTop1": clean[k] / n} for k in models}}


def main() -> None:
    ml = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(prog="python -m foodclass.endtoend")
    parser.add_argument("--seg", type=Path, required=True)
    parser.add_argument("--cls", nargs="+", required=True, help="name=checkpoint")
    parser.add_argument("--data", type=Path, default=ml / "data" / "FoodSeg103")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    classifiers = {k: Path(v) for k, v in (c.split("=", 1) for c in args.cls)}
    print(json.dumps(run(args.seg, classifiers, args.data, args.limit), indent=1))


if __name__ == "__main__":
    main()

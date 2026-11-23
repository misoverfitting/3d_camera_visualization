"""FoodSeg103 semantic segmentation for the app's item finder.

The app's hand-built segmenter (PlateItems: color/texture against the bare plate) misses
food that looks like the plate (rice on white), bleeds into shadows and splits or merges
items by color. This learns it instead: per pixel, background or one of 103 foods.
The app takes "food" = 1 - P(background) for the mask and splits it into items by
the per-pixel food class.

Backbone and decoder start from torchvision's COCO-trained MobileNetV3 segmentation
weights (only the last 1x1 class layer is new), which run comfortably on a phone.

Loss: cross-entropy over 104 classes plus a binary food/background term on
logsumexp(food logits) - background logit, since the food mask matters most.

Input at eval/export is the whole image squashed to size x size, like the app's plate crop.
"""
from __future__ import annotations

import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASSES = 104  # 0 = background
IGNORE = 255
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MIN_ITEM_FRACTION = 0.01  # Same as the classifier's crops.
MIN_PREDICTED_FRACTION = 0.005
OTHER_INGREDIENTS = 103


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_split(data: Path, split: str, limit: int | None = None) -> list[tuple[bytes, bytes]]:
    pairs: list[tuple[bytes, bytes]] = []
    prefix = "train" if split == "train" else "validation"
    for file in sorted((data / "data").glob(f"{prefix}-*.parquet")):
        table = pq.read_table(file, columns=["image", "label"])
        pairs += [(i["bytes"], m["bytes"]) for i, m in zip(table.column("image").to_pylist(), table.column("label").to_pylist())]
        if limit and len(pairs) >= limit:
            return pairs[:limit]
    return pairs


def decode(pair: tuple[bytes, bytes]) -> tuple[Image.Image, Image.Image]:
    image = Image.open(io.BytesIO(pair[0])).convert("RGB")
    mask = Image.open(io.BytesIO(pair[1]))
    if mask.mode not in ("L", "P"):
        mask = Image.fromarray(np.asarray(mask)[..., 0])
    if image.size != mask.size:
        image = image.resize(mask.size, Image.Resampling.BILINEAR)
    return image, mask


def to_tensor(image: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class Pairs(Dataset):
    def __init__(self, pairs: list[tuple[bytes, bytes]], size: int, train: bool):
        self.pairs, self.size, self.train = pairs, size, train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        image, mask = decode(self.pairs[i])
        if not self.train:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
            return to_tensor(image), torch.from_numpy(np.asarray(mask, dtype=np.int64))
        rng = random.Random()
        # Squash like the app (random aspect), then random scale and crop.
        w, h = image.size
        base = self.size / min(w, h) if rng.random() < 0.5 else self.size / max(w, h) * rng.uniform(1.0, 1.4)
        scale = base * rng.uniform(0.6, 1.8)
        aspect = rng.uniform(0.8, 1.25)
        nw, nh = max(8, round(w * scale * aspect)), max(8, round(h * scale / aspect))
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        mask = mask.resize((nw, nh), Image.Resampling.NEAREST)
        pw, ph = max(self.size, nw), max(self.size, nh)
        if (pw, ph) != (nw, nh):
            canvas = Image.new("RGB", (pw, ph), tuple(int(255 * m) for m in MEAN))
            canvas.paste(image, (0, 0)); image = canvas
            m = Image.new("L", (pw, ph), IGNORE); m.paste(mask, (0, 0)); mask = m
        x0, y0 = rng.randint(0, pw - self.size), rng.randint(0, ph - self.size)
        image = image.crop((x0, y0, x0 + self.size, y0 + self.size))
        mask = mask.crop((x0, y0, x0 + self.size, y0 + self.size))
        if rng.random() < 0.5:
            image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        turn = rng.choice([None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270])
        if turn is not None:  # Overhead plates have no "up".
            image, mask = image.transpose(turn), mask.transpose(turn)
        if rng.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
        x = to_tensor(image)
        # Color jitter in normalized space: brightness, contrast, per-channel gain (white balance).
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-0.3, 0.3)
        x = x * torch.tensor([rng.uniform(0.9, 1.1) for _ in range(3)]).view(3, 1, 1)
        if rng.random() < 0.3:
            x = x + torch.randn_like(x) * rng.uniform(0.02, 0.08)
        return x, torch.from_numpy(np.asarray(mask, dtype=np.int64))


class Segmenter(nn.Module):
    """torchvision segmentation model returning plain logits (N, 104, H, W)."""

    def __init__(self, arch: str = "lraspp", pretrained: bool = True):
        super().__init__()
        seg = torchvision.models.segmentation
        if arch == "lraspp":
            net = seg.lraspp_mobilenet_v3_large(weights=seg.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None)
            net.classifier.low_classifier = nn.Conv2d(net.classifier.low_classifier.in_channels, CLASSES, 1)
            net.classifier.high_classifier = nn.Conv2d(net.classifier.high_classifier.in_channels, CLASSES, 1)
        elif arch == "deeplabv3":
            net = seg.deeplabv3_mobilenet_v3_large(weights=seg.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                                                   aux_loss=True if pretrained else None)
            net.aux_classifier = None
            net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, CLASSES, 1)
        else:
            raise ValueError(arch)
        self.arch, self.net = arch, net

    def forward(self, x):
        return self.net(x)["out"]


def food_logit(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE)
    valid = target != IGNORE
    food = F.binary_cross_entropy_with_logits(food_logit(logits)[valid], (target[valid] > 0).float())
    return ce + 0.5 * food


def autocast(dev: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage
    return ndimage.label(mask)


def proposals(food: np.ndarray, classes: np.ndarray) -> list[np.ndarray]:
    """What the app does with the outputs: food pixels split by class into connected items."""
    out = []
    area = food.size
    for c in np.unique(classes[food]):
        labeled, n = components(food & (classes == c))
        for k in range(1, n + 1):
            item = labeled == k
            if item.sum() >= MIN_PREDICTED_FRACTION * area:
                out.append(item)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, pairs: list[tuple[bytes, bytes]], size: int, dev: torch.device, items: bool = True) -> dict:
    model.eval()
    confusion = torch.zeros(CLASSES, CLASSES, dtype=torch.int64)
    found = found_class = total = 0
    loader = DataLoader(Pairs(pairs, size, False), batch_size=32, num_workers=4)
    for x, y in loader:
        with autocast(dev):
            logits = model(x.to(dev)).float()
        pred = logits.argmax(1).cpu()
        valid = y != IGNORE
        confusion += torch.bincount(y[valid] * CLASSES + pred[valid], minlength=CLASSES * CLASSES).view(CLASSES, CLASSES)
        if not items:
            continue
        food = (torch.sigmoid(food_logit(logits)) > 0.5).cpu().numpy()
        cls = (logits[:, 1:].argmax(1) + 1).cpu().numpy()
        for k in range(len(y)):
            truth = y[k].numpy()
            predicted = proposals(food[k], cls[k])
            for c in np.unique(truth):
                if c in (0, IGNORE, OTHER_INGREDIENTS):
                    continue
                item = truth == c
                if item.mean() < MIN_ITEM_FRACTION:
                    continue
                total += 1
                best = max(((iou(item, p), p) for p in predicted), default=(0.0, None), key=lambda t: t[0])
                found += best[0] >= 0.5
                found_class += best[0] >= 0.5 and int(np.bincount(cls[k][best[1]]).argmax()) == c
    inter = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - inter
    present = confusion.sum(1) > 0
    food_inter = confusion[1:, 1:].sum().float()
    food_union = confusion[1:, :].sum().float() + confusion[:, 1:].sum().float() - food_inter
    out = {"mIoU": float((inter[present] / union[present]).mean()), "foodIoU": float(food_inter / food_union),
           "pixelAccuracy": float(inter.sum() / confusion.sum())}
    if items and total:
        out.update({"items": int(total), "itemRecall@0.5": float(found / total), "itemRecallWithClass@0.5": float(found_class / total)})
    return out


def train(data: Path, work: Path, arch: str = "lraspp", size: int = 384, epochs: int = 80, batch: int = 32, lr: float = 2e-3,
          workers: int = 8, limit: int | None = None, out: str = "checkpoint.pt") -> Path:
    work.mkdir(parents=True, exist_ok=True)
    dev = device()
    train_pairs, val_pairs = load_split(data, "train", limit), load_split(data, "val", limit and max(8, limit // 4))
    model = Segmenter(arch).to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    loader = DataLoader(Pairs(train_pairs, size, True), batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=dev.type == "cuda")
    # New class layers learn faster than the pretrained body.
    head = [p for n, p in model.named_parameters() if "classifier" in n]
    body = [p for n, p in model.named_parameters() if "classifier" not in n]
    optimizer = torch.optim.AdamW([{"params": body, "lr": lr * 0.5}, {"params": head, "lr": lr}], weight_decay=0.01)
    steps = epochs * len(loader)
    warmup = min(500, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    checkpoint, history, start = work / out, [], 0
    config = {"arch": arch, "size": size}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state.get("config") == config:
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"]); history = state["history"]; start = state["epoch"] + 1
            print(f"resuming after epoch {start}")
    for epoch in range(start, epochs):
        model.train()
        began, running = time.time(), 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if dev.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with autocast(dev):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(); schedule.step()
            running += loss.item()
        entry = {"epoch": epoch + 1, "loss": running / max(1, len(loader)), "minutes": (time.time() - began) / 60}
        last = epoch + 1 == epochs
        if last or (epoch + 1) % 10 == 0:
            entry["val"] = evaluate(model, val_pairs, size, dev, items=last)
        history.append(entry)
        print(json.dumps(entry), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                    "config": config, "epoch": epoch, "history": history}, checkpoint)
    (work / f"{Path(out).stem}-metrics.json").write_text(json.dumps({"config": config, "train": len(train_pairs),
                                                                    "val": len(val_pairs), "history": history}, indent=1))
    return checkpoint


def load(checkpoint: Path) -> tuple[Segmenter, dict, list]:
    state = torch.load(checkpoint, map_location="cpu")
    model = Segmenter(state["config"]["arch"], pretrained=False)
    model.load_state_dict(state["model"])
    return model.eval(), state["config"], state.get("history", [])


def evaluate_checkpoint(data: Path, checkpoint: Path, limit: int | None = None) -> dict:
    model, config, _ = load(checkpoint)
    dev = device()
    return evaluate(model.to(dev), load_split(data, "val", limit), config["size"], dev)

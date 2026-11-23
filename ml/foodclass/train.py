"""Fine-tunes an ImageNet-pretrained backbone with two heads on the prepared manifest.

- **Dish head:** Food-101 classes, from whole-dish photos. The app asks it about the
  whole plate.
- **Item head:** FoodSeg103 / Nutrition5k / auto-labeled ingredient classes, from item
  crops. The app asks it about each region its segmenter finds.

Dish and ingredient names used to compete in one softmax (v1), so a crop of rice could
lose to "fried rice" and a whole plate to one of its ingredients. Separate heads remove
that, and each head is evaluated only on its own task.

Input is the whole crop squashed to the training size (the app asks Vision for
.scaleFill), so no center crop at eval time either. Checkpoints after every epoch;
rerunning resumes from the last one.

FoodSeg103 training crops are cut fresh every epoch from the whole photos with roughened
masks (crops.app_crop) when regions.csv exists, and Food-101 photos are sometimes
dimmed outside an ellipse like the app's whole-plate input (crops.plate_dim).

`build_model` (single head, v1) stays for the Nutrition5k pseudo-labeler, whose teacher
checkpoint is a v1 model.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .crops import app_crop, plate_dim

SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DISH_SOURCES = {"food101"}
DISH, ITEM = 0, 1


def read_manifest(root: Path) -> list[dict]:
    with (root / "manifest.csv").open() as f:
        return list(csv.DictReader(f))


def read_regions(root: Path) -> list[dict]:
    path = root / "regions.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def task_of(row: dict) -> int:
    return DISH if row["source"] in DISH_SOURCES else ITEM


def build_model(classes: int, pretrained: bool = True, freeze_blocks: int = 0) -> nn.Module:
    """v1: one MobileNetV3-Large softmax over every label."""
    weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    for block in model.features[:freeze_blocks]:
        for p in block.parameters():
            p.requires_grad = False
    return model


ARCHS = {
    # name: (constructor, weights, feature width)
    "mobilenet_v3_large": (torchvision.models.mobilenet_v3_large, "MobileNet_V3_Large_Weights.IMAGENET1K_V2"),
    "efficientnet_v2_s": (torchvision.models.efficientnet_v2_s, "EfficientNet_V2_S_Weights.IMAGENET1K_V1"),
    "convnext_tiny": (torchvision.models.convnext_tiny, "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"),
}


class TwoHead(nn.Module):
    """Shared backbone, a dish head and an item head. Returns (dish_logits, item_logits)."""

    def __init__(self, arch: str, dishes: int, items: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        make, weights = ARCHS[arch]
        net = make(weights=weights if pretrained else None)
        self.arch = arch
        if arch == "mobilenet_v3_large":
            # Keep the 960 -> 1280 hardswish layer; it carries much of the ImageNet head's quality.
            self.features, self.pool = net.features, net.avgpool
            self.neck = nn.Sequential(nn.Flatten(1), *list(net.classifier.children())[:3])
            width = net.classifier[0].out_features
        elif arch == "efficientnet_v2_s":
            self.features, self.pool = net.features, net.avgpool
            self.neck = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout))
            width = net.classifier[1].in_features
        else:  # convnext_tiny
            self.features, self.pool = net.features, net.avgpool
            self.neck = nn.Sequential(*list(net.classifier.children())[:2], nn.Dropout(dropout))  # LayerNorm2d, Flatten
            width = net.classifier[2].in_features
        self.dish = nn.Linear(width, dishes)
        self.item = nn.Linear(width, items)

    def forward(self, x):
        z = self.neck(self.pool(self.features(x)))
        return self.dish(z), self.item(z)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def normalize_tensor() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class Images(Dataset):
    """Rows are manifest rows (files) or regions.csv rows (fresh crops). Yields
    (image, label index within its head, task)."""

    def __init__(self, root: Path, rows: list[dict], dish_index: dict[str, int], item_index: dict[str, int],
                 train: bool, size: int = SIZE, plate_dim_p: float = 0.4):
        self.root, self.rows, self.train = root, rows, train
        self.dish_index, self.item_index, self.plate_dim_p = dish_index, item_index, plate_dim_p
        tail = [T.ToTensor(), T.Normalize(MEAN, STD)]
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.6, 1.66)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.3),
                T.RandomApply([T.RandomRotation(180)], p=0.3),  # Overhead plates have no "up".
                T.ColorJitter(0.35, 0.3, 0.25, 0.04),
                T.RandomApply([T.GaussianBlur(5, (0.1, 1.6))], p=0.25),
                *tail, T.RandomErasing(0.1),
            ])
        else:
            self.transform = T.Compose([T.Resize((size, size)), *tail])

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self, relative: str) -> Image.Image:
        path = Path(relative)
        return Image.open(path if path.is_absolute() else self.root / path)

    def __getitem__(self, i: int):
        row = self.rows[i]
        rng = random.Random()
        if "mask" in row and row.get("mask"):
            photo = self._open(row["image"]).convert("RGB")
            classes = np.asarray(self._open(row["mask"]))
            item = classes == int(row["class_id"])
            others = (classes != 0) & ~item
            image = app_crop(photo, item, others, rng)
            return self.transform(image), self.item_index[row["label"]], ITEM
        image = self._open(row["path"]).convert("RGB")
        task = task_of(row)
        if task == DISH:
            if self.train and rng.random() < self.plate_dim_p:
                image = plate_dim(image, rng)
            return self.transform(image), self.dish_index[row["label"]], DISH
        return self.transform(image), self.item_index[row["label"]], ITEM


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, rows: list[dict], dev: torch.device) -> dict:
    model.eval()
    top1, top5, total = defaultdict(int), defaultdict(int), defaultdict(int)
    offset = 0
    for images, targets, tasks in loader:
        with autocast(dev):
            dish, item = model(images.to(dev, non_blocking=True))
        dish, item = dish.float().cpu(), item.float().cpu()
        for k in range(len(targets)):
            logits = dish[k] if tasks[k] == DISH else item[k]
            best = logits.topk(5).indices
            source = rows[offset + k]["source"]
            for key in (source, "dish" if tasks[k] == DISH else "item"):
                total[key] += 1
                top1[key] += int(best[0] == targets[k])
                top5[key] += int(targets[k] in best)
        offset += len(targets)
    return {key: {"top1": top1[key] / total[key], "top5": top5[key] / total[key], "images": total[key]} for key in total}


def autocast(dev: torch.device):
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def label_sets(rows: list[dict], regions: list[dict]) -> tuple[list[str], list[str]]:
    dishes = sorted({r["label"] for r in rows if task_of(r) == DISH})
    items = sorted({r["label"] for r in rows if task_of(r) == ITEM} | {r["label"] for r in regions})
    return dishes, items


def train(root: Path, epochs: int = 30, batch: int = 128, lr: float = 1e-3, workers: int = 8, limit: int | None = None,
          size: int = SIZE, arch: str = "mobilenet_v3_large", fresh_crops: bool = True, out: str = "checkpoint.pt",
          item_weight: float = 1.0) -> Path:
    rows = read_manifest(root)
    regions = read_regions(root) if fresh_crops else []
    dishes, items = label_sets(rows, regions)
    dish_index = {name: i for i, name in enumerate(dishes)}
    item_index = {name: i for i, name in enumerate(items)}
    train_rows = [r for r in rows if r["split"] == "train"]
    if regions:  # Fresh crops replace FoodSeg103's fixed training crops.
        train_rows = [r for r in train_rows if r["source"] != "foodseg103"] + regions
    val_rows = [r for r in rows if r["split"] == "val"]
    if limit:
        rng = random.Random(0)
        train_rows, val_rows = rng.sample(train_rows, min(limit, len(train_rows))), rng.sample(val_rows, min(max(1, limit // 4), len(val_rows)))
    dev = device()
    model = TwoHead(arch, len(dishes), len(items)).to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    loader_args = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=dev.type == "cuda",
                       prefetch_factor=4 if workers else None)
    train_loader = DataLoader(Images(root, train_rows, dish_index, item_index, True, size), batch_size=batch,
                              shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(Images(root, val_rows, dish_index, item_index, False, size), batch_size=batch * 2,
                            shuffle=False, **loader_args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * len(train_loader)
    warmup = min(1000, steps // 15)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")

    checkpoint, history, start = root / out, [], 0
    config = {"arch": arch, "size": size, "dishes": dishes, "items": items}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state.get("config") == config:
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"]); history = state["history"]; start = state["epoch"] + 1
            print(f"resuming after epoch {start}")

    for epoch in range(start, epochs):
        model.train()
        began, seen, running = time.time(), 0, 0.0
        for step, (images, targets, tasks) in enumerate(train_loader):
            images = images.to(dev, non_blocking=True)
            if dev.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            targets, tasks = targets.to(dev, non_blocking=True), tasks.to(dev, non_blocking=True)
            with autocast(dev):
                dish, item = model(images)
            is_dish = tasks == DISH
            loss = torch.zeros((), device=dev)
            if is_dish.any():
                loss = loss + loss_fn(dish[is_dish].float(), targets[is_dish]).sum()
            if (~is_dish).any():
                loss = loss + item_weight * loss_fn(item[~is_dish].float(), targets[~is_dish]).sum()
            loss = loss / len(targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(); schedule.step()
            seen += len(targets); running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {running / (step + 1):.3f} "
                      f"{seen / (time.time() - began):.0f} img/s", flush=True)
        metrics = evaluate(model, val_loader, val_rows, dev)
        history.append({"epoch": epoch + 1, "loss": running / max(1, len(train_loader)), "val": metrics,
                        "minutes": (time.time() - began) / 60})
        print(json.dumps(history[-1]), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                    "config": config, "epoch": epoch, "history": history}, checkpoint)
    (root / f"{Path(out).stem}-metrics.json").write_text(json.dumps(
        {"config": {k: v for k, v in config.items() if k not in ("dishes", "items")}, "dishes": len(dishes), "items": len(items),
         "train": len(train_rows), "val": len(val_rows), "history": history}, indent=1))
    return checkpoint

"""Fine-tunes an ImageNet-pretrained MobileNetV3-Large on the prepared manifest.

Input is the whole crop squashed to 224x224 (the app asks Vision for .scaleFill),
so no center crop at eval time either. Checkpoints after every epoch; rerunning
resumes from the last one.

Speed on a base M4 (PyTorch MPS) is the constraint: full fine-tuning at 224 px
runs ~60 img/s. Training at 192 px with the first 7 feature blocks frozen runs
~140 img/s (about 11 min per epoch here). Evaluation and export stay at 224, which
the fully convolutional net handles (and the slightly larger test size helps).
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def read_manifest(root: Path) -> list[dict]:
    with (root / "manifest.csv").open() as f:
        return list(csv.DictReader(f))


class Images(Dataset):
    def __init__(self, root: Path, rows: list[dict], index: dict[str, int], train: bool, train_size: int = SIZE):
        self.root, self.rows, self.index = root, rows, index
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(train_size, scale=(0.4, 1.0), ratio=(0.6, 1.66)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2),
                T.ColorJitter(0.25, 0.25, 0.2, 0.03),
                T.ToTensor(), T.Normalize(MEAN, STD), T.RandomErasing(0.15),
            ])
        else:
            self.transform = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        path = Path(row["path"])
        image = Image.open(path if path.is_absolute() else self.root / path).convert("RGB")
        return self.transform(image), self.index[row["label"]]


def build_model(classes: int, pretrained: bool = True, freeze_blocks: int = 0) -> nn.Module:
    weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    for block in model.features[:freeze_blocks]:
        for p in block.parameters():
            p.requires_grad = False
    return model


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, rows: list[dict], dev: torch.device) -> dict:
    model.eval()
    top1, top5, total = defaultdict(int), defaultdict(int), defaultdict(int)
    offset = 0
    for images, targets in loader:
        logits = model(images.to(dev)).float().cpu()
        best = logits.topk(5, dim=1).indices
        for k in range(len(targets)):
            source = rows[offset + k]["source"]
            for key in (source, "all"):
                total[key] += 1
                top1[key] += int(best[k, 0] == targets[k])
                top5[key] += int(targets[k] in best[k])
        offset += len(targets)
    return {key: {"top1": top1[key] / total[key], "top5": top5[key] / total[key], "images": total[key]} for key in total}


def train(root: Path, epochs: int = 6, batch: int = 96, lr: float = 1.5e-3, workers: int = 8, limit: int | None = None,
          train_size: int = 192, freeze_blocks: int = 7) -> Path:
    rows = read_manifest(root)
    labels = sorted({r["label"] for r in rows})
    index = {name: i for i, name in enumerate(labels)}
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if limit:
        train_rows, val_rows = train_rows[:limit], val_rows[: max(1, limit // 4)]
    dev = device()
    model = build_model(len(labels), freeze_blocks=freeze_blocks).to(dev)
    loader_args = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=False)
    train_loader = DataLoader(Images(root, train_rows, index, True, train_size), batch_size=batch, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(Images(root, val_rows, index, False), batch_size=batch * 2, shuffle=False, **loader_args)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.02)
    steps = epochs * len(train_loader)
    warmup = min(500, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    checkpoint, history, start = root / "checkpoint.pt", [], 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state["labels"] == labels:
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"]); history = state["history"]; start = state["epoch"] + 1
            print(f"resuming after epoch {start}")

    for epoch in range(start, epochs):
        model.train()
        for block in model.features[:freeze_blocks]:
            block.eval()  # Frozen blocks keep their ImageNet batch-norm statistics.
        began, seen, running = time.time(), 0, 0.0
        for step, (images, targets) in enumerate(train_loader):
            images, targets = images.to(dev), targets.to(dev)
            loss = loss_fn(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(); schedule.step()
            seen += len(targets); running += float(loss)
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {running / (step + 1):.3f} "
                      f"{seen / (time.time() - began):.0f} img/s", flush=True)
        metrics = evaluate(model, val_loader, val_rows, dev)
        history.append({"epoch": epoch + 1, "loss": running / max(1, len(train_loader)), "val": metrics,
                        "minutes": (time.time() - began) / 60})
        print(json.dumps(history[-1]), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                    "labels": labels, "epoch": epoch, "history": history}, checkpoint)
    (root / "metrics.json").write_text(json.dumps({"labels": len(labels), "train": len(train_rows), "val": len(val_rows),
                                                   "history": history}, indent=1))
    return checkpoint

"""Turns the source datasets into one folder of training images plus a manifest.

- Food-101 (ethz/food101 parquet): whole-dish photos, used as they are.
- FoodSeg103 (parquet with per-pixel masks): one crop per ingredient on the plate,
  padded and with everything else dimmed, exactly like the app's item crops
  (autolabel.verify.crop, which the auto-labeled dataset uses too).
- Optionally the auto-labeled dataset's classify/<split>/<class>/ crops (our own data).

Output: <out>/images/<source>/<split>/<n>.jpg and <out>/manifest.csv (path,label,source,split).
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from autolabel.verify import crop

from .labels import SKIP, normalize

MIN_ITEM_FRACTION = 0.01  # Ingredients smaller than 1% of the photo are too small to name.
MAX_SIDE_PX = 384


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    scale = MAX_SIDE_PX / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    image.save(path, quality=90)


def _label_names(parquet: Path) -> list[str]:
    """ClassLabel names stored by Hugging Face in the parquet schema metadata."""
    meta = pq.read_schema(parquet).metadata or {}
    info = json.loads(meta.get(b"huggingface", b"{}"))
    return info["info"]["features"]["label"]["names"]


def food101(root: Path, out: Path, rows: list[dict], limit: int | None) -> None:
    files = sorted((root / "data").glob("*.parquet"))
    names = _label_names(files[0])
    for file in files:
        split = "train" if file.name.startswith("train") else "val"
        table = pq.read_table(file, columns=["image", "label"])
        count = 0
        for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
            if limit and count >= limit:
                break
            path = out / "images" / "food101" / split / f"{file.stem}_{count}.jpg"
            if not path.exists():
                _save(Image.open(io.BytesIO(image["bytes"])), path)
            rows.append({"path": str(path.relative_to(out)), "label": normalize(names[label]), "source": "food101", "split": split})
            count += 1
        print(f"food101 {file.name}: {count}", flush=True)


def foodseg103(root: Path, out: Path, rows: list[dict], limit: int | None) -> None:
    id2label = {int(k): v for k, v in json.loads((root / "id2label.json").read_text()).items()}
    for file in sorted((root / "data").glob("*.parquet")):
        split = "train" if file.name.startswith("train") else "val"
        table = pq.read_table(file, columns=["image", "label"])
        count = 0
        for n, (image, mask) in enumerate(zip(table.column("image").to_pylist(), table.column("label").to_pylist())):
            if limit and n >= limit:
                break
            photo = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
            classes = np.asarray(Image.open(io.BytesIO(mask["bytes"])))
            if classes.ndim == 3:
                classes = classes[..., 0]
            h, w = classes.shape
            if photo.size != (w, h):
                photo = photo.resize((w, h))
            for class_id in np.unique(classes):
                name = id2label.get(int(class_id), "background")
                if name.lower() in SKIP:
                    continue
                item = classes == class_id
                if item.mean() < MIN_ITEM_FRACTION:
                    continue
                ys, xs = np.nonzero(item)
                box = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
                path = out / "images" / "foodseg103" / split / f"{file.stem}_{n}_{class_id}.jpg"
                if not path.exists():
                    _save(crop(photo, box, item), path)
                rows.append({"path": str(path.relative_to(out)), "label": normalize(name), "source": "foodseg103", "split": split})
                count += 1
        print(f"foodseg103 {file.name}: {count} crops", flush=True)


def autolabeled(root: Path, out: Path, rows: list[dict]) -> None:
    """classify/<split>/<class>/*.jpg from `python -m autolabel export`. Its test and
    golden splits stay out of training."""
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        split = {"train": "train", "val": "val"}.get(split_dir.name)
        if split is None:
            continue
        for file in sorted(split_dir.glob("*/*.jpg")):
            rows.append({"path": str(file.resolve()), "label": normalize(file.parent.name), "source": "autolabel", "split": split})


def prepare(out: Path, food101_dir: Path | None, foodseg_dir: Path | None, extra: list[Path], limit: int | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if foodseg_dir:
        foodseg103(foodseg_dir, out, rows, limit)
    if food101_dir:
        food101(food101_dir, out, rows, limit)
    for root in extra:
        autolabeled(root, out, rows)
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)
    labels = sorted({r["label"] for r in rows})
    print(f"{len(rows)} images, {len(labels)} classes -> {manifest}")
    return manifest

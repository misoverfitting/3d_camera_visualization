"""Converts the trained two-head checkpoint to Core ML and compiles it for the app.

The model takes an RGB image (Vision scales the crop to the training size) and returns
two probability vectors: "dish" (Food-101 classes, for the whole plate) and "item"
(ingredient classes, for each region). ImageNet normalization and softmax are baked in.
Label lists and the label -> category table ride along in the model's metadata and in
a labels.json next to it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .labels import category
from .train import DISH, MEAN, STD, TwoHead, read_manifest, task_of


class Deployable(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image):  # image in [0, 1]
        dish, item = self.model((image - self.mean) / self.std)
        return torch.softmax(dish, dim=1), torch.softmax(item, dim=1)


def load(checkpoint: Path) -> tuple[TwoHead, dict, list]:
    state = torch.load(checkpoint, map_location="cpu")
    config = state["config"]
    model = TwoHead(config["arch"], len(config["dishes"]), len(config["items"]), pretrained=False)
    model.load_state_dict(state["model"])
    return model.eval(), config, state.get("history", [])


def export(root: Path, assets: Path, name: str = "FoodClassifier", checkpoint: str = "checkpoint.pt") -> Path:
    import coremltools as ct

    model, config, history = load(root / checkpoint)
    size, dishes, items = config["size"], config["dishes"], config["items"]
    deploy = Deployable(model).eval()
    traced = torch.jit.trace(deploy, torch.rand(1, 3, size, size))
    categories = {label: category(label) for label in sorted(set(dishes) | set(items))}
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, size, size), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="dish"), ct.TensorType(name="item")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = ("Food classifier (Tare). 'dish': Food-101 dish for a whole plate. "
                                 "'item': ingredient for one region crop (padded, background dimmed).")
    mlmodel.version = f"{config['arch']}-2head-e{len(history)}"
    meta = mlmodel.user_defined_metadata
    meta["categories"] = json.dumps(categories)
    meta["dishes"] = json.dumps(dishes)
    meta["items"] = json.dumps(items)
    if history:
        meta["val"] = json.dumps(history[-1]["val"])

    package = root / f"{name}.mlpackage"
    if package.exists():
        shutil.rmtree(package)
    mlmodel.save(str(package))
    check(mlmodel, deploy, root, dishes, items, size)

    assets.mkdir(parents=True, exist_ok=True)
    compiled = assets / f"{name}.mlmodelc"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), tmp], check=True, capture_output=True)
        if compiled.exists():
            shutil.rmtree(compiled)
        shutil.move(str(Path(tmp) / f"{name}.mlmodelc"), compiled)
    (assets / f"{name}.labels.json").write_text(json.dumps(
        {"format": "two-head", "dishes": dishes, "items": items, "categories": categories, "size": size,
         "version": mlmodel.version, "val": history[-1]["val"] if history else None}, indent=1))
    print(f"wrote {compiled}")
    return compiled


def check(mlmodel, deploy: nn.Module, root: Path, dishes: list[str], items: list[str], size: int, count: int = 24) -> None:
    """Core ML and PyTorch must agree on the top class for a sample of val images."""
    rows = [r for r in read_manifest(root) if r["split"] == "val"][:: 499][:count]
    agree = 0
    for row in rows:
        path = Path(row["path"])
        image = Image.open(path if path.is_absolute() else root / path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        head = "dish" if task_of(row) == DISH else "item"
        with torch.no_grad():
            dish, item = deploy(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None])
        expected = int((dish if head == "dish" else item).argmax())
        got = int(np.asarray(mlmodel.predict({"image": image})[head]).reshape(-1).argmax())
        agree += got == expected
    print(f"Core ML matches PyTorch on {agree}/{len(rows)} val images")
    if rows and agree < 0.9 * len(rows):
        raise RuntimeError("Core ML conversion disagrees with PyTorch")

"""Converts the trained checkpoint to Core ML and compiles it for the app.

The model takes an RGB image (Vision scales the item crop to 224x224) and returns
class probabilities with labels, so VNCoreMLRequest yields VNClassificationObservation.
ImageNet normalization is baked in. The label -> category table rides along in the
model's metadata and in a labels.json next to it.
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
from .train import MEAN, SIZE, STD, build_model, read_manifest


class Deployable(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image):  # image in [0, 1]
        return torch.softmax(self.model((image - self.mean) / self.std), dim=1)


def export(root: Path, assets: Path, name: str = "FoodClassifier") -> Path:
    import coremltools as ct

    state = torch.load(root / "checkpoint.pt", map_location="cpu")
    labels: list[str] = state["labels"]
    model = build_model(len(labels), pretrained=False)
    model.load_state_dict(state["model"])
    deploy = Deployable(model).eval()
    traced = torch.jit.trace(deploy, torch.rand(1, 3, SIZE, SIZE))
    categories = {label: category(label) for label in labels}
    history = state.get("history", [])
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(labels),
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = "Names one food item from a padded crop with the background dimmed (Tare)."
    mlmodel.version = f"mobilenet_v3_large-e{len(history)}"
    mlmodel.user_defined_metadata["categories"] = json.dumps(categories)
    if history:
        mlmodel.user_defined_metadata["val"] = json.dumps(history[-1]["val"])

    package = root / f"{name}.mlpackage"
    if package.exists():
        shutil.rmtree(package)
    mlmodel.save(str(package))
    check(mlmodel, deploy, root, labels)

    assets.mkdir(parents=True, exist_ok=True)
    compiled = assets / f"{name}.mlmodelc"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), tmp], check=True, capture_output=True)
        if compiled.exists():
            shutil.rmtree(compiled)
        shutil.move(str(Path(tmp) / f"{name}.mlmodelc"), compiled)
    (assets / f"{name}.labels.json").write_text(json.dumps(
        {"labels": labels, "categories": categories, "version": mlmodel.version,
         "val": history[-1]["val"] if history else None}, indent=1))
    print(f"wrote {compiled}")
    return compiled


def check(mlmodel, deploy: nn.Module, root: Path, labels: list[str], count: int = 24) -> None:
    """Core ML and PyTorch must agree on the top class for a sample of val images."""
    rows = [r for r in read_manifest(root) if r["split"] == "val"][:: 997][:count]
    agree = 0
    for row in rows:
        path = Path(row["path"])
        image = Image.open(path if path.is_absolute() else root / path).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        with torch.no_grad():
            expected = labels[int(deploy(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]).argmax())]
        got = mlmodel.predict({"image": image})["classLabel"]
        agree += got == expected
    print(f"Core ML matches PyTorch on {agree}/{len(rows)} val images")
    if rows and agree < 0.9 * len(rows):
        raise RuntimeError("Core ML conversion disagrees with PyTorch")

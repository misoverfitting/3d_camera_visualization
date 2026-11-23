"""Converts the segmenter to Core ML and compiles it into the app's Assets.

Input: RGB image squashed to size x size (the app's upright plate crop).
Outputs, both size x size:
- "food": probability that the pixel is food (1 - P(background)).
- "classes": the most likely food class at the pixel (1..103), whether or not it is
  food; the app splits the food mask into items by this.
Normalization and softmax are baked in; class names ride along in the metadata.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .train import MEAN, STD, load

ML = Path(__file__).resolve().parents[1]


class Deployable(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image):  # image in [0, 1]
        logits = self.model((image - self.mean) / self.std)
        food = 1 - torch.softmax(logits, dim=1)[:, 0]
        classes = torch.argmax(logits[:, 1:], dim=1).to(torch.int32) + 1
        return food, classes


def export(checkpoint: Path, assets: Path, name: str = "FoodSegmenter") -> Path:
    import coremltools as ct

    model, config, history = load(checkpoint)
    size = config["size"]
    deploy = Deployable(model).eval()
    example = torch.rand(1, 3, size, size)
    traced = torch.jit.trace(deploy, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, size, size), scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="food"), ct.TensorType(name="classes")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )
    id2label = json.loads((ML / "data" / "FoodSeg103" / "id2label.json").read_text())
    from foodclass.labels import normalize
    names = [normalize(id2label[str(i)]) for i in range(len(id2label))]
    mlmodel.short_description = "Food segmenter (Tare): per-pixel food probability and food class for an upright plate crop."
    mlmodel.version = f"{config['arch']}-{size}-e{len(history)}"
    mlmodel.user_defined_metadata["classes"] = json.dumps(names)
    if history and "val" in history[-1]:
        mlmodel.user_defined_metadata["val"] = json.dumps(history[-1]["val"])

    # Parity: Core ML and PyTorch agree on the food mask for a random image.
    with torch.no_grad():
        food, _ = deploy(example)
    from PIL import Image
    pixels = (example[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    got = np.asarray(mlmodel.predict({"image": Image.fromarray(pixels)})["food"]).reshape(size, size)
    agree = float(((got > 0.5) == (food[0].numpy() > 0.5)).mean())
    print(f"Core ML food mask agrees with PyTorch on {agree:.1%} of pixels")
    if agree < 0.97:
        raise RuntimeError("Core ML conversion disagrees with PyTorch")

    package = checkpoint.parent / f"{name}.mlpackage"
    if package.exists():
        shutil.rmtree(package)
    mlmodel.save(str(package))
    compiled = assets / f"{name}.mlmodelc"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), tmp], check=True, capture_output=True)
        if compiled.exists():
            shutil.rmtree(compiled)
        shutil.move(str(Path(tmp) / f"{name}.mlmodelc"), compiled)
    (assets / f"{name}.json").write_text(json.dumps({"classes": names, "size": size, "version": mlmodel.version,
                                                     "val": history[-1].get("val") if history else None}, indent=1))
    print(f"wrote {compiled}")
    return compiled

"""Auto-labeling pipeline with Claude and SAM replaced by fakes that read the
synthetic images themselves, so every stage runs for real without network or keys."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from autolabel import claude, label, pipeline, verify
from autolabel import taxonomy as taxonomy_module
from autolabel.export import export, rle, split_of
from autolabel.ingest import ingest, select_frames
from autolabel.store import Workspace, read_json

TAXONOMY = taxonomy_module.load()
# Synthetic foods: pure colors on a gray plate over a dark table.
FOODS = {"broccoli": (40, 200, 40), "salmon": (250, 120, 90), "white_rice": (250, 250, 250)}


def meal(seed: int, foods=("broccoli", "salmon", "white_rice"), size=(640, 480)) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", size, (60, 45, 35))
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.ellipse([w * 0.1, h * 0.05, w * 0.9, h * 0.95], fill=(150, 150, 160))
    spots = [(0.3, 0.35), (0.65, 0.35), (0.48, 0.7)]
    for name, (cx, cy) in zip(foods, spots):
        rx, ry = w * (0.08 + 0.03 * rng.random()), h * (0.1 + 0.03 * rng.random())
        draw.ellipse([cx * w - rx, cy * h - ry, cx * w + rx, cy * h + ry], fill=FOODS[name])
    return image


def color_mask(pixels: np.ndarray, rgb) -> np.ndarray:
    return (np.abs(pixels.astype(int) - np.array(rgb)).sum(axis=2) < 60)


def decode(params: dict) -> Image.Image:
    block = next(b for b in params["messages"][0]["content"] if b["type"] == "image")
    return Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))).convert("RGB")


class FakeMessages:
    """Answers label requests by finding the colored blobs, and verify requests by the
    crop's dominant bright color. `miss` makes the first label pass drop an item."""

    def __init__(self):
        self.calls = []
        self.batches = FakeBatches(self)

    def create(self, **params):
        self.calls.append(params)
        image = decode(params)
        pixels = np.asarray(image)
        h, w, _ = pixels.shape
        if "system" in params:
            items = []
            for name, rgb in FOODS.items():
                ys, xs = np.nonzero(color_mask(pixels, rgb))
                if xs.size < 50:
                    continue
                x0, x1, y0, y1 = xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                items.append({"label": name.replace("_", " "), "food_class": name, "confidence": 0.9,
                              "box": [x0, y0, x1, y1], "points": [{"x": cx, "y": cy}, {"x": cx + 0.01, "y": cy}],
                              "partly_hidden": False})
            text = json.dumps({"image_quality": "good", "items": items})
        else:
            counts = {name: color_mask(pixels, rgb).sum() for name, rgb in FOODS.items()}
            best = max(counts, key=counts.get)
            text = json.dumps({"is_single_food_item": True, "food_class": best, "label_correct": True, "mask_quality": "good"})
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                               stop_reason="end_turn", model=params["model"],
                               usage=SimpleNamespace(input_tokens=1500, output_tokens=300))


class FakeBatches:
    def __init__(self, messages):
        self.messages, self.stored = messages, {}

    def create(self, requests):
        batch_id = f"msgbatch_{len(self.stored)}"
        self.stored[batch_id] = [(r["custom_id"], self.messages.create(**r["params"])) for r in requests]
        return SimpleNamespace(id=batch_id)

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended", request_counts=SimpleNamespace(processing=0, succeeded=1))

    def results(self, batch_id):
        for custom_id, message in self.stored[batch_id]:
            yield SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message))


class ColorSegmenter:
    name = "color-test"

    def masks(self, image, prompts):
        pixels = np.asarray(image)
        out = []
        for box, _ in prompts:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            region = np.zeros(pixels.shape[:2], dtype=bool)
            region[y0:y1, x0:x1] = True
            best = max(FOODS.values(), key=lambda rgb: (color_mask(pixels, rgb) & region).sum())
            out.append((color_mask(pixels, best) & region, 0.95))
        return out


@pytest.fixture
def fake(monkeypatch):
    messages = FakeMessages()
    monkeypatch.setattr(claude, "_client", SimpleNamespace(messages=messages))
    return messages


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "work")


def add_meals(tmp_path, ws, count=6):
    folder = tmp_path / "photos"
    for i in range(count):
        (folder / f"dish{i % 3}").mkdir(parents=True, exist_ok=True)
        meal(i).save(folder / f"dish{i % 3}" / f"meal{i}.jpg", quality=95)
    for group in sorted(p for p in folder.iterdir()):
        ingest(ws, [group])


def test_end_to_end_direct(tmp_path, ws, fake):
    add_meals(tmp_path, ws)
    assert len(ws.records()) == 6 and {r.group for r in ws.records()} == {"dish0", "dish1", "dish2"}
    assert pipeline.run_label(ws, TAXONOMY, passes=2, batch=False, workers=4, limit=None) == 12
    # Pass 1 sees a mirrored image; its boxes come back in original coordinates and agree.
    p0 = read_json(ws.label_path(ws.records()[0].id, 0))["labels"]["items"]
    p1 = read_json(ws.label_path(ws.records()[0].id, 1))["labels"]["items"]
    by_class = {i["food_class"]: i["box"] for i in p1}
    assert all(label.iou(i["box"], by_class[i["food_class"]]) > 0.9 for i in p0)

    assert pipeline.run_segment(ws, TAXONOMY, 2, ColorSegmenter()) == 6
    assert pipeline.run_verify(ws, TAXONOMY, batch=False, workers=4, limit=None) == 18
    assert pipeline.run_label(ws, TAXONOMY, 2, False, 4, None) == 0      # Idempotent.
    assert pipeline.run_verify(ws, TAXONOMY, False, 4, None) == 0

    out = tmp_path / "dataset"
    summary = export(ws, TAXONOMY, out, golden={"dish2"})
    assert summary["stats"]["items_accepted"] == 18 and summary["stats"]["images_complete"] == 6
    assert summary["decisions"] == {"accepted_all_agree": 18}
    golden = json.loads((out / "annotations" / "golden.json").read_text())
    assert {i["group"] for i in golden["images"]} == {"dish2"}
    train_val_test = [json.loads((out / "annotations" / f"{s}.json").read_text()) for s in ("train", "val", "test")]
    assert all(i["group"] != "dish2" for split in train_val_test for i in split["images"])
    # Masks decode back to the item's pixels and the label image matches.
    annotation = golden["annotations"][0]
    size, counts = annotation["segmentation"]["size"], annotation["segmentation"]["counts"]
    flat = np.repeat(np.tile([0, 1], len(counts))[: len(counts)], counts).astype(bool)
    assert flat.sum() == annotation["area"]
    label_image = np.asarray(Image.open(out / "masks" / f"{annotation['image_id']}.png"))
    assert (flat.reshape(size, order="F") == (label_image == 1)).all()
    crops = list((out / "classify").rglob("*.jpg"))
    assert len(crops) == 18 and {c.parent.name for c in crops} == {"broccoli", "salmon", "white_rice"}
    assert "Accepted items: 18" in (out / "report.md").read_text()
    assert summary["cost_usd"]["total"] > 0


def test_batches_submit_and_collect(tmp_path, ws, fake):
    add_meals(tmp_path, ws, count=3)
    assert pipeline.run_label(ws, TAXONOMY, 2, batch=True, workers=1, limit=None) == 6
    assert list(ws.batches().values())[0]["stage"] == "label"
    assert pipeline.run_label(ws, TAXONOMY, 2, True, 1, None) == 0      # Pending: not resubmitted.
    assert pipeline.collect(ws, TAXONOMY, wait=False) == 0
    assert ws.batches() == {}
    assert all(read_json(ws.label_path(r.id, p))["labels"]["items"] for r in ws.records() for p in (0, 1))
    assert all(entry["batch"] for entry in ws.usage())


def test_decisions_need_independent_agreement():
    base = {"classes": ["salmon"], "labels": ["salmon"], "agreement": "both"}
    yes = {"is_single_food_item": True, "food_class": "salmon", "label_correct": True, "mask_quality": "good"}
    assert verify.decide(base, yes) == ("salmon", "accepted_all_agree")
    assert verify.decide(base, None) == (None, "unverified")
    assert verify.decide(base, {**yes, "food_class": "tuna"})[0] is None
    assert verify.decide(base, {**yes, "mask_quality": "too_large"}) == (None, "mask_too_large")
    tie = {**base, "agreement": "class_differs", "classes": ["salmon", "tuna"]}
    assert verify.decide(tie, {**yes, "food_class": "tuna"}) == ("tuna", "accepted_verifier_broke_tie")
    single = {**base, "agreement": "pass1_only"}
    assert verify.decide(single, {**yes, "label_correct": False})[0] is None
    assert verify.decide({**base, "rejected": "mask_spills_out_of_box"}, yes) == (None, "mask_spills_out_of_box")


def test_consensus_matches_by_overlap():
    item = lambda cls, box: {"label": cls, "food_class": cls, "category": "x", "confidence": 0.8, "box": box,
                             "points": [[box[0] + 0.01, box[1] + 0.01]], "partly_hidden": False}
    a = {"items": [item("salmon", [0.1, 0.1, 0.3, 0.3]), item("peas", [0.6, 0.6, 0.7, 0.7])]}
    b = {"items": [item("tuna", [0.11, 0.1, 0.31, 0.3]), item("rice", [0.4, 0.1, 0.5, 0.2])]}
    out = {c["agreement"]: c for c in label.consensus([a, b])}
    assert out["class_differs"]["classes"] == ["salmon", "tuna"]
    assert out["pass0_only"]["food_class"] == "peas"
    assert out["pass1_only"]["food_class"] == "rice"


def test_parse_unmirrors_and_cleans():
    text = json.dumps({"image_quality": "good", "items": [
        {"label": "x", "food_class": "not_a_class", "confidence": 2, "box": [0.5, 0.2, 0.1, 0.4],
         "points": [{"x": 0.3, "y": 0.9}], "partly_hidden": False},
        {"label": "tiny", "food_class": "peas", "confidence": 1, "box": [0.1, 0.1, 0.105, 0.2], "points": [], "partly_hidden": False}]})
    items = label.parse(text, TAXONOMY, mirrored=True)["items"]
    assert len(items) == 1
    assert items[0]["food_class"] == "other_food" and items[0]["confidence"] == 1
    assert items[0]["box"] == pytest.approx([0.5, 0.2, 0.9, 0.4])
    assert items[0]["points"] == [pytest.approx([0.7, 0.4])]   # Clipped into the box, then mirrored.


def test_rle_round_trip():
    mask = np.zeros((5, 7), dtype=bool)
    mask[0, 0] = mask[1:3, 2:5] = mask[4, 6] = True
    encoded = rle(mask)
    flat = np.repeat(np.tile([0, 1], len(encoded["counts"]))[: len(encoded["counts"])], encoded["counts"]).astype(bool)
    assert encoded["counts"][0] == 0 and (flat.reshape(encoded["size"], order="F") == mask).all()


def test_splits_are_by_group_and_stable():
    assert split_of("dish-a", set()) == split_of("dish-a", set())
    assert split_of("dish-a", {"dish-a"}) == "golden"
    splits = [split_of(f"dish-{i}", set()) for i in range(2000)]
    assert 0.65 < splits.count("train") / 2000 < 0.75 and 0.17 < splits.count("test") / 2000 < 0.23


def test_ingests_recording_zip_upright_with_diverse_frames(tmp_path, ws):
    frames = []
    archive = tmp_path / "scan.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for i, tilt in enumerate([0, 5, 30, 31, 60, 61]):
            buffer = io.BytesIO()
            Image.new("RGB", (400, 300), (i * 30, 0, 0)).save(buffer, format="JPEG")
            z.writestr(f"ABC/images/{i:05d}.jpg", buffer.getvalue())
            t = np.radians(tilt)
            forward = [0, -np.cos(t), -np.sin(t)]  # Looking down, tilted toward -Z.
            matrix = [1, 0, 0, 0, 0, 1, 0, 0, -forward[0], -forward[1], -forward[2], 0, 0, 0, 0, 1]
            frames.append({"image": f"images/{i:05d}.jpg", "camera_to_world": matrix, "tracking": "normal"})
        z.writestr("ABC/manifest.json", json.dumps({"frames": frames, "image_orientation": "native_sensor_unrotated"}))
    assert ingest(ws, [archive], frames_per_capture=3) == 3
    records = ws.records()
    assert all((r.width, r.height) == (300, 400) for r in records)      # Rotated upright.
    assert {r.group for r in records} == {"scan"}
    chosen = [f["image"] for f in select_frames(frames, 3)]
    assert chosen[0] == "images/00000.jpg"                              # Most overhead first,
    assert {"images/00004.jpg", "images/00005.jpg"} & set(chosen)       # then the steepest angle,
    assert not {"images/00000.jpg", "images/00001.jpg"} <= set(chosen)  # not near-duplicates.

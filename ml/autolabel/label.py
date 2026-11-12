"""Label passes: Claude lists every food item with a class, a box and interior points.

Two independent passes per image, the second on a mirrored copy (coordinates are
mirrored back), so agreement between them is real evidence rather than a repeat of
the same answer. `consensus` matches the passes' items into candidates.
"""
from __future__ import annotations

import json

from PIL import Image, ImageOps

from .claude import image_block, json_format
from .config import LABEL_EFFORT, LABEL_MODEL, MATCH_IOU
from .taxonomy import OTHER, Taxonomy

QUALITY = ["good", "blurry", "too_dark", "food_cut_off", "no_food"]


def schema(taxonomy: Taxonomy) -> dict:
    point = {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
             "required": ["x", "y"], "additionalProperties": False}
    item = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "food_class": {"type": "string", "enum": taxonomy.classes},
            "confidence": {"type": "number"},
            "box": {"type": "array", "items": {"type": "number"}},
            "points": {"type": "array", "items": point},
            "partly_hidden": {"type": "boolean"},
        },
        "required": ["label", "food_class", "confidence", "box", "points", "partly_hidden"],
        "additionalProperties": False,
    }
    return {"type": "object",
            "properties": {"image_quality": {"type": "string", "enum": QUALITY},
                           "items": {"type": "array", "items": item}},
            "required": ["image_quality", "items"], "additionalProperties": False}


def system_prompt(taxonomy: Taxonomy) -> str:
    sauce = ("Sauces, dressings, dips and spreads are separate items when they form a visible region of their own."
             if taxonomy.sauce_policy == "separate" else
             "Fold sauces and toppings into the item they sit on.")
    return f"""You label photos of meals for a food-recognition training set. Each photo shows food on a plate, in a bowl or on a table, usually from above or at an angle.

List every distinct food item you can see:
- One item per physically separate food or clearly different food. Several pieces of the same food lying together (fries, broccoli florets) are one item. A mixed dish that can't be separated (stir-fry, soup) is one item.
- {sauce}
- Ignore plates, bowls, cutlery, napkins, packaging text, hands, and food that is mostly outside the photo.

For each item:
- label: a short specific name ("grilled salmon fillet", "steamed white rice").
- food_class: the best match from the classes below; use {OTHER} only when nothing fits.
- confidence: 0 to 1, how sure you are of the food_class.
- box: [x_min, y_min, x_max, y_max] tightly around the item.
- points: 2 or 3 points clearly on the item, well inside its outline and spread across it, never on an edge, on garnish, or on another item. Use 2 for small items.
- partly_hidden: true if another item or object covers part of it.

All coordinates are fractions of image width (x) and height (y) from 0 to 1, with (0, 0) at the top-left.
image_quality: "good" unless the photo is blurry, too dark, has the food cut off, or shows no food. When there is no food, return no items.

Classes by category:
{taxonomy.prompt_block()}"""


def request(image: Image.Image, taxonomy: Taxonomy, pass_index: int) -> dict:
    shown = ImageOps.mirror(image) if pass_index == 1 else image
    return {
        "model": LABEL_MODEL,
        "max_tokens": 8000,
        "system": system_prompt(taxonomy),
        "output_config": {"effort": LABEL_EFFORT, "format": json_format(schema(taxonomy))},
        "messages": [{"role": "user", "content": [image_block(shown), {"type": "text", "text": "Label the food items in this photo."}]}],
    }


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def parse(text: str, taxonomy: Taxonomy, mirrored: bool) -> dict:
    """Validates and cleans one pass; mirrored passes are mapped back to the original."""
    data = json.loads(text)
    items = []
    for raw in data.get("items", []):
        box = raw.get("box", [])
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = (_clamp(v) for v in box)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            continue
        # Points must lie in the box; pull strays onto it.
        points = [[min(x1, max(x0, _clamp(p["x"]))), min(y1, max(y0, _clamp(p["y"])))] for p in raw.get("points", [])][:3]
        if not points:
            points = [[(x0 + x1) / 2, (y0 + y1) / 2]]
        if mirrored:
            x0, x1 = 1 - x1, 1 - x0
            points = [[1 - x, y] for x, y in points]
        food_class = raw.get("food_class") if raw.get("food_class") in taxonomy.category_of else OTHER
        items.append({"label": str(raw.get("label", "")).strip(), "food_class": food_class,
                      "category": taxonomy.category_of[food_class], "confidence": _clamp(raw.get("confidence", 0)),
                      "box": [x0, y0, x1, y1], "points": points, "partly_hidden": bool(raw.get("partly_hidden"))})
    return {"image_quality": data.get("image_quality", "good"), "items": items}


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def consensus(passes: list[dict]) -> list[dict]:
    """Greedy one-to-one matching of pass items by box IoU. Matched items average their
    boxes and keep both classes; unmatched items stay as single-pass candidates for the
    verifier to accept or reject."""
    if len(passes) == 1:
        return [{**item, "agreement": "single", "classes": [item["food_class"]], "labels": [item["label"]]}
                for item in passes[0]["items"]]
    a, b = passes[0]["items"], passes[1]["items"]
    pairs = sorted(((iou(x["box"], y["box"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, out = set(), set(), []
    for overlap, i, j in pairs:
        if overlap < MATCH_IOU or i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        x, y = a[i], b[j]
        box = [(p + q) / 2 for p, q in zip(x["box"], y["box"])]
        same = x["food_class"] == y["food_class"]
        best = x if x["confidence"] >= y["confidence"] else y
        out.append({**best, "box": box, "points": x["points"], "confidence": (x["confidence"] + y["confidence"]) / 2,
                    "agreement": "both" if same else "class_differs", "classes": [x["food_class"], y["food_class"]],
                    "labels": [x["label"], y["label"]], "match_iou": overlap,
                    "partly_hidden": x["partly_hidden"] or y["partly_hidden"]})
    for items, used, name in ((a, used_a, "pass0_only"), (b, used_b, "pass1_only")):
        out += [{**item, "agreement": name, "classes": [item["food_class"]], "labels": [item["label"]]}
                for k, item in enumerate(items) if k not in used]
    return out

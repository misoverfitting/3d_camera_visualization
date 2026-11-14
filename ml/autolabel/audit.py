"""Automated quality estimate: a stronger model reviews a random sample of accepted
labels drawn over the image and grades each one, plus anything missed. The report
turns this into precision/recall estimates with confidence intervals, so label
quality is measured without anyone labeling by hand."""
from __future__ import annotations

import json
import math

from PIL import Image, ImageDraw

from .claude import client, image_block, json_format
from .config import AUDIT_EFFORT, AUDIT_MODEL

COLORS = ["#E4572E", "#17BEBB", "#FFC914", "#76B041", "#B07CFF", "#FF6F91", "#2E86AB", "#F18F01"]


def overlay(image: Image.Image, items: list[dict]) -> Image.Image:
    shown = image.convert("RGB").copy()
    draw = ImageDraw.Draw(shown)
    w, h = shown.size
    for n, item in enumerate(items, start=1):
        color = COLORS[(n - 1) % len(COLORS)]
        x0, y0, x1, y1 = item["box"]
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=color, width=max(2, w // 300))
        draw.rectangle([x0 * w, y0 * h, x0 * w + 28, y0 * h + 22], fill=color)
        draw.text((x0 * w + 6, y0 * h + 4), str(n), fill="black")
    return shown


SCHEMA = {"type": "object",
          "properties": {"items": {"type": "array", "items": {
              "type": "object",
              "properties": {"number": {"type": "integer"}, "is_food": {"type": "boolean"},
                             "class_correct": {"type": "boolean"}, "box_ok": {"type": "boolean"}},
              "required": ["number", "is_food", "class_correct", "box_ok"], "additionalProperties": False}},
              "missed_items": {"type": "integer"}},
          "required": ["items", "missed_items"], "additionalProperties": False}


def request(image: Image.Image, items: list[dict]) -> dict:
    listing = "\n".join(f"{n}. {item['food_class']} (\"{item['label']}\")" for n, item in enumerate(items, start=1))
    text = f"""You are auditing automatic food labels for a training set. The numbered boxes on the photo were labeled:
{listing or "(no items)"}

For each number: is_food (the box is on a food item, not the plate or table), class_correct (the class fits that food), box_ok (the box covers that item reasonably tightly).
missed_items: how many clearly visible food items have no box. Be strict: a label that is only roughly right is not correct."""
    return {"model": AUDIT_MODEL, "max_tokens": 16000, "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default", "output_config": {"effort": AUDIT_EFFORT, "format": json_format(SCHEMA)},
            "messages": [{"role": "user", "content": [image_block(overlay(image, items)), {"type": "text", "text": text}]}]}


def run(params: dict) -> tuple[dict | None, object]:
    """Audits run directly (server-side fallbacks are not available on batches)."""
    message = client().beta.messages.create(**params)
    if message.stop_reason in ("refusal", "max_tokens"):
        return None, message.usage
    text = next((b.text for b in message.content if b.type == "text"), None)
    return (json.loads(text) if text else None), message.usage


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centre - half), min(1.0, centre + half))

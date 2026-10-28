"""Food identification proxy: one overhead photo of a plate in, labeled items out.

The phone never holds the Anthropic key; it posts a JPEG here and gets back each
food item's label, a box, and 2-3 points well inside it (normalized image
coordinates, origin top-left), which the app lifts into 3D as probe points.
"""
from __future__ import annotations

import base64
import os
from typing import Literal

import anthropic
from fastapi import APIRouter, Header, HTTPException, UploadFile
from pydantic import BaseModel

MODEL = os.environ.get("FOOD_VISION_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("FOOD_VISION_EFFORT", "medium")
# Optional shared secret so the deployed proxy is not an open Claude endpoint.
API_TOKEN = os.environ.get("TARE_API_TOKEN")
MAX_IMAGE_BYTES = 5 * 1024 * 1024

Category = Literal["protein", "starch", "vegetable", "fruit", "dairy", "sauce", "dessert", "drink", "other"]


class Point(BaseModel):
    x: float
    y: float


class Item(BaseModel):
    label: str
    category: Category
    confidence: float
    box: list[float]
    points: list[Point]


class Scene(BaseModel):
    items: list[Item]


PROMPT = """This photo is taken from above a plate or bowl of food. Identify each distinct food item on it.

For every item:
- label: a short specific name a nutrition app would use ("grilled chicken breast", "white rice", "steamed broccoli"). Separate items that are physically separate or clearly different foods; one portion of a mixed dish (e.g. a stir-fry) is one item.
- category: the closest food category.
- confidence: 0 to 1, how sure you are of the label.
- box: [x_min, y_min, x_max, y_max] around the item.
- points: 2 or 3 points that lie clearly on the item, well inside its outline (not on edges, garnish, sauce pooled around it, or another item), spread across it. Use 2 for small items.

All coordinates are fractions of the image width (x) and height (y), from 0 to 1, with (0, 0) at the top-left corner.
Ignore the plate itself, cutlery, napkins, and anything off the plate. Return an empty list if there is no food."""

router = APIRouter()
_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def clean(scene: Scene) -> list[dict]:
    """Clamps coordinates and keeps 2-3 points per item, falling back to the box center."""
    items = []
    for item in scene.items:
        if len(item.box) != 4:
            continue
        x0, y0, x1, y1 = (_clamp(v) for v in item.box)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        points = [[_clamp(p.x), _clamp(p.y)] for p in item.points][:3]
        if len(points) < 2:
            points.append([(x0 + x1) / 2, (y0 + y1) / 2])
        items.append({"label": item.label.strip(), "category": item.category,
                      "confidence": _clamp(item.confidence), "box": [x0, y0, x1, y1], "points": points})
    return items


def identify(jpeg: bytes) -> dict:
    response = client().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": EFFORT},
        output_format=Scene,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(jpeg).decode("ascii")}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    if response.stop_reason == "refusal":
        raise HTTPException(422, "The model declined to label this image")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise HTTPException(502, "The model returned no usable result")
    return {"items": clean(response.parsed_output), "model": response.model}


@router.post("/api/food/identify")
def identify_food(image: UploadFile, x_tare_token: str | None = Header(default=None)):
    if API_TOKEN and x_tare_token != API_TOKEN:
        raise HTTPException(401, "Missing or wrong X-Tare-Token")
    jpeg = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(jpeg) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large")
    if not jpeg.startswith(b"\xff\xd8"):
        raise HTTPException(400, "Expected a JPEG image")
    try:
        return identify(jpeg)
    except anthropic.AuthenticationError as exc:
        raise HTTPException(503, "Food vision is not configured on the server") from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(429, "Food vision is busy; try again shortly") from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Food vision failed ({exc.status_code})") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(502, "Could not reach the food vision service") from exc

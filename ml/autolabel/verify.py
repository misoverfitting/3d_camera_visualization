"""Second look at every candidate: Claude sees the item's crop with everything outside
its mask dimmed (the same crop the on-device classifier will see, tare_impl.md §6.11)
and confirms the class and the mask. `decide` combines this with pass agreement."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .claude import image_block, json_format
from .config import CROP_BACKGROUND_DIM, CROP_PADDING, VERIFY_EFFORT, VERIFY_MODEL
from .taxonomy import OTHER, Taxonomy

MASK_QUALITY = ["good", "partial", "too_large", "wrong_object"]


def crop(image: Image.Image, box: list[float], mask: np.ndarray | None) -> Image.Image:
    """Padded crop around the item, background dimmed (not blacked out)."""
    w, h = image.size
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    pad_x, pad_y = (x1 - x0) * CROP_PADDING, (y1 - y0) * CROP_PADDING
    left, top = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
    right, bottom = min(w, int(x1 + pad_x + 1)), min(h, int(y1 + pad_y + 1))
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    if mask is None:
        mask = np.zeros((h, w), dtype=bool)
        mask[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
    dimmed = np.where(mask[..., None], pixels, pixels * CROP_BACKGROUND_DIM)
    return Image.fromarray(dimmed[top:bottom, left:right].astype(np.uint8))


def schema(taxonomy: Taxonomy) -> dict:
    return {"type": "object",
            "properties": {"is_single_food_item": {"type": "boolean"},
                           "food_class": {"type": "string", "enum": taxonomy.classes},
                           "label_correct": {"type": "boolean"},
                           "mask_quality": {"type": "string", "enum": MASK_QUALITY}},
            "required": ["is_single_food_item", "food_class", "label_correct", "mask_quality"],
            "additionalProperties": False}


def request(item_crop: Image.Image, candidate: dict, taxonomy: Taxonomy) -> dict:
    labels = " / ".join(dict.fromkeys(candidate["labels"]))
    classes = " / ".join(dict.fromkeys(candidate["classes"]))
    text = f"""This crop is from a photo of a meal. The bright region is one item that was labeled "{labels}" (class: {classes}); everything else is dimmed.

- is_single_food_item: true if the bright region is one food item (or one pile of the same food), not the plate, table, cutlery, packaging or several different foods.
- food_class: the best class for the bright region from the allowed list; {OTHER} if none fits.
- label_correct: true if "{labels}" correctly names it.
- mask_quality: "good" if the bright region covers the item and little else; "partial" if it misses some of the item; "too_large" if it includes a lot that isn't the item; "wrong_object" if it is mostly something else.

Allowed classes by category:
{taxonomy.prompt_block()}"""
    return {"model": VERIFY_MODEL, "max_tokens": 4000,
            "output_config": {"effort": VERIFY_EFFORT, "format": json_format(schema(taxonomy))},
            "messages": [{"role": "user", "content": [image_block(item_crop), {"type": "text", "text": text}]}]}


def parse(text: str, taxonomy: Taxonomy) -> dict:
    data = json.loads(text)
    if data.get("food_class") not in taxonomy.category_of:
        data["food_class"] = OTHER
    return data


def decide(candidate: dict, verdict: dict | None) -> tuple[str | None, str]:
    """(final class or None, reason). Accepts only what independent looks agree on:
    both passes and the verifier, or one pass whose class the verifier confirms."""
    if candidate.get("rejected"):
        return None, candidate["rejected"]
    if verdict is None:
        return None, "unverified"
    if not verdict["is_single_food_item"]:
        return None, "not_a_single_food_item"
    if verdict["mask_quality"] in ("too_large", "wrong_object"):
        return None, f"mask_{verdict['mask_quality']}"
    verified = verdict["food_class"]
    if candidate["agreement"] == "both" and verified == candidate["classes"][0]:
        return verified, "accepted_all_agree"
    if candidate["agreement"] == "class_differs" and verified in candidate["classes"]:
        return verified, "accepted_verifier_broke_tie"
    if candidate["agreement"] in ("single", "pass0_only", "pass1_only") and verified == candidate["classes"][0] \
            and verdict["label_correct"]:
        return verified, "accepted_one_pass_verified"
    return None, "classes_disagree"

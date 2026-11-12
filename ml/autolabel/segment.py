"""Turns each candidate's box + interior points into a pixel mask with SAM, then
rejects masks that are clearly wrong (grabbed the whole plate, tiny specks).

SAM is optional: `--segmenter none` keeps boxes only, which still trains a detector
and the classifier but not the item segmenter (tare_impl.md §6.9)."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from .config import MAX_MASK_TO_BOX, MIN_MASK_FRACTION, MIN_MASK_TO_BOX


class Segmenter(Protocol):
    name: str

    def masks(self, image: Image.Image, prompts: list[tuple[list[float], list[list[float]]]]) -> list[tuple[np.ndarray, float]]:
        """One (bool mask HxW, score) per (pixel box, pixel points) prompt."""


class SamSegmenter:
    """Segment Anything via Hugging Face transformers (Apache-2.0 weights)."""

    def __init__(self, checkpoint: str = "facebook/sam-vit-base"):
        import torch
        from transformers import SamModel, SamProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SamModel.from_pretrained(checkpoint).to(self.device).eval()
        self.processor = SamProcessor.from_pretrained(checkpoint)
        self.name = checkpoint

    def masks(self, image, prompts):
        if not prompts:
            return []
        torch = self.torch
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        out = []
        for box, points in prompts:
            prompt = self.processor(image, input_boxes=[[box]], input_points=[[points]],
                                    input_labels=[[[1] * len(points)]], return_tensors="pt")
            prompt.pop("pixel_values", None)
            # MPS has no float64; coordinates arrive as float64 tensors.
            prompt = {k: (v.float() if v.dtype == torch.float64 else v).to(self.device) for k, v in prompt.items()}
            with torch.no_grad():
                result = self.model(image_embeddings=embeddings, multimask_output=True, **prompt)
            masks = self.processor.image_processor.post_process_masks(
                result.pred_masks.cpu(), prompt["original_sizes"].cpu(), prompt["reshaped_input_sizes"].cpu())[0][0]
            scores = result.iou_scores.cpu()[0, 0]
            best = int(scores.argmax())
            out.append((masks[best].numpy().astype(bool), float(scores[best])))
        return out


def check_mask(mask: np.ndarray, box_px: list[float]) -> str | None:
    """Reason to reject a mask, or None."""
    area = int(mask.sum())
    if area < MIN_MASK_FRACTION * mask.size:
        return "mask_too_small"
    box_area = max(1.0, (box_px[2] - box_px[0]) * (box_px[3] - box_px[1]))
    if area > MAX_MASK_TO_BOX * box_area:
        return "mask_spills_out_of_box"   # Usually the plate or the table.
    if area < MIN_MASK_TO_BOX * box_area:
        return "mask_covers_little_of_box"
    return None


def segment_candidates(image: Image.Image, candidates: list[dict], segmenter: Segmenter) -> tuple[list[dict], np.ndarray | None]:
    """Adds mask stats to candidates and returns (candidates, stacked masks). Overlaps go
    to the smaller mask, so a small item on top of a larger one keeps its pixels."""
    w, h = image.size
    prompts = [([c["box"][0] * w, c["box"][1] * h, c["box"][2] * w, c["box"][3] * h], [[x * w, y * h] for x, y in c["points"]])
               for c in candidates]
    results = segmenter.masks(image, prompts)
    if not results:
        return candidates, None
    stack = np.stack([m for m, _ in results])
    order = np.argsort([m.sum() for m, _ in results])
    claimed = np.zeros(stack.shape[1:], dtype=bool)
    for k in order:
        stack[k] &= ~claimed
        claimed |= stack[k]
    out = []
    for k, (candidate, (_, score), (box_px, _)) in enumerate(zip(candidates, results, prompts)):
        reject = check_mask(stack[k], box_px)
        out.append({**candidate, "mask_index": k, "mask_score": score, "mask_area_px": int(stack[k].sum()),
                    **({"rejected": reject} if reject else {})})
    return out, stack

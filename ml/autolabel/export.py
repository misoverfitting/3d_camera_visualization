"""Writes the training dataset and a report.

    <out>/images/<id>.jpg
    <out>/masks/<id>.png                  instance index per pixel (0 = background), when masks exist
    <out>/annotations/<split>.json        COCO instances (boxes, RLE masks, food classes)
    <out>/classify/<split>/<class>/*.jpg  item crops for the classifier (tare_impl.md §6.11)
    <out>/dataset.json                    versions, models, counts, split of every group
    <out>/report.md                       acceptance, agreement, audit estimates, cost

Detection/segmentation annotations include only *complete* images: images where every
candidate was decided with confidence. An image with a rejected item would teach the
detector that a real food is background. Accepted items from incomplete images still
become classifier crops.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from . import verify
from .audit import wilson
from .config import (AUDIT_MODEL, LABEL_MODEL, LABEL_PROMPT_VERSION, PRICES, SPLIT_FRACTIONS, VERIFY_MODEL,
                     VERIFY_PROMPT_VERSION)
from .pipeline import decisions, load_masks
from .store import Workspace, read_json, write_json
from .taxonomy import Taxonomy

# Rejections that mean "not a food item": safe to drop without making the image incomplete.
HARMLESS = {"not_a_single_food_item", "mask_wrong_object"}


def split_of(group: str, golden: set[str]) -> str:
    """Deterministic split by group hash; golden groups never reach train/val (§9.1, §13)."""
    if group in golden:
        return "golden"
    u = int(hashlib.sha256(group.encode()).hexdigest()[:12], 16) / 16 ** 12
    cumulative = 0.0
    for name, fraction in SPLIT_FRACTIONS.items():
        cumulative += fraction
        if u < cumulative:
            return name
    return "test"


def rle(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE: run lengths in column-major order, starting with a (possibly
    empty) run of zeros."""
    flat = mask.flatten(order="F").astype(bool)
    counts: list[int] = []
    if flat.size:
        boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        edges = np.concatenate([[0], boundaries, [flat.size]])
        runs = np.diff(edges).tolist()
        counts = ([0] if flat[0] else []) + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(c) for c in counts]}


def export(ws: Workspace, taxonomy: Taxonomy, out: Path, golden: set[str]) -> dict:
    out = Path(out)
    for sub in ("images", "masks", "annotations", "classify"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    categories = [{"id": i + 1, "name": c, "supercategory": taxonomy.category_of[c]} for i, c in enumerate(taxonomy.classes)]
    category_id = {c["name"]: c["id"] for c in categories}
    coco = {s: {"images": [], "annotations": [], "categories": categories} for s in (*SPLIT_FRACTIONS, "golden")}
    stats = Counter()
    reasons, agreement, classes = Counter(), Counter(), Counter()
    groups: dict[str, str] = {}
    annotation_id = 0

    for record in ws.records():
        decided = decisions(ws, record)
        if decided is None:
            stats["images_unprocessed"] += 1
            continue
        split = groups.setdefault(record.group, split_of(record.group, golden))
        stats["images_processed"] += 1
        accepted = [(k, c, final) for k, (c, final, _) in enumerate(decided) if final]
        for c, final, reason in decided:
            reasons[reason] += 1
            agreement[c["agreement"]] += 1
        complete = all(final or reason in HARMLESS for _, final, reason in decided)
        if not accepted and not complete:
            continue
        image = Image.open(ws.image_path(record.id)).convert("RGB")
        masks = load_masks(ws, record.id)
        shutil.copyfile(ws.image_path(record.id), out / "images" / f"{record.id}.jpg")
        for k, candidate, final in accepted:
            mask = masks[candidate["mask_index"]] if masks is not None and "mask_index" in candidate else None
            folder = out / "classify" / split / final
            folder.mkdir(parents=True, exist_ok=True)
            verify.crop(image, candidate["box"], mask).save(folder / f"{record.id}_{k}.jpg", quality=92)
            classes[final] += 1
        stats["items_accepted"] += len(accepted)
        if not complete:
            stats["images_incomplete"] += 1
            continue
        stats["images_complete"] += 1
        w, h = image.size
        coco[split]["images"].append({"id": record.id, "file_name": f"images/{record.id}.jpg", "width": w, "height": h,
                                      "group": record.group, "source_kind": record.kind})
        label_image = np.zeros((h, w), dtype=np.uint8)
        for n, (k, candidate, final) in enumerate(accepted, start=1):
            x0, y0, x1, y1 = candidate["box"]
            mask = masks[candidate["mask_index"]] if masks is not None and "mask_index" in candidate else None
            annotation_id += 1
            annotation = {"id": annotation_id, "image_id": record.id, "category_id": category_id[final],
                          "bbox": [x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h], "iscrowd": 0,
                          "label": candidate["label"], "agreement": candidate["agreement"]}
            if mask is not None:
                ys, xs = np.nonzero(mask)
                annotation.update(segmentation=rle(mask), area=int(mask.sum()),
                                  bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)])
                label_image[mask] = n
            else:
                annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
            coco[split]["annotations"].append(annotation)
        if masks is not None:
            Image.fromarray(label_image).save(out / "masks" / f"{record.id}.png")

    for split, data in coco.items():
        write_json(out / "annotations" / f"{split}.json", data)
    summary = {"taxonomy_version": taxonomy.version, "label_prompt": LABEL_PROMPT_VERSION,
               "verify_prompt": VERIFY_PROMPT_VERSION, "models": {"label": LABEL_MODEL, "verify": VERIFY_MODEL, "audit": AUDIT_MODEL},
               "stats": dict(stats), "decisions": dict(reasons), "agreement": dict(agreement), "classes": dict(classes),
               "splits": {s: {"images": len(d["images"]), "annotations": len(d["annotations"])} for s, d in coco.items()},
               "groups": groups, "audit": audit_summary(ws), "cost_usd": cost(ws)}
    write_json(out / "dataset.json", summary)
    (out / "report.md").write_text(report(summary))
    return summary


def audit_summary(ws: Workspace) -> dict:
    graded = correct = food = boxes = missed = images = 0
    for path in sorted((ws.root / "audit").glob("*.json")):
        result = (read_json(path) or {}).get("result")
        if not result:
            continue
        images += 1
        missed += result.get("missed_items", 0)
        for item in result["items"]:
            graded += 1
            food += item["is_food"]
            correct += item["is_food"] and item["class_correct"]
            boxes += item["box_ok"]
    if not graded:
        return {"images": images}
    precision = correct / graded
    return {"images": images, "items": graded, "precision": precision, "precision_95ci": wilson(correct, graded),
            "is_food_rate": food / graded, "box_ok_rate": boxes / graded,
            "recall_estimate": correct / max(1, correct + missed), "missed_items": missed}


def cost(ws: Workspace) -> dict:
    totals: Counter = Counter()
    for entry in ws.usage():
        price_in, price_out = PRICES.get(entry["model"], (0.0, 0.0))
        dollars = (entry["input"] * price_in + entry["output"] * price_out) / 1e6
        totals[entry["model"]] += dollars * (0.5 if entry.get("batch") else 1)
    return {model: round(v, 4) for model, v in totals.items()} | {"total": round(sum(totals.values()), 4)}


def report(s: dict) -> str:
    stats, audit = s["stats"], s["audit"]
    lines = [
        "# Auto-labeled food dataset", "",
        f"Taxonomy `{s['taxonomy_version']}`, prompts `{s['label_prompt']}` / `{s['verify_prompt']}`, "
        f"label model `{s['models']['label']}`, verifier `{s['models']['verify']}`, auditor `{s['models']['audit']}`.", "",
        "## Images", "",
        f"- Processed: {stats.get('images_processed', 0)} (not yet through every stage: {stats.get('images_unprocessed', 0)})",
        f"- Complete (used for detection/segmentation): {stats.get('images_complete', 0)}",
        f"- Incomplete (some item undecided; crops only): {stats.get('images_incomplete', 0)}",
        f"- Accepted items: {stats.get('items_accepted', 0)}", "",
        "## Splits (by group)", "", "| Split | Images | Annotations |", "| --- | --- | --- |",
        *[f"| {k} | {v['images']} | {v['annotations']} |" for k, v in s["splits"].items()], "",
        "## Candidate decisions", "", "| Decision | Count |", "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in sorted(s["decisions"].items(), key=lambda kv: -kv[1])], "",
        "## Pass agreement", "", "| Agreement | Count |", "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in sorted(s["agreement"].items(), key=lambda kv: -kv[1])], "",
        "## Audit (stronger model on a random sample)", "",
    ]
    if audit.get("items"):
        lo, hi = audit["precision_95ci"]
        lines += [f"- Images audited: {audit['images']}, items graded: {audit['items']}",
                  f"- Label precision: {audit['precision']:.1%} (95% CI {lo:.1%}–{hi:.1%})",
                  f"- Box acceptable: {audit['box_ok_rate']:.1%}",
                  f"- Recall estimate (missed items): {audit['recall_estimate']:.1%} ({audit['missed_items']} missed)"]
    else:
        lines.append("- Not run yet (`python -m autolabel audit`).")
    lines += ["", "## Classes", "", "| Class | Items |", "| --- | --- |",
              *[f"| {k} | {v} |" for k, v in sorted(s["classes"].items(), key=lambda kv: -kv[1])], "",
              "## Cost (USD, from logged token usage)", "", *[f"- {k}: ${v:.4f}" for k, v in s["cost_usd"].items()], ""]
    return "\n".join(lines)

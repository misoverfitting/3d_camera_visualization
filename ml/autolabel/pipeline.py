"""Stage orchestration. Each stage finds the work still missing in the workspace, does
it, and saves results immediately, so stages are idempotent and resumable."""
from __future__ import annotations

import random
import threading
from typing import Callable, Iterator

import numpy as np
from PIL import Image

from . import audit, label, verify
from .claude import Outcome, collect_batches, run_direct, submit_batches
from .config import AUDIT_PROMPT_VERSION, LABEL_PROMPT_VERSION, VERIFY_PROMPT_VERSION
from .segment import Segmenter, segment_candidates
from .store import ImageRecord, Workspace, read_json, write_json
from .taxonomy import Taxonomy

MAX_ATTEMPTS = 3
CHUNK = 400  # Requests built (images encoded) at a time, to bound memory.
_lock = threading.Lock()


def label_version(taxonomy: Taxonomy) -> str:
    return f"{LABEL_PROMPT_VERSION}/{taxonomy.version}"


def _needs(saved: dict | None, version: str) -> bool:
    if saved is None or saved.get("version") != version:
        return True
    return "error" in saved and saved["error"] != "refusal" and saved.get("attempts", 1) < MAX_ATTEMPTS


def _pending_ids(ws: Workspace) -> set[str]:
    return {i for info in ws.batches().values() for i in info.get("ids", [])}


def _dispatch(ws: Workspace, stage: str, requests: Iterator[tuple[str, Callable[[], dict]]],
              handler: Callable[[str, Outcome], None], batch: bool, workers: int, limit: int | None) -> int:
    """Builds requests lazily in chunks and runs or submits each chunk."""
    pending, count, chunk = _pending_ids(ws), 0, {}

    def flush():
        if not chunk:
            return
        if batch:
            for batch_id in submit_batches(ws, chunk, stage):
                batches = ws.batches()
                batches[batch_id]["ids"] = list(chunk)
                ws.save_batches(batches)
        else:
            run_direct(ws, chunk, handler, workers)
        chunk.clear()

    for custom_id, build in requests:
        if custom_id in pending:
            continue
        if limit is not None and count >= limit:
            break
        chunk[custom_id] = build()
        count += 1
        if len(chunk) >= CHUNK:
            flush()
    flush()
    return count


# Label ------------------------------------------------------------------------

def label_handler(ws: Workspace, taxonomy: Taxonomy) -> Callable[[str, Outcome], None]:
    def handle(custom_id: str, outcome: Outcome) -> None:
        pass_index, image_id = int(custom_id[1]), custom_id[3:]
        path = ws.label_path(image_id, pass_index)
        version = label_version(taxonomy)
        if outcome.text:
            try:
                write_json(path, {"version": version, "labels": label.parse(outcome.text, taxonomy, mirrored=pass_index == 1)})
                return
            except (ValueError, KeyError, TypeError) as exc:
                outcome = Outcome(None, outcome.stop_reason, f"unparseable: {exc}")
        previous = read_json(path, {})
        attempts = previous.get("attempts", 0) + 1 if previous.get("version") == version else 1
        write_json(path, {"version": version, "error": outcome.error, "attempts": attempts})
    return handle


def run_label(ws: Workspace, taxonomy: Taxonomy, passes: int, batch: bool, workers: int, limit: int | None) -> int:
    version = label_version(taxonomy)

    def requests():
        for record in ws.records():
            for p in range(passes):
                if _needs(read_json(ws.label_path(record.id, p)), version):
                    yield f"L{p}_{record.id}", lambda r=record, p=p: label.request(Image.open(ws.image_path(r.id)), taxonomy, p)

    return _dispatch(ws, "label", requests(), label_handler(ws, taxonomy), batch, workers, limit)


# Segment ----------------------------------------------------------------------

def _passes(ws: Workspace, record: ImageRecord, passes: int, version: str) -> list[dict] | None:
    saved = [read_json(ws.label_path(record.id, p)) for p in range(passes)]
    if any(s is None or s.get("version") != version or "labels" not in s for s in saved):
        return None
    return [s["labels"] for s in saved]


def run_segment(ws: Workspace, taxonomy: Taxonomy, passes: int, segmenter: Segmenter | None) -> int:
    version, done = f"{label_version(taxonomy)}/{segmenter.name if segmenter else 'boxes'}", 0
    for record in ws.records():
        if (read_json(ws.candidates_path(record.id)) or {}).get("version") == version:
            continue
        labels = _passes(ws, record, passes, label_version(taxonomy))
        if labels is None:
            continue
        candidates = label.consensus(labels)
        qualities = [l["image_quality"] for l in labels]
        if segmenter and candidates:
            candidates, masks = segment_candidates(Image.open(ws.image_path(record.id)).convert("RGB"), candidates, segmenter)
            if masks is not None:
                np.savez_compressed(ws.masks_path(record.id), masks=masks)
        write_json(ws.candidates_path(record.id), {"version": version, "image_quality": qualities,
                                                   "segmenter": segmenter.name if segmenter else None, "items": candidates})
        done += 1
    return done


def load_masks(ws: Workspace, image_id: str) -> np.ndarray | None:
    path = ws.masks_path(image_id)
    return np.load(path)["masks"] if path.exists() else None


# Verify -----------------------------------------------------------------------

def verify_handler(ws: Workspace, taxonomy: Taxonomy) -> Callable[[str, Outcome], None]:
    def handle(custom_id: str, outcome: Outcome) -> None:
        index_text, image_id = custom_id[1:].split("_", 1)
        with _lock:
            path = ws.verify_path(image_id)
            saved = read_json(path, {})
            if saved.get("version") != VERIFY_PROMPT_VERSION:
                saved = {"version": VERIFY_PROMPT_VERSION, "items": {}}
            entry = saved["items"].get(index_text, {})
            if outcome.text:
                try:
                    saved["items"][index_text] = {"verdict": verify.parse(outcome.text, taxonomy)}
                except (ValueError, KeyError, TypeError) as exc:
                    saved["items"][index_text] = {"error": f"unparseable: {exc}", "attempts": entry.get("attempts", 0) + 1}
            else:
                saved["items"][index_text] = {"error": outcome.error, "attempts": entry.get("attempts", 0) + 1}
            write_json(path, saved)
    return handle


def run_verify(ws: Workspace, taxonomy: Taxonomy, batch: bool, workers: int, limit: int | None) -> int:
    def requests():
        for record in ws.records():
            candidates = read_json(ws.candidates_path(record.id))
            if not candidates:
                continue
            saved = read_json(ws.verify_path(record.id), {})
            done = saved.get("items", {}) if saved.get("version") == VERIFY_PROMPT_VERSION else {}
            masks = None
            for k, candidate in enumerate(candidates["items"]):
                entry = done.get(str(k))
                finished = entry and ("verdict" in entry or entry.get("error") == "refusal" or entry.get("attempts", 0) >= MAX_ATTEMPTS)
                if candidate.get("rejected") or finished:
                    continue
                if masks is None:
                    masks = load_masks(ws, record.id)
                mask = masks[candidate["mask_index"]] if masks is not None and "mask_index" in candidate else None
                yield f"V{k}_{record.id}", lambda r=record, c=candidate, m=mask: verify.request(
                    verify.crop(Image.open(ws.image_path(r.id)), c["box"], m), c, taxonomy)

    return _dispatch(ws, "verify", requests(), verify_handler(ws, taxonomy), batch, workers, limit)


# Decide -------------------------------------------------------------------------

def decisions(ws: Workspace, record: ImageRecord) -> list[tuple[dict, str | None, str]] | None:
    """(candidate, final class or None, reason) per candidate, or None if unprocessed."""
    candidates = read_json(ws.candidates_path(record.id))
    if candidates is None:
        return None
    saved = read_json(ws.verify_path(record.id), {})
    verdicts = saved.get("items", {}) if saved.get("version") == VERIFY_PROMPT_VERSION else {}
    out = []
    for k, candidate in enumerate(candidates["items"]):
        final, reason = verify.decide(candidate, verdicts.get(str(k), {}).get("verdict"))
        out.append((candidate, final, reason))
    return out


# Audit ------------------------------------------------------------------------

def run_audit(ws: Workspace, fraction: float, maximum: int, seed: int = 7) -> int:
    eligible = []
    for record in ws.records():
        decided = decisions(ws, record)
        if decided is None or (read_json(ws.audit_path(record.id)) or {}).get("version") == AUDIT_PROMPT_VERSION:
            continue
        accepted = [{**c, "food_class": final} for c, final, _ in decided if final]
        if accepted:
            eligible.append((record, accepted))
    random.Random(seed).shuffle(eligible)
    sample = eligible[: min(maximum, max(1, round(fraction * len(eligible))))] if eligible else []
    for record, accepted in sample:
        result, usage = audit.run(audit.request(Image.open(ws.image_path(record.id)), accepted))
        if usage is not None:
            ws.log_usage({"id": f"A_{record.id}", "model": audit.AUDIT_MODEL, "batch": False,
                          "input": usage.input_tokens, "output": usage.output_tokens})
        write_json(ws.audit_path(record.id), {"version": AUDIT_PROMPT_VERSION, "items": accepted, "result": result})
    return len(sample)


# Batches ------------------------------------------------------------------------

def collect(ws: Workspace, taxonomy: Taxonomy, wait: bool) -> int:
    return collect_batches(ws, {"label": label_handler(ws, taxonomy), "verify": verify_handler(ws, taxonomy)}, wait)

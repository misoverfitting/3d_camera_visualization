"""Runs Messages API requests either directly (small jobs, quick feedback) or as
Message Batches (half price, usually done within an hour). Both paths take the same
request params and produce the same Outcome, so stages don't care which ran."""
from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from PIL import Image

from .config import MAX_BATCH_BYTES
from .store import Workspace


@dataclass
class Outcome:
    text: str | None          # The JSON text from structured output, when it succeeded.
    stop_reason: str | None
    error: str | None = None


_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def image_block(image: Image.Image, quality: int = 88) -> dict:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")}}


def json_format(schema: dict) -> dict:
    return {"type": "json_schema", "schema": schema}


def _outcome(message) -> Outcome:
    # Content can start with thinking blocks; take the text block.
    text = next((block.text for block in message.content if block.type == "text"), None)
    if message.stop_reason in ("refusal", "max_tokens"):
        return Outcome(None, message.stop_reason, message.stop_reason)
    return Outcome(text, message.stop_reason, None if text else "no text in response")


def _log(ws: Workspace, custom_id: str, model: str, usage, batch: bool) -> None:
    if usage is not None:
        ws.log_usage({"id": custom_id, "model": model, "batch": batch, "input": usage.input_tokens,
                      "output": usage.output_tokens, "time": time.time()})


def run_direct(ws: Workspace, requests: dict[str, dict], on_result: Callable[[str, Outcome], None],
               workers: int = 8) -> None:
    """Runs requests concurrently; the SDK retries 429/5xx with backoff."""
    def one(item: tuple[str, dict]) -> None:
        custom_id, params = item
        try:
            message = client().messages.create(**params)
            _log(ws, custom_id, params["model"], message.usage, batch=False)
            outcome = _outcome(message)
        except anthropic.BadRequestError as exc:
            outcome = Outcome(None, None, f"invalid request: {exc.message}")
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            outcome = Outcome(None, None, f"api error: {exc}")
        on_result(custom_id, outcome)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(one, requests.items()):
            pass


def submit_batches(ws: Workspace, requests: dict[str, dict], stage: str) -> list[str]:
    """Submits requests in size-limited chunks and records the batch IDs for `collect`."""
    chunks: list[list[Request]] = [[]]
    size = 0
    for custom_id, params in requests.items():
        request_bytes = len(json.dumps(params))
        if chunks[-1] and (size + request_bytes > MAX_BATCH_BYTES or len(chunks[-1]) >= 100_000):
            chunks.append([])
            size = 0
        chunks[-1].append(Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params)))
        size += request_bytes
    batches = ws.batches()
    ids = []
    for chunk in chunks:
        if not chunk:
            continue
        batch = client().messages.batches.create(requests=chunk)
        batches[batch.id] = {"stage": stage, "requests": len(chunk), "created": time.time()}
        ws.save_batches(batches)
        ids.append(batch.id)
        print(f"submitted {stage} batch {batch.id} ({len(chunk)} requests)")
    return ids


def collect_batches(ws: Workspace, handlers: dict[str, Callable[[str, Outcome], None]], wait: bool,
                    poll_s: float = 60) -> int:
    """Saves results of every ended batch; with `wait`, polls until none are pending.
    Returns how many batches are still processing."""
    while True:
        batches = ws.batches()
        pending = 0
        for batch_id, info in list(batches.items()):
            batch = client().messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                pending += 1
                counts = batch.request_counts
                print(f"{info['stage']} batch {batch_id}: {counts.processing} processing, {counts.succeeded} done")
                continue
            handle = handlers[info["stage"]]
            for result in client().messages.batches.results(batch_id):
                match result.result.type:
                    case "succeeded":
                        message = result.result.message
                        _log(ws, result.custom_id, message.model, message.usage, batch=True)
                        handle(result.custom_id, _outcome(message))
                    case "errored":
                        handle(result.custom_id, Outcome(None, None, f"batch error: {result.result.error.type}"))
                    case other:  # canceled / expired: leave unsaved so the next run resubmits.
                        print(f"{result.custom_id}: {other}, will be resubmitted")
            del batches[batch_id]
            ws.save_batches(batches)
            print(f"collected {info['stage']} batch {batch_id}")
        if not pending or not wait:
            return pending
        time.sleep(poll_s)

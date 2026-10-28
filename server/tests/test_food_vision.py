"""Food vision proxy, with the Anthropic client stubbed out (no network, no key)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import food_vision
from app.food_vision import Item, Point, Scene

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def api(monkeypatch):
    def install(response):
        messages = FakeMessages(response)
        monkeypatch.setattr(food_vision, "_client", SimpleNamespace(beta=SimpleNamespace(messages=messages)))
        app = FastAPI()
        app.include_router(food_vision.router)
        return TestClient(app), messages
    return install


def scene(*items):
    return SimpleNamespace(stop_reason="end_turn", parsed_output=Scene(items=list(items)), model="claude-opus-5-5")


def test_identify_returns_clamped_items_with_two_or_three_points(api):
    chicken = Item(label=" grilled chicken ", category="protein", confidence=0.9, box=[0.6, 0.2, 0.3, 0.5],
                   points=[Point(x=0.4, y=0.3), Point(x=0.45, y=0.35), Point(x=0.5, y=0.4), Point(x=1.4, y=-0.2)])
    rice = Item(label="rice", category="starch", confidence=1.3, box=[0.1, 0.6, 0.3, 0.8], points=[Point(x=0.2, y=0.7)])
    client, messages = api(scene(chicken, rice))
    response = client.post("/api/food/identify", files={"image": ("plate.jpg", JPEG, "image/jpeg")})
    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["label"] == "grilled chicken"
    assert items[0]["box"] == [0.3, 0.2, 0.6, 0.5]
    assert len(items[0]["points"]) == 3
    assert items[1]["confidence"] == 1.0
    assert items[1]["points"] == [[0.2, 0.7], [0.2, 0.7]]
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


def test_rejects_non_jpeg_and_refusals(api):
    client, _ = api(SimpleNamespace(stop_reason="refusal", parsed_output=None, model="x"))
    assert client.post("/api/food/identify", files={"image": ("a.png", b"\x89PNG", "image/png")}).status_code == 400
    assert client.post("/api/food/identify", files={"image": ("a.jpg", JPEG, "image/jpeg")}).status_code == 422


def test_token_is_enforced_when_configured(api, monkeypatch):
    monkeypatch.setattr(food_vision, "API_TOKEN", "secret")
    client, _ = api(scene())
    files = {"image": ("a.jpg", JPEG, "image/jpeg")}
    assert client.post("/api/food/identify", files=files).status_code == 401
    assert client.post("/api/food/identify", files=files, headers={"X-Tare-Token": "secret"}).json() == {"items": [], "model": "claude-opus-5-5"}

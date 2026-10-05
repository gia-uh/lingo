"""Integration tests for the Playground Starlette application."""

import sys
from types import ModuleType

from starlette.testclient import TestClient

from lingo import Lingo
from lingo.mock import MockLLM
from lingo.playground.app import create_playground_app
from lingo.playground.loader import load_target


def _factory_target(monkeypatch, name: str, responses: list[str]):
    module = ModuleType(name)
    module.create_app = lambda: Lingo(
        "Visual assistant",
        description="A browser test assistant.",
        llm=MockLLM(list(responses)),
    )
    monkeypatch.setitem(sys.modules, name, module)
    return load_target(f"{name}:create_app")


def _receive_until(websocket, event_type: str) -> list[dict]:
    events = []
    while not events or events[-1]["type"] != event_type:
        events.append(websocket.receive_json())
    return events


def test_health_and_static_shell(monkeypatch):
    app = create_playground_app(
        _factory_target(monkeypatch, "playground_http_target", ["Hello"])
    )

    with TestClient(app) as client:
        health = client.get("/health")
        index = client.get("/")
        styles = client.get("/static/styles.css")
        script = client.get("/static/app.js")

    assert health.json() == {"status": "ok"}
    assert index.status_code == 200
    assert "Lingo Playground" in index.text
    assert index.headers["content-security-policy"].startswith("default-src 'self'")
    assert styles.status_code == 200
    assert "--accent" in styles.text
    assert ".reasoning-status" in styles.text
    assert script.status_code == 200
    assert "new WebSocket" in script.text
    assert "Reasoning…" in script.text


def test_websocket_runs_lingo_turn_and_streams_events(monkeypatch):
    app = create_playground_app(
        _factory_target(monkeypatch, "playground_websocket_target", ["Hello there"])
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            ready = websocket.receive_json()
            websocket.send_json({"type": "user_message", "content": "Hi"})
            events = _receive_until(websocket, "turn_completed")

    assert ready["type"] == "ready"
    assert ready["app"]["name"] == "Visual assistant"
    assert ready["mode"] == "factory"
    assert [event["type"] for event in events] == [
        "turn_started",
        "token",
        "token",
        "message",
        "turn_completed",
    ]


def test_websocket_rejects_invalid_client_event_without_disconnecting(monkeypatch):
    app = create_playground_app(
        _factory_target(monkeypatch, "playground_invalid_event_target", ["Hello"])
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "unknown"})
            error = websocket.receive_json()
            websocket.send_json({"type": "user_message", "content": "Hi"})
            events = _receive_until(websocket, "turn_completed")

    assert error == {
        "type": "error",
        "code": "invalid_event",
        "message": "Unsupported Playground event.",
        "recoverable": True,
    }
    assert events[-1]["type"] == "turn_completed"


def test_websocket_reset_recreates_factory_application(monkeypatch):
    app = create_playground_app(
        _factory_target(monkeypatch, "playground_ws_reset_target", ["Hello"])
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "reset"})
            reset = websocket.receive_json()

    assert reset["type"] == "reset_completed"
    assert reset["mode"] == "factory"


def test_websocket_rejects_blank_and_oversized_messages(monkeypatch):
    app = create_playground_app(
        _factory_target(monkeypatch, "playground_validation_target", ["Hello"])
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "user_message", "content": "  "})
            blank = websocket.receive_json()
            websocket.send_json({"type": "user_message", "content": "x" * 20_001})
            oversized = websocket.receive_json()

    assert blank["code"] == "invalid_message"
    assert oversized["code"] == "message_too_large"

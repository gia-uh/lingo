"""Tests for Playground session isolation, callbacks, and lifecycle."""

from collections.abc import Callable
import sys
from types import ModuleType

import pytest

from lingo import Context, Engine, Lingo
from lingo.mock import MockLLM
from lingo.playground.events import PlaygroundEvent, event_payload
from lingo.playground.loader import load_target
from lingo.playground.session import SessionManager, SessionUnavailableError


class EventCollector:
    """Async event sink used by session tests."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def __call__(self, event: PlaygroundEvent) -> None:
        self.events.append(event_payload(event))


def _install_factory(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    factory: Callable[[], Lingo],
) -> str:
    module = ModuleType(name)
    module.create_app = factory
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:create_app"


@pytest.mark.asyncio
async def test_factory_sessions_receive_isolated_lingo_instances(monkeypatch):
    reference = _install_factory(
        monkeypatch,
        "playground_isolated_sessions",
        lambda: Lingo(llm=MockLLM(["Hello"])),
    )
    manager = SessionManager(load_target(reference))

    first = await manager.open(EventCollector())
    second = await manager.open(EventCollector())

    assert first.app is not second.app
    assert first.app.messages is not second.app.messages
    await manager.close(first.id)
    await manager.close(second.id)


@pytest.mark.asyncio
async def test_singleton_target_rejects_second_live_session(monkeypatch):
    module = ModuleType("playground_singleton_sessions")
    module.app = Lingo(llm=MockLLM(["Hello"]))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    manager = SessionManager(load_target(f"{module.__name__}:app"))
    first = await manager.open(EventCollector())

    with pytest.raises(SessionUnavailableError, match="one browser session"):
        await manager.open(EventCollector())

    await manager.close(first.id)


@pytest.mark.asyncio
async def test_session_streams_events_and_preserves_existing_callbacks(monkeypatch):
    original_tokens: list[str] = []

    def create_app() -> Lingo:
        return Lingo(
            "Callback app",
            llm=MockLLM(["Hello there"], on_token=original_tokens.append),
        )

    reference = _install_factory(monkeypatch, "playground_callback_session", create_app)
    collector = EventCollector()
    manager = SessionManager(load_target(reference))
    session = await manager.open(collector)

    await session.submit("Hi")

    event_types = [event["type"] for event in collector.events]
    assert event_types[0] == "ready"
    assert "turn_started" in event_types
    assert "token" in event_types
    assert "message" in event_types
    assert event_types[-1] == "turn_completed"
    assert "".join(original_tokens) == "Hello there "
    app = session.app
    original_callback = session._original_callbacks["_on_token"]
    await manager.close(session.id)
    assert app.llm._on_token is original_callback


@pytest.mark.asyncio
async def test_reset_replaces_factory_backed_app(monkeypatch):
    reference = _install_factory(
        monkeypatch,
        "playground_reset_session",
        lambda: Lingo(llm=MockLLM(["Hello"])),
    )
    collector = EventCollector()
    manager = SessionManager(load_target(reference))
    session = await manager.open(collector)
    original = session.app

    await session.reset()

    assert session.app is not original
    assert collector.events[-1]["type"] == "reset_completed"
    await manager.close(session.id)


@pytest.mark.asyncio
async def test_execution_failure_becomes_safe_error_event(monkeypatch):
    class FailingLLM(MockLLM):
        async def chat(self, messages, **kwargs):
            raise RuntimeError("secret provider detail")

    reference = _install_factory(
        monkeypatch,
        "playground_failing_session",
        lambda: Lingo(llm=FailingLLM()),
    )
    collector = EventCollector()
    manager = SessionManager(load_target(reference))
    session = await manager.open(collector)

    await session.submit("Fail")

    error = collector.events[-1]
    assert error == {
        "type": "error",
        "code": "execution_error",
        "message": "The Lingo application could not complete this turn.",
        "recoverable": True,
    }
    await manager.close(session.id)


@pytest.mark.asyncio
async def test_session_preserves_ask_pause_and_resume(monkeypatch):
    def create_app() -> Lingo:
        app = Lingo(llm=MockLLM(["What is your name?", "Nice to meet you."]))

        @app.skill
        async def onboarding(context: Context, engine: Engine):
            name = await engine.ask(context, "Ask for the user's name.")
            response = await engine.reply(context, f"The user's name is {name}.")
            context.append(response)

        return app

    reference = _install_factory(monkeypatch, "playground_ask_session", create_app)
    collector = EventCollector()
    manager = SessionManager(load_target(reference))
    session = await manager.open(collector)

    await session.submit("Start")
    assert session.app._runner_task is not None
    assert not session.app._runner_task.done()

    await session.submit("Ada")

    assert session.app._runner_task is None
    messages = [
        event["message"]["content"]
        for event in collector.events
        if event["type"] == "message"
    ]
    assert messages == ["What is your name?", "Nice to meet you."]
    await manager.close(session.id)

"""Tests for the typed JSON protocol used by Lingo Playground."""

from lingo.llm import Message, ToolCall, Usage
from lingo.playground.events import (
    AppMetadata,
    ErrorEvent,
    ReadyEvent,
    TokenEvent,
    event_payload,
    message_payload,
)


def test_message_payload_preserves_assistant_metadata():
    message = Message.assistant(
        "Checking the weather.",
        thinking="I should use the weather tool.",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="get_weather",
                arguments={"city": "Havana"},
            )
        ],
        stop_reason="tool_calls",
        usage=Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
    )

    assert message_payload(message) == {
        "role": "assistant",
        "content": "Checking the weather.",
        "tool_calls": [
            {
                "id": "call-1",
                "name": "get_weather",
                "arguments": {"city": "Havana"},
            }
        ],
        "thinking": "I should use the weather tool.",
        "stop_reason": "tool_calls",
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }


def test_message_payload_omits_absent_optional_metadata():
    assert message_payload(Message.user("Hello")) == {
        "role": "user",
        "content": "Hello",
    }


def test_event_payload_uses_discriminator_and_omits_none():
    event = TokenEvent(text="Hello")

    assert event_payload(event) == {"type": "token", "text": "Hello"}


def test_ready_event_describes_application_and_session_mode():
    event = ReadyEvent(
        session_id="session-1",
        app=AppMetadata(name="Assistant", description="A helpful assistant."),
        mode="factory",
    )

    assert event_payload(event) == {
        "type": "ready",
        "session_id": "session-1",
        "app": {
            "name": "Assistant",
            "description": "A helpful assistant.",
        },
        "mode": "factory",
    }


def test_error_event_exposes_only_safe_protocol_fields():
    event = ErrorEvent(
        code="model_error",
        message="The model request failed.",
        recoverable=True,
    )

    assert event_payload(event) == {
        "type": "error",
        "code": "model_error",
        "message": "The model request failed.",
        "recoverable": True,
    }

"""Typed wire events for communication with the Playground browser client."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel

from lingo.llm import Content, Message


SessionMode: TypeAlias = Literal["factory", "singleton"]


class AppMetadata(BaseModel):
    """Public application details displayed by the Playground."""

    name: str
    description: str


class ReadyEvent(BaseModel):
    """Initial event sent after a browser session is established."""

    type: Literal["ready"] = "ready"
    session_id: str
    app: AppMetadata
    mode: SessionMode


class TurnStartedEvent(BaseModel):
    """Signal that a user message is being processed."""

    type: Literal["turn_started"] = "turn_started"


class TokenEvent(BaseModel):
    """A streamed assistant text fragment."""

    type: Literal["token"] = "token"
    text: str


class ReasoningTokenEvent(BaseModel):
    """A streamed model reasoning fragment."""

    type: Literal["reasoning_token"] = "reasoning_token"
    text: str


class ToolCallStartedEvent(BaseModel):
    """Signal that the model started emitting a tool call."""

    type: Literal["tool_call_started"] = "tool_call_started"
    id: str
    name: str


class ToolCallDeltaEvent(BaseModel):
    """The cumulative JSON argument text emitted for a tool call."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    id: str
    arguments: str


class ToolCallCompletedEvent(BaseModel):
    """A fully assembled tool call emitted by the model."""

    type: Literal["tool_call_completed"] = "tool_call_completed"
    id: str
    arguments: dict[str, Any]


class MessageEvent(BaseModel):
    """A completed model message observed through the LLM callback."""

    type: Literal["message"] = "message"
    message: dict[str, Any]


class StructuredOutputEvent(BaseModel):
    """A structured Pydantic result created by the model."""

    type: Literal["structured_output"] = "structured_output"
    value: Any


class TurnCompletedEvent(BaseModel):
    """Signal that `Lingo.chat` returned for the submitted user message."""

    type: Literal["turn_completed"] = "turn_completed"
    message: dict[str, Any]


class ResetCompletedEvent(BaseModel):
    """Signal that the current browser session was reset."""

    type: Literal["reset_completed"] = "reset_completed"
    app: AppMetadata
    mode: SessionMode


class ErrorEvent(BaseModel):
    """A safe error description suitable for display in the browser."""

    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool


PlaygroundEvent: TypeAlias = (
    ReadyEvent
    | TurnStartedEvent
    | TokenEvent
    | ReasoningTokenEvent
    | ToolCallStartedEvent
    | ToolCallDeltaEvent
    | ToolCallCompletedEvent
    | MessageEvent
    | StructuredOutputEvent
    | TurnCompletedEvent
    | ResetCompletedEvent
    | ErrorEvent
)


def message_payload(message: Message) -> dict[str, Any]:
    """Serialize a Lingo message without losing Playground metadata."""
    content = message.content
    serialized_content = (
        content.model_dump(mode="json") if isinstance(content, Content) else content
    )
    payload: dict[str, Any] = {
        "role": message.role,
        "content": serialized_content,
    }
    if message.tool_calls is not None:
        payload["tool_calls"] = [
            call.model_dump(mode="json") for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.thinking is not None:
        payload["thinking"] = message.thinking
    if message.stop_reason is not None:
        payload["stop_reason"] = message.stop_reason
    if message.usage is not None:
        payload["usage"] = message.usage.model_dump(mode="json")
    return payload


def event_payload(event: PlaygroundEvent) -> dict[str, Any]:
    """Return a compact JSON-compatible event dictionary."""
    return event.model_dump(mode="json", exclude_none=True)

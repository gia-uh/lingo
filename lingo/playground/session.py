"""Session isolation and LLM event bridging for Lingo Playground."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import inspect
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .events import (
    AppMetadata,
    ErrorEvent,
    MessageEvent,
    PlaygroundEvent,
    ReadyEvent,
    ReasoningTokenEvent,
    ResetCompletedEvent,
    StructuredOutputEvent,
    TokenEvent,
    ToolCallCompletedEvent,
    ToolCallDeltaEvent,
    ToolCallStartedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
    message_payload,
)
from .loader import LoadedTarget


logger = logging.getLogger(__name__)
EventSink = Callable[[PlaygroundEvent], Awaitable[None]]


class SessionUnavailableError(RuntimeError):
    """Raised when a target cannot accept another browser session."""


async def _invoke_callback(callback: Callable | None, *args: Any) -> None:
    """Invoke an existing sync or async application callback."""
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


class PlaygroundSession:
    """Own one Lingo runtime and stream its activity to a browser client."""

    def __init__(self, target: LoadedTarget, sink: EventSink) -> None:
        self.id = uuid4().hex
        self.target = target
        self.sink = sink
        self.app = target.create()
        self._lock = asyncio.Lock()
        self._original_callbacks: dict[str, Callable | None] = {}
        self._bridge_callbacks: dict[str, Callable] = {}
        self._closed = False
        self._install_bridge()

    @property
    def metadata(self) -> AppMetadata:
        """Return application details safe to expose to the browser."""
        return AppMetadata(name=self.app.name, description=self.app.description)

    async def start(self) -> None:
        """Announce a newly opened browser session."""
        await self.sink(
            ReadyEvent(
                session_id=self.id,
                app=self.metadata,
                mode=self.target.mode,
            )
        )

    async def submit(self, content: str) -> None:
        """Process one user message and emit its complete event lifecycle."""
        async with self._lock:
            if self._closed:
                raise SessionUnavailableError("This Playground session is closed.")

            await self.sink(TurnStartedEvent())
            try:
                result = await self.app.chat(content)
            except Exception:
                logger.exception("Lingo Playground session %s failed", self.id)
                await self.sink(
                    ErrorEvent(
                        code="execution_error",
                        message="The Lingo application could not complete this turn.",
                        recoverable=True,
                    )
                )
                return

            await self.sink(TurnCompletedEvent(message=message_payload(result)))

    async def reset(self) -> None:
        """Reset this browser session without affecting any other session."""
        async with self._lock:
            if self._closed:
                raise SessionUnavailableError("This Playground session is closed.")

            await self._release_runtime()
            if self.target.mode == "factory":
                self.app = self.target.create()
            else:
                self.app.messages.clear()
                self._clear_runtime_state()
            self._install_bridge()
            await self.sink(
                ResetCompletedEvent(app=self.metadata, mode=self.target.mode)
            )

    async def close(self) -> None:
        """Release callbacks and any suspended flow owned by the session."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            await self._release_runtime()

    async def _release_runtime(self) -> None:
        """Restore application callbacks and stop a suspended flow."""
        self._restore_bridge()
        task = self.app._runner_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._clear_runtime_state()

    def _clear_runtime_state(self) -> None:
        """Clear private runner references after cancellation or reset."""
        self.app._runner_task = None
        self.app._active_engine = None
        self.app._active_context = None
        self.app._context_sync_offset = 0
        self.app._context_synced_count = 0

    def _install_bridge(self) -> None:
        """Compose Playground event callbacks with existing LLM callbacks."""
        llm = self.app.llm
        self._original_callbacks = {
            "_on_token": llm._on_token,
            "_on_reasoning_token": llm._on_reasoning_token,
            "_on_create": llm._on_create,
            "_on_message": llm._on_message,
            "_on_toolcall_start": llm._on_toolcall_start,
            "_on_toolcall_delta": llm._on_toolcall_delta,
            "_on_toolcall_end": llm._on_toolcall_end,
        }

        async def on_token(text: str) -> None:
            await _invoke_callback(self._original_callbacks["_on_token"], text)
            await self.sink(TokenEvent(text=text))

        async def on_reasoning_token(text: str) -> None:
            await _invoke_callback(
                self._original_callbacks["_on_reasoning_token"], text
            )
            await self.sink(ReasoningTokenEvent(text=text))

        async def on_create(value: BaseModel) -> None:
            await _invoke_callback(self._original_callbacks["_on_create"], value)
            await self.sink(StructuredOutputEvent(value=value.model_dump(mode="json")))

        async def on_message(message) -> None:
            await _invoke_callback(self._original_callbacks["_on_message"], message)
            await self.sink(MessageEvent(message=message_payload(message)))

        async def on_toolcall_start(call_id: str, name: str) -> None:
            await _invoke_callback(
                self._original_callbacks["_on_toolcall_start"], call_id, name
            )
            await self.sink(ToolCallStartedEvent(id=call_id, name=name))

        async def on_toolcall_delta(call_id: str, arguments: str) -> None:
            await _invoke_callback(
                self._original_callbacks["_on_toolcall_delta"], call_id, arguments
            )
            await self.sink(ToolCallDeltaEvent(id=call_id, arguments=arguments))

        async def on_toolcall_end(call_id: str, arguments: dict) -> None:
            await _invoke_callback(
                self._original_callbacks["_on_toolcall_end"], call_id, arguments
            )
            await self.sink(ToolCallCompletedEvent(id=call_id, arguments=arguments))

        self._bridge_callbacks = {
            "_on_token": on_token,
            "_on_reasoning_token": on_reasoning_token,
            "_on_create": on_create,
            "_on_message": on_message,
            "_on_toolcall_start": on_toolcall_start,
            "_on_toolcall_delta": on_toolcall_delta,
            "_on_toolcall_end": on_toolcall_end,
        }
        for name, callback in self._bridge_callbacks.items():
            setattr(llm, name, callback)

    def _restore_bridge(self) -> None:
        """Restore callbacks only when the Playground still owns each slot."""
        llm = self.app.llm
        for name, callback in self._bridge_callbacks.items():
            if getattr(llm, name) is callback:
                setattr(llm, name, self._original_callbacks[name])
        self._bridge_callbacks.clear()
        self._original_callbacks.clear()


class SessionManager:
    """Create and track browser sessions for one loaded target."""

    def __init__(self, target: LoadedTarget) -> None:
        self.target = target
        self._sessions: dict[str, PlaygroundSession] = {}
        self._lock = asyncio.Lock()

    async def open(self, sink: EventSink) -> PlaygroundSession:
        """Open and announce a browser session."""
        async with self._lock:
            if self.target.mode == "singleton" and self._sessions:
                raise SessionUnavailableError(
                    "A singleton Lingo target supports only one browser session. "
                    "Expose a zero-argument factory for isolated sessions."
                )
            session = PlaygroundSession(self.target, sink)
            self._sessions[session.id] = session

        try:
            await session.start()
        except Exception:
            await self.close(session.id)
            raise
        return session

    async def close(self, session_id: str) -> None:
        """Remove and close a session if it is still registered."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close()

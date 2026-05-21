"""Unit tests for the reasoning passthrough on `LLM.chat()`.

We don't hit a real provider — we patch `client.chat.completions.create`
to return an async iterable of synthetic chunks that look like what
OpenRouter / OpenAI / DeepSeek / Gemini stream. The point of these tests
is the wiring: the callback fires for reasoning fragments, the
`reasoning` body kwarg is injected only on `chat()`, and `model_extra`
fallback works for SDKs that didn't model the field.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lingo import LLM, Message
from lingo.llm import _read_reasoning


def _delta(**fields: Any) -> SimpleNamespace:
    """A fake `chunk.choices[0].delta` carrying arbitrary attrs."""
    return SimpleNamespace(**fields)


def _chunk(delta: SimpleNamespace, usage=None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=None)],
        usage=usage,
    )


class _FakeStream:
    """Async iterable that yields the prepared chunks once."""

    def __init__(self, chunks: list[SimpleNamespace]):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _patch_stream(llm: LLM, chunks: list[SimpleNamespace]) -> AsyncMock:
    """Replace the OpenAI client's create() with an AsyncMock returning our stream.

    Returns the AsyncMock so the test can inspect call kwargs.
    """
    create = AsyncMock(return_value=_FakeStream(chunks))
    llm.client = MagicMock()
    llm.client.chat = MagicMock()
    llm.client.chat.completions = MagicMock()
    llm.client.chat.completions.create = create
    return create


# --- _read_reasoning unit tests -----------------------------------------


def test_read_reasoning_typed_field():
    assert _read_reasoning(_delta(reasoning="thinking...")) == "thinking..."


def test_read_reasoning_alternate_names():
    assert _read_reasoning(_delta(reasoning_content="rc")) == "rc"
    assert _read_reasoning(_delta(thoughts="t")) == "t"


def test_read_reasoning_priority():
    # reasoning wins over reasoning_content over thoughts
    d = _delta(reasoning="r", reasoning_content="rc", thoughts="t")
    assert _read_reasoning(d) == "r"


def test_read_reasoning_falls_back_to_model_extra():
    # SDK didn't model the field; it lands in model_extra.
    d = SimpleNamespace(model_extra={"reasoning": "from-extra"})
    assert _read_reasoning(d) == "from-extra"


def test_read_reasoning_returns_none_when_absent():
    assert _read_reasoning(_delta(content="hello")) is None


def test_read_reasoning_ignores_empty_strings():
    # Empty fragments shouldn't count as reasoning — providers send
    # empty deltas as no-ops.
    assert _read_reasoning(_delta(reasoning="")) is None


# --- LLM.chat() integration with the fakes ------------------------------


@pytest.mark.asyncio
async def test_chat_dispatches_reasoning_then_content():
    reasoning_tokens: list[str] = []
    content_tokens: list[str] = []

    llm = LLM(
        model="x",
        api_key="dummy",
        on_reasoning_token=lambda t: reasoning_tokens.append(t),
        on_token=lambda t: content_tokens.append(t),
    )
    _patch_stream(
        llm,
        [
            _chunk(_delta(reasoning="let me think... ")),
            _chunk(_delta(reasoning="ok.")),
            _chunk(_delta(content="hello ")),
            _chunk(_delta(content="world")),
            _chunk(
                _delta(content=None),
                usage=SimpleNamespace(
                    prompt_tokens=1,
                    completion_tokens=2,
                    total_tokens=3,
                ),
            ),
        ],
    )

    msg = await llm.chat([Message.user("hi")])

    assert reasoning_tokens == ["let me think... ", "ok."]
    assert content_tokens == ["hello ", "world"]
    assert msg.content == "hello world"
    assert msg.usage is not None
    assert msg.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_chat_injects_reasoning_via_extra_body():
    """OpenAI's SDK rejects `reasoning` as a direct kwarg. It must be
    routed through `extra_body`, which the SDK merges into the request
    JSON and forwards to the upstream provider (OpenRouter, etc.)."""
    llm = LLM(
        model="x",
        api_key="dummy",
        reasoning={"effort": "high"},
    )
    create = _patch_stream(llm, [_chunk(_delta(content="ok"))])

    await llm.chat([Message.user("hi")])

    sent_kwargs = create.call_args.kwargs
    # Never as a direct kwarg — that would TypeError against the real SDK.
    assert "reasoning" not in sent_kwargs
    assert sent_kwargs["extra_body"]["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_chat_omits_reasoning_when_not_configured():
    llm = LLM(model="x", api_key="dummy")
    create = _patch_stream(llm, [_chunk(_delta(content="ok"))])

    await llm.chat([Message.user("hi")])

    sent_kwargs = create.call_args.kwargs
    assert "reasoning" not in sent_kwargs
    assert "extra_body" not in sent_kwargs


@pytest.mark.asyncio
async def test_chat_call_kwarg_overrides_constructor_reasoning():
    llm = LLM(model="x", api_key="dummy", reasoning={"effort": "low"})
    create = _patch_stream(llm, [_chunk(_delta(content="ok"))])

    await llm.chat([Message.user("hi")], reasoning={"effort": "high"})

    assert create.call_args.kwargs["extra_body"]["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_chat_preserves_caller_extra_body():
    """If the caller supplies their own extra_body, our reasoning entry
    is added without clobbering existing keys."""
    llm = LLM(model="x", api_key="dummy", reasoning={"effort": "high"})
    create = _patch_stream(llm, [_chunk(_delta(content="ok"))])

    await llm.chat(
        [Message.user("hi")],
        extra_body={"provider": {"order": ["a", "b"]}},
    )

    sent = create.call_args.kwargs["extra_body"]
    assert sent["reasoning"] == {"effort": "high"}
    assert sent["provider"] == {"order": ["a", "b"]}


@pytest.mark.asyncio
async def test_chat_no_reasoning_callback_doesnt_crash():
    # Reasoning streamed but no callback — must not raise.
    llm = LLM(model="x", api_key="dummy")
    _patch_stream(
        llm,
        [
            _chunk(_delta(reasoning="silent thinking")),
            _chunk(_delta(content="hi")),
        ],
    )

    msg = await llm.chat([Message.user("hi")])
    assert msg.content == "hi"

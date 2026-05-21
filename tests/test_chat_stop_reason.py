import pytest
from unittest.mock import AsyncMock, MagicMock
from lingo.llm import LLM


def _chunk(content=None, finish_reason=None):
    delta = MagicMock()
    delta.content = content
    # Explicitly None out reasoning fields so the MagicMock auto-attr doesn't
    # leak truthy non-string values into _read_reasoning (see A4 fixture note).
    delta.reasoning = None
    delta.reasoning_content = None
    delta.thoughts = None
    delta.tool_calls = None
    delta.model_extra = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "content_filter"])
async def test_stop_reason_captured(reason):
    chunks = [_chunk("Hi"), _chunk(finish_reason=reason)]

    async def gen():
        for c in chunks:
            yield c

    llm = LLM(model="x", api_key="k")
    llm.client.chat.completions.create = AsyncMock(return_value=gen())

    msg = await llm.chat([])
    assert msg.stop_reason == reason


@pytest.mark.asyncio
async def test_stop_reason_none_when_absent():
    """No finish_reason in any chunk → Message.stop_reason is None."""
    chunk = _chunk("Hi")
    async def gen():
        yield chunk
    llm = LLM(model="x", api_key="k")
    llm.client.chat.completions.create = AsyncMock(return_value=gen())
    msg = await llm.chat([])
    assert msg.stop_reason is None

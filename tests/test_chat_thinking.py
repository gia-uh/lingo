import pytest
from unittest.mock import AsyncMock, MagicMock
from lingo.llm import LLM


def _chunk_with_reasoning(reasoning):
    delta = MagicMock()
    delta.content = None
    delta.reasoning = reasoning
    delta.tool_calls = None
    delta.model_extra = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


@pytest.mark.asyncio
async def test_reasoning_accumulated_onto_message():
    chunks = [
        _chunk_with_reasoning("I need to "),
        _chunk_with_reasoning("read the file"),
        _chunk_with_reasoning(" then summarize."),
    ]

    async def gen():
        for c in chunks:
            yield c

    llm = LLM(model="x", api_key="k")
    llm.client.chat.completions.create = AsyncMock(return_value=gen())

    msg = await llm.chat([])
    assert msg.thinking == "I need to read the file then summarize."


@pytest.mark.asyncio
async def test_thinking_none_when_no_reasoning():
    """No reasoning chunks → Message.thinking stays None (not empty string)."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = "hi"
    delta.reasoning = None
    delta.tool_calls = None
    delta.model_extra = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk.choices = [choice]
    chunk.usage = None

    async def gen():
        yield chunk

    llm = LLM(model="x", api_key="k")
    llm.client.chat.completions.create = AsyncMock(return_value=gen())

    msg = await llm.chat([])
    assert msg.thinking is None

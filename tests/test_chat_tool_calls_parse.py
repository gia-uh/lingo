import pytest
from unittest.mock import AsyncMock, MagicMock
from lingo.llm import LLM


def _make_chunk(content=None, tool_call_chunks=None, finish_reason=None,
                reasoning=None, usage=None):
    """Build one streaming chunk shaped like the OpenAI SDK."""
    delta = MagicMock()
    delta.content = content
    delta.reasoning = reasoning
    delta.model_extra = None
    if tool_call_chunks is not None:
        delta.tool_calls = tool_call_chunks
    else:
        delta.tool_calls = None

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _tc_chunk(index, id_=None, name=None, args=None):
    """Build one tool_calls chunk fragment."""
    tc = MagicMock()
    tc.index = index
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


@pytest.mark.asyncio
async def test_streaming_tool_calls_parsed():
    """OpenAI streams tool_calls as deltas with name + chunked JSON arguments.
    Lingo must accumulate them into Message.tool_calls with parsed dict args."""
    chunks = [
        _make_chunk(tool_call_chunks=[_tc_chunk(0, id_="call_1", name="read")]),
        _make_chunk(tool_call_chunks=[_tc_chunk(0, args='{"path"')]),
        _make_chunk(tool_call_chunks=[_tc_chunk(0, args=': "foo.py"}')]),
        _make_chunk(finish_reason="tool_calls"),
    ]

    async def gen():
        for c in chunks:
            yield c

    llm = LLM(model="x", api_key="k")
    llm.client.chat.completions.create = AsyncMock(return_value=gen())

    msg = await llm.chat([])
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "read"
    assert tc.arguments == {"path": "foo.py"}

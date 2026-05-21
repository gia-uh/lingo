import pytest
from unittest.mock import AsyncMock, patch
from lingo.llm import LLM
from lingo.tools import tool


@tool
async def read(path: str) -> str:
    """Read a file."""
    return open(path).read()


def _schema_for(tool_obj):
    """Build the OpenAI tools entry the way LLM.chat does."""
    from lingo.llm import tool_to_openai_schema

    return tool_to_openai_schema(tool_obj)


def test_tool_to_openai_schema_shape():
    entry = _schema_for(read)
    assert entry["type"] == "function"
    fn = entry["function"]
    assert fn["name"] == "read"
    assert fn["description"].strip() == "Read a file."
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "path" in params["properties"]
    assert params["properties"]["path"]["type"] == "string"
    assert "path" in params["required"]


@pytest.mark.asyncio
async def test_chat_passes_tools_kwarg():
    """LLM.chat(tools=[...]) routes the serialized schemas to the SDK."""

    # Build an async-iterable empty stream.
    async def empty_stream():
        return
        yield  # unreachable; marks this as an async generator

    llm = LLM(model="x", api_key="k")
    with patch.object(
        llm.client.chat.completions,
        "create",
        new=AsyncMock(return_value=empty_stream()),
    ) as create:
        await llm.chat([], tools=[read])

    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert "tools" in kwargs
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["function"]["name"] == "read"

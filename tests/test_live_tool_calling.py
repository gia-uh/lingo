import os
import pytest
from lingo.llm import LLM, Message
from lingo.tools import tool


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for live API",
)
async def test_native_tool_call_round_trip_live():
    """Real OpenRouter call: model emits a tool call → lingo parses it."""
    llm = LLM(
        model="anthropic/claude-haiku-4-5",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    msg = await llm.chat(
        [Message.user("What's the weather in Havana? Use the tool.")],
        tools=[get_weather],
    )
    assert msg.tool_calls is not None, "model should have emitted a tool call"
    assert len(msg.tool_calls) >= 1
    tc = msg.tool_calls[0]
    assert tc.name == "get_weather"
    assert "city" in tc.arguments
    assert "havana" in tc.arguments["city"].lower()
    assert msg.stop_reason in ("tool_calls", "stop")

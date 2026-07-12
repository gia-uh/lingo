import pytest
from unittest.mock import AsyncMock
from lingo import tool, Message, Lingo
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM

@tool
async def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a city.

    Args:
        city: The name of the city.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'.
    """
    return f"The weather in {city} is 22 {unit}."

def test_tool_metadata():
    assert get_weather.name == "get_weather"
    assert "city" in get_weather.parameters()
    assert "unit" in get_weather.defaults()
    assert get_weather.defaults()["unit"] == "celsius"

@pytest.mark.asyncio
async def test_tool_runs():
    result = await get_weather.run(city="Havana", unit="celsius")
    assert "Havana" in result
    assert "22" in result

@pytest.mark.asyncio
async def test_engine_invoke_tool():
    from pydantic import create_model

    # engine.invoke calls engine.infer which calls engine.create
    # Mock create to return the inferred parameters as a Pydantic model
    InferredParams = create_model("get_weather", city=(str, ...), unit=(str, ...))
    inferred = InferredParams(city="Havana", unit="celsius")

    engine = Engine(MockLLM())
    engine.create = AsyncMock(return_value=inferred)

    c = Context([Message.user("What's the weather in Havana?")])
    result = await engine.invoke(c, get_weather)
    assert result.tool == "get_weather"
    assert result.error is None
    assert "Havana" in str(result.result)

def make_tool_bot() -> Lingo:
    bot = Lingo(name="WeatherBot", llm=MockLLM(["It's sunny!"]))

    @bot.tool
    async def check_weather(city: str) -> str:
        """Check the weather for a city."""
        return f"Sunny in {city}."

    return bot

def test_tool_bot_has_tool():
    bot = make_tool_bot()
    assert len(bot.tools) == 1
    assert bot.tools[0].name == "check_weather"

@tool
async def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

def test_scope_adds_tools():
    base_engine = Engine(MockLLM())
    scoped = base_engine.scope([search_web])
    assert len(scoped._tools) == 1
    assert len(base_engine._tools) == 0
    assert scoped._llm is base_engine._llm  # same LLM instance


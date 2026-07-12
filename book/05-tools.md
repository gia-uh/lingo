# Chapter 5: Tools

A `Tool` is a function the LLM can call. lingo handles schema generation,
argument inference, and execution automatically.

## Defining a tool with @tool

The `@tool` decorator converts an async (or sync) function into a `Tool`.
The function's name becomes the tool name; its docstring becomes the description.
Parameter types and a Google-style `Args:` block in the docstring generate the JSON Schema.

```python {name=ch05_weather_tool}
@tool
async def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a city.

    Args:
        city: The name of the city.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'.
    """
    return f"The weather in {city} is 22 {unit}."
```

```python {name=ch05_test_tool_metadata}
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
```

## Tools on the Engine

`Engine.equip(context, *tools)` asks the LLM to select the best tool from a list.
`Engine.invoke(context, tool)` infers arguments from context and runs the tool.
`Engine.act(context, *tools)` combines equip + invoke in one call.

Both `equip` and `invoke` use `engine.create()` internally for structured output.
In tests, mock `engine.create` to control tool selection and argument inference:

```python {name=ch05_test_invoke}
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
```

## Tools on Lingo — the @bot.tool decorator

The `Lingo.tool` decorator registers a tool at the bot level.
The bot passes it to the engine via native tool-calling on every turn.

```python {name=ch05_test_bot_tool}
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
```

## Scoped tools — Engine.scope

`engine.scope(tools)` returns a new Engine with extra tools for a sub-flow,
without altering the parent engine:

```python {name=ch05_test_scope}
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
```

## Test file

```python {export=tests/test_ch05.py}
import pytest
from unittest.mock import AsyncMock
from lingo import tool, Message, Lingo
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM

<<ch05_weather_tool>>
<<ch05_test_tool_metadata>>
<<ch05_test_invoke>>
<<ch05_test_bot_tool>>
<<ch05_test_scope>>
```

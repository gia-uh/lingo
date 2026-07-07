"""index_llm.py — native tool-calling with lingo's LLM.

Demonstrates:
  - LLM.chat() with tools=[...]
  - Reading Message.tool_calls
  - Feeding tool results back as Message.tool(...)
  - The stop_reason field to detect when the model is done calling tools

Run:
    API_KEY=... python examples/index_llm.py
"""
import asyncio
from lingo import LLM, Message, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"sunny, 22°C in {city}"


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def _main_async():
    llm = LLM()
    messages = [Message.user("What's the weather in Havana, and what is 21 + 21?")]

    while True:
        msg = await llm.chat(messages, tools=[get_weather, add])
        messages.append(msg)

        # stop_reason tells us why the model stopped:
        #   "stop"       → natural end, no more tool calls
        #   "tool_calls" → model wants to call tools
        if msg.stop_reason != "tool_calls":
            print(msg.content)
            break

        for call in msg.tool_calls or []:
            result = await {"get_weather": get_weather, "add": add}[call.name].run(
                **call.arguments
            )
            # tool_call_id is required: the model uses it to match result to call
            messages.append(Message.tool(str(result), tool_call_id=call.id))


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()

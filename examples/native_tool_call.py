"""native_tool_call.py — Lingo 2.0 native tool-calling, manual dispatch loop.

Lingo's new ``LLM.chat(messages, tools=[...])`` serializes your ``@tool``
schemas into OpenAI's native tools field and parses returned tool calls
back into ``Message.tool_calls``. **Lingo does not execute the tools or
loop for you** — that's the developer's job. This example shows the
minimal manual dispatch loop.

For an agentic framework that owns the loop, see apiad/lovelaice
(or build your own on top of lingo).

Run:
    # Using a .env file (MODEL, BASE_URL, API_KEY):
    python examples/native_tool_call.py

    # Or pointing at OpenRouter directly:
    MODEL=anthropic/claude-haiku-4-5 \\
    BASE_URL=https://openrouter.ai/api/v1 \\
    API_KEY=... \\
    python examples/native_tool_call.py
"""

import asyncio
import os
import dotenv
from lingo import LLM, Message, tool


dotenv.load_dotenv()

# Allow a convenience override: set OPENROUTER_API_KEY and the example
# will inject the standard OpenRouter base URL automatically.
if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("API_KEY"):
    os.environ["API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    os.environ.setdefault("BASE_URL", "https://openrouter.ai/api/v1")
    os.environ.setdefault("MODEL", "anthropic/claude-haiku-4-5")


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real app this would hit a weather API. Returning canned data
    # keeps the example self-contained.
    return f"sunny, 22°C in {city}"


@tool
async def add(a: int, b: int) -> int:
    """Compute a + b."""
    return a + b


# Index by name so we can dispatch calls by the string the model returns.
TOOLS_BY_NAME = {"get_weather": get_weather, "add": add}


async def main():
    llm = LLM()  # reads MODEL / BASE_URL / API_KEY from environment

    messages = [
        Message.user(
            "Use the tools to answer: what is the weather in Havana? "
            "Also compute 21 + 21."
        ),
    ]

    # Manual dispatch loop: keep calling the LLM until it stops emitting tool calls.
    for turn in range(5):  # safety cap — well-prompted models finish in 1-2 turns
        msg = await llm.chat(messages, tools=list(TOOLS_BY_NAME.values()))
        messages.append(msg)

        # Print any free-form text the model produced this turn.
        if isinstance(msg.content, str) and msg.content.strip():
            print(f"Assistant: {msg.content}")

        if not msg.tool_calls:
            # Final assistant message — no further tool calls requested.
            print(f"\n[stop reason: {msg.stop_reason}]")
            break

        # Execute each requested tool and feed the result back.
        for call in msg.tool_calls:
            print(f"  -> tool: {call.name}({call.arguments})")
            fn = TOOLS_BY_NAME[call.name]
            result = await fn.run(**call.arguments)
            print(f"  <- result: {result}")
            # Tool-role messages MUST carry tool_call_id so the API can link
            # the result back to the originating tool call.
            messages.append(Message.tool(str(result), tool_call_id=call.id))
    else:
        print("[reached safety cap of 5 turns]")


if __name__ == "__main__":
    asyncio.run(main())

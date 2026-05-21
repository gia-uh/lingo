"""native_tool_call_streaming.py — Lingo 2.0 native tool-calling with streaming.

Demonstrates the per-event callbacks that fire AS the LLM streams:

  - on_token(text)               — content chunks
  - on_reasoning_token(text)     — reasoning/thinking chunks (some models)
  - on_toolcall_start(id, name)  — when a new tool call begins
  - on_toolcall_delta(id, args)  — accumulated args string so far (NOT the
                                   incremental fragment — cumulative, so you
                                   can render a live JSON preview without
                                   re-concatenating fragments yourself)
  - on_toolcall_end(id, args)    — finalized parsed dict

Useful for building live UIs (chat panels, IDE integrations) where you
want to render partial state before the message finalizes.

Run:
    # Using a .env file (MODEL, BASE_URL, API_KEY):
    python examples/native_tool_call_streaming.py

    # Or pointing at OpenRouter directly:
    MODEL=anthropic/claude-haiku-4-5 \\
    BASE_URL=https://openrouter.ai/api/v1 \\
    API_KEY=... \\
    python examples/native_tool_call_streaming.py
"""

import asyncio
import os
import sys
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
    return f"sunny, 22°C in {city}"


# --- Streaming callbacks ---


def _on_token(t: str):
    sys.stdout.write(t)
    sys.stdout.flush()


def _on_reasoning_token(t: str):
    # Render thinking in a dimmed style via ANSI escape codes.
    sys.stdout.write(f"\033[90m{t}\033[0m")
    sys.stdout.flush()


def _on_toolcall_start(call_id: str, name: str):
    print(f"\n[tool_call.start  id={call_id}  name={name}]", flush=True)


def _on_toolcall_delta(call_id: str, cumulative_args: str):
    # `cumulative_args` is the full accumulated args string so far — not
    # the incremental fragment. For a live JSON preview this is more useful
    # than re-concatenating fragments yourself.
    print(f"  args... {cumulative_args!r}", flush=True)


def _on_toolcall_end(call_id: str, args: dict):
    print(f"[tool_call.end    id={call_id}  args={args}]", flush=True)


async def _main_async():
    llm = LLM(  # reads MODEL / BASE_URL / API_KEY from environment
        on_token=_on_token,
        on_reasoning_token=_on_reasoning_token,
        on_toolcall_start=_on_toolcall_start,
        on_toolcall_delta=_on_toolcall_delta,
        on_toolcall_end=_on_toolcall_end,
    )

    # Single turn — we're demonstrating the streaming events, not the dispatch loop.
    # For the full dispatch loop, see native_tool_call.py.
    msg = await llm.chat(
        [Message.user("What's the weather in Havana? Use the tool.")],
        tools=[get_weather],
    )

    print("\n\n[final message]")
    print(f"  content:     {msg.content!r}")
    print(f"  thinking:    {msg.thinking!r}")
    print(f"  stop_reason: {msg.stop_reason}")
    print(f"  tool_calls:  {msg.tool_calls}")


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()

# The LLM — lingo's model interface

This chapter covers `lingo/llm.py`: the `LLM` class, the `Message` model, and the wire between lingo and any OpenAI-compatible provider.

---

## What it is

`LLM` is the boundary between lingo and the model. Everything that reaches the provider goes through here; everything that comes back is parsed here. No other module speaks to the network.

Two methods:

- **`chat(messages, tools=None)`** — free-form response. Always streams internally; accumulates into a complete `Message`. Fires streaming callbacks as tokens, reasoning fragments, and tool-call chunks arrive.
- **`create(model, messages)`** — structured output. Uses OpenAI's non-streaming `parse()` endpoint. Forces the provider to return JSON that validates against a Pydantic schema.

These two paths exist because they have genuinely different semantics: `chat()` is for responses the user sees (streaming feels alive); `create()` is for machine-readable output (correctness matters more than latency). They use different underlying API endpoints and cannot be unified without losing one of those properties.

---

## Message

`Message` is the unit of conversation. Every message has a role (`user`, `system`, `assistant`, `tool`) and content. The assistant variant carries optional extras: `tool_calls`, `thinking`, `stop_reason`, `usage`.

Constructor shortcuts:

```python
Message.user("hello")
Message.system("You are a helpful assistant.")
Message.assistant("I can help with that.", usage=Usage(...))
Message.tool(result_str, tool_call_id="call_abc123")
```

For multimodal content, factory methods encode the media and return a ready-to-send `Message`:

```python
Message.local_image("diagram.png")       # base64-encodes, attaches as image_url
Message.online_image("https://...")      # attaches URL directly
Message.local_audio("recording.mp3")    # base64-encodes, attaches as input_audio
```

One serialization subtlety worth knowing: when the assistant message has `tool_calls` but no text content, `model_dump()` emits `"content": null` rather than `"content": ""`. The OpenAI spec permits null here; strict providers (Qwen via OpenRouter) reject the empty string and misinterpret the conversation, stalling the agent loop. This is a provider bug, but lingo works around it.

---

## Streaming and callbacks

`LLM.chat()` always uses the streaming API. The accumulated result is returned as a complete `Message`, so callers don't have to handle streaming themselves. Callbacks are the hook for anything that needs tokens in real time (typing indicators, live rendering):

| Callback | Fires when |
|----------|-----------|
| `on_token(token)` | Each content chunk arrives |
| `on_reasoning_token(token)` | Each reasoning/thinking chunk (see below) |
| `on_toolcall_start(call_id, name)` | A new tool call begins streaming |
| `on_toolcall_delta(call_id, args_so_far)` | Arguments accumulate (cumulative, not incremental) |
| `on_toolcall_end(call_id, args_dict)` | A tool call is complete and parsed |
| `on_message(message)` | The full `Message` is assembled (after streaming) |
| `on_create(obj)` | A `create()` result is parsed (structured output only) |

All callbacks accept sync or async functions. `LLM` detects which with `inspect.iscoroutine`.

---

## Reasoning tokens

Several providers expose chain-of-thought reasoning as a separate stream alongside content: OpenRouter uses `delta.reasoning`, OpenAI o-series uses the same field, DeepSeek and some Anthropic paths use `delta.reasoning_content`, and Gemini uses `delta.thoughts`.

The OpenAI SDK preserves unknown fields via `model_extra`. `_read_reasoning()` checks both the typed attribute path and `model_extra`, so provider variance is transparent to callers. The reasoning fragments are accumulated onto `Message.thinking` and dispatched to `on_reasoning_token` independently from content tokens.

---

## Native tool-calling

Pass `tools=[...]` to `chat()`. lingo serializes each tool's schema to OpenAI's `tools[]` format, then parses streamed `tool_calls` back into `Message.tool_calls` (a list of `ToolCall(id, name, arguments)`). The tool-call callbacks fire as arguments arrive.

lingo does not execute tools, does not loop, does not decide what to do with tool results. That is your responsibility. The minimal loop:

```python {export=examples/index_llm.py}
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
```

The loop pattern: call → check `stop_reason` → if `tool_calls`, execute and append results → loop. If `stop_reason` is `stop` or anything other than `tool_calls`, the model is done.

---

## Structured output via `create()`

`create()` uses OpenAI's `parse()` endpoint, which forces the provider to emit JSON validating against the given Pydantic schema. It is non-streaming and does not accept the `reasoning` kwarg (the `parse` endpoint rejects unknown fields). Use it when you need a machine-readable result, not a user-visible response.

```python
# Illustrative only — called by Engine.create() internally
from pydantic import BaseModel

class ExtractedDate(BaseModel):
    year: int
    month: int
    day: int

result: ExtractedDate = await llm.create(ExtractedDate, messages)
```

`Engine.create()` wraps this with a prompt that describes the schema to the model. Direct `llm.create()` calls skip that prompt — you're responsible for guiding the model yourself.

---

## Configuration

`LLM` reads credentials from environment by default:

```bash
MODEL=openai/gpt-4o
BASE_URL=https://openrouter.ai/api/v1
API_KEY=sk-...
```

All three can be overridden per-instance:

```python
llm = LLM(model="anthropic/claude-opus-4-7", api_key="...", base_url="...")
```

Extra kwargs passed to `LLM.__init__` are forwarded to every API call (e.g., `temperature=0.2`). Per-call kwargs to `chat()` merge with and override them.

---

## Success criteria for this chapter

1. A developer reading this chapter understands why `chat()` and `create()` are two separate methods rather than one — and when to reach for each.

2. The `content: null` subtlety is documented. A developer debugging a stalled tool-calling loop on Qwen can find the explanation here.

3. The reasoning field variance is explained once. A developer adding a new provider does not have to dig through `_read_reasoning()` to understand why it checks three different field names.

4. The exported example (`index_llm.py`) runs end-to-end against `MockLLM` in CI and demonstrates the complete tool-calling loop with `stop_reason` handling.

5. A developer who only needs `LLM` directly (no `Lingo` shell) has everything they need in this chapter.

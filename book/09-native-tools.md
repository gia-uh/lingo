# Chapter 9: Native Tool Calling

In Chapters 5–6 the `@tool` decorator was used with the *structured-output path*:
you pick the tool, the engine infers arguments, the engine runs it.

lingo also exposes a *native* path: `LLM.chat(messages, tools=[...])` passes the
tool schemas directly to the model, which decides whether (and which) tools to call.
**lingo does not execute the tools or loop** — you write the agentic loop yourself.

This path is best when:

- You want the LLM to decide *whether and which* tools to call.
- You need parallel tool calls in one model response.
- You're building an agent that should run autonomously.

## The loop

```python
import asyncio
from lingo import LLM, Message, tool

@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny, 22°C in {city}"

@tool
async def add(a: int, b: int) -> int:
    """Compute a + b."""
    return a + b

TOOLS = {"get_weather": get_weather, "add": add}

async def main():
    llm = LLM()
    messages = [Message.user("What's the weather in Havana? Then compute 21+21.")]

    for _ in range(5):   # safety cap prevents infinite loops
        msg = await llm.chat(messages, tools=list(TOOLS.values()))
        messages.append(msg)

        if msg.content:
            print(f"Assistant: {msg.content}")

        if not msg.tool_calls:
            break  # model is done

        for call in msg.tool_calls:
            result = await TOOLS[call.name].run(**call.arguments)
            # tool_call_id links the result back to the call that requested it
            messages.append(Message.tool(str(result), tool_call_id=call.id))

asyncio.run(main())
```

The `Message` returned by `chat()` carries:

- `msg.tool_calls` — `list[ToolCall] | None`. Each `ToolCall` has `.id`, `.name`, `.arguments`.
- `msg.stop_reason` — `"stop" | "tool_calls" | "length" | ...`
- `msg.thinking` — accumulated reasoning text, if the model emits it.

## Streaming callbacks

Pass callbacks to `LLM()` to handle tool events as they stream in:

```python
llm = LLM(
    on_token=lambda t: print(t, end="", flush=True),
    on_toolcall_start=lambda call_id, name: print(f"\n[calling {name}]"),
    on_toolcall_delta=lambda call_id, args: None,   # cumulative args string
    on_toolcall_end=lambda call_id, args: print(f"  args: {args}"),
)
```

## Structured output — `LLM.create`

For single-shot structured extraction (no tool loop needed), use `LLM.create`:

```python
from pydantic import BaseModel

class Ticket(BaseModel):
    priority: str
    summary: str

result: Ticket = await llm.create(Ticket, [Message.user("Urgent: server is down")])
print(result.priority, result.summary)
```

This uses the model's `response_format` (JSON mode) to force a valid Pydantic instance.

## Native vs structured-output — when to use which

| You want… | Use |
|---|---|
| LLM decides whether/which tools to call | **Native** (`LLM.chat(tools=...)`) |
| You pick the tool; LLM fills args | **Structured** (`engine.equip` + `engine.invoke`) |
| Autonomous agent loop | **Native** + write your own loop |
| Structured conversation driven by your code | **Structured** + `Lingo` chatbot |
| Parallel tool calls in one response | **Native** |
| Broadest model compatibility | **Structured** (uses JSON mode, not function-calling) |

See `examples/native_tool_call.py` and `examples/native_tool_call_streaming.py` for
runnable end-to-end demos.

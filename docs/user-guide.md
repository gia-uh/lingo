# User Guide: Building Applications with Lingo

This guide provides "recipes" or "blueprints" for common LLM application patterns. Each recipe is based on core Lingo primitives like `Lingo`, `Engine`, `Context`, and `Flow`.

> **Two tool-calling paths.** Recipe 2 below shows the *structured-output* path (you pick the tool; the LLM fills the args) via `engine.equip` + `engine.invoke`. Recipe 8 shows the *native* path (new in 2.0: the LLM picks the tool itself, optionally in parallel batches) via `LLM.chat(tools=...)` + `Message.tool_calls`. The two paths coexist on purpose — they serve different application shapes. The `Lingo` chatbot class uses the structured path; agent frameworks that want the LLM to decide use the native path.

## 👩‍🍳 Recipe 1: The Stateful Wizard (Linear Flows)

**Best for: Multi-step data collection, onboarding, forms.**

Use `await engine.ask()` to pause and wait for user input. Variables persist in memory across turns, so you don't need to manage session state manually.

```python
@app.skill
async def register(ctx, eng):
    name = await eng.ask(ctx, "Name?")
    email = await eng.ask(ctx, f"Hi {name}, what's your email?")
    
    # engine.decide() forces a boolean response from the LLM
    if await eng.decide(ctx, f"Is {email} a valid corporate email?"):
        await eng.reply(ctx, "Success!")
    else:
        await eng.reply(ctx, "Personal emails not allowed.")
```

> 📂 Runnable: [`examples/wizard.py`](../examples/wizard.py) extends this with `engine.choose` (pick from a list) and `engine.create` (extract a typed Pydantic object).

## 🛠️ Recipe 2: The Tool-User (Function Calling)

**Best for: Interacting with external APIs, databases, or local scripts.**

Register functions as tools and use `engine.equip()` and `engine.invoke()` to execute them.

```python
@app.tool
def get_weather(city: str):
    """Returns the weather for a city."""
    return f"It's sunny in {city}."

@app.skill
async def weather_assistant(ctx, eng):
    tool = await eng.equip(ctx) # LLM selects the best tool
    result = await eng.invoke(ctx, tool) # Executes it and returns the result
    await eng.reply(ctx, f"The weather report is: {result}")
```

## 🏗️ Recipe 3: The Structured Architect (StateMachine)

**Best for: Complex business logic, secure environments, and deterministic workflows.**

Use `StateMachine` to define strict states and transitions.

```python
from lingo.fsm import StateMachine

fsm = StateMachine(app.registry)

@fsm.state
async def home(ctx, eng):
    await eng.reply(ctx, "Main menu. Say 'Help' or 'Settings'.")
    fsm.goto(settings) # Transition to another state

@app.skill
async def main_loop(ctx, eng):
    await fsm.execute(ctx, eng)
```

## ⚡ Recipe 4: The Reactive Agent (Reflexive Patterns)

**Best for: Guardrails, interruptions, global commands.**

Use `@app.when()` to intercept messages before they reach regular skills.

```python
@app.when("User wants to stop or cancel")
async def stop_handler(ctx, eng):
    await eng.reply(ctx, "Stopping everything.")
    eng.stop() # Immediately terminates the current flow
```

## 🏗️ Recipe 5: Scaling with Sub-skills

**Best for: Large applications with many components (e.g., a smart home).**

Organize logic into parent and child skills with scoped tools.

```python
@app.skill
async def kitchen(ctx, eng):
    """Manages the kitchen."""
    pass

@kitchen.tool
def start_oven(temp: int):
    return f"Oven heating to {temp}."
```

## 💾 Recipe 6: Managing Persistent State

**Best for: Games, RPGs, or applications with complex, structured data.**

Subclass `State` (a Pydantic model) to define your application's data schema.

```python
from lingo import State

class GameData(State):
    hp: int = 100
    gold: int = 50

bot = Lingo("GameBot", state=GameData())

@bot.tool
def take_damage(damage: int, state=depends(GameData)):
    state.hp -= damage
    return f"Ouch! HP: {state.hp}"
```

## 📝 Recipe 7: Prompt Engineering with Lingo

Lingo uses the `Context` object to manage the conversation history. You can steer the LLM by manually appending messages to the context.

- **`Message.system(text)`**: Internal instructions or status updates.
- **`Message.user(text)`**: Simulates user input.
- **`Message.assistant(text)`**: Simulates the bot's response.

```python
@app.skill
async def steering_skill(ctx, eng):
    # Inject a hidden instruction before the LLM generates a reply
    ctx.append(Message.system("Respond in the style of a pirate."))
    await eng.reply(ctx)
```

### Pro Tip: `context.fork()`
Use `with context.fork():` to create a temporary branch of the conversation history. Any messages appended inside the block are discarded after the block exits, which is perfect for "scratchpad" reasoning or speculative execution.

## 🔧 Recipe 8: The Native Tool-Caller (new in 2.0)

**Best for: agent-style applications where the LLM decides which tools to call and in what order, possibly in parallel batches.**

Lingo 2.0 added a direct wire to OpenAI's native tools API. `LLM.chat(messages, tools=[...])` serializes the schemas, the model emits zero or more tool calls in its response, and lingo parses them back into `Message.tool_calls`. **Lingo does not execute the tools or loop** — you write the loop yourself.

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


TOOLS_BY_NAME = {"get_weather": get_weather, "add": add}


async def main():
    llm = LLM()  # picks up MODEL / BASE_URL / API_KEY from env
    messages = [Message.user("What's the weather in Havana? Then compute 21+21.")]

    for _ in range(5):  # safety cap
        msg = await llm.chat(messages, tools=list(TOOLS_BY_NAME.values()))
        messages.append(msg)
        if msg.content:
            print(f"Assistant: {msg.content}")
        if not msg.tool_calls:
            break
        for call in msg.tool_calls:
            fn = TOOLS_BY_NAME[call.name]
            result = await fn.run(**call.arguments)
            # tool_call_id is REQUIRED — OpenAI uses it to link the result
            # back to the tool call that requested it.
            messages.append(Message.tool(str(result), tool_call_id=call.id))


asyncio.run(main())
```

### Streaming the partial state

Pass per-event callbacks to `LLM()` to render the assistant's output as it streams. The new tool-call callbacks mirror the existing `on_token`/`on_reasoning_token` pattern:

```python
def on_toolcall_start(call_id: str, name: str):
    print(f"[tool: {name}]")

def on_toolcall_delta(call_id: str, cumulative_args: str):
    # `cumulative_args` is the full accumulated args string so far —
    # not the incremental fragment. Useful for live JSON-args rendering.
    print(f"  …{cumulative_args}")

def on_toolcall_end(call_id: str, args: dict):
    print(f"  ✓ {args}")


llm = LLM(
    on_toolcall_start=on_toolcall_start,
    on_toolcall_delta=on_toolcall_delta,
    on_toolcall_end=on_toolcall_end,
)
```

The finalized `Message` also carries:

* `message.tool_calls`: `list[ToolCall] | None` — `None` if the model didn't emit any.
* `message.thinking`: `str | None` — accumulated reasoning fragments (when the model supports it).
* `message.stop_reason`: `"stop" | "length" | "tool_calls" | "content_filter" | "error" | "aborted" | None` — the OpenAI `finish_reason`.

See `examples/native_tool_call.py` and `examples/native_tool_call_streaming.py` for runnable end-to-end demos.

### Choosing native vs structured-output

Quick guide:

| You want… | Use |
|---|---|
| The LLM to decide *whether and which* tools to call, possibly in parallel | **Native** (`LLM.chat(tools=...)`) |
| To pick exactly one tool yourself and ask the LLM to fill its arguments | **Structured-output** (Recipe 2 above, `engine.equip` + `engine.invoke`) |
| To build an agentic loop that runs autonomously | **Native** + write your loop on top |
| To drive a structured conversation where each step is decided in your code | **Structured-output** + the `Lingo` chatbot class |
| To support any model that has structured output (broader compatibility) | **Structured-output** |
| To support parallel tool calls in one model response | **Native** |

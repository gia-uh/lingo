# lingo

A lightweight Python toolkit for building LLM-powered applications. lingo gives you the building blocks for **conversational modeling** — defining exactly how your system perceives, processes, and advances a dialogue — without locking you into a framework's idea of what an agent should look like.

lingo is deliberately minimal. It wires the pipes (schema serialization, tool-call parsing, streaming) and steps aside. The shape of your agent is yours.

---

## The four objects

Everything in lingo composes around four objects.

**`LLM`** — one-call interface to the model. Streams content, reasoning tokens, and tool calls; returns a rich `Message`. Reads `MODEL`, `BASE_URL`, and `API_KEY` from environment. Stateless across calls.

**`Context`** — the conversation history for one session. A mutable list of `Message` objects. The LLM sees whatever is in the context; you control what goes in and when.

**`Engine`** — the actuator. Wraps an `LLM` and a set of tools and exposes typed operations: `reply`, `ask`, `decide`, `choose`, `create`, `equip`, `invoke`. Reads from a `Context` but does not own it.

**`Lingo`** — the application shell. Manages the global conversation history, routes messages to skills, and handles pause/resume across turns.

The relationship in one sentence: *Lingo* holds history and skills; *Engine* drives the LLM; *Context* carries the messages for one flow execution; *LLM* talks to the model.

---

## Getting started

Install:

```bash
pip install lingo-ai
```

Set credentials:

```bash
export API_KEY=...
export MODEL=openai/gpt-4o
export BASE_URL=https://openrouter.ai/api/v1
```

The minimal runnable application:

```python {export=examples/index_hello.py}
"""index_hello.py — the minimal lingo application.

A Lingo with no skills configured simply replies to every message.
This is the floor — the simplest thing that works.

Run:
    API_KEY=... python examples/index_hello.py
"""
from lingo import Lingo
from lingo.cli import loop

app = Lingo("Assistant", description="A helpful assistant.")

def main():
    loop(app)

if __name__ == "__main__":
    main()
```

`loop()` is a minimal REPL for development. In production, call `await app.chat(user_message)` directly from your web framework.

---

## Pattern 1 — Procedural skills

The most common pattern. A skill is a Python coroutine. The Python call stack is your state machine: `await engine.ask(...)` suspends the coroutine until the next user message arrives. Local variables persist across the pause.

```python {export=examples/index_wizard.py}
"""index_wizard.py — procedural skill with pause/resume.

Demonstrates the four structured-output engine methods:
  - eng.ask(ctx, question)     pause execution, wait for user input
  - eng.decide(ctx, ...)       LLM returns bool (with chain-of-thought)
  - eng.choose(ctx, options)   LLM picks one item from a typed list
  - eng.create(ctx, Model)     LLM fills a Pydantic model

Run:
    API_KEY=... python examples/index_wizard.py
"""
from pydantic import BaseModel, Field
from lingo import Lingo, Context, Engine
from lingo.llm import Message
from lingo.cli import loop

app = Lingo("OnboardingWizard", description="Walks a new user through account setup.")


class UserProfile(BaseModel):
    """Information to collect about the new user."""
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    use_case: str = Field(description="What they want to use this for, one sentence")


@app.skill
async def onboarding(ctx: Context, eng: Engine):
    """Guide a new user through account setup.

    The skill is a linear script — await-points are the only state.
    No session IDs, no database lookups mid-flow, no manual resume logic.
    """
    role = await eng.choose(
        ctx,
        ["developer", "researcher", "founder", "student"],
        "What best describes the user's role based on the conversation?",
    )
    await eng.reply(ctx, f"Setting up the {role} experience.")

    raw = await eng.ask(ctx, "Tell me your name, email, and what you want to use this for.")
    ctx.append(Message.user(raw))

    profile = await eng.create(ctx, UserProfile)
    await eng.reply(ctx, f"Got it, {profile.name}. I'll reach you at {profile.email}.")

    if await eng.decide(ctx, "Did the user mention a team or enterprise context?"):
        await eng.reply(ctx, "I'll flag this as a team account.")
    else:
        await eng.reply(ctx, "Setting up a personal account.")

    await eng.reply(ctx, "All done!")


def main():
    loop(app)

if __name__ == "__main__":
    main()
```

The four structured-output operations:

| Method | Returns | When to use |
|--------|---------|-------------|
| `eng.ask(ctx, question)` | `str` | You need free-text input from the user |
| `eng.decide(ctx, ...)` | `bool` | The LLM must make a yes/no judgment |
| `eng.choose(ctx, options)` | `T` | The LLM must pick one item from a typed list |
| `eng.create(ctx, Model)` | `Model` | The LLM must extract a structured Pydantic object |

All four use chain-of-thought internally: the LLM reasons first, then commits. See `→ engine.md`.

---

## Pattern 2 — Symbolic states (FSM)

When your conversation has strict transition rules — security boundaries, multi-step workflows where backtracking is not allowed — use `StateMachine`. Each state is a coroutine. The only way to move between states is an explicit `fsm.goto(next_state)` call.

```python
# Illustrative only — for a runnable version see examples/fsm.py
from lingo.fsm import StateMachine

fsm = StateMachine(app.registry)

@fsm.state
async def triage(ctx, eng):
    topic = await eng.ask(ctx, "Is your issue about billing or tech?")
    fsm.goto(billing if "bill" in topic.lower() else tech, restart=True)

@fsm.state
async def billing(ctx, eng):
    await eng.reply(ctx, "Billing department. How can I help?")
```

The key guarantee: the LLM cannot transition between states by talking its way into one. All transitions are explicit Python calls. See `→ fsm.md`.

---

## Pattern 3 — Reflexive patterns (guardrails)

Use `@app.when` for conditions that should intercept the conversation before any skill runs. The LLM evaluates the condition on every turn; the first match wins.

```python
# Illustrative only — for a runnable version see examples/when.py
@app.when("User wants to stop, quit, or cancel")
async def emergency_stop(ctx, eng):
    await eng.reply(ctx, "Stopping.")
    eng.stop()
```

See `→ core.md`.

---

## Two tool-calling paths

**Native (LLM drives):** Pass `tools=[...]` to `llm.chat()`. The LLM decides which tools to call and when. You execute the calls and loop.

```python
# Illustrative only — for a runnable version see examples/native_tool_call.py
from lingo import LLM, Message, tool

@tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"sunny, 22°C in {city}"

llm = LLM()
messages = [Message.user("Weather in Havana?")]
while True:
    msg = await llm.chat(messages, tools=[get_weather])
    messages.append(msg)
    if not msg.tool_calls:
        break
    for call in msg.tool_calls:
        result = await get_weather.run(**call.arguments)
        messages.append(Message.tool(str(result), tool_call_id=call.id))
```

**Structured dispatch (developer drives):** Use `engine.equip` + `engine.invoke`. The developer decides when a tool gets called; the LLM only fills in the parameters.

```python
# Illustrative only — for a runnable version see examples/banker.py
@bot.skill
async def transact(ctx, eng):
    tool = await eng.equip(ctx)
    result = await eng.invoke(ctx, tool)
    await eng.reply(ctx, result, "Summarise the result for the user.")
```

See `→ llm.md` for native tool-calling. See `→ engine.md` for structured dispatch.

---

## Chapter map

| Chapter | Module | What lives there |
|---------|--------|-----------------|
| `→ llm.md` | `lingo/llm.py` | `LLM`, `Message`, streaming, native tool-calling |
| `→ engine.md` | `lingo/engine.py` | Engine primitives: reply, decide, choose, equip, invoke |
| `→ flow.md` | `lingo/flow.py` | Flow graph: fork, join, retry, route |
| `→ fsm.md` | `lingo/fsm.py` | `StateMachine`: states, transitions, per-state tools |
| `→ core.md` | `lingo/core.py` | `Lingo`, `Context`, hooks, routing |
| `→ tools.md` | `lingo/tools.py` | `@tool` decorator, schema extraction, `ToolResult` |
| `→ state.md` | `lingo/state.py` | Persistent `State` shared across turns |

---

## Success criteria for this document

1. A developer new to lingo can read this top to bottom in under 10 minutes and know which pattern to reach for their use case.

2. Every exported code block (`{export=...}`) runs end-to-end against `MockLLM` in CI. No example here can silently break.

3. Illustrative snippets (marked "Illustrative only") are clearly distinguished from exported ones. A reader never confuses a fragment for a complete program.

4. The chapter map is sufficient navigation — a developer looking for streaming knows to open `llm.md` without asking anyone.

5. When lingo's public API changes, this document is the first thing that needs updating — because it is the contract, not a description of it.

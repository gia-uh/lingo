# lingo

**lingo** is a minimal, async-native Python library for building LLM-powered applications.
It gives you typed, composable primitives — `LLM`, `Message`, `Context`, `Engine`, `Flow` —
and stays out of the way. No magic agents. No hidden chains.

## Install

```bash
pip install lingo-ai
# or with uv (recommended)
uv add lingo-ai
```

Requires Python 3.12+.

## Setup

lingo talks to any OpenAI-compatible API. Set these environment variables
(or pass them explicitly to `LLM()`):

```bash
export MODEL=gpt-4o-mini       # required
export BASE_URL=...            # optional, defaults to OpenAI
export API_KEY=sk-...          # required
```

For local models via LM Studio, Ollama, or similar:

```bash
export MODEL=qwen3:8b
export BASE_URL=http://localhost:1234/v1
export API_KEY=unused
```

## Quick start

The simplest possible bot:

```python
import asyncio
from lingo import Lingo

bot = Lingo(name="Assistant", description="A helpful AI.")

async def main():
    reply = await bot.chat("Hello!")
    print(reply.content)

asyncio.run(main())
```

A bot that collects input mid-conversation:

```python
from lingo import Lingo

bot = Lingo(name="Wizard")

@bot.skill
async def onboarding(ctx, eng):
    """Greet the user and ask their name."""
    name = await eng.ask(ctx, "What is your name?")
    ctx.append(f"The user's name is {name}.")
    await eng.reply(ctx, f"Welcome, {name}!")
```

Run it in the terminal with `lingo.cli.loop`:

```python
from lingo.cli import loop
loop(bot)
```

## How it works

```
LLM ──► Engine ──► Flow ──► Context
                              │
                              ▼
                         Lingo (orchestrator)
```

- **`LLM`** — wraps any OpenAI-compatible API. Streams tokens, fires callbacks.
- **`Message`** — a single typed conversation turn. Supports text, images, audio, video.
- **`Context`** — the mutable message window for one interaction. Supports fork/clone/atomic.
- **`Engine`** — performs LLM operations on a context: reply, decide, choose, create, invoke.
- **`Flow`** — a declarative, chainable workflow: sequential, conditional, looping, parallel.
- **`Lingo`** — the chatbot facade. Owns history, builds flows from skills, exposes `.chat()`.

## This book

The chapters below are the complete API reference for lingo, written as literate programming:
every code block is executable and tested. `make book` compiles and verifies them.

Navigate the chapters in order, or jump directly to what you need.

| Chapter | Topic |
|---|---|
| [1. Hello, lingo](01-hello-lingo.md) | LLM, Message, your first chatbot |
| [2. Messages and Context](02-messages-and-context.md) | Multimodal content, context manipulation |
| [3. The Engine](03-the-engine.md) | reply, decide, choose, create |
| [4. Flows](04-flows.md) | Declarative, chainable workflows |
| [5. Tools](05-tools.md) | Functions as LLM-callable tools |
| [6. Skills and Routing](06-skills-and-routing.md) | Multi-skill bots |
| [7. State](07-state.md) | Conversation state with atomic semantics |
| [8. Patterns](08-patterns.md) | End-to-end examples |
| [9. Native Tool Calling](09-native-tools.md) | Direct LLM tool-calling API |

## Source and license

[github.com/gia-uh/lingo](https://github.com/gia-uh/lingo) — MIT license.

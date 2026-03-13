# Lingo: The Architect in the Machine

**Lingo** is a minimal, async-native, and unopinionated toolkit for building modern LLM applications. It moves beyond generic "agents" and "chains" to focus on **Conversational Modeling**—the discipline of defining exactly how a system perceives, processes, and advances a dialogue state.

## 🧠 The Philosophy

Lingo is built on the principle that **The Python stack is your state machine.** 

Most frameworks force you to define complex graphs, manage session IDs manually, or use "black-box" agents that are hard to steer. Lingo takes the opposite approach: it uses standard Python `async/await` to pause and resume execution, making the conversation history a first-class citizen of your code.

We call this **The Architect in the Machine**: a design pattern where the LLM is not just a chatbot, but an active component in a structured, typed architecture.

## ⚡ Core Values

- **💾 Stateful by Default**: Use `await engine.ask()` to pause execution. Lingo automatically suspends the stack and preserves variables in memory across turns.
- **🛡️ Type-Safe**: Built on Pydantic. All inputs, outputs, and tool calls are validated against your schemas.
- **🌊 Multi-Paradigm**: Mix rigid business rules (States) with flexible reasoning (Skills) and event-driven guardrails (Reflexive Patterns).
- **🏗️ Unopinionated**: No forced "agent" abstractions. Use as much or as little of the framework as you need.

## 🚀 Quick Start

To get a feel for Lingo, check out the [User Guide](user-guide.md) for step-by-step recipes, or the [Deployment Guide](deploy.md) to set up your environment.

```bash
pip install lingo-ai
```

### The "Hello World" (Stateful Wizard)

```python
import asyncio
from lingo import Lingo

app = Lingo("Wizard")

@app.skill
async def onboarding(ctx, eng):
    await eng.reply(ctx, "Welcome!")
    # PAUSE execution and wait for user input
    name = await eng.ask(ctx, "What is your name?")
    # Resume and use context from previous turns
    await eng.reply(ctx, f"Hi {name}, nice to meet you!")

if __name__ == "__main__":
    from lingo.cli import loop
    loop(app)
```

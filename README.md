<p align="center"> <img src="https://github.com/user-attachments/assets/27a24307-cda0-4fa8-ba6c-9b5ca9b27efe" alt="lingo library logo" width="300"/> </p>

<p align="center"> <strong>A minimal, async-native, and unopinionated toolkit for modern LLM applications.</strong> </p>

![PyPI - Version](https://img.shields.io/pypi/v/lingo-ai)
![PyPi - Python Version](https://img.shields.io/pypi/pyversions/lingo-ai)
![Github - Open Issues](https://img.shields.io/github/issues-raw/gia-uh/lingo)
![PyPi - Downloads (Monthly)](https://img.shields.io/pypi/dm/lingo-ai)
![Github - Commits](https://img.shields.io/github/commit-activity/m/gia-uh/lingo)

> **Stop wiring graphs. Start modeling conversations.**

Lingo is a lightweight, type-safe Python framework for building LLM-powered applications. It moves beyond generic "agents" and "chains" to focus on **Conversational Modeling**—the discipline of defining exactly how a system perceives, processes, and advances a dialogue state.

It unifies three powerful paradigms in a single, typed architecture:

1.  **Procedural Skills** (Linear, script-like flows)
2.  **Symbolic States** (Deterministic FSMs)
3.  **Reflexive Patterns** (Event-driven guardrails)

## ⚡ Features

* **💾 Stateful by Default:** The Python stack *is* your state machine. Use `await engine.ask()` to pause execution and wait for user input naturally.
* **🧠 Cognitive Architecture:** Mix rigid business rules (States) with flexible reasoning (Skills).
* **🛡️ Type-Safe:** Built on Pydantic. All inputs, outputs, and tool calls are validated schemas.
* **🌊 Low-Level Flow Control:** Direct access to the underlying `Flow` graph for complex orchestration (Fork/Join, Retry, Loops).

## 🚀 Quickstart

### Installation

```bash
pip install lingo-ai

```

### The "Hello World" (Stateful Wizard)

Lingo allows you to model conversations as linear scripts. You don't need to manage session IDs or database steps manually—variables persist in memory across turns.

```python
import asyncio
from lingo import Lingo

# Initialize the application
app = Lingo("Wizard", description="A helpful setup wizard")

@app.skill
async def onboarding(ctx, eng):
    # 1. Output a message
    await eng.reply(ctx, "Welcome to the system.")

    # 2. PAUSE execution and wait for user input
    # Lingo automatically suspends the stack here.
    # The variable 'name' is preserved in memory when the user replies!
    name = await eng.ask(ctx, "What is your name?")

    # 3. Resume and use context from previous turns
    email = await eng.ask(ctx, f"Hi {name}, what is your email?")

    # 4. Use structured decision making (LLM is forced to return bool)
    if await eng.decide(ctx, f"Is {email} a valid corporate email address?"):
        await eng.reply(ctx, "Registration complete.")
    else:
        await eng.reply(ctx, "Personal emails are not allowed.")

if __name__ == "__main__":
    from lingo.cli import loop
    loop(app)

```

---

## 🧠 The Three Modeling Paradigms

Lingo gives you the right abstraction for every type of logic.

### 1. Symbolic States (Finite State Machine)

**Best for: Business Logic, Security Boundaries, Multi-Step Workflows.**

Use the `StateMachine` to enforce strict rules about allowed transitions.

```python
from lingo.fsm import StateMachine

# 1. Initialize the FSM with the bot's registry
fsm = StateMachine(app.registry)

@fsm.state
async def login(ctx, eng):
    await eng.reply(ctx, "Please log in.")
    # Deterministic transition to the next state
    fsm.goto(dashboard, restart=True)

@fsm.state
async def dashboard(ctx, eng):
    await eng.reply(ctx, "Welcome to your dashboard.")
    # Logic restricted to this state...

# Register the FSM as a skill
@app.skill
async def run_workflow(ctx, eng):
    await fsm.execute(ctx, eng)

```

### 2. Reflexive Patterns (Event-Driven)

**Best for: Guardrails, Interruptions, Global Commands.**

Use `@app.when` to define high-priority listeners that intercept messages *before* they reach skills.

```python
@app.when("User wants to quit or cancel the operation")
async def emergency_stop(ctx, eng):
    await eng.reply(ctx, "Stopping immediately.")
    eng.stop() # Terminates the flow and clears the stack

```

### 3. Structured Flows (Low-Level Graph)

**Best for: Parallel Processing, Retries, Complex Orchestration.**

You can drop down to the `Flow` API to build complex execution graphs explicitly.

```python
from lingo import Flow

# Define two sub-flows
research = Flow("Research").reply("Searching for info...")
draft = Flow("Draft").reply("Drafting content...")

# Build a flow that runs them in parallel (Fork)
# and summarizes the result
complex_flow = (
    Flow("ParallelWorker")
    .fork(
        research,
        draft,
        aggregator="Combine the research and draft into a final report."
    )
)

```

## 📦 Architecture

* **`Context`**: Mutable ledger of the conversation history.
* **`Engine`**: The "actuator" that drives the LLM. It exposes methods like `.ask()`, `.decide()`, `.choose()`, and `.create()`.
* **`Flow`**: The underlying graph representation of all skills.

## 🤝 Contribution

We welcome contributions! Please see [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for details on how to set up your development environment and submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# AI Agent Skill: Mastering Lingo-AI

This document is a technical guide for AI coding agents (like Gemini, Claude, or GPT) on how to correctly install, implement, test, and maintain applications using the **Lingo-AI** library.

---

## 🏗️ Project Overview for Agents
Lingo-AI is a "Conversational Modeling" framework. Unlike agentic frameworks that rely on autonomous loops, Lingo uses a **stateful execution model** where the Python async stack *is* the conversation state. 

**Key Concept**: "The Architect in the Machine." The LLM is a component within a typed, structured architecture, not just a chatbot.

---

## 🛠️ Installation & Setup
To set up a Lingo-AI project, use `uv` (preferred) or `pip`:
```bash
uv add lingo-ai
# or
pip install lingo-ai
```

**Environment Variables**:
- `OPENAI_API_KEY`: Required for LLM operations.
- `OPENAI_MODEL`: Defaults to `gpt-4o-mini`.

---

## 🧠 The Idiomatic Lingo-AI Agent

### 1. Defining Skills (`@bot.skill`)
Skills are the entry points for conversation. They MUST be `async`.
- **Linear Logic**: Use `await engine.ask(ctx, "Question?")` to pause and wait for user input.
- **Branching**: Use `if await engine.decide(ctx, "Is the user angry?"):` for semantic branching.
- **Nesting**: Use `@skill.subskill` to organize large apps.

### 2. Writing Tools (`@bot.tool`)
Tools are functions the LLM can invoke.
- **Rich Docstrings**: The LLM reads the docstring to understand *when* and *how* to use the tool. Be descriptive.
- **Hidden Parameters**: Prefix parameters with `_` (e.g., `_api_key`) to hide them from the LLM's schema.
- **Dependency Injection**: Use `depends()` for registry-managed objects (e.g., `db=depends(Database)`).

### 3. State Management (`State`)
Subclass `State` for persistent, typed data.
- **`state.atomic()`**: Use this context manager to ensure state changes are rolled back if an error occurs.
- **`state.fork()`**: Use this to speculatively modify state; it ALWAYS rolls back.

---

## 🌊 Control Flow Patterns

### Reflexive Patterns (`@bot.when`)
Reflexes intercept messages before skills.
- **Guardrails**: `@bot.when("User uses profanity")`. Call `engine.stop()` to abort the current flow.
- **Context Injection**: `@bot.when("User mentions a product")`. Append information to the context without stopping.

### Early Exit
Use `engine.stop()` to immediately terminate the current `Flow` or `Skill` execution.

---

## ✅ Dos and Don'ts

### 👍 DO
- **Use `async/await`** for ALL engine and skill calls.
- **Provide clear type hints** for all tool parameters.
- **Use `engine.create(ctx, PydanticModel)`** for structured data extraction.
- **Use `context.fork()`** for "thought chains" or reasoning steps you don't want the user to see in the final history.
- **Mock the LLM** in unit tests using `MockLLM`.

### 👎 DON'T
- **Do NOT use global variables** for conversation state. Use the `State` class.
- **Do NOT forget to `await`** `engine.reply()`, `engine.ask()`, or `engine.invoke()`.
- **Do NOT use underscores** for parameters the LLM *needs* to provide (e.g., `query`).
- **Do NOT use complex loops** inside skills if you can use a `StateMachine`.

---

## 🧪 Testing Strategy
Use `MockLLM` from `lingo.mock` to write deterministic tests.

```python
from lingo.mock import MockLLM
from lingo import Context, Message

async def test_my_skill():
    # 1. Setup Mock LLM with pre-programmed responses
    llm = MockLLM(["Hello!", "I am an AI."])
    bot = Lingo(llm=llm)
    ctx = Context()
    
    # 2. Execute
    await bot.chat(Message.user("Hi"), ctx)
    
    # 3. Assert on history
    assert "Hello!" in ctx.render()
```

---

## 🚀 Example: Idiomatic Lingo-AI App
A customer support bot with a safety guardrail.

```python
from lingo import Lingo, State, Message, depends

class UserData(State):
    name: str = "Guest"

app = Lingo("SupportBot", state=UserData())

@app.when("The user is being abusive or using profanity")
async def safety_guardrail(ctx, eng):
    await eng.reply(ctx, "I cannot continue this conversation due to language.")
    eng.stop() # Abort the main skill

@app.tool
def update_name(new_name: str, state=depends(UserData)):
    """Updates the user's name in the system."""
    state.name = new_name
    return f"Name updated to {new_name}"

@app.skill
async def support_skill(ctx, eng, state=depends(UserData)):
    """Main support flow."""
    await eng.reply(ctx, f"Hello {state.name}, how can I help?")
    
    # Orchestrate tools and interaction
    tool = await eng.equip(ctx)
    if tool:
        await eng.invoke(ctx, tool)
        await eng.reply(ctx)
```

<p align="center"> <img src="https://github.com/user-attachments/assets/27a24307-cda0-4fa8-ba6c-9b5ca9b27efe" alt="lingo library logo" width="300"/> </p>

<p align="center"> <strong>A minimal, async-native, and unopinionated toolkit for modern LLM applications.</strong> </p>

![PyPI - Version](https://img.shields.io/pypi/v/lingo-ai)
![PyPi - Python Version](https://img.shields.io/pypi/pyversions/lingo-ai)
![Github - Open Issues](https://img.shields.io/github/issues-raw/gia-uh/lingo)
![PyPi - Downloads (Monthly)](https://img.shields.io/pypi/dm/lingo-ai)
![Github - Commits](https://img.shields.io/github/commit-activity/m/gia-uh/lingo)

---

**Lingo** is a framework for creating LLM-based applications built on the concept of **Prompt Flows**. It offers two distinct patterns for building AI logic: the **Flow API** (declarative) and the **Bot API** (imperative). You can mix and match these approaches as needed, using flows for reusable logic and the Bot API for stateful, interactive agents.

## 1. The Flow API (Declarative)

The Flow API is designed for building reusable, stateless sequences of operations. Using a fluent interface, you chain nodes that represent logical steps. Because these flows use Python 3.12 generics (`Flow[T]`), the return type is tracked throughout the entire chain.

### Example: A Research & Extraction Flow

This flow performs parallel research, handles potential errors atomically, and extracts structured data.

```python
from lingo import Flow, Engine, LLM
from pydantic import BaseModel

class ResearchData(BaseModel):
    summary: str
    confidence: float

# Define a 'fixer' for retries
fixer = Flow().append(lambda ctx: f"Error encountered: {ctx.metadata['last_exception']}")

# Declarative Flow
research_flow = (Flow[ResearchData]("Researcher")
    .append("Topic: {topic}")
    .fork(
        Flow().append("Search news...").act(news_tool),
        Flow().append("Search wiki...").act(wiki_tool),
        aggregator="Synthesize these findings"
    )
    .retry(fixer, max_retries=2)
    .create(ResearchData, "Generate the final JSON object")
)
```

## 2. The Bot API (Imperative)

The Bot API allows you to build stateful agents by inheriting from the `Lingo` class. It provides a robust **Dependency Injection** system, allowing your skills and tools to request resources (like the `LLM`, `Context`, or custom services) automatically.

### Example: The Banker Bot with Dependency Injection

This bot demonstrates how to inject dependencies directly into tools, keeping your logic clean and testable.

```python
from lingo import Lingo, Context, Engine, Message, skill, tool, LLM
from purely import depends

# 1. Initialize Bot
bot = Lingo(name="Banker", description="A bank assistant")

# 2. Define Tools with Injection
@bot.tool
async def analyze_spending(
    category: str,
    # Automatically inject the LLM instance
    llm: LLM = depends(LLM)
) -> str:
    """Analyze spending history for a category."""
    # The LLM is available here without manual passing
    response = await llm.chat([Message.user(f"Analyze {category}")])
    return response.content

# 3. Define Skills
@bot.skill
async def banker_skill(context: Context, engine: Engine):
    """Interact with the bank account."""

    # Engine.equip automatically respects injected dependencies
    selected_tool = await engine.equip(context)

    # Engine.invoke merges LLM-generated args with manual overrides
    # You can pass internal flags (starting with _) that the LLM won't see
    result = await engine.invoke(context, selected_tool, _internal_flag=True)

    await engine.reply(context, Message.system(result.model_dump_json()))
```

## 3. Middleware & Hooks

Lingo supports a middleware system that allows you to execute logic before and after the main skill routing. This is ideal for logging, context preparation, or cleanup.

```python
@bot.before
async def log_interaction(context: Context, engine: Engine):
    print(f"New interaction started with {len(context)} messages.")

@bot.after
async def cleanup(context: Context, engine: Engine):
    # Perform cleanup or analytics
    print("Interaction finished.")
```

## 4. Key Differences at a Glance

| Feature          | Flow API (Declarative)                | Bot API (Imperative)                     |
| ---------------- | ------------------------------------- | ---------------------------------------- |
| **Logic Type**   | Reusable, stateless sequences.        | Stateful, dynamic agents.                |
| **Control**      | Orchestrated via `Node` components.   | Direct access to `Engine` and `Context`. |
| **Branching**    | Handled by `When` and `Branch` nodes. | Handled by the **Skill Router**.         |
| **Tool Use**     | Managed via the `act()` node.         | Manual `equip()` and `invoke()` calls.   |
| **Dependencies** | Passed via `Flow` arguments.          | **Automatic Dependency Injection**.      |
| **Hooks**        | N/A                                   | **@before** and **@after** middleware.   |

## 5. Resilience & Memory Management

Both APIs benefit from Lingo's v1.0 core primitives:

* **Atomic Transactions**: Use `context.atomic()` to roll back history if a segment of logic fails, ensuring a clean history.
* **Context Compression**: Use `compress()` to prune the message history (summarizing or sliding window) to stay within token limits.
* **Usage Auditing**: Every interaction tracks token counts via `Usage` objects and optional `on_message` callbacks for the `LLM`.

## 6. Contribution & License

### Contribution

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on submitting PRs or reporting issues.

### License

Lingo is released under the **MIT License**.

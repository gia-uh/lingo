<p align="center"> <img src="https://github.com/user-attachments/assets/27a24307-cda0-4fa8-ba6c-9b5ca9b27efe" alt="lingo library logo" width="300"/> </p>

<p align="center"> <strong>A minimal, async-native, and unopinionated toolkit for modern LLM applications.</strong> </p>

---

<!-- Project badges -->
![PyPI - Version](https://img.shields.io/pypi/v/lingo-ai)
![PyPi - Python Version](https://img.shields.io/pypi/pyversions/lingo-ai)
![Github - Open Issues](https://img.shields.io/github/issues-raw/gia-uh/lingo)
![PyPi - Downloads (Monthly)](https://img.shields.io/pypi/dm/lingo-ai)
![Github - Commits](https://img.shields.io/github/commit-activity/m/gia-uh/lingo)


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

The Bot API allows you to build stateful agents by inheriting from the `Lingo` class. Here, you manually interact with the `Engine` and `Context`. The primary building block is the **Skill**, which acts as a top-level router. The bot automatically selects the most appropriate skill based on the user's input.

### Example: The Banker Bot

This bot uses skills to handle different intents and includes manual **Tool Calling**.

```python
from lingo import Lingo, Context, Engine, Message, skill, tool

bot = Lingo(name="Banker", description="A bank assistant")

@bot.skill
async def banker_skill(context: Context, engine: Engine):
    """Interact with the bank account."""
    # Manual tool selection and invocation
    selected_tool = await engine.equip(context) # Inspects available @bot.tools
    result = await engine.invoke(context, selected_tool)

    # Imperative response generation
    await engine.reply(
        context,
        Message.system(result),
        Message.system("Inform the user of the result.")
    )

@bot.tool
async def check_balance() -> dict:
    """Returns the current account balance."""
    return {"balance": 1000}

# You can also call a declarative Flow from within a Skill
@bot.skill
async def specialized_task(context: Context, engine: Engine):
    """Runs a pre-defined declarative flow."""
    result = await my_declarative_flow.run(engine, context)
    context.append(Message.assistant(f"Task complete: {result}"))
```

## 3. Key Differences at a Glance

| Feature | Flow API (Declarative) | Bot API (Imperative) |
| --- | --- | --- |
| **Logic Type** | Reusable, stateless sequences. | Stateful, dynamic agents. |
| **Control** | Orchestrated via `Node` components. | Direct access to `Engine` and `Context`. |
| **Branching** | Handled by `When` and `Branch` nodes. | Handled by the **Skill Router**. |
| **Tool Use** | Managed via the `act()` node. | Manual `equip()` and `invoke()` calls. |
| **Error Handling** | Transactional `retry()` and `attempt()`. | Manual try/except or `context.atomic()`. |

## 4. Resilience & Memory Management

Both APIs benefit from Lingo's v1.0 core primitives:

* **Atomic Transactions**: Use `context.atomic()` to roll back history if a segment of logic fails, ensuring a clean history.
* **Context Compression**: Use `compress()` to prune the message history (summarizing or sliding window) to stay within token limits.
* **Usage Auditing**: Every interaction tracks token counts via `Usage` objects and optional `on_message` callbacks for the `LLM`.

## 5. Contribution & License

### Contribution

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on submitting PRs or reporting issues.

### License

Lingo is released under the **MIT License**.

Would you like me to generate a more complex example where a **Bot API** agent manages multiple **Flow API** routines?

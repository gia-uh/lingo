<p align="center"> <img src="https://github.com/user-attachments/assets/27a24307-cda0-4fa8-ba6c-9b5ca9b27efe" alt="lingo library logo" width="300"/> </p>

<p align="center"> <strong>A minimal, async-native, and unopinionated toolkit for modern LLM applications.</strong> </p>

---

<!-- Project badges -->
![PyPI - Version](https://img.shields.io/pypi/v/lingo-ai)
![PyPi - Python Version](https://img.shields.io/pypi/pyversions/lingo-ai)
![Github - Open Issues](https://img.shields.io/github/issues-raw/gia-uh/lingo)
![PyPi - Downloads (Monthly)](https://img.shields.io/pypi/dm/lingo-ai)
![Github - Commits](https://img.shields.io/github/commit-activity/m/gia-uh/lingo)

-----

`lingo` provides a powerful, three-layered API for building, testing, and deploying complex LLM workflows with precision and clarity.

## The Philosophy: A Three-Layer API

`lingo` is built on the idea that developers need different levels of abstraction for different tasks.

1.  **The High-Level `Lingo` API**: For purely declarative, ready-to-use LLM assistants. This is the fastest way to get a chatbot running.
2.  **The Mid-Level `Flow` API**: For declarative, reusable context engineering workflows. This allows you to define complex, composable logic with branching, tool use, and subroutines.
3.  **The Low-Level (`LLM`, `Engine`, `Context`) API**: For direct, explicit context engineering. This gives you full, imperative control over the message history and LLM interactions.

## Installation

```bash
pip install lingo-ai
```

You will also need to set your environment variables (e.g., in a `.env` file) for your LLM provider:

```.env
# Example for OpenAI
MODEL="gpt-4o"
API_KEY="sk-..."
```

## Quickstart: A 5-Line Chatbot

This is the fastest way to get a `lingo` assistant running using the high-level `Lingo` class.

```python
from lingo import Lingo
from lingo.cli import loop
import dotenv

# Load .env variables (API_KEY, MODEL)
dotenv.load_dotenv()

# 1. Initialize the assistant
bot = Lingo(
    name="Assistant",
    description="A simple, helpful chatbot."
)

# 2. Run the chat loop in your terminal
loop(bot)
```

That's it\! You now have a fully interactive chatbot.

```bash
Name: Assistant
Description: A simple, helpful chatbot.

[Press Ctrl+D to exit]

>>> Hello!
Hello! How can I help you today?

>>>
```

## The Three API Layers

`lingo` gives you the flexibility to choose the right level of abstraction.

### 1\. High-Level API: The `Lingo` Class

This is the "batteries-included" approach. The `Lingo` class manages the `LLM`, `Engine`, and `Flow` for you. You just define **skills** (reusable flows) and **tools**, and `lingo` handles routing the conversation to the correct one.

This is the recommended starting point for most applications.

```python
from lingo import Lingo, Context, Engine
from lingo.cli import loop
import dotenv

dotenv.load_dotenv()

bot = Lingo(
    name="Greeter",
    description="A bot that just says hello."
)

# A "skill" is a complete, self-contained workflow
@bot.skill
async def greet(context: Context, engine: Engine):
    """A skill to greet the user."""
    await engine.reply(
        context,
        "You are a friendly greeter. Reply with a warm welcome."
    )

loop(bot)
```

### 2\. Mid-Level API: The `Flow` Class

The `Flow` class provides a fluent, chainable interface for declaratively building reusable workflows. You define the *steps* of the conversation, and `lingo` handles the execution. This is perfect for defining complex, stateful logic.

```python
import asyncio
from lingo import Lingo, Flow, Message, Engine, LLM
from lingo.tools import tool

llm = LLM(model="gpt-4o")

@tool
async def get_weather(location: str) -> str:
    """Gets the current weather for a specified location."""
    return "It's 75°F and sunny."

# A flow is a composable, reusable blueprint
weather_flow = (
    Flow(name="Weather")
    .system("You only answer with the weather.")
    .invoke(get_weather)  # -> ToolResult is added to context
    .reply()              # -> LLM generates reply based on ToolResult
)

# You can nest flows inside other flows
main_flow = (
    Flow(name="Main")
    .choose(
        prompt="Is the user asking about weather or stocks?",
        choices=dict(
            weather=weather_flow,
            stocks=Flow().reply("I don't know about stocks."),
        )
    )
)

async def main():
    engine = Engine(llm)
    messages = [Message.user("What's the weather like?")]

    # Run the flow
    final_context = await main_flow(engine, messages)
    print(final_context.messages[-1].content)

asyncio.run(main())
```

### 3\. Low-Level API: `LLM`, `Engine`, & `Context`

For maximum control, you can use the imperative API.

  * **`Context`**: A simple, mutable object holding the `list[Message]`.
  * **`LLM`**: The client for interacting with the LLM API (e.g., `chat`, `create`).
  * **`Engine`**: The "behavior" layer. It holds the `LLM` and performs operations *on* a `Context` (e.g., `engine.reply(context)`, `engine.invoke(context, tool)`).

This is ideal for building custom chatbot loops or integrating `lingo` into an existing application.

```python
import asyncio
from lingo import LLM, Engine, Context, Message

async def main():
    # 1. Setup the components
    llm = LLM(model="gpt-4o")
    engine = Engine(llm)

    # 2. The Context is a pure, mutable state object
    context = Context([
        Message.system("You are a helpful assistant.")
    ])

    # 3. Manually build the conversation
    user_input = "Hello!"
    context.append(Message.user(user_input))

    # 4. Call the Engine to perform an LLM operation
    response = await engine.reply(context)

    # 5. Mutate the context with the new message
    context.append(response)

    print(f"Bot: {response.content}")

asyncio.run(main())
```

## Contributing

Contributions are welcome\! `lingo` is an open-source project, and we'd love your help in making it better. Please feel free to open an issue or submit a pull request.

## License

`lingo` is licensed under the **MIT License**. See the `LICENSE` file for details.

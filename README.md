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

`lingo` provides a powerful, two-layered API for building, testing, and deploying complex LLM workflows with precision and clarity.

## The Philosophy: A Dual-Layer API

`lingo` is built on the idea that developers need both a high-level, declarative way to _build_ workflows and a low-level, imperative way to _control_ them.

1. **The High-Level `Flow` API**: For most applications, you'll use the fluent, chainable `Flow` API. This allows you to define complex, reusable conversation logic with branching, tool use, and subroutines in a declarative, readable way.

2. **The Low-Level `Context` API**: For full, fine-grained control, `lingo` provides the `LLM` and `Context` classes. This imperative layer is perfect for building custom chatbot loops or implementing your own conversational state logic from scratch.


## Installation

```
pip install lingo-ai
```

## Quickstart: Building a Declarative `Flow`

Let's build a simple assistant that can check the weather, but will first decide if the user is being polite.

### 1. Define Tools & LLM

```python
import asyncio
from lingo.llm import LLM, Message
from lingo.tools import tool
from lingo.flow import Flow, NoOp

llm = LLM(model="gpt-4o")

@tool
async def get_weather(location: str) -> str:
    """Gets the current weather for a specified location."""
    if "boston" in location.lower():
        return "It's 75°F and sunny in Boston."
    else:
        return f"Weather for {location} is unknown."
```

### 2. Define Reusable Sub-Flows


```python
# A sub-flow is just another 'Flow' instance.
polite_flow = Flow().system(
    "Acknowledge the user's politeness before proceeding."
)
```

### 3. Define the Main Workflow


```python
main_flow = (
    Flow()
    # Step 1: Add a system message to the context
    .system("You are a helpful assistant. You can check the weather.")

    # Step 2: Use an LLM to make a True/False decision
    .decide(
        prompt="Is the user being polite (e.g., asking 'please')?",
        on_true=polite_flow,  # Run the sub-flow if True
        on_false=NoOp()       # Do nothing if False
    )

    # Step 3: Equip and invoke a tool from the available list
    .invoke(get_weather)

    # Step 4: Generate a final reply using the tool's output
    .reply()
)
```

### 4. Execute the Flow


```python
async def main():
    # Example 1: The "polite" branch
    user_query_1 = "Could you please tell me the weather in Boston?"

    # .execute() runs the full pipeline on the initial messages
    final_context = await main_flow.run(llm, [Message.user(user_query_1)])

    print(f"User: {user_query_1}")
    print(f"Assistant: {final_context.messages[-1].content}")
    # Assistant: It's 75°F and sunny in Boston. I'm happy to help!

    print("\n" + "-"*20 + "\n")

    # Example 2: The "impolite" branch
    user_query_2 = "boston weather now"
    final_context_2 = await main_flow.run(llm, [Message.user(user_query_2)])

    print(f"User: {user_query_2}")
    print(f"Assistant: {final_context_2.messages[-1].content}")
    # Assistant: It's 75°F and sunny in Boston.

if __name__ == "__main__":
    asyncio.run(main())
```

## The Two APIs

`lingo` gives you the flexibility to choose the right level of abstraction.

### 1. High-Level API: `Flow`

The `Flow` class provides a fluent, chainable interface for declaratively building reusable workflows. You define the _steps_ of the conversation, and `lingo` handles the execution.

```python
# A flow is a composable, reusable blueprint
weather_flow = (
    Flow()
    .system("You only answer with the weather.")
    .invoke(get_weather)
    .reply()
)

# You can nest flows inside other flows
main_flow = (
    Flow()
    .choose(
        prompt="Is the user asking about weather or stocks?",
        choices={
            "weather": weather_flow,
            "stocks": stock_flow,
        }
    )
)
```

### 2. Low-Level API: `LLM` & `Context`

For maximum control, you can use the imperative API. The `Context` object holds the message history, and you call its methods (`.reply()`, `.decide()`, `.invoke()`) directly. This is ideal for building custom chatbot loops.

```python
# Manually building a conversation turn by turn
llm = LLM(model="gpt-4o")
context = Context(llm, [Message.system("You are a chatbot.")])

while True:
    user_input = input("You: ")
    context.add(Message.user(user_input))

    # Call the LLM with the current context
    response = await context.reply()

    context.add(response)
    print(f"Bot: {response.content}")
```

## Contributing

Contributions are welcome! `lingo` is an open-source project, and we'd love your help in making it better. Please feel free to open an issue or submit a pull request.

## License

`lingo` is licensed under the **MIT License**. See the `LICENSE` file for details.

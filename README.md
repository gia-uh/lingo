<p align="center">
<img src="https://github.com/user-attachments/assets/27a24307-cda0-4fa8-ba6c-9b5ca9b27efe" alt="lingo library logo" width="300"/>
</p>

<p align="center">
<strong>A minimal, async-native, and unopinionated toolkit for modern LLM applications.</strong>
</p>

`lingo` provides a powerful, fluent API for building, testing, and deploying complex prompt engineering workflows. Instead of locking you into a rigid framework, `lingo` gives you a set of composable, functional building blocks to orchestrate LLM interactions with precision and clarity.

## The Philosophy: Context Engineering

Other libraries often focus on "prompt engineering" as the process of writing a better string of text. `lingo` elevates this idea to **Context Engineering**.

The quality of an LLM's output is a function of the entire context it receives—not just a single prompt. This context includes few-shot examples, retrieved documents (RAG), summaries of past conversations, and explicit instructions.

`lingo` is designed to make the process of building this context **declarative, readable, and reusable**. You define *what* you want in the context, not *how* to imperatively construct it. This is achieved through the `PromptFlow` API, a pipeline of composable transformations.

## Installation

```bash
pip install lingo
```

## Core Concepts

  * **`Lingo`**: The core class providing access to the LLM for fundamental operations like `.chat()`, `.create()` (for Pydantic models), `.decide()`, and `.choose()`.
  * **`Transformation`**: A single, reusable operation that takes a list of messages and returns a new, modified list (e.g., `AddSystemMessage`, `AddKShotExamples`).
  * **`PromptFlow`**: A fluent API for chaining `Transformation` objects together into a pipeline. It supports branching, parallel execution, and subroutines, allowing you to define complex logic in a clean, readable way.

## A Small Example

This example demonstrates how to build a `PromptFlow` that dynamically decides how to answer a user's question about a programming language.

```python
import asyncio
from typing import List, Tuple
from lingo import Lingo, Message, PromptFlow

# --- 1. Setup the LLM and any data/functions you'll need ---
llm = Lingo(model="gpt-4-turbo")

def get_python_docs() -> str:
    """Simulates retrieving documentation for Python."""
    return "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability..."

def get_javascript_docs() -> str:
    """Simulates retrieving documentation for JavaScript."""
    return "JavaScript is a high-level, often just-in-time compiled language that conforms to the ECMAScript specification..."

# --- 2. Define reusable "subroutines" as PromptFlows ---

# A subroutine to answer a question using Python context
python_explainer = PromptFlow(llm).rag(get_python_docs).system_message(
    "Explain this concept to a beginner using the provided Python documentation."
)

# A subroutine to answer a question using JavaScript context
javascript_explainer = PromptFlow(llm).rag(get_javascript_docs).system_message(
    "Explain this concept to a beginner using the provided JavaScript documentation."
)

# --- 3. Build the main flow with branching logic ---

# This flow will first decide which language the user is asking about,
# then execute the appropriate subroutine.
main_flow = PromptFlow(llm).choose(
    prompt="Is the user asking about 'Python' or 'JavaScript'?",
    choices={
        "Python": python_explainer,
        "JavaScript": javascript_explainer,
    }
)

# --- 4. Execute the flow with an initial message ---

async def main():
    user_query = "What is a decorator?"
    initial_messages = [Message.user(user_query)]
    
    # Execute the flow to get the final, engineered context
    final_messages = await main_flow.execute(initial_messages)
    
    # Send the rich context to the LLM for the final answer
    response = await llm.chat(final_messages)
    
    print(f"User Query: {user_query}")
    print("\n--- Lingo's Final Response ---")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

## Contributing

Contributions are welcome\! `lingo` is an open-source project, and we'd love your help in making it better. Please feel free to open an issue or submit a pull request.

## License

`lingo` is licensed under the **MIT License**. See the LICENSE file for details.

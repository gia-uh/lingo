# Architectural Design: Lingo Internals

This document describes the high-level architecture and the technical design patterns used in the `lingo-ai` library.

## 🏗️ Core Components

The system is built on four primary abstractions:

### 1. `Lingo` (The Orchestrator)
The central hub of your application. It manages the chat session, maintains the global message history (`Context`), and coordinates the execution of the "Main Flow." It also serves as the registry for all skills and tools.

### 2. `Engine` (The Actuator)
The interface between your code and the Large Language Model (LLM). Unlike a simple wrapper, the Engine provides high-level semantic primitives:
- **`ask()`**: Pauses execution to wait for user input.
- **`decide()`**: Forces the LLM to return a boolean based on a question.
- **`choose()`**: Forces the LLM to select an option from a list.
- **`create()`**: Uses Pydantic models for structured generation.
- **`equip/invoke()`**: Handles tool selection and execution.

### 3. `Flow` & `Node` (The Workflow Engine)
A declarative execution engine built on the **Composite pattern**. Every skill in Lingo is a `Flow`.
- **`Flow`**: A container of nodes.
- **`Node`**: A discrete unit of execution (e.g., a function call, a reply, a branch).
Flows are async-native and can be nested. They support complex control structures like parallel forks (`Fork`), retries (`Retry`), and state machines (`StateMachine`).

### 4. `Context` (The Mutable Ledger)
The source of truth for the conversation history. It encapsulates a list of `Message` objects and provides mechanisms for transactional operations:
- **`fork()`**: Creates a temporary state branch that can be discarded.
- **`atomic()`**: Ensures a series of operations are rolled back if an error occurs.

---

## 🛠️ Design Patterns

### 📋 Registry Pattern & Dependency Injection
Lingo uses the `purely` package for a registry-based dependency injection system. When a tool or skill is defined, it can request dependencies using `depends()`:

```python
@bot.tool
def my_tool(query: str, llm=depends(LLM), state=depends(MyState)):
    # The 'llm' and 'state' are automatically injected from the Lingo registry.
```

### 🌉 Composite Pattern
The `Flow` API uses the composite pattern to build complex execution trees. A `Flow` can contain other `Flows`, `States`, or `Nodes`, treating them as a single executable unit.

### 🔄 The Event-Loop / Runner Architecture
Lingo execution follows a "Run-Pause-Resume" cycle:
1.  **Run**: The `FlowRunner` starts executing nodes sequentially.
2.  **Pause**: When an `engine.input()` (via `ask()`) is encountered, the runner yields control back to the `Lingo.chat` loop.
3.  **Resume**: When the next user message arrives, the runner resumes from the exact point it was paused, with all local variables preserved.

---

## 🏗️ Technology Stack

- **Pydantic v2**: Core data validation for `State`, `Message`, and `Engine.create`.
- **OpenAI API**: Default LLM interface (supports structured generation and tool calling).
- **PyYAML**: Serialization for rendering `State` objects into prompt context.
- **Uv**: Preferred package manager for fast, reproducible environments.

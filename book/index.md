# lingo — The Book

A literate-programming guide to `lingo`, a Python library for context engineering.
Every code block in this book is executable. Run `make book` to compile and verify the full guide.

## What lingo is

`lingo` is a library for building LLM-powered applications.
Its core idea is that conversations are structured as typed messages inside a mutable `Context`,
operations on that context are performed by an `Engine` that wraps an `LLM`,
and the sequence of operations is described as a composable `Flow`.

```
LLM ──► Engine ──► Flow ──► Context
                              │
                              ▼
                         Lingo (orchestrator)
```

The top-level `Lingo` class is the chatbot facade: it owns a conversation history,
builds flows from skills, and exposes a single `.chat(msg)` coroutine.

## Chapters

1. [Hello, lingo](01-hello-lingo.md) — LLM, Message, and your first chatbot
2. [Messages and Context](02-messages-and-context.md) — Multimodal content, context manipulation
3. [The Engine](03-the-engine.md) — reply, decide, choose, create, ask
4. [Flows](04-flows.md) — Declarative, chainable workflows
5. [Tools](05-tools.md) — Functions as LLM-callable tools
6. [Skills and Routing](06-skills-and-routing.md) — Multi-skill bots
7. [State](07-state.md) — Conversation state with atomic semantics
8. [Patterns](08-patterns.md) — End-to-end examples

## Running the book

```bash
make book          # compile + test all chapters
make book-compile  # compile only (no tests)
```

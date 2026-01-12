---
number: 11
title: "Version 1.0 Roadmap — Resilient, Multi-Modal & Type-Safe Orchestration"
state: open
labels:
---

## Overview

Lingo 1.0 focuses on three core pillars: **Reliability** (error recovery and branching), **Performance** (parallel execution), and **Developer Experience** (type-safety and YAML serialization). We are moving away from simple linear chains toward a robust "Composite" architecture that supports complex agentic cycles.

## 1. Core Architecture Enhancements

### Typed Flows (`Flow[T]`)

Transform the `Flow` class into a generic `Flow[T]`.

* **Type Safety**: Use Python generics to track the return type of a flow through the fluent API.
* **Stateful Execution**: The `run(engine, context)` method will modify the `Context` in-place but return the specific value produced by the final node in the sequence.

### Context Management

* **Persistent History**: Add `to_json()` and `from_json()` to `Context` for conversation serialization.
* **Refined Forking**: Leverage the existing `context.fork()` to support isolated "scratchpad" reasoning where intermediate turns are discarded, but conclusions are integrated back into the main branch.

## 2. Advanced Logic & Control Flow

We will introduce a set of semantic logic nodes to replace basic conditionals:

* **`When` Node**: A binary branching primitive (`.when(prompt, then, otherwise)`) that uses `engine.decide`.
* **`Branch` Node**: A multi-way switch/case primitive. It will support both dictionary mappings and keyword arguments (`**branches`) for syntax sugar, using `engine.choose` for selection.
* **`Repeat` Node**: An iterative loop with a `max_repeats` safety limit and a break condition that can be a simple callable or another sub-flow.

## 3. Resilience & Agentic Autonomy

### Multi-Tool Support

* **Multi-Equip**: Update `Engine.equip` to allow the model to select a list of tools simultaneously for complex tasks.
* **Parallel `Invoke**`: An `Invoke` node capable of running multiple equipped tools concurrently via `asyncio.gather`.

### Self-Healing Workflows

* **`Act` Node**: Syntactic sugar combining `Equip` and `Invoke` into a single "agentic turn".
* **`Recover` Node**: A "try-except" for flows. If a node fails, it captures the exception, forks the context, and runs a recovery flow to attempt a fix without polluting the main history.

## 4. Parallel Orchestration (Map-Reduce)

* **`Fork(*flows, aggregator)`**: Launch multiple sub-flows in parallel forks.
* **Recursive Aggregators**: The `aggregator` will be a standard `Node` or `Flow`. Results from parallel branches will be injected into `context.metadata["fork_results"]` for the aggregator to synthesize.

## 5. YAML Serialization (Federalized Flows)

Allow entire workflows to be defined in YAML files for portability and non-developer editing.

* **Federalization**: Every node class will implement `to_dict()` and `from_dict()`.
* **Registry System**: A centralized registry to map string identifiers in YAML to Python `Tool` objects and `BaseModel` schemas.

## 6. Auditing & Multi-Modality

* **Usage Tracking**: Add a `Usage` object to assistant messages to track `prompt_tokens`, `completion_tokens`, and `total_tokens`.
* **Observability**: Add an `on_message` callback to the `LLM` class for global auditing and tracing.
* **Enhanced Multi-Modality**: Fully integrate `Image`, `Audio`, `Video`, and `File` content parts into the `Message` lifecycle.

---

## Implementation Priority

1. **Generics & Typing**: Update `Flow` and `Node` to support `Flow[T]`.
2. **Logic Nodes**: Implement `When`, `Branch`, and `Repeat`.
3. **Resilience**: Implement `Recover` and the parallel `Fork` with aggregator support.
4. **Serialization**: Build the YAML layer (`to_dict`/`from_dict`).
5. **Auditing**: Integrate `Usage` tracking and `on_message` hooks.
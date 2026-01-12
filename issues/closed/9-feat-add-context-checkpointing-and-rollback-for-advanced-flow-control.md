---
number: 9
title: "Feat: Add context checkpointing and rollback for advanced flow control"
state: closed
labels:
---

### Description

Developers building complex agents or diagnostic workflows often need to explore multiple reasoning paths in a single conversation. If a path fails or proves unproductive, they should be able to revert the conversation state to a prior point and try an alternative, without leaving partial or misleading messages in the history.

Currently, this requires manual use of `Context.clone()` in imperative code, which breaks the declarative flow style and becomes cumbersome for dynamic, multi-step backtracking.

### Proposed Solution

Add two lightweight, explicit methods to the `Context` class:

- `checkpoint() -> int`: Saves the current message list and returns a token (e.g., an index).
- `rollback_to(token: int) -> None`: Reverts the context’s message list to the state saved at `token`.

This enables in-memory “undo” semantics for use cases like:
- Agent reasoning with trial-and-error
- Diagnostic flows that test and discard hypotheses
- Interactive conversation repair

The implementation is minimal:
- Initialize a new private attribute `_checkpoints: list[list[Message]]` in `Context.__init__`
- `checkpoint()` appends a shallow copy of `_messages` to `_checkpoints`
- `rollback_to()` restores `_messages` from `_checkpoints[token]`

### Key Benefits

- Keeps conversation history logically consistent with only the actually followed path
- Complements `clone()` by enabling active, iterative editing (vs. passive isolation)
- Remains unopinionated—developers choose when and how to use it

### Relevant Files

- `lingo/context.py`: Add `_checkpoints` in `__init__`, and implement `checkpoint()` and `rollback_to()` methods
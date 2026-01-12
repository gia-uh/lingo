---
number: 1
title: "Branching nodes (`Decide`, `Choose`) lack atomicity and can pollute context on failure"
state: closed
labels:
---

**Description**

The current implementation of the `Decide` and `Choose` nodes executes their sub-nodes (e.g., the `on_true` or `on_false` branch) directly on the main `Context` object.

This creates a significant problem: the branch execution is not atomic. If a sub-flow partially succeeds but then fails, it leaves the main `Context` in a corrupted, "polluted" state.

**Example Failure**

1.  A `Flow` uses `.decide(on_true=sub_flow, ...)`
2.  The `sub_flow` is a `Sequence` that first runs `.system("Adding info...")` and then `.invoke(failing_tool)`.
3.  The `Decide` node executes `sub_flow`.
4.  The `.system("Adding info...")` node *succeeds*, and the system message is added to the main `context.messages` list.
5.  The `.invoke(failing_tool)` node *fails*, raising an exception.
6.  The exception propagates, but the main `Context` is now permanently modified. The message "Adding info..." is still in the message history, even though the logical branch it belonged to failed and did not complete.

This polluted context can confuse subsequent LLM calls, which will see messages from a logical path that was never successfully finished.

**Proposed Solution**

Branching nodes should not execute on the shared `Context`. They should use `context.clone()` to create an isolated, independent context for the sub-flow.

If the sub-flow executes *successfully*, its new message list should be merged back into the parent `Context`. If the sub-flow *fails*, the cloned context is simply discarded, and the parent `Context` remains clean, as if the branch was never attempted.

This would make branch execution "atomic" and prevent failed branches from corrupting the main conversation state.

**Relevant Files**

* `lingo/flow.py`: See `Decide.execute` and `Choose.execute`.
* `lingo/context.py`: See the `clone()` method, which should be used to isolate the branch.
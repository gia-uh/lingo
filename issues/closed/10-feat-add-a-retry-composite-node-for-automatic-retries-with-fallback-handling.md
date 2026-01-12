---
number: 10
title: "Feat: Add a Retry composite node for automatic retries with fallback handling"
state: closed
labels:
---

### Description

When using tools (e.g., via `Invoke`), transient failures—such as network errors, rate limits, or temporary service outages—are common. Currently, if a node fails:

- Execution stops.
- The context remains clean (thanks to atomic branch execution).
- But there is **no built-in way to retry the operation** or gracefully handle the failure within the declarative `Flow` API.

Developers must fall back to imperative loops or custom error-handling logic outside the flow, breaking composition and readability.

### Proposed Solution

Introduce a new composite `Node` called `Retry` that wraps another node and automatically retries it on failure.

Usage example:
```python
flow = Flow().then(
    Retry(
        Invoke(weather_tool),
        max_attempts=3,
        on_failure=Flow().append("Service is temporarily unavailable.")
    )
)
```

**Behavior:**
- On each attempt, executes the wrapped node in an isolated context (via `context.clone()`).
- If successful, commits the result to the parent context and exits.
- If all attempts fail, executes the optional `on_failure` node (which runs on the original, clean context).
- Propagates the last exception if `on_failure` is not provided.

### Key Benefits

- Makes flows resilient to transient tool failures
- Keeps error handling inside the `Flow` API no imperative workarounds
- Leverages existing `clone()` mechanism for context safety

### Relevant Files

- `lingo/flow.py`: Add `Retry` node class and  `.retry(...)` method to the `Flow` fluent API
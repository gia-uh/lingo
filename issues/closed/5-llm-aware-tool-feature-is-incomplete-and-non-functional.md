---
number: 5
title: "LLM-aware tool feature is incomplete and non-functional"
state: closed
labels:
---

**Description**

The codebase has logic that suggests an "LLM-aware" tool feature, where a tool function can type-hint `LLM` to get an instance of the client injected. However, the pieces of this feature are not connected, and it will not work as expected.

**The Problem**

There are three disconnected parts:

1.  **The Hiding Logic:** `DelegateTool.parameters()` correctly identifies an `LLM` type hint and hides it from the parameter list shown to the model. This is correct.
2.  **The Injection Logic:** `LLM.wrap()` is a decorator that correctly wraps a function to inject `self` (the `LLM` instance) as an argument.
3.  **The Missing Connection:** The `@tool` decorator *does not* use the `LLM.wrap()` method. Furthermore, `Context.invoke()` simply calls `tool.run()` without any special handling.

As a result, if a developer creates a tool like this:

```python
@tool
async def my_tool(query: str, llm: LLM):
    # This will fail
    response = await llm.chat(...)
    return response.content
```

It will fail at runtime with a `TypeError: my_tool() missing 1 required positional argument: 'llm'`, because nothing is actually injecting the `LLM` instance.

**Proposed Solution**

The logic needs to be connected. The `LLM` class is the correct entity to manage this, as it's the one that should be injected.

A potential solution is to have `Context.invoke` call `llm.wrap(tool.run)` instead of calling `tool.run` directly. This would allow the `LLM` instance associated with the context to inject itself into the tool call.

Alternatively, the `@tool` decorator itself could be modified, but this is more complex as the `Tool` object is not aware of the specific `LLM` instance that will be used at runtime. The injection should happen at call time, in the `Context`.

**Relevant Files**

  * `lingo/llm.py`: See `LLM.wrap()`.
  * `lingo/tools.py`: See the `@tool` decorator and `DelegateTool.parameters()`.
  * `lingo/context.py`: See `Context.invoke()`, where the `tool.run()` call happens.
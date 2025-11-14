---
number: 3
title: "Structured generation (`.create`) lacks streaming, leading to poor UX"
state: closed
labels:
---

**Description**

The `LLM` class provides an excellent streaming `chat()` method with a token callback, which is ideal for responsive, conversational feedback.

However, the `LLM.create()` method—which powers *all* structured data generation (including `context.decide`, `context.choose`, `context.invoke`, and `context.create`)—is non-streaming. It relies on `client.beta.chat.completions.parse`, which only returns a value *after* the model has generated the full, complete JSON object.

**The Problem**

This creates a jarring and inconsistent user experience:

* A standard `flow.reply()` will stream tokens to the user.
* A `flow.decide()` or `flow.invoke()` will cause the application to "hang" for several seconds with no feedback. The `LLM`'s `callback` is never fired, making it useless for the most complex and time-consuming operations.

This defeats the purpose of an "async-native" framework, as the most important LLM calls are effectively blocking.

**Proposed Solution**

The `LLM.create()` method should be refactored to use the standard streaming `client.chat.completions.create` endpoint (just like `LLM.chat()`), but with the `response_format` set to JSON.

While this means the client will not *guarantee* valid Pydantic parsing, the `LLM` class can (and should) take on the responsibility of:

1.  Streaming the raw JSON token chunks.
2.  Aggregating the full JSON string.
3.  Parsing the string into the target Pydantic model (`model.model_validate_json(full_string)`).

This approach would:
* Allow the `callback` to fire for *all* LLM interactions, including JSON generation (e.g., streaming the raw `{"reasoning": "...",` tokens).
* Provide a consistent, responsive, stream-first user experience.
* Keep all streaming logic encapsulated within the `LLM` class.

**Relevant Files**

* `lingo/llm.py`: See `LLM.chat` (streaming) vs. `LLM.create` (non-streaming).
* `lingo/context.py`: All methods using `.create()` are affected (e.g., `decide`, `choose`, `invoke`, `create`).
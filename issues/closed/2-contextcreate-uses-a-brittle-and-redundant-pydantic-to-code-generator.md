---
number: 2
title: "`Context.create` uses a brittle and redundant Pydantic-to-code generator"
state: closed
labels:
---

**Description**

The `Context.create()` method uses a helper function, `generate_pydantic_code`, to dynamically generate a Python class signature as a string. This string is then passed to the LLM in the prompt to instruct it on the data format to return.

**The Problem**

This approach has two major flaws:

1.  **It is extremely brittle:** The `generate_pydantic_code` utility relies on introspecting internal `typing` attributes like `__origin__`, `__args__`, and `_name`. This is fragile and will likely fail with more complex types, forward references, or future changes to Pydantic or Python's typing system.
2.  **It is redundant:** The prompt for `Context.create()` *already* includes `model.model_json_schema()`. The JSON schema is the robust, machine-readable, and industry-standard way to define the shape of the expected JSON data. The additional, reverse-engineered Python code string is unnecessary, adds prompt tokens, and is a significant source of potential bugs.

**Proposed Solution**

The `generate_pydantic_code` utility should be removed entirely.

The `DEFAULT_CREATE_PROMPT` and the `Context.create()` method should be simplified to *only* use the `model_json_schema()`. The JSON schema provides all the information the LLM needs to generate the correct output format.

This will make the `create` feature more robust, less error-prone, and more efficient by reducing prompt size.

**Relevant Files**

* `lingo/context.py`: See the `create` method's prompt assembly.
* `lingo/utils.py`: See the `generate_pydantic_code` and `type_to_str` functions.
* `lingo/prompts.py`: See `DEFAULT_CREATE_PROMPT` (which uses the `{signature}`).
---
number: 4
title: "`Context.invoke` parameter generation logic is overly complex and error-prone"
state: closed
labels:
---

**Description**

The `Context.invoke` method implements a complex "diffing" logic to generate parameters for a `Tool`. It identifies which parameters are *missing* from `**kwargs`, dynamically creates a Pydantic model *only* for those missing parameters, and then asks the LLM to fill in that partial model.

**The Problem**

This design is complex and can lead to validation errors:

1.  **Loss of Context:** The LLM is only asked to generate a subset of parameters in isolation. It doesn't see the *full* tool signature or the parameters that were *already provided* (which are passed as "defaults" in the prompt, but this is less effective than including them in the schema).
2.  **Co-dependency Failures:** This can easily fail. For example, if a tool has two co-dependent arguments like `start_date` and `end_date`, and the user provides `start_date` in `**kwargs`, the LLM will be asked to generate `end_date` *without* knowing what `start_date` is. This makes it impossible for the LLM to ensure `end_date` is after `start_date`.

**Proposed Solution**

The logic should be simplified and made more robust. Instead of dynamically creating a partial model, `Context.invoke` should:

1.  Create a Pydantic model for the *entire* set of the tool's parameters.
2.  Pass this *full* model to `llm.create()`. The prompt should instruct the LLM to fill in all parameters based on the conversation.
3.  Once the LLM returns the *full* parameter model, `Context.invoke` can then merge this with the `**kwargs` provided, with `kwargs` taking precedence.

This approach gives the LLM the full context of *all* required arguments, allowing it to generate a valid, co-dependent set of parameters, which are then overridden by any explicit user-provided values.

**Relevant Files**

* `lingo/context.py`: See the `invoke` method.
* `lingo/prompts.py`: See `DEFAULT_INVOKE_PROMPT`.
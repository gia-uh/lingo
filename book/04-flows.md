# Chapter 4: Flows

A `Flow` is a declarative, chainable sequence of steps executed against a
`Context` by an `Engine`. Each step adds, transforms, or queries the context.

The `@flow` decorator converts any `async def f(context, engine)` function into a `Flow`.
The `Flow` class itself provides a fluent builder API.

## Sequential steps

`reply()` appends an LLM response. `append()` and `prepend()` inject static content.

```python {name=ch04_test_sequential}
@pytest.mark.asyncio
async def test_sequential_flow():
    f = (
        Flow("greeter")
        .append("Greet the user warmly.")
        .reply()
    )
    c = Context([Message.user("Hello")])
    await f.execute(c, Engine(MockLLM(["Hi there!"])))
    assert c.messages[-1].role == "assistant"
    assert c.messages[-1].content == "Hi there!"
```

## Conditional branching — when

`when(prompt, then, otherwise)` makes a boolean decision and routes to one of
two sub-flows. Under the hood it calls `engine.decide()`. In tests, mock that:

```python {name=ch04_test_when}
@pytest.mark.asyncio
async def test_when_takes_true_branch():
    positive = Flow("positive").append("Response is positive.")
    negative = Flow("negative").append("Response is negative.")

    f = Flow("sentiment").when("Is the sentiment positive?", then=positive, otherwise=negative)
    c = Context([Message.user("I love it!")])

    engine = Engine(MockLLM())
    cot = MagicMock(); cot.result = True
    engine.create = AsyncMock(return_value=cot)

    await f.execute(c, engine)
    assert any("positive" in str(m.content) for m in c.messages)
```

## Multi-way branching — branch

`branch(prompt, **choices)` routes to one of N sub-flows based on the LLM's
string choice:

```python {name=ch04_test_branch}
@pytest.mark.asyncio
async def test_branch_routes_to_correct_choice():
    math_flow  = Flow("math").append("Doing math.")
    code_flow  = Flow("code").append("Writing code.")

    f = Flow("router").branch("Which topic?", math=math_flow, code=code_flow)
    c = Context([Message.user("Help me with Python")])

    engine = Engine(MockLLM())
    cot = MagicMock(); cot.result = "code"
    engine.create = AsyncMock(return_value=cot)

    await f.execute(c, engine)
    assert any("Writing code" in str(m.content) for m in c.messages)
```

## Looping — repeat

`repeat(body, until, max_repeats)` runs `body` until the `until` condition
returns `True` or `max_repeats` is reached.

```python {name=ch04_test_repeat}
@pytest.mark.asyncio
async def test_repeat_runs_at_least_once():
    counter = {"n": 0}

    @flow
    async def increment(context: Context, engine: Engine):
        counter["n"] += 1

    engine = Engine(MockLLM())
    cot = MagicMock(); cot.result = True
    engine.create = AsyncMock(return_value=cot)

    f = Flow("loop").repeat(increment, until="Stop now?", max_repeats=3)
    c = Context([Message.user("go")])
    await f.execute(c, engine)
    assert counter["n"] >= 1
```

## Parallel branches — fork

`fork(*flows, aggregator)` runs multiple sub-flows in parallel on clones of
the current context. Their results are synthesized by an aggregator flow (or prompt).

```python {name=ch04_test_fork}
@pytest.mark.asyncio
async def test_fork_runs_branches_in_parallel():
    branch_a = Flow("a").reply("Branch A")
    branch_b = Flow("b").reply("Branch B")

    f = Flow("parallel").fork(branch_a, branch_b, aggregator="Combine the results.")
    c = Context([Message.user("Analyze from two angles.")])
    await f.execute(c, Engine(MockLLM(["A result", "B result", "Summary of both."])))
    assert c.messages[-1].role == "assistant"
```

## Compression — compress

`compress()` prunes context history to keep it within token budgets.
With `aggregator=None`, it acts as a sliding window; with an aggregator it summarizes.

```python {name=ch04_test_compress}
@pytest.mark.asyncio
async def test_compress_shrinks_context():
    long_history = [Message.user(f"msg {i}") for i in range(10)]
    c = Context(long_history)
    f = Flow("compress").compress(n=None, prefix_k=1, aggregator="Summarize.")
    await f.execute(c, Engine(MockLLM(["Summary here."])))
    # After compression: 1 prefix + 1 summary
    assert len(c.messages) <= 3
```

## Custom steps — @flow decorator

The `@flow` decorator is the escape hatch for any step that doesn't fit the
fluent API. The function receives `(context, engine)` directly.

```python {name=ch04_test_custom}
@flow
async def my_custom_step(context: Context, engine: Engine):
    """Logs the last user message and adds a note."""
    last = context.messages[-1]
    context.append(f"[log] Last message was: {last.content}")

@pytest.mark.asyncio
async def test_custom_flow_step():
    c = Context([Message.user("test input")])
    await my_custom_step.execute(c, Engine(MockLLM(["ok"])))
    assert any("[log]" in str(m.content) for m in c.messages)
```

## Test file

```python {export=tests/test_ch04.py}
import pytest
from unittest.mock import AsyncMock, MagicMock
from lingo import Message, Flow, flow
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM

<<ch04_test_sequential>>
<<ch04_test_when>>
<<ch04_test_branch>>
<<ch04_test_repeat>>
<<ch04_test_fork>>
<<ch04_test_compress>>
<<ch04_test_custom>>
```

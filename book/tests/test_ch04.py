import pytest
from unittest.mock import AsyncMock, MagicMock
from lingo import Message, Flow, flow
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM

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

@pytest.mark.asyncio
async def test_fork_runs_branches_in_parallel():
    branch_a = Flow("a").reply("Branch A")
    branch_b = Flow("b").reply("Branch B")

    f = Flow("parallel").fork(branch_a, branch_b, aggregator="Combine the results.")
    c = Context([Message.user("Analyze from two angles.")])
    await f.execute(c, Engine(MockLLM(["A result", "B result", "Summary of both."])))
    assert c.messages[-1].role == "assistant"

@pytest.mark.asyncio
async def test_compress_shrinks_context():
    long_history = [Message.user(f"msg {i}") for i in range(10)]
    c = Context(long_history)
    f = Flow("compress").compress(n=None, prefix_k=1, aggregator="Summarize.")
    await f.execute(c, Engine(MockLLM(["Summary here."])))
    # After compression: 1 prefix + 1 summary
    assert len(c.messages) <= 3

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


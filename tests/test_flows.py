# tests/test_flows.py
import pytest
from lingo import Flow, Engine, Context, Message
from lingo.mock import MockLLM
from pydantic import BaseModel


class User(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_value_return_vs_context_mutation():
    """Verify that calling the flow returns T, while execute modifies Context."""
    mock = MockLLM(["Hello World"])
    engine = Engine(mock)
    flow = Flow[str]().append("System Init").reply("Greet")

    # 1. Testing Value Return (returns the string from Reply)
    result_value = await flow.execute(Context([Message.user("Hi")]), engine)
    assert result_value == "Hello World"

    # 2. Testing Context Mutation (requires explicit Context and execute)
    ctx = Context([Message.user("Hi")])
    mock = MockLLM(["Hello World"])
    engine = Engine(mock)
    await flow.execute(ctx, engine)

    assert len(ctx.messages) == 3  # User + System Init + Assistant Reply
    assert ctx.messages[-1].content == "Hello World"


@pytest.mark.asyncio
async def test_atomic_retry_rollback():
    """Verify that retry rolls back context on failure before succeeding."""
    mock = MockLLM(["Retrying...", "Success"])

    # Custom engine to simulate a failure on the first try
    class FailOnceEngine(Engine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempts = 0

        async def reply(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise ValueError("Transient Error")
            return await super().reply(*args, **kwargs)

    engine = FailOnceEngine(mock)
    fixer = Flow().reply("Fix the error")

    # Flow that should return a string
    flow = Flow[str]().reply("Try this").retry(fixer, max_retries=2)

    ctx = Context([Message.user("Start")])
    final_val = await flow.execute(ctx, engine)

    assert final_val == "Success"
    # History should NOT contain the failed 'Try this' turn thanks to atomic()
    # It should contain: [User, FixerMessage, SuccessReply]
    assert "Retrying..." in str(ctx.messages[1].content)
    assert "Success" in str(ctx.messages[2].content)


@pytest.mark.asyncio
async def test_parallel_fork_isolation():
    """Verify that fork isolates branches and returns the aggregator's result."""
    mock = MockLLM(["Result A", "Result B", "Synthesis"])
    engine = Engine(mock)

    flow = Flow[str]().fork(
        Flow().reply("Branch A"), Flow().reply("Branch B"), aggregator="Summarize"
    )

    ctx = Context([Message.user("Input")])
    final_summary = await flow.execute(ctx, engine)

    assert final_summary == "Synthesis"
    # Main context should only have original input + 1 assistant summary
    assert len(ctx.messages) == 2
    assert ctx.messages[-1].role == "assistant"


@pytest.mark.asyncio
async def test_compression_pruning():
    """Verify that compress modifies context length in-place."""
    mock = MockLLM(["Short Summary"])
    engine = Engine(mock)

    flow = (
        Flow()
        .append("Instruction 1")
        .append("Instruction 2")
        .compress(prefix_k=1)  # Keep 1st msg, summarize rest
    )

    ctx = Context([])
    await flow.execute(ctx, engine)

    # Original length was 2. Now should be [Instruction 1, Summary]
    assert len(ctx.messages) == 2
    assert "SUMMARY" in str(ctx.messages[1].content)

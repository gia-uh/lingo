"""Unit tests for lingo/engine.py.

Covers all public methods of Engine without hitting any real LLM.
Uses MockLLM (lingo.mock) for structured-output paths and AsyncMock
for the chat path.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from lingo.engine import Engine, INPUT_SIGNAL
from lingo.context import Context
from lingo.llm import Message, LLM
from lingo.tools import Tool, ToolResult
from lingo.mock import MockLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_context(*messages: Message) -> Context:
    return Context(list(messages))


def empty_ctx() -> Context:
    return Context([Message.user("hello")])


class MyModel(BaseModel):
    """A simple test model."""

    value: str


class SimpleTool(Tool):
    """A minimal concrete Tool for testing."""

    def __init__(self, name: str = "simple_tool", should_raise: bool = False):
        super().__init__(name, f"Does {name}")
        self._should_raise = should_raise

    def parameters(self) -> dict:
        return {"x": str}

    async def run(self, **kwargs):
        if self._should_raise:
            raise RuntimeError("tool error")
        return f"ran with {kwargs}"


# ---------------------------------------------------------------------------
# Engine.__init__
# ---------------------------------------------------------------------------


def test_init_stores_llm_and_tools():
    llm = MockLLM()
    t1 = SimpleTool("t1")
    engine = Engine(llm, [t1])

    assert engine._llm is llm
    assert engine._tools == [t1]


def test_init_creates_queues():
    engine = Engine(MockLLM())
    assert isinstance(engine._input_queue, asyncio.Queue)
    assert isinstance(engine._signal_queue, asyncio.Queue)


def test_init_no_tools_defaults_to_empty():
    engine = Engine(MockLLM())
    assert engine._tools == []


# ---------------------------------------------------------------------------
# Engine.scope
# ---------------------------------------------------------------------------


def test_scope_returns_new_engine_with_combined_tools():
    llm = MockLLM()
    t1 = SimpleTool("t1")
    t2 = SimpleTool("t2")
    t3 = SimpleTool("t3")

    base = Engine(llm, [t1])
    scoped = base.scope([t2, t3])

    assert scoped is not base
    assert scoped._llm is llm
    assert set(t.name for t in scoped._tools) == {"t1", "t2", "t3"}
    # Original engine is not mutated
    assert len(base._tools) == 1


def test_scope_same_llm_instance():
    llm = MockLLM()
    engine = Engine(llm)
    scoped = engine.scope([SimpleTool()])
    assert scoped._llm is llm


# ---------------------------------------------------------------------------
# Engine._expand_content
# ---------------------------------------------------------------------------


def test_expand_content_string_instruction():
    engine = Engine(MockLLM())
    ctx = Context([Message.user("hi")])
    msgs = engine._expand_content(ctx, "be concise")
    assert msgs[-1].role == "system"
    assert msgs[-1].content == "be concise"


def test_expand_content_message_instruction():
    engine = Engine(MockLLM())
    ctx = Context([Message.user("hi")])
    extra = Message.system("extra instruction")
    msgs = engine._expand_content(ctx, extra)
    assert msgs[-1] is extra


def test_expand_content_basemodel_instruction():
    engine = Engine(MockLLM())
    ctx = Context([Message.user("hi")])
    obj = MyModel(value="test-value")
    msgs = engine._expand_content(ctx, obj)
    assert msgs[-1].role == "system"
    assert "test-value" in msgs[-1].content


def test_expand_content_includes_context_messages():
    engine = Engine(MockLLM())
    m1 = Message.user("first")
    m2 = Message.assistant("second")
    ctx = Context([m1, m2])
    msgs = engine._expand_content(ctx)
    assert msgs[0] is m1
    assert msgs[1] is m2


def test_expand_content_multiple_instructions():
    engine = Engine(MockLLM())
    ctx = Context([Message.user("q")])
    msgs = engine._expand_content(ctx, "inst1", "inst2", "inst3")
    assert len(msgs) == 4  # 1 ctx + 3 instructions
    assert msgs[1].content == "inst1"
    assert msgs[2].content == "inst2"
    assert msgs[3].content == "inst3"


# ---------------------------------------------------------------------------
# Engine.reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_calls_llm_chat():
    response_msg = Message.assistant("hello back")
    llm = MockLLM(responses=[response_msg])
    engine = Engine(llm)
    ctx = Context([Message.user("hello")])

    result = await engine.reply(ctx)

    assert result is response_msg
    assert len(llm.history) == 1


@pytest.mark.asyncio
async def test_reply_passes_expanded_messages():
    response_msg = Message.assistant("ok")
    llm = MockLLM(responses=[response_msg])
    engine = Engine(llm)
    ctx = Context([Message.user("q")])

    await engine.reply(ctx, "extra instruction")

    sent = llm.history[0]
    assert len(sent) == 2
    assert sent[-1].content == "extra instruction"


# ---------------------------------------------------------------------------
# Engine.input / wait / put  (queue synchronisation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_sends_signal_before_waiting():
    engine = Engine(MockLLM())

    # Arrange: pre-load the input queue so input() returns immediately
    await engine._input_queue.put("user typed this")

    result = await engine.input()

    # Signal should have been placed on the signal queue
    assert not engine._signal_queue.empty()
    signal = await engine._signal_queue.get()
    assert signal is INPUT_SIGNAL

    assert result == "user typed this"


@pytest.mark.asyncio
async def test_put_unblocks_input():
    """put() should unblock a coroutine that is awaiting input()."""
    engine = Engine(MockLLM())

    received: list[str] = []

    async def reader():
        received.append(await engine.input())

    # Start reader (it will block on _input_queue)
    task = asyncio.ensure_future(reader())
    # Drain the signal so reader doesn't stay blocked after put
    await asyncio.sleep(0)  # let reader reach _signal_queue.put

    # Consume the INPUT_SIGNAL
    signal = await engine._signal_queue.get()
    assert signal is INPUT_SIGNAL

    # Now unblock it
    await engine.put("unblocked!")
    await task

    assert received == ["unblocked!"]


@pytest.mark.asyncio
async def test_wait_blocks_until_signal():
    engine = Engine(MockLLM())

    finished = []

    async def waiter():
        await engine.wait()
        finished.append(True)

    task = asyncio.ensure_future(waiter())
    await asyncio.sleep(0)  # let waiter block

    assert not finished

    # Unblock by sending any object to the signal queue
    await engine._signal_queue.put(INPUT_SIGNAL)
    await task

    assert finished == [True]


# ---------------------------------------------------------------------------
# Engine.ask
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_calls_reply_then_input():
    engine = Engine(MockLLM())

    reply_result = Message.assistant("What's your name?")
    input_result = "Alice"

    # Mock both halves
    engine.reply = AsyncMock(return_value=reply_result)
    engine.input = AsyncMock(return_value=input_result)

    ctx = empty_ctx()
    result = await engine.ask(ctx, "What's your name?")

    engine.reply.assert_awaited_once_with(ctx, "What's your name?")
    engine.input.assert_awaited_once()
    assert result == input_result


# ---------------------------------------------------------------------------
# Engine.create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_returns_pydantic_model():
    expected = MyModel(value="from-llm")
    llm = MockLLM(responses=[expected])
    engine = Engine(llm)
    ctx = Context([Message.user("create it")])

    result = await engine.create(ctx, MyModel)

    assert isinstance(result, MyModel)
    assert result.value == "from-llm"


@pytest.mark.asyncio
async def test_create_appends_default_create_prompt():
    expected = MyModel(value="ok")
    llm = MockLLM(responses=[expected])
    engine = Engine(llm)
    ctx = Context([Message.user("q")])

    await engine.create(ctx, MyModel, "extra hint")

    sent_messages = llm.history[0]
    # The DEFAULT_CREATE_PROMPT is appended as the final message
    last_msg = sent_messages[-1]
    assert last_msg.role == "system"
    assert "MyModel" in last_msg.content


@pytest.mark.asyncio
async def test_create_passes_extra_instructions():
    expected = MyModel(value="ok")
    llm = MockLLM(responses=[expected])
    engine = Engine(llm)
    ctx = Context([Message.user("q")])

    await engine.create(ctx, MyModel, "hint A", "hint B")

    sent = llm.history[0]
    contents = [m.content for m in sent]
    assert any("hint A" in str(c) for c in contents)
    assert any("hint B" in str(c) for c in contents)


# ---------------------------------------------------------------------------
# Engine.choose
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_choose_returns_selected_option():
    options = ["alpha", "beta", "gamma"]
    engine = Engine(MockLLM())

    # Mock create to return a CoT-like object with result="beta"
    cot_result = MagicMock()
    cot_result.result = "beta"
    engine.create = AsyncMock(return_value=cot_result)

    ctx = empty_ctx()
    result = await engine.choose(ctx, options)

    assert result == "beta"


@pytest.mark.asyncio
async def test_choose_maps_result_back_to_original_object():
    """Non-string options must be mapped back via the str representation."""

    class Option:
        def __init__(self, n):
            self.n = n

        def __str__(self):
            return f"opt-{self.n}"

    opts = [Option(1), Option(2), Option(3)]
    engine = Engine(MockLLM())

    cot_result = MagicMock()
    cot_result.result = "opt-2"
    engine.create = AsyncMock(return_value=cot_result)

    ctx = empty_ctx()
    result = await engine.choose(ctx, opts)

    assert result is opts[1]


# ---------------------------------------------------------------------------
# Engine.decide
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_returns_true():
    engine = Engine(MockLLM())
    cot = MagicMock()
    cot.result = True
    engine.create = AsyncMock(return_value=cot)

    ctx = empty_ctx()
    assert await engine.decide(ctx, "Is this true?") is True


@pytest.mark.asyncio
async def test_decide_returns_false():
    engine = Engine(MockLLM())
    cot = MagicMock()
    cot.result = False
    engine.create = AsyncMock(return_value=cot)

    ctx = empty_ctx()
    assert await engine.decide(ctx, "Is this false?") is False


# ---------------------------------------------------------------------------
# Engine.equip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_equip_raises_with_no_tools():
    engine = Engine(MockLLM())
    ctx = empty_ctx()

    with pytest.raises(ValueError, match="No tools available"):
        await engine.equip(ctx)


@pytest.mark.asyncio
async def test_equip_single_tool_short_circuits():
    engine = Engine(MockLLM())
    t = SimpleTool("only_tool")
    ctx = empty_ctx()

    # create should NOT be called
    engine.create = AsyncMock()
    result = await engine.equip(ctx, t)

    assert result is t
    engine.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_equip_multiple_tools_calls_llm():
    t1 = SimpleTool("tool_one")
    t2 = SimpleTool("tool_two")
    engine = Engine(MockLLM())

    cot = MagicMock()
    cot.result = "tool_two"
    engine.create = AsyncMock(return_value=cot)

    ctx = empty_ctx()
    result = await engine.equip(ctx, t1, t2)

    assert result is t2


@pytest.mark.asyncio
async def test_equip_uses_engine_tools_when_none_passed():
    t1 = SimpleTool("engine_tool")
    engine = Engine(MockLLM(), [t1])

    ctx = empty_ctx()
    # Single engine tool → short-circuit
    result = await engine.equip(ctx)

    assert result is t1


# ---------------------------------------------------------------------------
# Engine.infer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_infer_returns_dict_with_generated_params():
    from pydantic import create_model

    t = SimpleTool("my_tool")
    engine = Engine(MockLLM())

    # Simulate create returning a model with x="generated"
    generated = MagicMock()
    generated.model_dump = MagicMock(return_value={"x": "generated"})
    engine.create = AsyncMock(return_value=generated)

    ctx = empty_ctx()
    result = await engine.infer(ctx, t)

    assert result == {"x": "generated"}


@pytest.mark.asyncio
async def test_infer_kwargs_take_precedence_over_generated():
    t = SimpleTool("my_tool")
    engine = Engine(MockLLM())

    generated = MagicMock()
    generated.model_dump = MagicMock(return_value={"x": "generated"})
    engine.create = AsyncMock(return_value=generated)

    ctx = empty_ctx()
    result = await engine.infer(ctx, t, x="override")

    assert result["x"] == "override"


# ---------------------------------------------------------------------------
# Engine.invoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_success_returns_tool_result():
    t = SimpleTool("my_tool")
    engine = Engine(MockLLM())
    engine.infer = AsyncMock(return_value={"x": "val"})

    ctx = empty_ctx()
    result = await engine.invoke(ctx, t)

    assert isinstance(result, ToolResult)
    assert result.tool == "my_tool"
    assert result.error is None
    assert result.result is not None


@pytest.mark.asyncio
async def test_invoke_exception_returns_tool_result_with_error():
    t = SimpleTool("bad_tool", should_raise=True)
    engine = Engine(MockLLM())
    engine.infer = AsyncMock(return_value={"x": "val"})

    ctx = empty_ctx()
    result = await engine.invoke(ctx, t)

    assert isinstance(result, ToolResult)
    assert result.tool == "bad_tool"
    assert result.error == "tool error"
    assert result.result is None


@pytest.mark.asyncio
async def test_invoke_passes_kwargs_to_infer():
    t = SimpleTool("my_tool")
    engine = Engine(MockLLM())

    infer_mock = AsyncMock(return_value={"x": "v"})
    engine.infer = infer_mock

    ctx = empty_ctx()
    await engine.invoke(ctx, t, x="forced")

    infer_mock.assert_awaited_once_with(ctx, t, x="forced")


# ---------------------------------------------------------------------------
# Engine.act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_act_combines_equip_and_invoke():
    t1 = SimpleTool("t1")
    t2 = SimpleTool("t2")
    engine = Engine(MockLLM())

    equip_mock = AsyncMock(return_value=t1)
    invoke_mock = AsyncMock(return_value=ToolResult(tool="t1", result="done"))
    engine.equip = equip_mock
    engine.invoke = invoke_mock

    ctx = empty_ctx()
    result = await engine.act(ctx, t1, t2)

    equip_mock.assert_awaited_once_with(ctx, t1, t2)
    invoke_mock.assert_awaited_once_with(ctx, t1)
    assert result.tool == "t1"
    assert result.result == "done"


# ---------------------------------------------------------------------------
# Engine.stop
# ---------------------------------------------------------------------------


def test_stop_raises_stop_flow():
    from lingo.flow import StopFlow

    engine = Engine(MockLLM())

    with pytest.raises(StopFlow):
        engine.stop()


# ---------------------------------------------------------------------------
# Integration: full equip→invoke pipeline with MockLLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_infer_pipeline():
    """Exercise infer end-to-end: create is called and kwargs override the result."""
    t = SimpleTool("my_tool")
    engine = Engine(MockLLM())

    # Patch create so we bypass the MockLLM type-matching issue for dynamic models.
    # We verify the merged output (kwargs win) and that create was actually called.
    from pydantic import create_model as _cm

    generated = MagicMock()
    generated.model_dump = MagicMock(return_value={"x": "generated"})
    create_mock = AsyncMock(return_value=generated)
    engine.create = create_mock

    ctx = empty_ctx()
    result = await engine.infer(ctx, t, x="override")

    create_mock.assert_awaited_once()
    # kwargs ("override") must win over generated value
    assert result["x"] == "override"


# ---------------------------------------------------------------------------
# Context-doubling regression tests (small-model context-window bug)
# ---------------------------------------------------------------------------
# decide/equip/infer/choose used to pre-expand the context with
# _expand_content() and then pass the expanded list as *instructions to
# Engine.create(), which calls _expand_content() again — sending 2*N
# context messages to the LLM instead of N.  For small models (Qwen 3.5 9b,
# etc.) with limited context windows this caused the agentic loop to break
# after the first tool call (tool results inflate context, doubled sends push
# past the model's limit).


def _ctx_with_tool_result() -> Context:
    """Three-message context simulating state after one tool call."""
    return Context([
        Message.system("You are an agent."),
        Message.user("Do the task"),
        Message.system("[Tool result] ran with {'x': 'hello'}"),
    ])


def _counting_llm() -> tuple[LLM, list[int]]:
    """Return a mock LLM and a list that accumulates message counts per create() call.

    The create() spy records how many messages were sent and then raises so the
    caller's try/except can handle it; only the count matters for these tests.
    """
    counts: list[int] = []
    llm = MagicMock(spec=LLM)

    async def fake_create(model, messages, **kwargs):
        counts.append(len(messages))
        raise StopIteration("count captured")  # bail out; caller catches

    llm.create = fake_create
    return llm, counts


@pytest.mark.asyncio
async def test_decide_does_not_double_context_messages():
    """decide() must send N+2 messages (context + instruction + schema), not 2N+2."""
    ctx = _ctx_with_tool_result()
    N = len(ctx.messages)  # 3

    llm, counts = _counting_llm()
    engine = Engine(llm)

    try:
        await engine.decide(ctx, "Is the task done?")
    except Exception:
        pass

    assert counts, "LLM.create was never called"
    # N context + 1 decide instruction + 1 decide-format prompt + 1 create schema = N+3
    assert counts[0] <= N + 4, (
        f"decide() sent {counts[0]} messages to LLM for N={N} context messages; "
        f"expected ≤{N + 4} (context was doubled: 2*{N} = {2*N})"
    )


@pytest.mark.asyncio
async def test_equip_does_not_double_context_messages():
    """equip() must send N+2 messages (context + equip_prompt + schema), not 2N+2."""
    ctx = _ctx_with_tool_result()
    N = len(ctx.messages)  # 3
    t1 = SimpleTool("t1")
    t2 = SimpleTool("t2")

    llm, counts = _counting_llm()
    engine = Engine(llm)

    try:
        await engine.equip(ctx, t1, t2)
    except Exception:
        pass

    assert counts, "LLM.create was never called"
    # N context + 1 equip prompt + 1 create schema = N+2
    assert counts[0] <= N + 3, (
        f"equip() sent {counts[0]} messages to LLM for N={N} context messages; "
        f"expected ≤{N + 3} (context was doubled: 2*{N} = {2*N})"
    )


@pytest.mark.asyncio
async def test_infer_does_not_double_context_messages():
    """infer() must send N+2 messages (context + invoke_prompt + schema), not 2N+2."""
    ctx = _ctx_with_tool_result()
    N = len(ctx.messages)  # 3
    t = SimpleTool("t1")

    llm, counts = _counting_llm()
    engine = Engine(llm)

    try:
        await engine.infer(ctx, t)
    except Exception:
        pass

    assert counts, "LLM.create was never called"
    # N context + 1 invoke prompt + 1 create schema = N+2
    assert counts[0] <= N + 3, (
        f"infer() sent {counts[0]} messages to LLM for N={N} context messages; "
        f"expected ≤{N + 3} (context was doubled: 2*{N} = {2*N})"
    )

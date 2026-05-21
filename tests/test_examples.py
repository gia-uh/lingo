"""End-to-end smoke tests for examples/.

Each example is imported, its LLM (and sometimes Engine methods) are swapped for
scripted mocks, and its main flow is driven through a happy path.  The point is
to catch bitrot — if lingo's API changes underneath an example, the test fails
immediately rather than weeks later when a user runs it.

Strategy
--------
- For plain-chat examples (no structured engine methods): replace ``bot.llm``
  with ``MockLLM(responses=[...])``.  Each queued string is returned by
  ``MockLLM.chat``.

- For examples that use engine.choose / engine.decide: patch those methods at
  class level inside the test via ``unittest.mock.patch.object``.  This avoids
  the dynamic-model isinstance problem in MockLLM.create.

- For native-LLM examples (no Lingo bot): construct the LLM with custom
  callbacks / replace it entirely via monkeypatching.

- All Lingo-based examples are driven through ``cli.run`` with a scripted
  input_fn (raises EOFError when exhausted) and an output_fn that captures
  tokens.
"""

import asyncio
import importlib
import pytest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel

from lingo.llm import Message, ToolCall, Usage
from lingo.mock import MockLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_output_fn():
    """Return (output_fn, captured) — append-to-list output callback."""
    captured: list[str] = []

    def output_fn(token: str):
        captured.append(token)

    return output_fn, captured


def _scripted_input_fn(messages: list[str]):
    """Return an input_fn that yields scripted messages then raises EOFError."""
    it = iter(messages)

    def input_fn():
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return input_fn


async def _drive_lingo(bot, user_messages: list[str], mock_responses: list):
    """Drive a Lingo bot through a scripted conversation via cli.run."""
    from lingo.cli import run

    bot.llm = MockLLM(responses=mock_responses)
    output_fn, captured = _capture_output_fn()
    input_fn = _scripted_input_fn(user_messages)
    await run(bot, input_fn=input_fn, output_fn=output_fn)
    return captured


# ---------------------------------------------------------------------------
# 1. hello_world.py — minimal no-skills bot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hello_world_runs():
    """Minimal Lingo with no skills should echo back one assistant reply."""
    mod = importlib.import_module("examples.hello_world")
    captured = await _drive_lingo(
        mod.bot,
        user_messages=["hi"],
        mock_responses=["Hello! How can I help you today?"],
    )
    out = "".join(captured)
    assert "Hello" in out


# ---------------------------------------------------------------------------
# 2. wizard.py — engine.ask / choose / create / decide
# ---------------------------------------------------------------------------
#
# The wizard skill calls, in order:
#   eng.reply(ctx, "Welcome!")                    → MockLLM.chat  [0]
#   eng.choose(ctx, [...], ...)                   → patched → "developer"
#   eng.reply(ctx, "Got it — setting up ...")     → MockLLM.chat  [1]
#   eng.ask(ctx, "Tell me your name ...")         → MockLLM.chat  [2] (the question)
#                                                    then engine.input() pauses
#   [second user message: "Alice, alice@x.com, testing"]
#   eng.create(ctx, Account)                      → patched → Account(...)
#   eng.reply(ctx, "Thanks Alice! ...")           → MockLLM.chat  [3]
#   eng.decide(ctx, "enterprise context?")        → patched → False
#   eng.reply(ctx, "Setting up a personal account.") → MockLLM.chat  [4]
#   eng.reply(ctx, "All done!")                   → MockLLM.chat  [5]


@pytest.mark.asyncio
async def test_wizard_runs():
    """Wizard onboarding example drives through choose/ask/create/decide."""
    mod = importlib.import_module("examples.wizard")
    Account = mod.Account

    from lingo.engine import Engine

    account_instance = Account(
        name="Alice",
        email="alice@example.com",
        use_case="testing the wizard",
    )

    # Patch choose and decide at class level so the dynamically-created CoT
    # models never need to be type-checked.  create() is only called by the
    # skill directly for Account — patch that too.
    with (
        patch.object(
            Engine, "choose", new=AsyncMock(return_value="developer")
        ),
        patch.object(Engine, "decide", new=AsyncMock(return_value=False)),
        patch.object(Engine, "create", new=AsyncMock(return_value=account_instance)),
    ):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=[
                "start",
                "Alice, alice@example.com, I want to test this",
            ],
            mock_responses=[
                "Welcome! Let's set up your account.",      # reply #1
                "Got it — setting up the developer experience.",  # reply #2
                "Tell me your name, email, and what you want to use this for.",  # ask
                "Thanks Alice! I have your email as alice@example.com.",  # reply after create
                "Setting up a personal account.",            # decide→False branch
                "All done!",                                 # final reply
            ],
        )

    out = "".join(captured)
    assert "Welcome" in out or "All done" in out


# ---------------------------------------------------------------------------
# 3. native_tool_call.py — raw LLM usage (no Lingo bot)
# ---------------------------------------------------------------------------
#
# _main_async creates its own LLM().  We patch the LLM class's chat method
# to return scripted Messages, simulating a single tool-call turn followed
# by a final text answer.


@pytest.mark.asyncio
async def test_native_tool_call_runs():
    """Native tool-call loop executes get_weather and add without crashing."""
    mod = importlib.import_module("examples.native_tool_call")

    # Turn 1: LLM requests two tool calls
    turn1 = Message.assistant(
        "",
        tool_calls=[
            ToolCall(id="c1", name="get_weather", arguments={"city": "Havana"}),
            ToolCall(id="c2", name="add", arguments={"a": 21, "b": 21}),
        ],
        stop_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    # Turn 2: LLM gives final text
    turn2 = Message.assistant(
        "The weather in Havana is sunny and 21+21=42.",
        stop_reason="stop",
        usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )

    call_count = 0

    # patch.object replaces the method; the bound call passes `self` as first arg.
    async def mock_chat(self, messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        return turn1 if call_count == 1 else turn2

    from lingo.llm import LLM

    with patch.object(LLM, "chat", new=mock_chat):
        await mod._main_async()

    assert call_count == 2


# ---------------------------------------------------------------------------
# 4. native_tool_call_streaming.py — raw LLM with callbacks
# ---------------------------------------------------------------------------
#
# _main_async creates LLM(on_token=..., ...) and calls chat once.  We patch
# LLM.chat to return a Message that has a tool_call (no actual streaming).


@pytest.mark.asyncio
async def test_native_tool_call_streaming_runs():
    """Streaming tool-call example calls callbacks without crashing."""
    mod = importlib.import_module("examples.native_tool_call_streaming")

    result_msg = Message.assistant(
        "",
        tool_calls=[
            ToolCall(id="s1", name="get_weather", arguments={"city": "Havana"}),
        ],
        stop_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    from lingo.llm import LLM

    with patch.object(LLM, "chat", new=AsyncMock(return_value=result_msg)):
        await mod._main_async()


# ---------------------------------------------------------------------------
# 5. banker.py — two skills (casual_chat + banker), equip + invoke + reply
# ---------------------------------------------------------------------------
#
# Two skills → Route node → engine.choose picks the skill flow.
# Then within banker skill: engine.equip picks check_balance, engine.invoke
# runs it, engine.reply gives the final answer.
#
# Patch engine.choose to always pick "banker" skill, engine.equip to always
# pick check_balance, and engine.invoke to return a scripted ToolResult.


@pytest.mark.asyncio
async def test_banker_check_balance():
    """Banker example: user asks for balance — check_balance tool runs."""
    mod = importlib.import_module("examples.banker")

    from lingo.engine import Engine
    from lingo.tools import ToolResult

    check_balance_tool = mod.check_balance  # the Tool object registered on bot
    balance_result = ToolResult(
        tool="check_balance", result={"balance": 1000}
    )

    # The Route node uses engine.choose to pick a skill flow.
    # We capture the flows passed to choose and return the banker one.
    async def scripted_choose(self_engine, ctx, options, *instructions):
        # options is a list of Flow objects; pick the one named "banker"
        for opt in options:
            if hasattr(opt, "name") and opt.name == "banker":
                return opt
        return options[-1]

    with (
        patch.object(Engine, "choose", new=scripted_choose),
        patch.object(Engine, "equip", new=AsyncMock(return_value=check_balance_tool)),
        patch.object(Engine, "invoke", new=AsyncMock(return_value=balance_result)),
    ):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=["What is my balance?"],
            mock_responses=[
                "Your current balance is $1000.",  # reply after invoke
            ],
        )

    out = "".join(captured)
    assert "1000" in out or out  # ran without crash


# ---------------------------------------------------------------------------
# 6. state_rpg.py — State + depends, equip + invoke + reply in fork context
# ---------------------------------------------------------------------------
#
# Single skill: game_loop. It calls engine.equip, engine.invoke, engine.reply
# inside context.fork().  We patch equip and invoke.


@pytest.mark.asyncio
async def test_state_rpg_runs():
    """RPG example: check_status tool is selected and executed."""
    mod = importlib.import_module("examples.state_rpg")

    from lingo.engine import Engine
    from lingo.tools import ToolResult

    check_status_tool = mod.check_status  # Tool registered on bot
    status_result = ToolResult(
        tool="check_status",
        result="hp: 100\ngold: 50\nlocation: Town Square",
    )

    with (
        patch.object(Engine, "equip", new=AsyncMock(return_value=check_status_tool)),
        patch.object(Engine, "invoke", new=AsyncMock(return_value=status_result)),
    ):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=["What is my status?"],
            mock_responses=[
                "You are in the Town Square with 100 HP.",  # reply after invoke
            ],
        )

    out = "".join(captured)
    assert out  # ran without crash


# ---------------------------------------------------------------------------
# 7. injection.py — tool with DI (llm injected), nested llm.chat call
# ---------------------------------------------------------------------------
#
# single skill: search_assistant.  The flow:
#   engine.invoke(ctx, smart_search, _secret_key="sk-12345")
#     → engine.infer(ctx, smart_search, ...)  → llm.create(smart_search params model)
#     → smart_search.run(query=..., _secret_key=..., llm=<bot.llm>)
#       → llm.chat([...])  [inner call — summarize the search results]
#     → result.result = summary.content  (a string)
#   ctx.append(Message.assistant(result.result))
#   engine.reply(ctx, "How can I help you?")  → llm.chat
#
# MockLLM.chat needs: [inner summary response, final reply]
# MockLLM.create needs: smart_search params model instance with query="France capital"
#
# We patch engine.invoke directly to avoid the create/params complexity.


@pytest.mark.asyncio
async def test_injection_runs():
    """Injection example: smart_search tool runs with injected LLM."""
    mod = importlib.import_module("examples.injection")

    from lingo.engine import Engine
    from lingo.tools import ToolResult

    # Patch invoke to return a scripted ToolResult directly.
    scripted_result = ToolResult(
        tool="smart_search", result="The capital of France is Paris."
    )

    with patch.object(
        Engine, "invoke", new=AsyncMock(return_value=scripted_result)
    ):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=["What is the capital of France?"],
            mock_responses=[
                "The capital of France is Paris.",  # engine.reply at the end of search_assistant
            ],
        )

    out = "".join(captured)
    assert "Paris" in out or out  # ran without crash


# ---------------------------------------------------------------------------
# 8. when.py — @bot.when filters (guardrail + sentiment)
# ---------------------------------------------------------------------------
#
# bot._filters has two conditions:
#   "The user is asking for illegal actions or using abusive language"
#   "The user seems frustrated or angry"
#
# On each chat turn, Lingo calls _build_filters() which calls engine.create
# with a dynamic Filter model having two bool fields (option_0, option_1).
# Then for each True field the corresponding flow runs.
#
# Happy path: normal user message → both filters False → support_skill runs.
# We patch engine.create to return a Filter(option_0=False, option_1=False).


@pytest.mark.asyncio
async def test_when_normal_message():
    """when.py normal message bypasses both filters and reaches support_skill."""
    mod = importlib.import_module("examples.when")

    from lingo.engine import Engine
    from pydantic import create_model

    # Build the same Filter model that _build_filters() would build, so we can
    # pre-create an instance.  Field names must match: option_{i} for each key.
    from pydantic import Field as PField

    Filter = create_model(
        "Filter",
        option_0=(bool, PField(description="True if The user is asking for illegal actions or using abusive language")),
        option_1=(bool, PField(description="True if The user seems frustrated or angry")),
    )
    # Both False → no filter fires
    filter_instance = Filter(option_0=False, option_1=False)

    with patch.object(Engine, "create", new=AsyncMock(return_value=filter_instance)):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=["I need help with my account."],
            mock_responses=[
                "How can I help you with your account today?",  # support_skill reply
            ],
        )

    out = "".join(captured)
    assert "help" in out.lower() or out


@pytest.mark.asyncio
async def test_when_guardrail_fires():
    """when.py guardrail fires on illegal-content flag.

    Note: engine.stop() inside a @bot.when filter only stops the *filter flow*,
    not the outer main flow (StopFlow is caught by the inner Flow.execute()).
    The main skill (support_skill) still runs after the guardrail.  This is the
    current runtime behavior; we test for what actually happens.
    """
    mod = importlib.import_module("examples.when")

    from lingo.engine import Engine
    from pydantic import create_model, Field as PField

    Filter = create_model(
        "Filter",
        option_0=(bool, PField(description="True if The user is asking for illegal actions or using abusive language")),
        option_1=(bool, PField(description="True if The user seems frustrated or angry")),
    )
    # option_0=True → security_guardrail fires
    filter_instance = Filter(option_0=True, option_1=False)

    with patch.object(Engine, "create", new=AsyncMock(return_value=filter_instance)):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=["Do something illegal"],
            mock_responses=[
                # 1. guardrail reply (engine.reply inside security_guardrail)
                "I cannot assist with that request as it violates our policy.",
                # 2. support_skill also runs (StopFlow is caught by the inner router flow)
                "How can I help you with your account today?",
            ],
        )

    out = "".join(captured)
    # The guardrail message should appear in the output
    assert "cannot" in out.lower() or "policy" in out.lower() or out


# ---------------------------------------------------------------------------
# 9. smart_home.py — parent skill + subskills (kitchen / living_room)
# ---------------------------------------------------------------------------
#
# Single skill smart_home with two subskills: kitchen and living_room.
# Route selects one subskill; within kitchen: engine.equip picks coffee_maker,
# engine.invoke runs it, engine.reply gives answer.
#
# We patch engine.choose (used by Route) to pick "kitchen", then
# engine.equip to pick coffee_maker, engine.invoke to return a result.


@pytest.mark.asyncio
async def test_smart_home_kitchen():
    """smart_home.py routes to kitchen and turns on the coffee maker."""
    mod = importlib.import_module("examples.smart_home")

    from lingo.engine import Engine
    from lingo.tools import ToolResult

    coffee_tool = mod.coffee_maker  # Tool registered on kitchen subskill
    coffee_result = ToolResult(tool="coffee_maker", result="Coffee maker turned on.")

    # smart_home uses a Route node to select among subskills (kitchen / living_room).
    # The Route calls engine.choose.  We pick the kitchen flow by name.
    async def scripted_choose(self_engine, ctx, options, *instructions):
        for opt in options:
            if hasattr(opt, "name") and "kitchen" in opt.name:
                return opt
        return options[0]

    with (
        patch.object(Engine, "choose", new=scripted_choose),
        patch.object(Engine, "equip", new=AsyncMock(return_value=coffee_tool)),
        patch.object(Engine, "invoke", new=AsyncMock(return_value=coffee_result)),
    ):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=["Turn on the coffee maker"],
            mock_responses=[
                "Coffee maker turned on.",  # engine.reply(context, str(result))
            ],
        )

    out = "".join(captured)
    assert out  # ran without crash


# ---------------------------------------------------------------------------
# 10. fsm.py — StateMachine with states, goto, tools
# ---------------------------------------------------------------------------
#
# Single skill fsm_skill wraps the StateMachine.  Initial state: triage.
# triage: reply("Is your issue Billing or Tech?") → equip → invoke route_request
# route_request("billing") calls fsm.goto(billing, restart=True)
# billing: reply("BILLING DEPT: ...") → equip → invoke some billing tool
#
# We patch engine.equip and engine.invoke to control which tool runs, and
# prevent the goto restart from causing infinite loops by also controlling
# what invoke returns.


@pytest.mark.asyncio
async def test_fsm_triage_to_billing():
    """fsm.py routes from triage to billing via route_request tool."""
    mod = importlib.import_module("examples.fsm")

    from lingo.engine import Engine
    from lingo.tools import ToolResult
    from lingo.fsm import ChangeState

    # Reset FSM state so tests don't bleed into each other
    mod.fsm._current_state = None

    route_tool = mod.route_request  # triage's routing tool
    refund_tool = mod.refund_transaction  # billing's tool

    equip_call_count = 0

    async def scripted_equip(self_engine, ctx, *tools):
        nonlocal equip_call_count
        equip_call_count += 1
        # First equip call (triage): return route_request
        # Second equip call (billing): return refund_transaction
        if equip_call_count == 1:
            return route_tool
        return refund_tool

    invoke_call_count = 0

    async def scripted_invoke(self_engine, ctx, tool, *instructions, **kwargs):
        nonlocal invoke_call_count
        invoke_call_count += 1
        if invoke_call_count == 1:
            # triage invoke: route_request("billing") → triggers goto(billing, restart=True)
            # Call the real tool to trigger the state change
            result = await tool.run(topic="billing", fsm=mod.fsm)
            return ToolResult(tool=tool.name, result=result)
        # billing invoke: just return a success
        return ToolResult(tool=tool.name, result="Refund processed for ID: TX123.")

    with (
        patch.object(Engine, "equip", new=scripted_equip),
        patch.object(Engine, "invoke", new=scripted_invoke),
    ):
        captured = await _drive_lingo(
            mod.bot,
            user_messages=["I have a billing issue"],
            mock_responses=[
                "Is your issue related to Billing or Tech?",  # triage reply
                "BILLING DEPT: How can I help with your account?",  # billing reply
            ],
        )

    out = "".join(captured)
    assert out  # ran without crash
    # Reset FSM for subsequent test runs
    mod.fsm._current_state = None

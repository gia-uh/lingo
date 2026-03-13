import pytest
from lingo import Lingo, Context, Engine
from lingo.fsm import StateMachine
from lingo.mock import MockLLM


@pytest.mark.asyncio
async def test_fsm_basic_transitions():
    """
    Tests basic state transitions in a StateMachine.
    """
    # 1. Setup bot and FSM
    bot = Lingo(llm=MockLLM(["Option 1", "Option 2"]))
    fsm = StateMachine(bot.registry)

    @fsm.state
    async def state_one(ctx, eng):
        await eng.reply(ctx, "In state one")
        # Deterministic transition
        fsm.goto(state_two)

    @fsm.state
    async def state_two(ctx, eng):
        await eng.reply(ctx, "In state two")

    # 2. Execute FSM
    ctx = Context([])
    engine = Engine(bot.llm)

    await fsm.execute(ctx, engine)

    assert fsm._current_state == state_two


@pytest.mark.asyncio
async def test_fsm_tool_routing():
    """
    Tests that the FSM correctly routes tools based on the current state.
    """
    bot = Lingo(llm=MockLLM())
    fsm = StateMachine(bot.registry)

    @fsm.state
    async def restricted_state(ctx, eng):
        pass

    @restricted_state.tool
    def state_tool():
        return "State Tool Result"

    fsm._current_state = restricted_state

    # Verify the tool is available in this state
    assert any(t.name == "state_tool" for t in restricted_state.tools)

import pytest
from unittest.mock import AsyncMock

from lingo import Flow, Message, Context
from lingo.flow import NoOp


@pytest.mark.asyncio
async def test_decide_branch_failure_does_not_pollute_context():
    """If the on_true branch fails midway, the parent context must remain unchanged."""

    # Mock engine to force the 'yes' branch
    mock_engine = AsyncMock()
    mock_engine.decide = AsyncMock(return_value=True)

    # Define a step that fails AFTER a message is added
    async def failing_step(ctx, eng):
        raise RuntimeError("Branch failed after partial execution")

    # Build a branch that: (1) adds a message, (2) fails
    failing_branch = Flow().append("This message must NOT appear").custom(failing_step)

    # Main flow with Decide
    main_flow = Flow().decide("Run failing branch?", yes=failing_branch, no=NoOp())

    # Initial clean context
    initial_messages = [Message.user("Start")]
    context = Context(initial_messages)

    # Execute → should raise exception
    with pytest.raises(RuntimeError, match="Branch failed after partial execution"):
        await main_flow.execute(context, mock_engine)

    # ASSERT: context is unchanged → atomicity achieved
    assert context.messages == initial_messages


@pytest.mark.asyncio
async def test_decide_branch_success_updates_context():
    """Successful branch should commit its messages to the parent context."""

    mock_engine = AsyncMock()
    mock_engine.decide = AsyncMock(return_value=True)

    success_branch = Flow().append("Committed message")

    main_flow = Flow().decide("Run success branch?", yes=success_branch, no=NoOp())

    context = Context([Message.user("Start")])
    await main_flow.execute(context, mock_engine)

    assert len(context.messages) == 2
    assert context.messages[1].content == "Committed message"


@pytest.mark.asyncio
async def test_choose_branch_failure_does_not_pollute_context():
    """Failed Choose branch must not leave messages in the parent context."""

    mock_engine = AsyncMock()
    mock_engine.choose = AsyncMock(return_value="risky_option")

    async def crashing_step(ctx, eng):
        raise ValueError("Chosen branch crashed")

    choices = {
        "safe_option": Flow().append("Safe path"),
        "risky_option": Flow().append("Dangerous message").custom(crashing_step),
    }

    main_flow = Flow().choose("Which path?", choices=choices)

    context = Context([Message.user("Begin")])

    with pytest.raises(ValueError, match="Chosen branch crashed"):
        await main_flow.execute(context, mock_engine)

    # Context must be untouched
    assert context.messages == [Message.user("Begin")]


@pytest.mark.asyncio
async def test_choose_branch_success_updates_context():
    """Successful Choose branch commits its messages."""

    mock_engine = AsyncMock()
    mock_engine.choose = AsyncMock(return_value="chosen")

    choices = {
        "ignored": Flow().append("Not this one"),
        "chosen": Flow().append("Correctly chosen!"),
    }

    main_flow = Flow().choose("Select", choices=choices)
    context = Context([Message.user("Hi")])
    await main_flow.execute(context, mock_engine)

    assert len(context.messages) == 2
    assert context.messages[1].content == "Correctly chosen!"
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from lingo import Lingo, Message

# We need to test the underlying 'run' coroutine because 'loop' calls 'asyncio.run'
# which cannot be called from an existing event loop (like the one pytest-asyncio provides).
from lingo.cli import run as cli_run


@pytest.mark.asyncio
async def test_cli_run_basic_interaction():
    """
    Tests the CLI internal run coroutine with a basic interaction.
    """
    # 1. Setup mock bot
    bot = Lingo(name="TestBot")

    async def mocked_chat(msg):
        # Simulate token emission
        await bot.llm.on_token("Hello ")
        await bot.llm.on_token("user!")
        return Message.assistant("Hello user!")

    bot.chat = AsyncMock(side_effect=mocked_chat)

    # 2. Mock input/output
    # We use a side effect that raises EOFError to stop the loop
    def input_side_effect(*args, **kwargs):
        if not hasattr(input_side_effect, "called"):
            input_side_effect.called = True
            return "Hi"
        raise EOFError()

    mock_input = MagicMock(side_effect=input_side_effect)
    mock_print = MagicMock()

    # 3. Run the internal 'run' coroutine
    with patch("builtins.input", mock_input), patch("builtins.print", mock_print):
        try:
            await cli_run(bot)
        except (EOFError, SystemExit):
            pass

    # 4. Verify interactions
    bot.chat.assert_called()
    # Check if any call to print contained "Hello user!"
    # The output_fn is called with "Hello ", "user!", then "\n\n"
    all_printed = "".join(str(call.args[0]) for call in mock_print.call_args_list)
    assert "Hello user!" in all_printed


@pytest.mark.asyncio
async def test_cli_run_custom_handlers():
    """
    Tests the CLI internal run coroutine with custom input and output handlers.
    """
    bot = Lingo(name="HandlerBot")

    async def mocked_chat(msg):
        await bot.llm.on_token("Handled!")
        return Message.assistant("Handled!")

    bot.chat = AsyncMock(side_effect=mocked_chat)

    def input_side_effect(*args, **kwargs):
        if not hasattr(input_side_effect, "called"):
            input_side_effect.called = True
            return "custom input"
        raise EOFError()

    mock_input_fn = MagicMock(side_effect=input_side_effect)
    mock_output_fn = MagicMock()

    try:
        await cli_run(bot, input_fn=mock_input_fn, output_fn=mock_output_fn)
    except EOFError:
        pass

    mock_input_fn.assert_called()
    # Verify that the tokens were sent to the custom output handler
    all_output = "".join(str(call.args[0]) for call in mock_output_fn.call_args_list)
    assert "Handled!" in all_output

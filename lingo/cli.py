import asyncio

from pydantic import BaseModel

from .utils import tee
from .core import Lingo


async def run(lingo: Lingo, input_fn=None, output_fn=None):
    """
    Runs this model in the terminal, using optional
    input and output callbacks.

    Args:
        input_fn: If provided, should be a function that
            returns the user message.
            Defaults to Python's builtin input().
        output_fn: If provided, should be a callback to stream
            the LLM chat response tokens (as strings).
            Defaults to print().
    """
    print("Name:", lingo.name)
    print("Description:", lingo.description)

    if lingo._verbose:
        print("Mode: Verbose")

    print("\n[Press Ctrl+D to exit]\n")

    if input_fn is None:
        input_fn = lambda: input(">>> ")

    if output_fn is None:
        # Default output_fn just takes a string
        output_fn = lambda token: print(token, end="", flush=True)

    # The handler is simplified, as it only receives strings
    cli_token_handler = output_fn

    original_on_token = lingo.llm._on_token
    original_on_create = lingo.llm._on_create

    lingo.llm._on_token = (
        tee(cli_token_handler, original_on_token)
        if original_on_token
        else cli_token_handler
    )

    def _verbose_on_create(model: BaseModel):
        """Callback to pretty-print parsed Pydantic models."""
        output_fn("\n------- [Thinking] -------\n")
        output_fn(repr(model))
        output_fn("\n--------------------------\n")

    if lingo._verbose:
        lingo.llm._on_create = (
            tee(original_on_create, _verbose_on_create)
            if original_on_create
            else _verbose_on_create
        )

    try:
        while True:
            msg = input_fn()
            await lingo.chat(msg)
            output_fn("\n\n")
    except EOFError:
        pass
    finally:
        # Restore the original on_token callback
        lingo.llm._on_token = original_on_token


def loop(lingo: Lingo, input_fn=None, output_fn=None):
    """
    Automatically creates an asyncio event loop and runs
    this chatbot in a simple CLI loop.

    Receives the same arguments as `run`.
    """
    asyncio.run(run(lingo, input_fn, output_fn))

"""Tests for examples exported from docs/lingo.md.

These tests exist to verify that the exported code blocks in the literate
index are not pseudocode — they are real programs that run against MockLLM.
If lingo's API changes and an example breaks, this test catches it immediately.
"""

import pytest
from unittest.mock import AsyncMock, patch
from pydantic import BaseModel, Field

from lingo.llm import Message, Usage
from lingo.mock import MockLLM


def _scripted_input_fn(messages):
    it = iter(messages)
    def input_fn():
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return input_fn


async def _drive(bot, user_messages, mock_responses):
    from lingo.cli import run
    bot.llm = MockLLM(responses=mock_responses)
    captured = []
    await run(
        bot,
        input_fn=_scripted_input_fn(user_messages),
        output_fn=lambda token: captured.append(token),
    )
    return "".join(captured)


@pytest.mark.asyncio
async def test_index_hello_runs():
    """index_hello.py: minimal app replies to a message."""
    import importlib
    mod = importlib.import_module("examples.index_hello")
    out = await _drive(mod.app, ["hi"], ["Hello! How can I help?"])
    assert "Hello" in out


@pytest.mark.asyncio
async def test_index_wizard_runs():
    """index_wizard.py: onboarding skill drives through choose/ask/create/decide."""
    import importlib
    mod = importlib.import_module("examples.index_wizard")

    from lingo.engine import Engine

    profile = mod.UserProfile(
        name="Alice",
        email="alice@corp.com",
        use_case="testing lingo",
    )

    with (
        patch.object(Engine, "choose", new=AsyncMock(return_value="developer")),
        patch.object(Engine, "decide", new=AsyncMock(return_value=False)),
        patch.object(Engine, "create", new=AsyncMock(return_value=profile)),
    ):
        out = await _drive(
            mod.app,
            user_messages=["start", "Alice, alice@corp.com, testing lingo"],
            mock_responses=[
                "Setting up the developer experience.",
                "Tell me your name, email, and what you want to use this for.",
                "Got it, Alice. I'll reach you at alice@corp.com.",
                "Setting up a personal account.",
                "All done!",
            ],
        )

    assert "All done" in out or "Alice" in out

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from lingo import Message
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM

@pytest.mark.asyncio
async def test_reply_returns_assistant_message():
    engine = Engine(MockLLM(["Hello!"]))
    c = Context([Message.user("Hi")])
    msg = await engine.reply(c)
    assert msg.role == "assistant"
    assert "Hello" in str(msg.content)

@pytest.mark.asyncio
async def test_reply_with_temporary_instruction():
    engine = Engine(MockLLM(["brief answer"]))
    c = Context([Message.user("Explain quantum physics")])
    # The instruction is temporary — it does not enter the context history.
    msg = await engine.reply(c, "Answer in one word.")
    assert msg.role == "assistant"
    assert len(c.messages) == 1  # context unchanged

@pytest.mark.asyncio
async def test_decide_returns_bool():
    engine = Engine(MockLLM())
    cot = MagicMock()
    cot.result = True
    engine.create = AsyncMock(return_value=cot)

    c = Context([Message.user("Is water wet?")])
    result = await engine.decide(c, "Answer True or False.")
    assert result is True

@pytest.mark.asyncio
async def test_choose_returns_one_of_options():
    options = ["red", "green", "blue"]
    engine = Engine(MockLLM())
    cot = MagicMock()
    cot.result = "green"
    engine.create = AsyncMock(return_value=cot)

    c = Context([Message.user("Pick a color")])
    result = await engine.choose(c, options, "Pick the most natural color.")
    assert result == "green"

class Sentiment(BaseModel):
    """Classify the sentiment of the last message."""
    label: str
    score: float

@pytest.mark.asyncio
async def test_create_returns_pydantic_model():
    expected = Sentiment(label="positive", score=0.9)
    engine = Engine(MockLLM([expected]))
    c = Context([Message.user("I love lingo!")])
    result = await engine.create(c, Sentiment)
    assert isinstance(result, Sentiment)
    assert result.label == "positive"


# Chapter 3: The Engine

`Engine` is the operator — it holds the `LLM` and performs all LLM operations
on a given `Context`. You never call the LLM directly in a flow; the engine
mediates every interaction.

## reply — the basic LLM call

`reply(context, *instructions)` sends the context to the LLM and returns the
assistant `Message`. It does NOT append that message to the context — the caller
decides whether to keep it.

```python {name=ch03_test_reply}
@pytest.mark.asyncio
async def test_reply_returns_assistant_message():
    engine = Engine(MockLLM(["Hello!"]))
    c = Context([Message.user("Hi")])
    msg = await engine.reply(c)
    assert msg.role == "assistant"
    assert "Hello" in str(msg.content)
```

Pass additional instructions as extra arguments to guide a specific reply
without polluting the permanent context:

```python {name=ch03_test_reply_temp}
@pytest.mark.asyncio
async def test_reply_with_temporary_instruction():
    engine = Engine(MockLLM(["brief answer"]))
    c = Context([Message.user("Explain quantum physics")])
    # The instruction is temporary — it does not enter the context history.
    msg = await engine.reply(c, "Answer in one word.")
    assert msg.role == "assistant"
    assert len(c.messages) == 1  # context unchanged
```

## decide — a boolean LLM call

`decide(context, *instructions)` asks the LLM a yes/no question and returns
a Python `bool`. It uses `engine.create()` internally to force a structured response.
In tests, mock `engine.create` directly with an `AsyncMock`:

```python {name=ch03_test_decide}
@pytest.mark.asyncio
async def test_decide_returns_bool():
    engine = Engine(MockLLM())
    cot = MagicMock()
    cot.result = True
    engine.create = AsyncMock(return_value=cot)

    c = Context([Message.user("Is water wet?")])
    result = await engine.decide(c, "Answer True or False.")
    assert result is True
```

## choose — select from options

`choose(context, options, *instructions)` picks one item from a list.
Like `decide`, it uses `engine.create` internally for a forced structured response:

```python {name=ch03_test_choose}
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
```

## create — structured output

`create(context, Model, *instructions)` forces the LLM to return a JSON
object that matches a Pydantic model. Use this when you need structured data,
not free-form text.

```python {name=ch03_sentiment_model}
class Sentiment(BaseModel):
    """Classify the sentiment of the last message."""
    label: str
    score: float
```

```python {name=ch03_test_create}
@pytest.mark.asyncio
async def test_create_returns_pydantic_model():
    expected = Sentiment(label="positive", score=0.9)
    engine = Engine(MockLLM([expected]))
    c = Context([Message.user("I love lingo!")])
    result = await engine.create(c, Sentiment)
    assert isinstance(result, Sentiment)
    assert result.label == "positive"
```

## ask and input — interactive flows

`ask(context, question)` sends a question to the user and waits for their reply.
`input()` just waits without sending anything first. These are used inside
multi-turn flows that need to pause mid-execution (see Chapter 6).

The result from `input()` / `ask()` is a plain string — it is NOT automatically
appended to the context. The flow decides what to do with it.

## Test file

```python {export=tests/test_ch03.py}
import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from lingo import Message
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM

<<ch03_test_reply>>
<<ch03_test_reply_temp>>
<<ch03_test_decide>>
<<ch03_test_choose>>
<<ch03_sentiment_model>>
<<ch03_test_create>>
```

# Chapter 1: Hello, lingo

This chapter covers the three building blocks every lingo program starts with:
the `LLM` client, the `Message` data model, and the `Lingo` chatbot.

## The LLM client

`LLM` is an async wrapper around the OpenAI-compatible API.
Configure it with a model, a base URL, and an API key — or leave them as `None`
and lingo reads `MODEL`, `BASE_URL`, and `API_KEY` from the environment.

The LLM exposes two async operations:

- `chat(messages)` — streams a conversation and returns an assistant `Message`.
- `create(model, messages)` — forces a JSON response matching a Pydantic model.

Callbacks let you react to tokens as they arrive:

```python {name=ch01_callbacks}
# LLM accepts callbacks for token-level events.
# Useful for streaming UIs or logging.
# llm = LLM(
#     on_token=lambda t: print(t, end="", flush=True),
#     on_message=lambda msg: print("\n--- done ---"),
# )
```

## Messages

`Message` is the unit of conversation. Build them with factory classmethods:

```python {name=ch01_messages}
def test_message_factories():
    system_msg  = Message.system("You are a helpful assistant.")
    user_msg    = Message.user("What is 2 + 2?")
    assist_msg  = Message.assistant("4")
    assert system_msg.role == "system"
    assert user_msg.role == "user"
    assert assist_msg.role == "assistant"
```

The role is always one of `"system"`, `"user"`, `"assistant"`, or `"tool"`.
An assistant `Message` can carry tool calls and token usage statistics — lingo
fills those automatically when it comes back from the LLM.

## The Lingo chatbot

`Lingo` is the high-level facade. It manages conversation history and exposes
a single `.chat(msg: str) -> Message` coroutine.

```python {name=ch01_make_bot}
def make_bot(*responses) -> Lingo:
    llm = MockLLM(list(responses) if responses else ["Hello!"])
    return Lingo(name="TestBot", description="A test bot.", llm=llm)
```

Calling `.chat()` appends the user message to history, runs the flow, and returns
the bot's reply. History grows with each call — the bot remembers the conversation.

```python {name=ch01_test_basic_chat}
@pytest.mark.asyncio
async def test_basic_chat():
    bot = make_bot("4")
    reply = await bot.chat("What is 2 + 2?")
    assert "4" in str(reply.content)

@pytest.mark.asyncio
async def test_history_grows():
    bot = make_bot("Hi!", "Doing well!")
    await bot.chat("Hello")
    await bot.chat("How are you?")
    # user + assistant for each turn
    assert len(bot.messages) == 4
```

## System prompt

Give the bot a custom personality via `system_prompt`:

```python {name=ch01_test_system_prompt}
@pytest.mark.asyncio
async def test_custom_system_prompt():
    bot = Lingo(
        name="Poet",
        description="A poetry bot.",
        system_prompt="You are {name}. {description} Speak only in verse.",
        llm=MockLLM(["Roses are red."]),
    )
    reply = await bot.chat("Say hello.")
    assert reply.content  # not empty
```

The `{name}` and `{description}` placeholders in `system_prompt` are filled
automatically from the constructor arguments.

## Test file

```python {export=tests/test_ch01.py}
import pytest
from lingo import LLM, Message, Lingo
from lingo.mock import MockLLM

<<ch01_callbacks>>
<<ch01_messages>>
<<ch01_make_bot>>
<<ch01_test_basic_chat>>
<<ch01_test_system_prompt>>
```

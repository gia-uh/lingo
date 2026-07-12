# Chapter 1: Hello, lingo

We want a Python program that talks to a language model. Not a one-shot script that fires a single prompt and exits — a real chatbot that holds a conversation, remembers what was said, and stays in character from the first message to the last.

This chapter builds **PyTutor**, a terminal bot that answers Python questions. It runs from your command line, keeps a tutor persona, and grows the conversation with every turn. By the end, `examples/01_hello_lingo.py` will be a working script you can run right now.

## The raw API

At the lowest level, lingo exposes one operation: `LLM.chat(messages)`. It takes a list of typed conversation turns and returns a single reply.

Those turns are `Message` objects, built with factory methods that make the role explicit:

- `Message.system(text)` — instructions the model follows but the user never sees
- `Message.user(text)` — what the user said
- `Message.assistant(text)` — a reply from the model, or a scripted one you inject

Here is the entire raw API for a single question-and-answer:

```python
import asyncio
from lingo import LLM, Message

async def ask_once():
    llm = LLM()   # reads MODEL, BASE_URL, API_KEY from environment
    reply = await llm.chat([
        Message.system("You are a concise Python tutor."),
        Message.user("What does the walrus operator do?"),
    ])
    print(reply.content)

asyncio.run(ask_once())
```

`LLM()` with no arguments reads three environment variables:

```
MODEL=gpt-4o-mini
BASE_URL=https://api.openai.com/v1   # override for local models or other providers
API_KEY=sk-...
```

This convention keeps credentials out of your code. Switching from OpenAI to a local Ollama instance means changing two env vars; nothing else changes.

The raw API is enough for a one-shot script. But a chatbot needs more: a growing message history, a persistent system prompt, and an input loop. That plumbing is what `Lingo` is for.

## Building PyTutor

`Lingo` is the high-level facade. You give it a name and a description; it handles everything else — creates the `LLM`, builds the system prompt, manages message history, and exposes a simple `.chat(str) → Message` coroutine.

Our script opens by loading credentials from a `.env` file, the conventional place for development secrets:

```python {export=examples/01_hello_lingo.py}
import dotenv
from lingo import Lingo
from lingo.cli import loop

dotenv.load_dotenv()
```

Next, the bot itself. `name` and `description` are not decorative — lingo injects them into the system prompt automatically, so the model knows its role before the first user message arrives:

```python {export=examples/01_hello_lingo.py}

bot = Lingo(
    name="PyTutor",
    description="A friendly Python tutor. Explain concepts clearly and use short code examples.",
)
```

## Running the conversation

`lingo.cli.loop` is a blocking REPL: it reads a line of input, calls `bot.chat()`, prints the reply, and repeats until the user types `exit` or hits Ctrl-C:

```python {export=examples/01_hello_lingo.py}

if __name__ == "__main__":
    loop(bot)
```

Run it:

```
python examples/01_hello_lingo.py
```

Ask it anything:

```
You > What is a generator in Python?
PyTutor > A generator is a function that yields values one at a time...

You > Can you show me an example?
PyTutor > Sure — here's a simple range-like generator...
```

The bot remembers what was said. Each call to `bot.chat()` appends the user message and the reply to the conversation history; the model sees the full exchange on every subsequent turn.

## How the pieces connect

Three components compose to make this work:

- **`LLM`** wraps the API. One call, one message back. Stateless by design.
- **`Message`** is the typed unit of conversation. Roles enforce the wire format every model expects.
- **`Lingo`** owns state. It manages the message list, the system prompt, and the event loop internally. `LLM` is a transport detail.

When `bot.chat("What is a list comprehension?")` is called, `Lingo` appends a `Message.user(...)` to its history, calls `llm.chat(history)`, appends the reply, and returns it. `loop` just drives that in a REPL. The script above — ten lines — is the complete application.

In [Chapter 2](02-messages-and-context.md) we go deeper into `Message` and `Context`: attaching images and audio, forking the conversation for speculative reasoning, and managing the message window when history grows too long.

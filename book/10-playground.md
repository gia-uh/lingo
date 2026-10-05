# Chapter 10: Lingo Playground

Terminal output is enough to prove that a conversational application works. It
is less useful when you need to understand *why* a turn ended, which tool the
model requested, or whether two browser tabs are accidentally sharing state.
Lingo Playground makes those details visible in a small local web interface.

The Playground is a development tool. It runs your Python application on your
machine, keeps model credentials on the server, and uses plain HTML, CSS, and
JavaScript in the browser. It is not a production chat service.

## Install the optional extra

The core `lingo-ai` package does not depend on a web framework. Install the
Playground extra when you need the visual interface:

```bash
pip install "lingo-ai[playground]"
```

Inside a checkout of this repository, use:

```bash
uv sync --all-groups --extra playground
```

The extra installs Starlette, Uvicorn, and its WebSocket transport. It does not
introduce a JavaScript package manager or frontend build step.

## Configure the model

The Playground uses the `LLM` configured by your application. A default `LLM`
reads the same environment variables as the rest of Lingo:

```bash
export MODEL=gpt-4o-mini
export API_KEY=sk-...
# Optional for OpenAI-compatible providers:
export BASE_URL=https://provider.example/v1
```

Credentials remain in the Python process. The Playground does not send them to
the browser or store them in browser storage.

## Expose an application factory

A Lingo application carries mutable conversation state. Give the Playground a
zero-argument factory so every browser connection receives a new instance:

```python
from lingo import Lingo


def create_app() -> Lingo:
    return Lingo(
        "Support Assistant",
        description="Answers product questions for the support team.",
    )
```

Save this as `support_app.py`, then identify the target as
`support_app:create_app`. The text before the colon is an importable Python
module; the text after it is the factory inside that module.

The repository includes a ready-to-run example at
`examples.playground_app:create_app`.

## Start the Playground

Run the bundled example:

```bash
lingo-playground examples.playground_app:create_app --open
```

The equivalent module command is:

```bash
python -m lingo.playground examples.playground_app:create_app --open
```

By default, Uvicorn listens on `127.0.0.1:8765`. Open
<http://127.0.0.1:8765> manually when you omit `--open`.

Use a different local port when the default is occupied:

```bash
lingo-playground support_app:create_app --port 9000
```

## Read the interface

The application card on the left shows the `Lingo.name`, description, and
session mode. The center column contains the conversation and streams assistant
tokens as they arrive.

When a provider streams reasoning tokens before its answer, the assistant card
shows **Reasoning…** with the elapsed turn time. The label changes to **Reasoned
for** when answer text begins. Providers that do not expose reasoning tokens do
not show this indicator. The Playground observes these events; it does not
enable or disable reasoning, rewrite the prompt, or change model parameters.

Completed assistant messages may include an **Inspect metadata** panel:

- **Reasoning** contains reasoning text exposed by a compatible provider.
- **Tool calls** lists native tool requests and their validated arguments.
- **Tool call ID** links a tool result to its request.
- **Stop reason** distinguishes normal completion, length limits, and tool calls.
- **Token usage** shows prompt, completion, and total tokens when the provider
  reports them.

Structured outputs and tool-call lifecycle events appear as compact inspector
rows. All model-provided content is inserted as text; the browser does not
interpret it as HTML.

Press **Enter** to send a message and **Shift+Enter** for a new line. The
interface follows the system color scheme, and the theme button stores only the
light/dark preference in browser storage.

## Pause and resume a skill

Interactive skills work without a special browser API. When a skill reaches
`Engine.ask()`, `Lingo.chat()` returns control after displaying the question.
The next browser message resumes the suspended Python coroutine, including its
local variables.

```python
from lingo import Lingo


def create_app() -> Lingo:
    app = Lingo("Registration")

    @app.skill
    async def register(context, engine):
        name = await engine.ask(context, "What is your name?")
        await engine.reply(context, f"Welcome, {name}.")

    return app
```

Closing the browser connection cancels a suspended flow owned by that session.
It is never reassigned to another browser.

## Reset a session

The **Reset** button discards the current browser conversation. For a factory
target, the server calls the factory again and installs a completely new `Lingo`
instance. This resets messages, state, callbacks, and suspended flow execution
together.

Reset is disabled while a message is being processed. It affects only the
current browser session.

## Use an existing singleton

The loader also accepts an existing module-level `Lingo` object:

```bash
lingo-playground examples.wizard:bot
```

This compatibility mode allows one live browser session. A second connection is
rejected because a singleton contains one message history and one active flow
runner. Reset clears its visible conversation and runner state, but a factory is
the reliable choice when custom closures or state machines hold additional
state.

Convert a singleton module by wrapping construction in a function:

```python
# Before
bot = Lingo("Assistant")

# Recommended for the Playground
def create_app() -> Lingo:
    return Lingo("Assistant")
```

## Command reference

```text
lingo-playground TARGET [--host HOST] [--port PORT] [--open]
                        [--log-level LEVEL]
```

| Argument | Meaning | Default |
|---|---|---|
| `TARGET` | Import path in `module:attribute` form | Required |
| `--host` | Address passed to Uvicorn | `127.0.0.1` |
| `--port` | TCP port from 1 through 65535 | `8765` |
| `--open` | Open the local URL in the default browser | Off |
| `--log-level` | `critical`, `error`, `warning`, `info`, `debug`, or `trace` | `info` |

Binding to a non-loopback host prints a warning. The Playground has no
authentication or TLS and is not hardened for public exposure.

## Troubleshooting

### The target cannot be imported

Run the command from a directory where Python can import your module. Use dots
for package paths and one colon before the attribute:

```bash
lingo-playground my_package.support:create_app
```

The loader reports malformed targets, missing modules, missing attributes,
factories with required arguments, and factories that return something other
than `Lingo`.

### Playground dependencies are missing

Install the optional extra in the same environment that provides the
`lingo-playground` command:

```bash
python -m pip install "lingo-ai[playground]"
```

### The browser disconnects immediately

A singleton target may already have another connected tab. Close that tab or
expose a factory. Also check the terminal for Uvicorn startup or application
import errors.

### A model request fails

The interface displays a safe error summary while the terminal retains the full
traceback for local debugging. Verify `MODEL`, `API_KEY`, `BASE_URL`, provider
availability, and model compatibility.

### A model takes a long time to show text

If the **Reasoning…** timer is visible, the provider is streaming reasoning
before answer text. The timer includes request startup and generation time from
the beginning of the turn. Expand **Inspect metadata** after completion to
compare the reasoning content and token usage with the visible answer.

Control reasoning and generation settings in your application or provider. The
Playground deliberately sends the application's original messages and model
configuration unchanged, so switching between a local server and a hosted API
does not require a Playground-specific mode.

### The port is already in use

Choose another port:

```bash
lingo-playground support_app:create_app --port 8766
```

## Security and scope

The Playground makes local development observable without pretending to be a
deployment platform:

- It binds to loopback by default.
- It validates WebSocket origins and incoming event shapes.
- It limits user messages to 20,000 characters.
- It serves a fixed set of packaged assets rather than arbitrary files.
- It applies a restrictive Content Security Policy.
- It never provides a browser control for importing modules or executing Python.

If an application needs public access, authentication, persistent sessions, or
multiple worker processes, place its own production API and identity model in
front of Lingo. The Playground should remain the small instrument you use while
building that application.

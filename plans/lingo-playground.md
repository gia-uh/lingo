# Lingo Playground implementation plan

## Objective

Build a lightweight, local-first visual application for running and inspecting
Lingo applications in a browser. The Playground will use a small Starlette /
Uvicorn backend and framework-free HTML, CSS, and JavaScript. It will ship as an
optional package extra so the core library keeps its current dependency weight.

The first release must let a user load a Lingo application, conduct a
conversation, observe streamed model activity and message metadata, reset an
isolated session, and understand the entire workflow through a published user
manual.

The Playground is a development tool, not a production chat server. It binds to
loopback by default, does not accept API keys in the browser, and does not add
authentication, durable persistence, or remote multi-user hosting in the MVP.

## Product scope

### MVP capabilities

1. Start the Playground with a `module:attribute` application target.
2. Accept either:
   - a zero-argument factory returning `Lingo` (recommended); or
   - an existing `Lingo` instance (single-session compatibility mode).
3. Create an isolated Lingo instance per browser session when a factory is used.
4. Display application name and description.
5. Send user messages and render streamed assistant text.
6. Display completed messages by role and expose optional metadata:
   `thinking`, `tool_calls`, `stop_reason`, and token usage.
7. Display structured-output and tool-call lifecycle events.
8. Reset a session. Factory-backed sessions receive a new `Lingo` instance;
   singleton-backed sessions receive a clearly documented limited reset.
9. Surface loading, model, and execution errors without terminating the server.
10. Provide a responsive, keyboard-accessible light/dark interface.

### Explicitly deferred

- Visual Flow editor or graph inspector.
- Arbitrary prompt, Python, or tool editing in the browser.
- Authentication, TLS, public hosting, or a production deployment contract.
- Database-backed conversations or restoration of suspended Python stacks.
- Native desktop packaging.
- Multiple concurrent sessions against one singleton Lingo object.
- Browser end-to-end test tooling or a JavaScript build pipeline.

## User-facing interface

### Installation

```bash
pip install "lingo-ai[playground]"
```

Development installation remains:

```bash
uv sync --all-groups --extra playground
```

### Launch command

```bash
lingo-playground examples.playground_app:create_app
```

Equivalent module invocation:

```bash
python -m lingo.playground examples.playground_app:create_app
```

Initial command options:

```text
TARGET              Python target in module:attribute form
--host HOST         Bind host; defaults to 127.0.0.1
--port PORT         Bind port; defaults to 8765
--open              Open the browser after startup
--log-level LEVEL   Uvicorn log level
```

The command fails before starting the server when the target cannot be imported,
is neither a `Lingo` instance nor a valid factory, or a factory returns the wrong
type.

### Recommended application contract

```python
def create_app() -> Lingo: ...
```

Factories are invoked once per browser session and again on reset. Existing
module-level applications such as `examples.wizard:bot` remain loadable, but the
UI and manual identify them as single-session mode.

## Architectural impact

### Package boundary

The feature lives under `lingo.playground` and imports Starlette/Uvicorn lazily.
Importing `lingo` must continue to work when the optional extra is absent. A
missing extra produces an actionable CLI error rather than an import-time failure
for the core package.

The core `Lingo`, `Engine`, `Flow`, and `LLM` APIs remain unchanged in the MVP.
The Playground adapts to the existing callback surface instead of adding UI
concerns to the orchestration layer.

### Runtime components

1. **Target loader**
   Resolves `module:attribute`, validates a singleton or factory, and records the
   target's session capabilities.
2. **Session manager**
   Owns browser session identifiers, Lingo instances, lifecycle locks, reset,
   and cleanup. It rejects a second active client for singleton targets.
3. **Event bridge**
   Temporarily composes the LLM callbacks used by the Playground with callbacks
   already configured by the application. It emits tokens, reasoning tokens,
   structured objects, completed messages, and tool-call events without
   suppressing application callbacks. Callback restoration is guaranteed during
   teardown.
4. **ASGI application**
   Serves packaged static assets, application metadata, a health endpoint, and a
   WebSocket transport.
5. **Browser client**
   Uses native WebSocket and DOM APIs. It has no Node.js dependency or frontend
   build step.

### Event protocol

Client-to-server events:

```text
user_message { content }
reset        {}
```

Server-to-client events:

```text
ready               { session_id, app, mode }
turn_started        {}
token               { text }
reasoning_token     { text }
tool_call_started   { id, name }
tool_call_delta     { id, arguments }
tool_call_completed { id, arguments }
message             { message }
structured_output   { value }
turn_completed      { message }
reset_completed     { app, mode }
error               { code, message, recoverable }
```

Each event is a JSON object with a `type` field. Pydantic models define and
validate the server-emitted shapes. Internal exceptions are logged server-side;
the browser receives a safe summary without secrets or tracebacks.

### Conversation and concurrency rules

- Each factory-backed WebSocket gets its own `Lingo` instance.
- A per-session async lock serializes messages for that instance.
- The normal `Lingo.chat()` pause/resume behavior remains authoritative: a turn
  that pauses at `Engine.input()` completes at the transport boundary, and the
  next `user_message` resumes the existing runner.
- Singleton targets allow only one live browser session and clearly expose that
  mode in the UI.
- Disconnecting removes the session. If its internal flow is still active, the
  manager cancels and awaits it during cleanup without reusing the instance.
- Reset is unavailable while a client command is being processed. It never
  mutates another session.

### Security boundaries

- Default binding is `127.0.0.1`; non-loopback binding emits a prominent warning.
- API credentials stay in the server environment and are never included in
  metadata, events, HTML, or browser storage.
- WebSocket origin validation accepts the served origin by default.
- Incoming message size and event type are validated.
- Static asset paths are fixed package resources, not arbitrary filesystem paths.
- Importing the explicit CLI target is treated as trusted local code; the UI
  cannot select or import arbitrary modules.
- Content is inserted into the DOM as text, never as unsanitized HTML.

## File operations

### Create

- `lingo/playground/__init__.py`
  Public Playground entry points without eagerly importing optional packages.
- `lingo/playground/__main__.py`
  Support `python -m lingo.playground`.
- `lingo/playground/cli.py`
  Argument parsing, optional-dependency error, target validation, server startup,
  and optional browser opening.
- `lingo/playground/loader.py`
  `module:attribute` resolution and factory/singleton target abstraction.
- `lingo/playground/events.py`
  Wire event models and safe message serialization.
- `lingo/playground/session.py`
  Session lifecycle, callback composition, locking, reset, and cleanup.
- `lingo/playground/app.py`
  Starlette factory, HTTP routes, WebSocket endpoint, security checks, and static
  resource responses.
- `lingo/playground/static/index.html`
  Semantic application shell.
- `lingo/playground/static/styles.css`
  Responsive theme and component styles.
- `lingo/playground/static/app.js`
  WebSocket state machine and safe DOM rendering.
- `examples/playground_app.py`
  Minimal factory-backed example used by the manual and smoke tests.
- `tests/test_playground.py`
  Loader, sessions, events, routes, WebSocket, errors, reset, and packaging-facing
  behavior using a mock LLM.
- `book/10-playground.md`
  Complete user manual.

### Modify

- `pyproject.toml`
  Add the `playground` optional dependency set, console script, and any test-only
  HTTP client dependency required by Starlette's test client.
- `mkdocs.yml`
  Add the Playground manual to the published navigation.
- `README.md`
  Add a concise Playground section linking to the manual; do not duplicate it.
- `CHANGELOG.md`
  Record the new optional visual application under Unreleased.
- `tests/test_examples.py`
  Include the factory-backed example in existing example smoke coverage when that
  is clearer than duplicating it in `test_playground.py`.

No existing core module is expected to change. If implementation reveals that a
core lifecycle or callback change is necessary, stop and revise this plan before
making that change.

## Execution steps

1. **Packaging and entry-point skeleton**
   Add the optional extra, console script, lazy import boundary, CLI parser, and
   early target validation. Verify that importing base `lingo` without the extra
   remains valid.
2. **Target loader**
   Implement strict `module:attribute` parsing, factory validation, singleton
   compatibility mode, and useful error messages. Add focused unit tests.
3. **Event schema and callback bridge**
   Define the JSON protocol and compose every existing LLM callback without
   losing synchronous or asynchronous application callbacks. Test ordering and
   restoration on success, model failure, and disconnect.
4. **Session lifecycle**
   Implement factory-backed isolation, singleton admission control, per-session
   locking, reset, and cleanup. Exercise `Engine.ask()` pause/resume behavior with
   scripted messages.
5. **Starlette application**
   Add health/metadata/static routes and the WebSocket handler. Validate origins,
   payloads, size limits, disconnects, and safe errors.
6. **Framework-free frontend**
   Build the chat layout, connection state, composer, message list, expandable
   metadata panels, token streaming, reset flow, theme preference, keyboard
   behavior, and accessible live regions.
7. **Example integration**
   Add a small `create_app()` example that is deterministic under `MockLLM` and
   suitable for the manual.
8. **User manual**
   Write installation, configuration, launch, factory conversion, interface
   guide, metadata interpretation, reset semantics, troubleshooting, security
   limitations, and command reference. Include lightweight screenshots only if
   they materially improve usability and can be maintained reliably.
9. **Project documentation**
   Link the manual from MkDocs and README, and update the changelog.
10. **Release-quality verification**
    Run formatting, lint, unit/integration tests, docs build, and wheel inspection.
    Confirm packaged wheels include static assets and the console entry point.

## Testing strategy

### Automated tests

- Loader accepts a factory and a singleton and rejects malformed/mistyped targets.
- Factory sessions receive distinct `Lingo`, `LLM`, message, and runner state.
- Singleton mode rejects concurrent clients deterministically.
- Existing callbacks still run exactly once alongside Playground events.
- Sync and async callbacks are both supported.
- Token, reasoning, structured-output, tool-call, message, usage, and stop-reason
  events serialize correctly.
- `Engine.ask()` pauses and resumes across two WebSocket messages.
- Reset recreates factory-backed sessions and clears browser-visible history.
- Execution errors become safe recoverable error events.
- Disconnect cleanup leaves no live session or runner task.
- Invalid origins, event types, non-string content, and oversized input are
  rejected.
- Health, metadata, and static routes return the expected content and headers.
- Base-package imports work without installing the Playground extra.
- Built wheels contain HTML/CSS/JS resources and expose `lingo-playground`.

### Verification commands

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -p no:cacheprovider
uv run mkdocs build --strict
uv build
```

Because the branch currently inherits known Ruff/format failures in test files,
verification will distinguish pre-existing failures from Playground regressions.
The feature should not add new lint or format violations. Whether to clean the
existing unrelated violations belongs in a separate commit or task.

### Manual acceptance

1. Launch the factory example using only documented commands.
2. Complete a normal multi-turn conversation.
3. Complete a conversation that pauses at `Engine.ask()` and resumes correctly.
4. Observe streaming content and expanded message metadata.
5. Reset and verify a clean factory-created application.
6. Open two tabs and verify factory isolation.
7. Confirm singleton mode communicates and enforces its limitation.
8. Trigger a model error and recover without restarting the server.
9. Confirm no credential or traceback appears in browser network payloads.
10. Check the UI at narrow and wide viewport sizes using keyboard navigation.

## Compatibility constraints

- Python 3.12 remains the minimum version.
- Starlette and Uvicorn are optional; core installations remain unchanged.
- Existing Lingo applications and decorators require no migration.
- Existing application LLM callbacks must retain their behavior.
- No JavaScript runtime or package manager is introduced.
- The published book remains the canonical user documentation.
- Static resources must work from both editable installs and built wheels.

## Risks and mitigations

1. **Lingo instances are stateful and not multi-session safe.**
   Use per-session factories; constrain singleton compatibility mode to one live
   session rather than attempting unsafe copying.
2. **Callback replacement can interfere with application behavior.**
   Compose and restore callbacks in a dedicated bridge with ordering tests and
   cleanup in `finally` blocks.
3. **`Engine.reply()` responses are not necessarily appended to `Context`.**
   Drive rendering from LLM callbacks as well as the final `Lingo.chat()` result,
   and deduplicate completed events by object/event identity where necessary.
4. **Disconnects can leave suspended flows.**
   Give sessions explicit ownership of runner tasks and deterministic cleanup.
5. **Optional imports can accidentally make the core package heavier.**
   Keep Starlette/Uvicorn imports below `lingo.playground` entry points and test a
   base-only environment/import path.
6. **A local development UI may be mistaken for a production server.**
   Label it clearly in CLI output and documentation; default to loopback and warn
   on external binding.
7. **Browser protocol changes can drift from Python models.**
   Centralize event names and cover each event with integration tests and manual
   documentation.

## Open decisions

The following choices can be made during implementation without changing the
architecture, unless the user prefers otherwise:

- Final product label: `Lingo Playground` versus `Lingo Studio Lite`.
- Whether `--open` defaults to false (recommended for predictable CLI behavior).
- Exact message-size limit and inactive-session cleanup timeout.
- Whether the first manual includes maintained screenshots or only diagrams and
  annotated interface descriptions.

## Delivery sequence

Keep commits independently reviewable:

1. `feat(playground): add optional server and target loader`
2. `feat(playground): add isolated sessions and event transport`
3. `feat(playground): add framework-free browser interface`
4. `docs(playground): add user manual and project links`

Implementation starts only after this plan is approved.

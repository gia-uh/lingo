# The LLM — lingo's model interface

`llm.chat(messages)` returns a `Message`. That is the contract this module upholds.

Everything else in this chapter is an answer to progressively deeper questions: what *is* a `Message`? How does multimodal content fit in? What happens when the model wants to call a tool? What goes over the wire? How does the stream turn into a complete result?

The reading order here is top-down by what a caller encounters first, not bottom-up by what Python requires to compile.

---

## The Message

`Message` is the unit of conversation. You build messages, pass them to `chat()`, and get one back. Everything else is details.

Every message has a role — who is speaking — and content — what they said. The assistant variant carries optional extras: `tool_calls` (when the model wants to call a function), `thinking` (streamed reasoning), `stop_reason` (why the model stopped), `usage` (token counts).

Four factory methods cover the common cases:

```python
Message.user("hello")
Message.system("You are a helpful assistant.")
Message.assistant("I can help.", usage=Usage(...))
Message.tool(result_str, tool_call_id="call_abc123")
```

The `tool_call_id` on a `tool` message is not optional in practice: the model uses it to match the result to the call it made. Omit it and the API rejects the conversation.

```python {name=message_model}
class Message(BaseModel):
    """A Pydantic model for a single chat message."""

    role: Literal["user", "system", "assistant", "tool"]
    content: Union[
        TextContent, ImageContent, AudioContent, VideoContent, FileContent, str
    ]
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # required for role="tool" by OpenAI API
    thinking: str | None = None
    stop_reason: (
        Literal[
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "error",
            "aborted",
        ]
        | None
    ) = None
    usage: Usage | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Union[Content, str]) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str,
        usage: Usage | None = None,
        tool_calls: list[ToolCall] | None = None,
        thinking: str | None = None,
        stop_reason: str | None = None,
    ) -> "Message":
        return cls(
            role="assistant",
            content=content,
            usage=usage,
            tool_calls=tool_calls,
            thinking=thinking,
            stop_reason=stop_reason,
        )

    @classmethod
    def tool(cls, content: Any, tool_call_id: str | None = None) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)
```

---

## Multimodal content

The `content` field above accepts a `str` or any `Content` subclass. Why the hierarchy? Because the API wire format differs by type: text is a bare string, images go in `image_url` dicts, audio in `input_audio` dicts. Having a typed class per content kind lets `model_dump()` serialize each correctly without a chain of `isinstance` checks at the call site.

The `type` literal on each class is the discriminator the API uses: `"image_url"`, `"input_audio"`, `"video_url"`, `"file_url"`. These are the exact strings OpenAI's schema expects.

For local files, factory methods on `Message` handle encoding:

```python
Message.local_image("diagram.png")    # base64-encodes, returns Message with ImageContent
Message.online_image("https://...")   # URL directly
Message.local_audio("recording.mp3") # base64-encodes, returns Message with AudioContent
```

```python {name=content_models}
class Content(BaseModel):
    """Base class for all message content types."""

    type: str

    def __str__(self) -> str:
        raise TypeError("Not a textual content.")


class TextContent(Content):
    """Standard text content."""

    type: Literal["text"] = "text"
    text: str

    def __str__(self):
        return self.text


class ImageContent(Content):
    """Image content supporting both URLs and base64 data."""

    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str] = Field(
        description="Dictionary containing 'url' (can be data:image/...)"
    )


class AudioContent(Content):
    """Audio content for multimodal models."""

    type: Literal["input_audio"] = "input_audio"
    input_audio: dict[str, str] = Field(
        description="Dictionary containing 'data' (base64) and 'format'"
    )


class VideoContent(Content):
    """Video content (supported by some OpenRouter/OpenAI models)."""

    type: Literal["video_url"] = "video_url"
    video_url: dict[str, str] = Field(description="Dictionary containing 'url'")


class FileContent(Content):
    """Generic file content."""

    type: Literal["file_url"] = "file_url"
    file_url: dict[str, str] = Field(description="Dictionary containing 'url'")
```

And the multimodal factory methods on `Message`:

```python {name=message_multimodal_factories}
@classmethod
def local_image(cls, path: str, detail: str = "auto") -> "Message":
    """Loads a local image and encodes it as base64."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"

    return cls.user(
        ImageContent(
            image_url={"url": f"data:{mime};base64,{data}", "detail": detail}
        )
    )

@classmethod
def online_image(cls, url: str, detail: str = "auto") -> "Message":
    """Creates a message with an online image URL."""
    return cls.user(ImageContent(image_url={"url": url, "detail": detail}))

@classmethod
def local_audio(cls, path: str, format: str | None = None) -> "Message":
    """Loads a local audio file and encodes it as base64."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    if not format:
        _, ext = os.path.splitext(path)
        format = ext.strip(".").lower() or "mp3"

    return cls.user(AudioContent(input_audio={"data": data, "format": format}))

@classmethod
def online_video(cls, url: str) -> "Message":
    """Creates a message with an online video URL."""
    return cls.user(VideoContent(video_url={"url": url}))
```

---

## What the model sends back: ToolCall

When the model responds, it might send back text — or it might send back a list of tool calls instead. A `ToolCall` is the model's request to invoke a function: here is the function name, here are the arguments, here is an id you must echo back in the result.

The `id` is provider-assigned. Omitting it in the result message (`Message.tool(result, tool_call_id=call.id)`) causes API errors on most providers. This is not a lingo constraint; it is how the protocol works.

```python {name=tool_call_model}
class ToolCall(BaseModel):
    """A tool-call request emitted by the LLM in an assistant message."""

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)
```

---

## Serialization: the wire format

`Message` overrides `model_dump()` because the OpenAI wire format does not match a straight Pydantic dump. Three cases:

**Text content:** A bare string in most cases. But when the assistant message has `tool_calls` and empty text content, the API requires `null`, not `""`. Strict providers (Qwen via OpenRouter) reject the empty string and misinterpret the conversation, stalling the agent loop. This is a provider bug, but lingo works around it unconditionally.

**Structured content (images, audio, etc.):** The API expects a list with one dict, not the dict directly: `[{"type": "image_url", "image_url": {...}}]`.

**Tool calls:** Serialized as the `tools[]` format the API expects, with `"type": "function"` wrapper and JSON-stringified arguments.

```python {name=message_serialization}
def model_dump(self) -> dict[str, Any]:
    """Custom model dump to handle structured Content and Pydantic models."""
    dump = dict(role=self.role)
    content = self.content

    if isinstance(content, str):
        # Tool-calling assistant messages with no text content must carry
        # content:null rather than content:"". The OpenAI spec allows null
        # here; strict providers (e.g. Qwen via OpenRouter) reject "" and
        # misinterpret the conversation, causing the agent loop to stall.
        if self.tool_calls and not content:
            dump["content"] = None
        else:
            dump["content"] = content
    elif isinstance(content, Content):
        dump["content"] = [content.model_dump()]
    elif isinstance(content, BaseModel):
        dump["content"] = content.model_dump_json()

    if self.tool_calls:
        dump["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in self.tool_calls
        ]

    if self.tool_call_id is not None:
        dump["tool_call_id"] = self.tool_call_id

    return dump
```

---

## Token usage

Every response carries a `Usage` record so callers can track costs. All fields default to 0 so partial responses (e.g., from mocks) don't require special-casing.

```python {name=usage_model}
class Usage(BaseModel):
    """Token usage statistics for an LLM interaction."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

---

## Tool schemas: bridging lingo to the OpenAI wire format

`chat()` accepts `tools=[...]` — a list of lingo `Tool` objects. Before sending to the API, each tool must be serialized into OpenAI's `tools[]` format: a function object with name, description, and a JSON Schema for its parameters.

`_python_type_to_json_schema` maps Python type annotations to JSON Schema fragments. It covers only the types that tool parameters actually use in practice. The `"string"` fallback for unknown types is intentional: the LLM can usually infer the right value from context even if the schema is imprecise.

All parameters are marked `required` because lingo tools have no optional parameters — every argument must either be inferred by the LLM or supplied explicitly via `**kwargs` in `Engine.invoke`.

```python {name=schema_helpers}
def _python_type_to_json_schema(t) -> dict:
    """Map a Python type annotation to a JSON Schema fragment."""
    import typing

    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}
    if t is list or typing.get_origin(t) is list:
        return {"type": "array"}
    if t is dict or typing.get_origin(t) is dict:
        return {"type": "object"}
    return {"type": "string"}  # safe fallback


def tool_to_openai_schema(tool_obj) -> dict:
    """Convert a lingo.Tool into an OpenAI tools[] entry."""
    params = tool_obj.parameters()
    properties = {
        name: _python_type_to_json_schema(ptype) for name, ptype in params.items()
    }
    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(params.keys()),
            },
        },
    }
```

---

## Reasoning tokens: a provider variance problem

Several providers expose chain-of-thought reasoning as a separate stream alongside content. There is no standard field name:

- OpenRouter unified and OpenAI o-series: `delta.reasoning`
- DeepSeek and some Anthropic paths: `delta.reasoning_content`
- Gemini: `delta.thoughts`

The OpenAI Python SDK discards fields it doesn't know about — except that it preserves them in `model_extra`. `_read_reasoning` checks both the typed attribute path and `model_extra`, so lingo is transparent to provider variance. New providers that use a different name require only a one-line addition to `_REASONING_FIELDS`.

```python {name=reasoning_fields}
_REASONING_FIELDS = ("reasoning", "reasoning_content", "thoughts")
```

```python {name=reasoning_helper}
def _read_reasoning(delta: Any) -> str | None:
    """Pull a streamed reasoning fragment from a delta, handling provider variance.

    Different providers expose reasoning under different field names:
    OpenRouter unified and OpenAI o-series use ``reasoning``, some
    DeepSeek/Anthropic paths use ``reasoning_content``, and some Gemini
    paths use ``thoughts``. The OpenAI SDK preserves unknown fields via
    ``model_extra``; we check both the typed attribute path and the
    extras dict so we catch fields the SDK didn't model directly.
    """
    for name in _REASONING_FIELDS:
        value = getattr(delta, name, None)
        if value:
            return value
    extra = getattr(delta, "model_extra", None) or {}
    for name in _REASONING_FIELDS:
        value = extra.get(name)
        if value:
            return value
    return None
```

---

## The LLM class: configuration

`LLM` wraps `AsyncOpenAI`. All credentials have environment-variable defaults so `LLM()` with no arguments works in the common case. Extra kwargs (e.g., `temperature=0.2`) are stored and forwarded to every API call; per-call kwargs merge with and override them.

Callbacks are optional and accept both sync and async functions — `inspect.iscoroutine` detects which at dispatch time, so callers don't need to think about it.

The `reasoning` kwarg deserves special mention: it is an OpenRouter body extension (`{"effort": "high"}` or `{"max_tokens": 1024}`) that controls how much reasoning the model is allowed to do. It cannot be passed to `create()` because OpenAI's `parse()` endpoint rejects unknown fields. We store it at the instance level and inject it only into the streaming `chat()` path via `extra_body`.

```python {name=llm_init}
def __init__(
    self,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    on_token: Callable[[str], Any] | None = None,
    on_reasoning_token: Callable[[str], Any] | None = None,
    on_create: Callable[[BaseModel], Any] | None = None,
    on_message: Callable[[Message], Any] | None = None,
    on_toolcall_start: Callable[[str, str], Any] | None = None,
    on_toolcall_delta: Callable[[str, str], Any] | None = None,
    on_toolcall_end: Callable[[str, dict], Any] | None = None,
    reasoning: dict[str, Any] | None = None,
    **extra_kwargs,
):
    """
    Initializes the LLM client.

    Args:
        model: The name of the model to use (e.g., "gpt-4").
        api_key: The API key. Defaults to os.getenv("API_KEY").
        base_url: The API base URL. Defaults to os.getenv("BASE_URL").
        on_token: A sync/async function called with each chat token.
        on_reasoning_token: A sync/async function called with each
            streamed reasoning/thinking fragment (OpenRouter
            ``delta.reasoning``, OpenAI o-series, etc.).
        on_create: A sync/async function called with the fully parsed
                   Pydantic model from a `create` call.
        on_message: A sync/async function called with every message (chat or create)
                    useful mostly for login usage.
        reasoning: Optional OpenRouter ``reasoning`` body kwarg
            (e.g. ``{"effort": "high"}`` or ``{"max_tokens": 1024}``).
            Injected only on the streaming ``chat()`` path; the
            structured-output ``create()`` path never carries it,
            because OpenAI's ``parse()`` rejects unknown kwargs.
        **extra_kwargs: Additional arguments for the client (e.g., temperature).
    """
    self._on_token = on_token
    self._on_reasoning_token = on_reasoning_token
    self._on_create = on_create
    self._on_message = on_message
    self._on_toolcall_start = on_toolcall_start
    self._on_toolcall_delta = on_toolcall_delta
    self._on_toolcall_end = on_toolcall_end
    self._reasoning = reasoning

    if model is None:
        model = os.getenv("MODEL")
    if base_url is None:
        base_url = os.getenv("BASE_URL")
    if api_key is None:
        api_key = os.getenv("API_KEY")

    self.model = model
    self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    self.extra_kwargs = extra_kwargs
```

---

## The LLM class: callback dispatch

The callbacks are thin wrappers that handle both sync and async callers. Every callback follows the same pattern: check if set, call, await if coroutine. They are methods rather than module-level functions so subclasses (e.g., lovelaice's `ThinkingLLM`) can override individual callbacks without replacing the whole dispatch mechanism.

```python {name=llm_callbacks}
async def on_token(self, token: str):
    if self._on_token:
        resp = self._on_token(token)
        if inspect.iscoroutine(resp):
            await resp

async def on_reasoning_token(self, token: str):
    if self._on_reasoning_token:
        resp = self._on_reasoning_token(token)
        if inspect.iscoroutine(resp):
            await resp

async def on_create(self, obj):
    if self._on_create:
        resp = self._on_create(obj)
        if inspect.iscoroutine(resp):
            await resp

async def on_message(self, msg: Message):
    if self._on_message:
        resp = self._on_message(msg)
        if inspect.iscoroutine(resp):
            await resp

async def on_toolcall_start(self, call_id: str, name: str):
    if self._on_toolcall_start:
        resp = self._on_toolcall_start(call_id, name)
        if inspect.iscoroutine(resp):
            await resp

async def on_toolcall_delta(self, call_id: str, partial_args: str):
    """Dispatch a streamed-tool-call args update.

    `partial_args` is the accumulated args string for this call so far
    (cumulative across all chunks received), not the incremental fragment.
    Consumers building a live-render of the partial-parsed JSON can use
    this string directly without re-concatenating fragments.
    """
    if self._on_toolcall_delta:
        resp = self._on_toolcall_delta(call_id, partial_args)
        if inspect.iscoroutine(resp):
            await resp

async def on_toolcall_end(self, call_id: str, args: dict):
    if self._on_toolcall_end:
        resp = self._on_toolcall_end(call_id, args)
        if inspect.iscoroutine(resp):
            await resp
```

---

## chat(): consuming the stream

`chat()` always uses the streaming API even though it returns a complete `Message`. Streaming is not more expensive than non-streaming; it just gives callbacks real-time access to tokens. Callers that don't register callbacks get the same complete result as if they had called a non-streaming endpoint.

The stream consists of chunks. Each chunk may contain content tokens, reasoning tokens, tool-call fragments, usage data, or a finish reason. The function accumulates each kind independently:

- Content tokens go into `result_chunks` (joined at the end into the message text).
- Reasoning fragments go into `reasoning_chunks`.
- Tool-call chunks go into `tool_call_accumulator`, keyed by the provider's index. Each slot tracks id, name, and the accumulating JSON string for arguments. The `_started` flag ensures `on_toolcall_start` fires exactly once per call regardless of how many chunks arrive with that index.

The `reasoning` kwarg requires special handling: it is an OpenRouter extension that must go through `extra_body`, not as a direct kwarg to `completions.create`. The OpenAI SDK would reject it otherwise. Per-call values override the instance default.

```python {name=chat_method}
async def chat(
    self,
    messages: list["Message"],
    tools: list | None = None,
    **kwargs,
) -> "Message":
    """
    Sends a message list and returns the full assistant Message.
    If an on_token callback is set, it will be triggered for each token.
    If an on_reasoning_token callback is set and the provider streams
    reasoning fragments (OpenRouter ``delta.reasoning``, OpenAI
    o-series, DeepSeek/Anthropic ``reasoning_content``, Gemini
    ``thoughts``), they are dispatched separately from content tokens.
    """
    result_chunks = []
    reasoning_chunks: list[str] = []
    usage: Usage | None = None
    last_finish_reason: str | None = None
    tool_call_accumulator: dict[int, dict] = {}
    api_messages = [msg.model_dump() for msg in messages]

    call_kwargs = self.extra_kwargs | kwargs
    extra_body = dict(call_kwargs.pop("extra_body", None) or {})
    per_call_reasoning = call_kwargs.pop("reasoning", None)
    reasoning = (
        per_call_reasoning if per_call_reasoning is not None else self._reasoning
    )
    if reasoning is not None and "reasoning" not in extra_body:
        extra_body["reasoning"] = reasoning
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    if tools:
        call_kwargs["tools"] = [tool_to_openai_schema(t) for t in tools]

    async for chunk in await self.client.chat.completions.create(
        model=self.model,  # type: ignore
        messages=api_messages,  # type: ignore
        stream=True,
        stream_options=dict(include_usage=True),  # type: ignore
        **call_kwargs,
    ):
        if chunk.usage:
            usage = Usage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )

        if not chunk.choices:
            continue

        fr = getattr(chunk.choices[0], "finish_reason", None)
        if fr is not None:
            last_finish_reason = fr

        delta = chunk.choices[0].delta

        reasoning = _read_reasoning(delta)
        if reasoning and isinstance(reasoning, str):
            reasoning_chunks.append(reasoning)
            await self.on_reasoning_token(reasoning)

        content = getattr(delta, "content", None)
        if content:
            await self.on_token(content)
            result_chunks.append(content)

        tc_chunks = getattr(delta, "tool_calls", None)
        if tc_chunks:
            for tc in tc_chunks:
                idx = tc.index
                if idx not in tool_call_accumulator:
                    tool_call_accumulator[idx] = {
                        "id": None,
                        "name": None,
                        "args": "",
                    }
                slot = tool_call_accumulator[idx]
                if tc.id and slot["id"] is None:
                    slot["id"] = tc.id
                if tc.function and tc.function.name and slot["name"] is None:
                    slot["name"] = tc.function.name
                if (
                    slot["id"] is not None
                    and slot["name"] is not None
                    and not slot.get("_started")
                ):
                    slot["_started"] = True
                    await self.on_toolcall_start(slot["id"], slot["name"])
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
                    await self.on_toolcall_delta(slot["id"] or "", slot["args"])

    tool_calls = None
    if tool_call_accumulator:
        tool_calls = []
        for idx in sorted(tool_call_accumulator.keys()):
            slot = tool_call_accumulator[idx]
            try:
                parsed_args = json.loads(slot["args"]) if slot["args"] else {}
            except json.JSONDecodeError:
                # Downstream consumers (e.g. lovelaice's harness) validate args against
                # the tool's Pydantic schema; a malformed-JSON case here becomes a
                # validation error one layer up, surfaced back to the LLM as a tool
                # error result. Silent {} keeps lingo agnostic about that policy.
                parsed_args = {}
            tc = ToolCall(
                id=slot["id"] or "", name=slot["name"] or "", arguments=parsed_args
            )
            tool_calls.append(tc)
            await self.on_toolcall_end(tc.id, tc.arguments)

    thinking = "".join(reasoning_chunks) if reasoning_chunks else None
    result = Message.assistant(
        "".join(result_chunks),
        usage=usage,
        tool_calls=tool_calls,
        thinking=thinking,
        stop_reason=last_finish_reason,
    )
    await self.on_message(result)

    return result
```

---

## create(): structured output

`create()` uses OpenAI's `parse()` endpoint rather than the streaming `completions.create`. The difference matters: `parse()` blocks until the response is complete and validates the JSON against the Pydantic schema before returning. Partial structured output is meaningless, so the non-streaming path is correct here.

`reasoning` is intentionally absent: the `parse()` endpoint rejects unknown kwargs. This is also why the same `reasoning` kwarg stored at the instance level is only injected in `chat()`, never here.

`Engine.create()` wraps this with a prompt that describes the schema and instructs the model to fill it. Direct calls to `llm.create()` skip that prompt — the model sees only what is already in `messages`.

```python {name=create_method}
async def create[T: BaseModel](
    self, model: type[T], messages: list["Message"], **kwargs
) -> T:
    """
    Sends a message list and forces the LLM to respond
    with a JSON object matching the Pydantic model
    using the non-streaming `parse` method.

    Fires the on_create callback with the parsed model.
    """
    api_messages = [msg.model_dump() for msg in messages]

    response = await self.client.chat.completions.parse(
        model=self.model,  # type: ignore
        messages=api_messages,  # type: ignore
        response_format=model,
        **(self.extra_kwargs | kwargs),
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse the response from the model.")

    if response.usage:
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    else:
        usage = None

    await self.on_message(Message.assistant(result.model_dump_json(), usage=usage))
    await self.on_create(result)
    return result
```

---

## Assembly

The sections above were written in reading order. This block assembles them in Python's required order. The `<<fragment>>` macros pull in named code blocks; the indentation here is what illiterate adds to each line of those blocks.

```python {name=imports}
import os
import json
import inspect
import base64
import mimetypes
from typing import Any, Callable, Literal, Union
from pydantic import BaseModel, Field
import openai
```

```python {export=lingo/llm.py}
<<imports>>

<<usage_model>>

<<content_models>>

<<tool_call_model>>

<<message_model>>

    <<message_multimodal_factories>>

    <<message_serialization>>

<<schema_helpers>>

<<reasoning_fields>>

<<reasoning_helper>>


class LLM:
    """
    A client for interacting with a Large Language Model.
    Wraps an OpenAI-compatible client.
    """

    <<llm_init>>

    <<llm_callbacks>>

    <<chat_method>>

    <<create_method>>
```

---

## The tool-calling example (exported and CI-tested)

```python {export=examples/index_llm.py}
"""index_llm.py — native tool-calling with lingo's LLM.

Demonstrates:
  - LLM.chat() with tools=[...]
  - Reading Message.tool_calls
  - Feeding tool results back as Message.tool(...)
  - The stop_reason field to detect when the model is done calling tools

Run:
    API_KEY=... python examples/index_llm.py
"""
import asyncio
from lingo import LLM, Message, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"sunny, 22°C in {city}"


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def _main_async():
    llm = LLM()
    messages = [Message.user("What's the weather in Havana, and what is 21 + 21?")]

    while True:
        msg = await llm.chat(messages, tools=[get_weather, add])
        messages.append(msg)

        # stop_reason tells us why the model stopped:
        #   "stop"       → natural end, no more tool calls
        #   "tool_calls" → model wants to call tools
        if msg.stop_reason != "tool_calls":
            print(msg.content)
            break

        for call in msg.tool_calls or []:
            result = await {"get_weather": get_weather, "add": add}[call.name].run(
                **call.arguments
            )
            # tool_call_id is required: the model uses it to match result to call
            messages.append(Message.tool(str(result), tool_call_id=call.id))


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
```

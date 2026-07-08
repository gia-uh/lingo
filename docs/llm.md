# The LLM — a streaming interface to the model

This module has one job: `llm.chat(messages)` takes a list of `Message` objects and returns a single `Message`. Everything here exists to make that one call work correctly across a wide and inconsistent landscape of providers.

Before examining any individual piece, here is the module skeleton. Read it as a table of contents in code form — each `<<named section>>` is defined in the correspondingly titled section below:

```python {export=lingo/llm.py}
<<imports>>

<<token_usage>>

<<content_hierarchy>>

<<tool_call_request>>

<<conversation_turn>>

    <<message_factories>>

    <<message_wire_format>>

<<tool_schema_conversion>>

<<reasoning_extraction>>


class LLM:
    """A client wrapping AsyncOpenAI for streaming chat and structured output."""

    <<llm_constructor>>

    <<llm_callbacks>>

    async def chat(
        self,
        messages: list["Message"],
        tools: list | None = None,
        **kwargs,
    ) -> "Message":
        """Stream a chat request and return the completed Message.

        Streams unconditionally — same cost as non-streaming, and callbacks
        fire in real time. Tool calls accumulate across multiple chunks; the
        returned Message carries them assembled and parsed.
        """
        <<initialize_stream_state>>
        <<prepare_api_call_kwargs>>
        async for chunk in await self.client.chat.completions.create(
            model=self.model,  # type: ignore
            messages=api_messages,  # type: ignore
            stream=True,
            stream_options=dict(include_usage=True),  # type: ignore
            **call_kwargs,
        ):
            <<process_one_stream_chunk>>
        <<finalize_tool_calls>>
        <<return_assembled_message>>

    <<create_structured_output>>
```

The module skeleton already shows something important: `chat()` is not a 130-line function. It is five named operations — initialize state, prepare the call, process each chunk, finalize tool calls, assemble the result — each explained in its own section below.

---

## What this module makes possible

Here is the loop that users of lingo eventually write:

```python
llm = LLM()
messages = [Message.user("What's the weather in Havana, and what is 21 + 21?")]

while True:
    msg = await llm.chat(messages, tools=[get_weather, add])
    messages.append(msg)
    if msg.stop_reason != "tool_calls":
        break
    for call in msg.tool_calls or []:
        result = await tools[call.name].run(**call.arguments)
        messages.append(Message.tool(str(result), tool_call_id=call.id))
```

Three things to notice. First, the conversation is a list of `Message` objects that grows with each turn — the caller owns the history, not the LLM. Second, the model can respond with tool calls instead of text; the caller executes them and feeds the results back. Third, `stop_reason` tells the caller why the model stopped, so it knows whether to loop or break.

Every section of this chapter is an answer to a question this loop raises. What is a `Message`? How does `stop_reason` arrive? How do `tool_calls` get there? What happens inside `chat()`?

---

## The conversation turn

Every API call to OpenAI starts with a list of messages. The API is opinionated: each message carries a `role` — one of `user`, `system`, `assistant`, or `tool` — and `content` that matches what that role is allowed to say. Invalid combinations (a `system` message with `tool_calls`, a `tool` message without `tool_call_id`) cause API errors.

We represent a message as a Pydantic `BaseModel` so invalid field combinations fail at construction time in Python, not at the API boundary where error messages are cryptic.

The assistant variant carries more than content: `tool_calls` (when the model wants to call functions), `thinking` (streamed chain-of-thought from reasoning models), `stop_reason` (why the model stopped generating), and `usage` (token counts for cost tracking). These are all `None` for non-assistant messages; Pydantic's optional fields let callers ignore what they don't use.

```python {name=conversation_turn}
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
```

---

## Constructing messages

Four roles, four factory methods for the common cases. The four classmethods below (`system`, `user`, `assistant`, `tool`) exist because `Message(role="user", content="hello")` is verbose, error-prone, and doesn't read well in application code.

`tool_call_id` on a `tool` message is not optional in practice: the model uses it to match each result to the call that requested it. The API rejects conversations where the id is absent or wrong. `Message.tool(result, tool_call_id=call.id)` makes the obligation visible.

For local files, factory methods handle base64 encoding internally — the caller passes a path and gets a correctly structured message back. The `mime` type is inferred from the extension; `"image/jpeg"` is the fallback when detection fails because most vision models accept JPEG framing even for other formats.

```python {name=message_factories}
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

## What happens at the wire

Pydantic's default `model_dump()` produces the wrong JSON for two cases that come up constantly, so `Message` overrides it.

**The content:null case.** When the model makes tool calls, the assistant message has no text content. Most providers accept `content: ""`, but some (Qwen via OpenRouter) reject the empty string — they misinterpret the conversation and the agent loop stalls with no obvious error. The OpenAI spec explicitly permits `content: null` in this position. We normalize to null whenever `tool_calls` is present and content is empty, unconditionally. It is correct on all providers, not just the strict ones.

**The structured content case.** When content is an `ImageContent`, `AudioContent`, or other `Content` subclass, the API expects a *list* with one dict, not the dict directly: `[{"type": "image_url", "image_url": {...}}]`. A Pydantic default dump gives the dict — the API rejects it.

**Tool calls.** The API's `tools[]` format requires a `"type": "function"` wrapper and JSON-stringified arguments. Pydantic would give structured dicts; the API wants `"arguments": "{\"city\": \"Havana\"}"`.

```python {name=message_wire_format}
def model_dump(self) -> dict[str, Any]:
    """Custom serialization to match the OpenAI wire format."""
    dump = dict(role=self.role)
    content = self.content

    if isinstance(content, str):
        # Tool-calling assistant messages with no text content must carry
        # content:null. Strict providers (e.g. Qwen via OpenRouter) reject
        # content:"" and misinterpret the conversation, stalling the loop.
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

## Content beyond text

Why is `content` in `Message` a `Union` of five types rather than just `str` or `bytes`? Because the API wire format differs structurally by content kind: text is a bare string, images go in `{"type": "image_url", "image_url": {...}}` dicts, audio in `{"type": "input_audio", "input_audio": {...}}` dicts. Without typed subclasses, `model_dump()` would need a chain of `isinstance` checks and string-based format decisions that would be easy to get wrong and hard to test.

Each subclass carries a `type` literal that matches the discriminator the API uses. These are the exact strings OpenAI's schema requires — not lingo's invention.

```python {name=content_hierarchy}
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

---

## What the model sends back

When the model finishes a turn, it may respond with text — or it may skip text entirely and send back a list of tool-call requests. A `ToolCall` is the model's instruction: call this function, pass these arguments, and send me the result with this id so I can match the result to the request.

The `id` field is provider-assigned. It must be echoed back in `Message.tool(result, tool_call_id=call.id)`. Missing or wrong ids cause API errors on every provider that implements the function-calling protocol — this is not a lingo requirement, it is the spec.

```python {name=tool_call_request}
class ToolCall(BaseModel):
    """A tool-call request emitted by the LLM in an assistant message."""

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)
```

---

## The streaming loop

We always use the streaming API, even though `chat()` returns a complete `Message`. This is not slower: the API streams whether you ask for it or not; the non-streaming endpoint just buffers and delivers the whole response at once. Streaming lets `on_token`, `on_reasoning_token`, and `on_toolcall_start` fire in real time without changing the caller's interface. A caller that registers no callbacks gets the same complete result.

Three things can arrive in a stream chunk, and each needs different handling:

- **Content tokens** — text fragments from the model's response. Simple: append, fire callback.
- **Reasoning tokens** — chain-of-thought fragments from reasoning models. Complicated: providers disagree on the field name. See the provider variance section below.
- **Tool-call fragments** — the model's function calls, spread across many chunks. Complicated: the id, name, and JSON argument string each arrive in separate chunks and must be stitched together by index.

### What we track during streaming

Before starting the stream, we initialize five accumulators. `result_chunks` and `reasoning_chunks` collect token strings to be joined at the end. `usage` and `last_finish_reason` are overwritten by the final chunk. `tool_call_accumulator` is keyed by the provider's index — typically 0, 1, 2, ... — and holds one slot per in-flight tool call.

```python {name=initialize_stream_state}
result_chunks: list[str] = []
reasoning_chunks: list[str] = []
usage: Usage | None = None
last_finish_reason: str | None = None
tool_call_accumulator: dict[int, dict] = {}
```

### Preparing the API call

The call kwargs are built by merging instance-level defaults (`self.extra_kwargs`) with per-call overrides (`kwargs`). This gives callers the ability to set defaults at construction time (`LLM(temperature=0.2)`) and override per call.

The `reasoning` kwarg requires special handling. It is an OpenRouter body extension — `{"effort": "high"}` or `{"max_tokens": 1024}` — that controls how much chain-of-thought the model does. It cannot go in `extra_body` directly because the caller might already have `extra_body` contents we need to preserve. It cannot go as a top-level kwarg because the OpenAI SDK would reject an unknown field. Instead, it goes into `extra_body["reasoning"]`, and we handle the merge carefully: a per-call value overrides the instance default, and we don't overwrite an `extra_body["reasoning"]` the caller already set explicitly.

```python {name=prepare_api_call_kwargs}
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
```

### What arrives in each chunk

Each chunk can carry any combination of these event types — they are not mutually exclusive, though in practice they seldom co-occur. `chunk.usage` arrives on the final chunk (via `stream_options={"include_usage": True}`). `chunk.choices` may be absent on the usage-only chunk, which is why we skip processing if it is empty.

The tool-call accumulator is keyed by `tc.index`, which the provider assigns to distinguish parallel tool calls. Each slot starts with `id`, `name`, and `args` set to empty/None and fills in as fragments arrive. The `_started` flag ensures `on_toolcall_start` fires exactly once per call, even though the id and name may arrive in different chunks.

```python {name=process_one_stream_chunk}
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
```

### After the stream: assembling tool calls

Once the stream ends, each slot in the accumulator holds the complete JSON string for that tool call's arguments. We parse it with `json.loads`. If parsing fails — some providers emit malformed JSON — we emit `{}` silently. The downstream consumer (`Engine.invoke`) validates the arguments against the tool's Pydantic schema; a `{}` from a bad stream becomes a validation error one layer up, surfaced back to the model as a tool error result. Keeping lingo agnostic about retry policy here is intentional.

```python {name=finalize_tool_calls}
tool_calls = None
if tool_call_accumulator:
    tool_calls = []
    for idx in sorted(tool_call_accumulator.keys()):
        slot = tool_call_accumulator[idx]
        try:
            parsed_args = json.loads(slot["args"]) if slot["args"] else {}
        except json.JSONDecodeError:
            parsed_args = {}
        tc = ToolCall(
            id=slot["id"] or "", name=slot["name"] or "", arguments=parsed_args
        )
        tool_calls.append(tc)
        await self.on_toolcall_end(tc.id, tc.arguments)
```

### Assembling the final message

The result is a `Message.assistant` built from everything accumulated during the stream. `thinking` is None when the model emitted no reasoning tokens — most models on most providers. `on_message` fires once, after the message is fully assembled, not per-token.

```python {name=return_assembled_message}
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

## A different path for structured output

`create()` uses OpenAI's `parse()` endpoint rather than the streaming `completions.create`. This is the right choice for structured output for a specific reason: partial structured JSON is meaningless. There is no valid intermediate state between "the model has not started" and "the model has produced a valid JSON object matching the schema." Streaming would force us to either buffer everything (paying the same latency as blocking) or deliver an invalid partial object. `parse()` blocks until the response is complete and validates the JSON against the Pydantic schema before returning.

The `reasoning` kwarg is absent from `create()` for a different reason: `parse()` rejects unknown kwargs. This is why `reasoning` is stored at the instance level (`self._reasoning`) and injected only on the `chat()` path via `extra_body`. If you need structured output from a reasoning model, you configure the reasoning budget at the `LLM()` constructor and call `chat()` with a system prompt that constrains the format.

```python {name=create_structured_output}
async def create[T: BaseModel](
    self, model: type[T], messages: list["Message"], **kwargs
) -> T:
    """
    Forces the LLM to respond with a JSON object matching the Pydantic model,
    using the non-streaming `parse` endpoint. Reasoning kwargs are not forwarded
    because `parse()` rejects unknown fields.
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

## The LLM constructor

`LLM()` with no arguments should work for the common case where credentials come from environment variables. `MODEL`, `BASE_URL`, and `API_KEY` are the three env vars because OpenRouter uses that naming convention and lingo was built primarily for OpenRouter.

The `reasoning` kwarg is stored separately from `extra_kwargs` because it needs different treatment: it goes into `extra_body["reasoning"]` on `chat()` calls, not as a top-level kwarg. If we stored it in `extra_kwargs`, the merging logic in `prepare_api_call_kwargs` would try to pass it directly to the SDK, which would reject it.

Callbacks are stored as private attributes (`_on_token`, etc.) and dispatched through methods (`on_token`, etc.). This separation lets subclasses override the dispatch behavior — `ThinkingLLM` in lovelaice does this to prepend reasoning tokens to the response — without replacing the storage.

```python {name=llm_constructor}
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

## Callback dispatch

Every callback follows the same pattern: check if the callback is set, call it, and await if it returned a coroutine. The `inspect.iscoroutine` check makes callbacks transparent to the async/sync distinction — callers don't need to think about whether their callback is an async function.

Callbacks are methods rather than module-level functions because subclasses can override individual dispatch methods without replacing callback storage. `on_toolcall_delta` notes that `partial_args` is the *cumulative* args string, not the incremental fragment — a caller rendering a live partial-JSON view can use it directly without re-concatenating.

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
    """partial_args is the accumulated args string so far — cumulative, not incremental."""
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

## Token accounting

Every response carries a `Usage` record. All three fields default to 0 so partial responses — from mocks, from interrupted streams, from `MockLLM` in tests — don't require callers to special-case a missing usage object. A caller that doesn't care about cost can ignore it; a caller that does can accumulate it across turns.

```python {name=token_usage}
class Usage(BaseModel):
    """Token usage statistics for an LLM interaction."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

---

## Describing tools to the model

When `tools=[...]` is passed to `chat()`, each lingo `Tool` object must be converted to the OpenAI `tools[]` format: a function descriptor with name, description, and a JSON Schema for the parameters.

`_python_type_to_json_schema` handles four cases beyond bare scalars: `Literal[...]` becomes an `enum`, `Optional[X]` / `X | None` becomes a nullable type, `list[X]` becomes `array` with an `items` schema, and unknown types fall back to `"string"`. The string fallback is deliberate — the model can usually infer the right value from context even when the schema is imprecise, and failing hard on an unrecognized annotation would be worse.

`_SCALAR_JSON` is a module-level map rather than a chain of `if/elif` because the same lookup is needed in two places inside the function: once for the Literal case (to infer the enum element type) and once for the base scalar case.

`tool_to_openai_schema` honors three things a naive implementation misses: pre-built schemas (MCP tools ship their own JSON Schema, which we use verbatim), optional defaults (parameters with defaults are excluded from `required`), and docstring descriptions (the `Args:` section in tool docstrings becomes `"description"` on each property).

```python {name=tool_schema_conversion}
_SCALAR_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _python_type_to_json_schema(t) -> dict:
    """Map a Python type annotation to a JSON Schema fragment.

    Handles scalars, Literal (→ enum), list[X] (→ array + items),
    and Optional[X] / X | None (→ nullable). Falls back to string.
    """
    import types as _types
    import typing

    origin = typing.get_origin(t)
    args = typing.get_args(t)

    # Literal[...] -> enum
    if origin is typing.Literal:
        vals = list(args)
        elem = _SCALAR_JSON.get(type(vals[0]), "string") if vals else "string"
        return {"type": elem, "enum": vals}

    # Optional[X] / X | None -> nullable
    union_types = (typing.Union, getattr(_types, "UnionType", None))
    if origin in union_types and origin is not None:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _python_type_to_json_schema(non_none[0])
            it = inner.get("type", "string")
            if isinstance(it, str):
                inner["type"] = [it, "null"]
            return inner

    # list[X] -> array + items
    if t is list or origin is list:
        items = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": items}

    if t is dict or origin is dict:
        return {"type": "object"}

    if t in _SCALAR_JSON:
        return {"type": _SCALAR_JSON[t]}
    return {"type": "string"}  # safe fallback


def tool_to_openai_schema(tool_obj) -> dict:
    """Convert a lingo Tool into an OpenAI tools[] entry.

    If the tool carries a pre-built json_schema (e.g. an MCP tool that
    already ships a full JSON Schema), it is used verbatim. Otherwise the
    schema is derived from parameters() — honoring optional defaults and
    docstring Args: descriptions.
    """
    prebuilt = getattr(tool_obj, "json_schema", None)
    if prebuilt:
        params_schema = prebuilt
    else:
        params = tool_obj.parameters()
        defaults = tool_obj.defaults()
        docs = tool_obj.param_docs()
        properties: dict = {}
        for name, ptype in params.items():
            prop = _python_type_to_json_schema(ptype)
            if name in docs:
                prop["description"] = docs[name]
            if name in defaults:
                prop["default"] = defaults[name]
            properties[name] = prop
        params_schema = {
            "type": "object",
            "properties": properties,
            "required": [n for n in params if n not in defaults],
        }
    return {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description.strip(),
            "parameters": params_schema,
        },
    }
```

---

## Provider variance: reasoning tokens

Several providers expose chain-of-thought reasoning as a separate stream alongside content. There is no standard field name:

- OpenRouter unified and OpenAI o-series: `delta.reasoning`
- DeepSeek and some Anthropic paths: `delta.reasoning_content`
- Gemini: `delta.thoughts`

The OpenAI Python SDK drops fields it doesn't model — except that it preserves unknown fields in `model_extra`. `_read_reasoning` checks both the typed attribute path and `model_extra`, so lingo handles any provider without changes to the streaming loop. Adding a new provider's field name requires one entry in `_REASONING_FIELDS`.

```python {name=reasoning_extraction}
_REASONING_FIELDS = ("reasoning", "reasoning_content", "thoughts")


def _read_reasoning(delta: Any) -> str | None:
    """Extract a reasoning fragment from a stream delta, across provider variance."""
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

## Imports

Imports appear last in this document because no narrative section requires knowing what was imported — Python does, but readers don't. The assembly block at the top places them first.

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

---

## The tool-calling example

This block is both documentation and CI: illiterate exports it to `examples/index_llm.py`, and `tests/test_lingo_md.py` runs it against `MockLLM` on every push. If the public API changes and this example breaks, the test catches it.

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

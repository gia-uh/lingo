import os
import json
import inspect
import base64
import mimetypes
from typing import Any, Callable, Literal, Union
from pydantic import BaseModel, Field
import openai


class Usage(BaseModel):
    """Token usage statistics for an LLM interaction."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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


class ToolCall(BaseModel):
    """A tool-call request emitted by the LLM in an assistant message."""

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


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


class LLM:
    """A client wrapping AsyncOpenAI for streaming chat and structured output."""

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
        result_chunks: list[str] = []
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

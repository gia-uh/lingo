import os
import inspect
import functools
import base64
import mimetypes
from typing import Any, Callable, Literal, Union, Self
from pydantic import BaseModel, Field
import openai


class Usage(BaseModel):
    """Token usage statistics for an LLM interaction."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# --- Multimodal Content Models ---


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


# --- Message Model ---


class Message(BaseModel):
    """A Pydantic model for a single chat message."""

    role: Literal["user", "system", "assistant", "tool"]
    content: Union[
        TextContent, ImageContent, AudioContent, VideoContent, FileContent, str
    ]
    usage: Usage | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Union[Content, str]) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, usage: Usage | None = None) -> "Message":
        return cls(role="assistant", content=content, usage=usage)

    @classmethod
    def tool(cls, content: Any) -> "Message":
        return cls(role="tool", content=content)

    # --- Multimodal Helper Methods ---

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
            # Try to get format from extension (e.g., .mp3 -> mp3)
            _, ext = os.path.splitext(path)
            format = ext.strip(".").lower() or "mp3"

        return cls.user(AudioContent(input_audio={"data": data, "format": format}))

    @classmethod
    def online_video(cls, url: str) -> "Message":
        """Creates a message with an online video URL."""
        return cls.user(VideoContent(video_url={"url": url}))

    def model_dump(self) -> dict[str, Any]:
        """
        Custom model dump to handle structured Content and Pydantic models.
        """
        dump = dict(role=self.role)
        content = self.content

        # 1. Handle raw strings (Standard Text)
        if isinstance(content, str):
            dump["content"] = content

        # 2. Handle structured Content objects (Images, Audio, etc.)
        elif isinstance(content, Content):
            # We dump the Content object as a dictionary
            # OpenAI/OpenRouter expect a single-item list for multimodal content parts
            # or the dict directly depending on the specific API version.
            # To follow the most common multimodal schema:
            dump["content"] = [content.model_dump()]

        # 3. Handle legacy Pydantic model serialization (for structured output responses)
        elif isinstance(content, BaseModel):
            dump["content"] = content.model_dump_json()

        return dump


class LLM:
    """
    A client for interacting with a Large Language Model.
    Wraps an OpenAI-compatible client.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        on_token: Callable[[str], Any] | None = None,
        on_create: Callable[[BaseModel], Any] | None = None,
        on_message: Callable[[Message], Any] | None = None,
        **extra_kwargs,
    ):
        """
        Initializes the LLM client.

        Args:
            model: The name of the model to use (e.g., "gpt-4").
            api_key: The API key. Defaults to os.getenv("API_KEY").
            base_url: The API base URL. Defaults to os.getenv("BASE_URL").
            on_token: A sync/async function called with each chat token.
            on_create: A sync/async function called with the fully parsed
                       Pydantic model from a `create` call.
            on_message: A sync/async function called with every message (chat or create)
                        useful mostly for login usage.
            **extra_kwargs: Additional arguments for the client (e.g., temperature).
        """
        self._on_token = on_token
        self._on_create = on_create
        self._on_message = on_message

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

    async def chat(self, messages: list["Message"], **kwargs) -> "Message":
        """
        Sends a message list and returns the full assistant Message.
        If an on_token callback is set, it will be triggered for each token.
        """
        result_chunks = []
        usage: Usage | None = None
        # Convert Message objects to dictionaries for the API
        api_messages = [msg.model_dump() for msg in messages]

        async for chunk in await self.client.chat.completions.create(
            model=self.model,  # type: ignore
            messages=api_messages,  # type: ignore
            stream=True,
            stream_options=dict(include_usage=True),  # type: ignore
            **(self.extra_kwargs | kwargs),
        ):
            if chunk.usage:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )

            content = chunk.choices[0].delta.content
            if content is None:
                continue

            await self.on_token(content)
            result_chunks.append(content)

        result = Message.assistant("".join(result_chunks), usage=usage)
        await self.on_message(result)

        return result

    async def create[T: BaseModel](
        self, model: type[T], messages: list["Message"], **kwargs
    ) -> T:
        """
        Sends a message list and forces the LLM to respond
        with a JSON object matching the Pydantic model
        using the non-streaming `parse` method.

        Fires the on_create callback with the parsed model.
        """
        # Convert Message objects to dictionaries for the API
        api_messages = [msg.model_dump() for msg in messages]

        # Use the non-streaming, async `parse` method as requested
        response = await self.client.chat.completions.parse(
            model=self.model,  # type: ignore
            messages=api_messages,  # type: ignore
            response_format=model,
            **(self.extra_kwargs | kwargs),
        )
        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError("Failed to parse the response from the model.")

        # Capture usage data
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

import os
import inspect
import base64
import mimetypes
from typing import Any, Callable, Literal, Union
from pydantic import BaseModel, Field
import openai


class Usage(BaseModel):
    """
    Token usage statistics for an LLM interaction.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# --- Multimodal Content Models ---


class Content(BaseModel):
    """
    Base class for all structured message content types (Multimodal).
    """

    type: str

    def __str__(self) -> str:
        """
        By default, structured content is not directly representable as a string.
        Subclasses like TextContent override this.
        """
        raise TypeError(f"Content type '{self.type}' is not a textual content.")


class TextContent(Content):
    """
    Standard text content part for multimodal messages.
    """

    type: Literal["text"] = "text"
    text: str

    def __str__(self) -> str:
        return self.text


class ImageContent(Content):
    """
    Image content part supporting both URLs and base64 data.
    """

    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str] = Field(
        description="Dictionary containing 'url' (can be data:image/...) and 'detail'."
    )


class AudioContent(Content):
    """
    Audio content part for multimodal models.
    """

    type: Literal["input_audio"] = "input_audio"
    input_audio: dict[str, str] = Field(
        description="Dictionary containing 'data' (base64) and 'format'."
    )


class VideoContent(Content):
    """
    Video content part (supported by some OpenRouter/OpenAI models).
    """

    type: Literal["video_url"] = "video_url"
    video_url: dict[str, str] = Field(description="Dictionary containing 'url'.")


class FileContent(Content):
    """
    Generic file content part.
    """

    type: Literal["file_url"] = "file_url"
    file_url: dict[str, str] = Field(description="Dictionary containing 'url'.")


# --- Message Model ---


class Message(BaseModel):
    """
    A Pydantic model representing a single chat message in a conversation.
    """

    role: Literal["user", "system", "assistant", "tool"]
    content: Union[
        TextContent, ImageContent, AudioContent, VideoContent, FileContent, str
    ]
    usage: Usage | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        """Creates a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Union[Content, str]) -> "Message":
        """Creates a user message (can be text or multimodal content)."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, usage: Usage | None = None) -> "Message":
        """Creates an assistant message."""
        return cls(role="assistant", content=content, usage=usage)

    @classmethod
    def tool(cls, content: Any) -> "Message":
        """Creates a tool message."""
        return cls(role="tool", content=content)

    # --- Multimodal Helper Methods ---

    @classmethod
    def local_image(cls, path: str, detail: str = "auto") -> "Message":
        """
        Loads a local image and encodes it as base64 in a user message.

        Args:
            path: Path to the local image file.
            detail: Fidelity level ('low', 'high', 'auto').
        """
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
        """Creates a user message with an online image URL."""
        return cls.user(ImageContent(image_url={"url": url, "detail": detail}))

    @classmethod
    def local_audio(cls, path: str, format: str | None = None) -> "Message":
        """
        Loads a local audio file and encodes it as base64 in a user message.

        Args:
            path: Path to the local audio file.
            format: Audio format (e.g., 'mp3', 'wav'). Inferred if None.
        """
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        if not format:
            _, ext = os.path.splitext(path)
            format = ext.strip(".").lower() or "mp3"

        return cls.user(AudioContent(input_audio={"data": data, "format": format}))

    @classmethod
    def online_video(cls, url: str) -> "Message":
        """Creates a user message with an online video URL."""
        return cls.user(VideoContent(video_url={"url": url}))

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Custom model dump to transform Pydantic structure into an OpenAI-compatible dict.

        Handles:
        - Raw strings for text.
        - Structured Content objects for multimodal parts.
        - Pydantic models (serialized to JSON).
        """
        dump = dict(role=self.role)
        content = self.content

        if isinstance(content, str):
            dump["content"] = content
        elif isinstance(content, Content):
            # OpenAI expects a list of content parts for multimodal input
            dump["content"] = [content.model_dump()]
        elif isinstance(content, BaseModel):
            # For tool results or structured outputs stored in content
            dump["content"] = content.model_dump_json()
        else:
            dump["content"] = str(content)

        return dump


class LLM:
    """
    A client wrapper for interacting with Large Language Models.
    Defaults to OpenAI's AsyncOpenAI client.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        on_token: Callable[[str], Any] | None = None,
        on_create: Callable[[BaseModel], Any] | None = None,
        on_message: Callable[[Message], Any] | None = None,
        **extra_kwargs: Any,
    ):
        """
        Initializes the LLM client.

        Args:
            model: The name of the model (e.g., "gpt-4o"). Defaults to MODEL env var.
            api_key: The API key. Defaults to API_KEY env var.
            base_url: The API base URL. Defaults to BASE_URL env var.
            on_token: Callback triggered for each token in streaming mode.
            on_create: Callback triggered with the parsed model from a `create` call.
            on_message: Callback triggered with every final Message.
            **extra_kwargs: Additional arguments for the OpenAI client.
        """
        self._on_token = on_token
        self._on_create = on_create
        self._on_message = on_message

        self.model = model or os.getenv("MODEL")
        self.client = openai.AsyncOpenAI(
            base_url=base_url or os.getenv("BASE_URL"),
            api_key=api_key or os.getenv("API_KEY"),
        )
        self.extra_kwargs = extra_kwargs

    async def _trigger_callback(self, callback: Callable | None, *args: Any) -> None:
        """Internal helper to execute sync or async callbacks."""
        if callback:
            resp = callback(*args)
            if inspect.iscoroutine(resp):
                await resp

    async def on_token(self, token: str) -> None:
        """Triggers the on_token callback."""
        await self._trigger_callback(self._on_token, token)

    async def on_create(self, obj: BaseModel) -> None:
        """Triggers the on_create callback."""
        await self._trigger_callback(self._on_create, obj)

    async def on_message(self, msg: Message) -> None:
        """Triggers the on_message callback."""
        await self._trigger_callback(self._on_message, msg)

    async def chat(self, messages: list[Message], **kwargs: Any) -> Message:
        """
        Sends a conversation history and returns the assistant's response.

        Args:
            messages: List of Message objects forming the conversation history.
            **kwargs: Overrides for the chat completion call.

        Returns:
            The assistant's Message.
        """
        result_chunks: list[str] = []
        usage: Usage | None = None
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

            content = chunk.choices[0].delta.content if chunk.choices else None
            if content is None:
                continue

            await self.on_token(content)
            result_chunks.append(content)

        result = Message.assistant("".join(result_chunks), usage=usage)
        await self.on_message(result)

        return result

    async def create[T: BaseModel](
        self, model: type[T], messages: list[Message], **kwargs: Any
    ) -> T:
        """
        Forces the LLM to respond with a structured JSON object matching the model.

        Uses the non-streaming `beta.chat.completions.parse` method.

        Args:
            model: The Pydantic model class defining the expected structure.
            messages: List of Message objects for context.
            **kwargs: Overrides for the completion call.

        Returns:
            An instance of the requested Pydantic model.
        """
        api_messages = [msg.model_dump() for msg in messages]

        response = await self.client.beta.chat.completions.parse(
            model=self.model,  # type: ignore
            messages=api_messages,  # type: ignore
            response_format=model,
            **(self.extra_kwargs | kwargs),
        )

        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError(
                "LLM response could not be parsed into the requested model."
            )

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        await self.on_message(Message.assistant(result.model_dump_json(), usage=usage))
        await self.on_create(result)
        return result

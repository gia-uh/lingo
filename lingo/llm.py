import os
import inspect
import functools
from pydantic import BaseModel
from typing import Any, Callable, Literal, Type
import openai
from enum import Enum, auto  # <-- Add this import


class StreamType(Enum):
    """Indicates the type of content being streamed from the LLM."""

    STRUCTURED = auto()  # Structured data (JSON) for internal use
    UNSTRUCTURED = auto()  # Unstructured text for the final reply


class Message(BaseModel):
    """A Pydantic model for a single chat message."""

    role: Literal["user", "system", "assistant", "tool"]
    content: Any

    @classmethod
    def system(cls, content: Any) -> "Message":
        """Factory for a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: Any) -> "Message":
        """Factory for a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: Any) -> "Message":
        """Factory for an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def tool(cls, content: Any) -> "Message":
        """Factory for a tool message."""
        return cls(role="tool", content=content)

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """
        Custom model dump to handle Pydantic models in 'content'.
        """
        # Get the standard model dump
        dump = super().model_dump(*args, **kwargs)

        # If content is a Pydantic model, serialize it to JSON string
        if isinstance(self.content, BaseModel):
            dump["content"] = self.content.model_dump_json()
        else:
            # Otherwise, just convert content to string
            dump["content"] = str(self.content)

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
        # vvv Update callback signature vvv
        callback: Callable[[str, StreamType], Any] | None = None,
        **extra_kwargs,
    ):
        """
        Initializes the LLM client.

        Args:
            model: The name of the model to use (e.g., "gpt-4").
            api_key: The API key. Defaults to os.getenv("API_KEY").
            base_url: The API base URL. Defaults to os.getenv("BASE_URL").
            callback: A sync/async function called with each token chunk
                      and its StreamType.
            **extra_kwargs: Additional arguments for the client (e.g., temperature).
        """
        self.callback = callback

        if model is None:
            model = os.getenv("MODEL")
        if base_url is None:
            base_url = os.getenv("BASE_URL")
        if api_key is None:
            api_key = os.getenv("API_KEY")

        if model is None:
            raise ValueError(
                "Model must be provided or set in the MODEL environment variable."
            )

        self.model = model
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.extra_kwargs = extra_kwargs

    async def chat(self, messages: list["Message"], **kwargs) -> "Message":
        """
        Sends a message list and returns the full assistant Message.
        If a callback is set, it will be triggered for each token
        with StreamType.RESPONSE.
        """
        result_chunks = []

        # Convert Message objects to dictionaries for the API
        api_messages = [msg.model_dump() for msg in messages]

        async for chunk in await self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,  # type: ignore
            stream=True,
            **(self.extra_kwargs | kwargs),
        ):
            content = chunk.choices[0].delta.content
            if content is None:
                continue

            if self.callback:
                # Handle both sync and async callbacks
                # vvv Pass StreamType.RESPONSE vvv
                if inspect.iscoroutinefunction(self.callback):
                    await self.callback(content, StreamType.UNSTRUCTURED)
                else:
                    self.callback(content, StreamType.UNSTRUCTURED)

            result_chunks.append(content)

        return Message.assistant("".join(result_chunks))

    async def create[T: BaseModel](
        self, model: Type[T], messages: list["Message"], **kwargs
    ) -> T:
        """
        Sends a message list and forces the LLM to respond
        with a JSON object matching the Pydantic model.

        This method is streaming-capable and will fire the
        callback with StreamType.THINKING.
        """
        result_chunks = []
        # Convert Message objects to dictionaries for the API
        api_messages = [msg.model_dump() for msg in messages]

        # Use the standard streaming endpoint with JSON mode
        async for chunk in await self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,  # type: ignore
            response_format=dict(type="json_schema", json_schema=model.model_json_schema()),  # type: ignore
            stream=True,
            **(self.extra_kwargs | kwargs),
        ):  # type: ignore
            content = chunk.choices[0].delta.content
            if content is None:
                continue

            if self.callback:
                # Handle both sync and async callbacks
                if inspect.iscoroutinefunction(self.callback):
                    await self.callback(content, StreamType.STRUCTURED)
                else:
                    self.callback(content, StreamType.STRUCTURED)

            result_chunks.append(content)

        full_json_string = "".join(result_chunks)

        try:
            # Parse the aggregated JSON string
            return model.model_validate_json(full_json_string)
        except Exception as e:
            # Handle parsing errors
            raise ValueError(
                f"Failed to parse LLM JSON response: {e}\nResponse: {full_json_string}"
            )

    def wrap(self, target: Callable) -> Callable:
        """
        Wraps a tool function to inject self (the LLM) if type-hinted.
        """
        llm_param_name: str | None = None
        parameters = inspect.signature(target).parameters

        # Find the parameter annotated as LingoLLM
        for name, param in parameters.items():
            # Check for both the class and its string name
            if param.annotation is LLM or param.annotation == "LLM":
                llm_param_name = name
                break

        if llm_param_name is None:
            # Not an LLM-aware tool, return original function
            return target

        @functools.wraps(target)
        async def wrapper(*args, **kwargs):
            # Inject self as the LLM instance
            kwargs[llm_param_name] = self
            return await target(*args, **kwargs)

        # Remove the LLM param from wrapper's annotations
        # to avoid confusion in later inspections (e.g., by MethodTool)
        if hasattr(wrapper, "__annotations__"):
            wrapper.__annotations__.pop(llm_param_name, None)

        return wrapper

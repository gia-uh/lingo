# tests/mocks.py

import asyncio
from typing import Any, List, Optional, Union
from pydantic import BaseModel
from lingo.llm import LLM, Message, Usage


class MockLLM(LLM):
    """
    A mock LLM for testing. Allows pre-programming a queue of responses.
    """

    def __init__(self, responses: Optional[List[Any]] = None, **kwargs):
        super().__init__(model="mock-model", api_key="mock-key", **kwargs)
        self.responses = responses or []
        self.history: List[List[Message]] = []

    async def chat(self, messages: List[Message], **kwargs) -> Message:
        """Simulates a streaming chat response."""
        self.history.append(messages)

        if not self.responses:
            raise ValueError("MockLLM: No responses left in queue.")

        resp = self.responses.pop(0)

        # If the programmed response is already a Message, use it
        if isinstance(resp, Message):
            response_msg = resp
        else:
            # Otherwise, wrap the string in an assistant message with mock usage
            response_msg = Message.assistant(
                str(resp),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        # Simulate streaming tokens
        if isinstance(response_msg.content, str):
            for token in response_msg.content.split():
                await self.on_token(token + " ")

        await self.on_message(response_msg)

        return response_msg

    async def create[T: BaseModel](
        self, model: type[T], messages: List[Message], **kwargs
    ) -> T:
        """Simulates a structured 'parse' response."""
        self.history.append(messages)

        if not self.responses:
            raise ValueError("MockLLM: No responses left in queue.")

        resp = self.responses.pop(0)

        # Validate that the mock response matches the expected model type
        if not isinstance(resp, model):
            raise TypeError(
                f"MockLLM: Expected {model.__name__}, got {type(resp).__name__}"
            )

        # Log virtual message for auditing
        usage = Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        audit_msg = Message.assistant(resp.model_dump_json(), usage=usage)
        await self.on_message(audit_msg)
        await self.on_create(resp)

        return resp

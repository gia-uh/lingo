import asyncio
import contextlib
from pydantic import BaseModel, create_model
from typing import Any
from enum import Enum

from .llm import LLM, Message
from .tools import Tool, ToolResult
from .prompts import (
    DEFAULT_EQUIP_PROMPT,
    DEFAULT_INVOKE_PROMPT,
    DEFAULT_DECIDE_PROMPT,
    DEFAULT_CHOOSE_PROMPT,
    DEFAULT_CREATE_PROMPT,
)
from .utils import generate_pydantic_code


class Context:
    """
    A *mutable* object representing a single interaction.
    It holds the message history and a reference to its parent Flow.
    """

    def __init__(self, llm: LLM, messages: list[Message]):
        self._llm = llm
        self._messages = messages  # This is a mutable list
        self._state_stack: list[list[Message]] = []  # For the fork() context manager

    @property
    def llm(self) -> LLM:
        """Gets the LLM from the parent Flow."""
        return self._llm

    @property
    def messages(self) -> list[Message]:
        """Gets the mutable list of messages for this turn."""
        return self._messages

    def _expand_content(self, *instructions) -> list[Message]:
        """Helper to combine permanent and temporary messages."""
        # Start with a copy of the *current* messages
        all_messages = list(self._messages)

        # Add temporary instructions
        for inst in instructions:
            if isinstance(inst, Message):
                all_messages.append(inst)
            elif isinstance(inst, BaseModel):
                # Serialize Pydantic models to JSON
                all_messages.append(Message.system(inst.model_dump_json()))
            else:
                all_messages.append(Message.system(str(inst)))
        return all_messages

    # --- 1. Context Manipulation ---
    def add(self, message: Message) -> None:
        """
        Mutates the context by appending a message to its
        internal list.
        """
        self._messages.append(message)

    def add_user(self, content: str) -> None:
        """Helper to add a user message."""
        self.add(Message.user(content))

    def add_system(self, content: str) -> None:
        """Helper to add a system message."""
        self.add(Message.system(content))

    def clone(self) -> "Context":
        """
        Returns a new, independent Context instance with a *shallow copy*
        of the current messages, allowing for durable branching.
        """
        return Context(self._llm, list(self._messages))

    @contextlib.contextmanager
    def fork(self):
        """
        A context manager for temporary, "what-if" state.
        All mutations (like .add()) inside the 'with'
        block will be discarded upon exit.
        """
        self._state_stack.append(list(self._messages))

        try:
            yield self
        finally:
            self._messages = self._state_stack.pop()

    # --- 2. Async LLM Calls (Read-Ops) ---
    async def reply(self, *instructions: str | Message) -> Message:
        """
        Calls the LLM with current context + temporary instructions.
        Does NOT mutate the context.
        """
        call_messages = self._expand_content(*instructions)
        return await self.llm.chat(call_messages)

    async def create[T: BaseModel](
        self, model: type[T], *instructions: str | Message
    ) -> T:
        """
        Calls LLM to create a Pydantic model.
        Does NOT mutate the context.
        """
        call_messages = self._expand_content(*instructions)
        model_code = generate_pydantic_code(model)
        prompt = DEFAULT_CREATE_PROMPT.format(
            type=model.__name__,
            signature=model_code,
            docs=model.__doc__ or "",
            format=model.model_json_schema(),
        )
        call_messages.append(Message.system(prompt))
        return await self.llm.create(model, call_messages)

    # --- Internal helper for CoT models ---
    def _create_cot_model(self, name: str, result_cls: type | Enum) -> type[BaseModel]:
        """Creates a dynamic Pydantic model for Chain-of-Thought reasoning."""
        return create_model(name, reasoning=(str, ...), result=(result_cls, ...))

    async def choose[T](self, options: list[T], *instructions: str | Message) -> T:
        """
        Calls the LLM to choose one item from a list of options.
        Does NOT mutate context.
        """
        # Create a mapping of string representations to original objects
        mapping = {str(option): option for option in options}
        enum_type = Enum("Choices", {c: c for c in mapping.keys()})
        model_cls = self._create_cot_model("Choose", enum_type)

        prompt = DEFAULT_CHOOSE_PROMPT.format(
            options="\n".join([f"- {opt}" for opt in mapping.keys()]),
            format=model_cls.model_json_schema(),
        )
        call_messages = self._expand_content(*instructions, Message.system(prompt))

        response = await self.llm.create(model_cls, call_messages)
        return mapping[response.result.value]  # type: ignore

    async def decide(self, *instructions: str | Message) -> bool:
        """
        Calls the LLM to make a True/False decision.
        Does NOT mutate context.
        """
        model_cls = self._create_cot_model("Decide", bool)
        prompt = DEFAULT_DECIDE_PROMPT.format(format=model_cls.model_json_schema())
        call_messages = self._expand_content(*instructions, Message.system(prompt))

        response = await self.llm.create(model_cls, call_messages)
        return response.result  # type: ignore

    async def equip(self, *tools: Tool) -> Tool:
        """
        Calls the LLM to select the most appropriate Tool
        from the parent Flow's tool list.
        Does NOT mutate context.
        """
        tool_map = {tool.name: tool for tool in tools}
        if not tool_map:
            raise ValueError("No tools available in the flow to equip.")

        enum_type = Enum("ToolChoices", {t: t for t in tool_map.keys()})
        model_cls = self._create_cot_model("Equip", enum_type)

        prompt = DEFAULT_EQUIP_PROMPT.format(
            tools="\n".join([f"- {t.name}: {t.description}" for t in tools]),
            format=model_cls.model_json_schema(),
        )
        call_messages = self._expand_content(Message.system(prompt))

        response = await self.llm.create(model_cls, call_messages)
        return tool_map[response.result.value]  # type: ignore

    async def invoke(
        self, tool: Tool, *instructions: str | Message, **kwargs
    ) -> ToolResult:
        """
        1. Calls the LLM to generate parameters for the given Tool.
        2. Merges with **kwargs.
        3. Executes the Tool.
        4. Returns a ToolResult (with data or error).
        Does NOT mutate context.
        """
        parameters: dict[str, Any] = tool.parameters()

        # Filter for params that don't have a value in kwargs
        missing_params = {}
        for k, v_type in parameters.items():
            if k not in kwargs:
                missing_params[k] = (v_type, ...)  # (type, default=Required)

        # Create a dynamic Pydantic model for the *missing* parameters
        model_cls: type[BaseModel] = create_model(
            f"{tool.name.capitalize()}Params", **missing_params
        )

        prompt = DEFAULT_INVOKE_PROMPT.format(
            name=tool.name,
            defaults=kwargs,
            parameters=missing_params,
            description=tool.description,
            format=model_cls.model_json_schema(),
        )

        call_messages = self._expand_content(*instructions, Message.system(prompt))

        try:
            # Generate *only* the missing parameters
            if missing_params:
                generated_params: BaseModel = await self.llm.create(
                    model_cls, call_messages
                )
                generated_dict = generated_params.model_dump()
            else:
                generated_dict = {}

            # Combine provided and generated params
            all_params = {**kwargs, **generated_dict}

            result = await tool.run(**all_params)
            return ToolResult(tool=tool.name, result=result)

        except Exception as e:
            return ToolResult(tool=tool.name, error=str(e))

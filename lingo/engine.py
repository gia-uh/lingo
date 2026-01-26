import asyncio
import json
from pydantic import BaseModel, create_model
from typing import Any, Literal, Self

from .llm import LLM, Message
from .tools import Tool, ToolResult
from .context import Context
from .prompts import (
    DEFAULT_EQUIP_PROMPT,
    DEFAULT_DECIDE_PROMPT,
    DEFAULT_CHOOSE_PROMPT,
    DEFAULT_INVOKE_PROMPT,
    DEFAULT_CREATE_PROMPT,
)


INPUT_SIGNAL = object()


class Engine:
    """
    Holds the LLM and tools, and performs all LLM-related
    operations on a given Context.
    """

    def __init__(self, llm: LLM, tools: list[Tool] | None = None):
        self._llm = llm
        self._tools = list(tools or [])

        # Communication channels for stateful sessions
        self._input_queue = asyncio.Queue[str]()
        self._signal_queue = asyncio.Queue()

    def scope(self, tools: list[Tool]) -> Self:
        """
        Returns a new Engine instance with the additional tools available.
        This creates a lightweight copy, ensuring that parallel flows
        do not interfere with each other's tool sets.
        """
        # Combine current tools with new tools
        # (You might want to add logic here to handle duplicate names if needed)
        new_tool_set = self._tools + tools
        return self.__class__(self._llm, new_tool_set)

    def _expand_content(self, context: Context, *instructions) -> list[Message]:
        """Helper to combine context messages with temporary instructions."""
        # Get messages from the state object
        all_messages = list(context.messages)

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

    # --- 2. Async LLM Calls (Read-Ops) ---
    async def reply(self, context: Context, *instructions: str | Message) -> Message:
        """
        Calls the LLM with current context + temporary instructions.
        """
        call_messages = self._expand_content(context, *instructions)
        return await self._llm.chat(call_messages)

    async def input(self) -> str:
        """
        Pauses the flow and waits for user input from the chat loop.
        The result message is NOT automatically appended to the Context!
        """
        # Signal Lingo.chat that we are waiting
        await self._signal_queue.put(INPUT_SIGNAL)

        # Block until input arrives
        user_text = await self._input_queue.get()

        return user_text

    async def ask(self, context: Context, question: str) -> str:
        """
        Composite method: Replies with a question, then waits for input.
        The result message is NOT automatically appended to the Context!
        """
        await self.reply(context, question)
        return await self.input()

    async def create[T: BaseModel](
        self, context: Context, model: type[T], *instructions: str | Message
    ) -> T:
        """
        Calls LLM to create a Pydantic model.
        (Fixes Issue #2 by removing Pydantic-to-code generation)
        """
        call_messages = self._expand_content(context, *instructions)

        # Addressing Issue #2: Using a simplified prompt without code generation
        prompt_str = DEFAULT_CREATE_PROMPT.format(
            type=model.__name__,
            docs=model.__doc__ or "N/A",
            schema=model.model_json_schema(),
        )

        call_messages.append(Message.system(prompt_str))

        # This will now use the simplified prompt
        return await self._llm.create(model, call_messages)

    # --- Internal helper for CoT models ---
    def _create_cot_model(self, name: str, result_cls) -> type[BaseModel]:
        """Creates a dynamic Pydantic model for Chain-of-Thought reasoning."""
        return create_model(name, reasoning=(str, ...), result=(result_cls, ...))

    async def choose[T](
        self, context: Context, options: list[T], *instructions: str | Message
    ) -> T:
        """
        Calls the LLM to choose one item from a list of options.
        """
        # Create a mapping of string representations to original objects
        mapping = {str(option): option for option in options}
        enum_type = Literal[*mapping.keys()]
        model_cls = self._create_cot_model("Choose", enum_type)

        prompt = DEFAULT_CHOOSE_PROMPT.format(
            options="\n".join([f"- {opt}" for opt in mapping.keys()]),
            format=model_cls.model_json_schema(),
        )
        call_messages = self._expand_content(
            context, *instructions, Message.system(prompt)
        )

        response = await self.create(context, model_cls, *call_messages)
        return mapping[response.result]  # type: ignore

    async def decide(self, context: Context, *instructions: str | Message) -> bool:
        """
        Calls the LLM to make a True/False decision.
        """
        model_cls = self._create_cot_model("Decide", bool)
        prompt = DEFAULT_DECIDE_PROMPT.format(format=model_cls.model_json_schema())
        call_messages = self._expand_content(
            context, *instructions, Message.system(prompt)
        )

        response = await self.create(context, model_cls, *call_messages)
        return response.result  # type: ignore

    async def equip(self, context: Context, *tools: Tool) -> Tool:
        """
        Calls the LLM to select the most appropriate Tool
        from the available tool list.
        """
        _tools = list(tools) or self._tools  # Use engine's tools if none passed

        if not _tools:
            raise ValueError("No tools available.")

        if len(_tools) == 1:
            return _tools[0]

        tool_map = {tool.name: tool for tool in _tools}

        # enum_type = Enum("ToolChoices", {t: t for t in tool_map.keys()})
        enum_type = Literal[*tool_map.keys()]
        model_cls = self._create_cot_model("Equip", enum_type)

        prompt = DEFAULT_EQUIP_PROMPT.format(
            tools="\n".join([f"- {t.name}: {t.description}" for t in _tools]),
            format=model_cls.model_json_schema(),
        )
        call_messages = self._expand_content(context, Message.system(prompt))

        response = await self.create(context, model_cls, *call_messages)
        return tool_map[response.result]  # type: ignore

    async def invoke(
        self, context: Context, tool: Tool, *instructions: str | Message, **kwargs
    ) -> ToolResult:
        """
        1. Calls the LLM to generate parameters for the given Tool.
        2. Merges with **kwargs.
        3. Executes the Tool.
        4. Returns a ToolResult (with data or error).
        """
        try:
            all_params = await self.infer(context, tool, *instructions, **kwargs)

            result = await tool.run(**all_params)
            return ToolResult(tool=tool.name, result=result)

        except Exception as e:
            return ToolResult(tool=tool.name, error=str(e))

    async def infer(
        self, context: Context, tool: Tool, *instructions: str | Message, **kwargs
    ) -> dict[str, Any]:
        """
        Infers parameters for a given tool, and returns a dictionary with
        all the parameters. Can be passed a list of custom parameter values
        if you want to fix some of them.
        """
        parameters: dict[str, Any] = tool.parameters()

        # --- Fix for Issue #4 ---
        # 1. Create a Pydantic model for the *entire* set of parameters
        param_fields = {name: (p_type, ...) for name, p_type in parameters.items()}
        model_cls: type[BaseModel] = create_model(tool.name, **param_fields)

        # 2. Ask the LLM to fill in this *full* model.
        prompt_str = DEFAULT_INVOKE_PROMPT.format(
            name=tool.name,
            description=tool.description,
            parameters=parameters,
            defaults=json.dumps(kwargs),
            schema=model_cls.model_json_schema(),
        )

        call_messages = self._expand_content(
            context, *instructions, Message.system(prompt_str)
        )

        # The LLM generates its "best guess" for all params
        generated_params: BaseModel = await self.create(
            context, model_cls, *call_messages
        )
        generated_dict = generated_params.model_dump()

        # 3. Merge, with **kwargs taking precedence
        all_params = {**generated_dict, **kwargs}
        return all_params

    async def act(self, context: Context, *tools: Tool) -> ToolResult:
        """
        Shortcut for equip/invoke. Selects a tool and runs it immediately,
        returning the tool result.
        """
        tool = await self.equip(context, *tools)
        return await self.invoke(context, tool)

    def stop(self):
        """
        Stops the current flow by raising `StopFlow`, which is
        captured by Flow.execute(...).

        DO NOT USE outside a Flow `execute` method.
        """
        from .flow import StopFlow

        raise StopFlow()

    async def wait(self):
        """
        Wait on the internal queue.
        Used to sincronize with input().
        """
        await self._signal_queue.get()

    async def put(self, msg: str):
        """
        Put a message on the internal queue.
        Used to synchronize with input().
        """
        await self._signal_queue.put(msg)

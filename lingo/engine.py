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
        """
        Initializes the Engine with an LLM and an optional list of tools.

        Args:
            llm: The LLM client to use for operations.
            tools: An optional list of Tool objects available to the engine.
        """
        self._llm = llm
        self._tools = list(tools or [])

        # Communication channels for stateful sessions
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._signal_queue: asyncio.Queue = asyncio.Queue()

    def scope(self, tools: list[Tool]) -> Self:
        """
        Returns a new Engine instance with the additional tools available.
        This creates a lightweight copy, ensuring that parallel flows
        do not interfere with each other's tool sets.

        Args:
            tools: A list of Tool objects to add to the new scoped engine.

        Returns:
            A new Engine instance with the expanded tool set.
        """
        # Combine current tools with new tools
        new_tool_set = self._tools + tools
        return self.__class__(self._llm, new_tool_set)

    def _expand_content(
        self, context: Context, *instructions: str | Message | BaseModel
    ) -> list[Message]:
        """
        Helper to combine context messages with temporary instructions.

        Args:
            context: The conversation context.
            *instructions: Additional instructions to prepend to the LLM call.

        Returns:
            A list of Message objects combining history and instructions.
        """
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

        Args:
            context: The conversation context.
            *instructions: Additional instructions or system messages for the LLM.

        Returns:
            The assistant Message response from the LLM.
        """
        call_messages = self._expand_content(context, *instructions)
        return await self._llm.chat(call_messages)

    async def input(self) -> str:
        """
        Pauses the flow and waits for user input from the chat loop.
        The result message is NOT automatically appended to the Context!

        Returns:
            The raw string input from the user.
        """
        # Signal Lingo.chat that we are waiting
        await self._signal_queue.put(INPUT_SIGNAL)

        # Block until input arrives
        user_text = await self._input_queue.get()

        return user_text

    async def ask(self, context: Context, question: str) -> str:
        """
        Composite method: Replies with a question, then waits for input.
        The result message IS automatically appended to the Context!

        Args:
            context: The conversation context.
            question: The question to ask the user.

        Returns:
            The raw string input from the user.
        """
        response = await self.reply(context, question)
        context.append(response)
        return await self.input()

    async def create[T: BaseModel](
        self, context: Context, model: type[T], *instructions: str | Message
    ) -> T:
        """
        Calls LLM to create a structured Pydantic model.

        Args:
            context: The conversation context.
            model: The Pydantic model class to instantiate.
            *instructions: Additional instructions for the creation process.

        Returns:
            An instance of the requested Pydantic model.
        """
        call_messages = self._expand_content(context, *instructions)

        # Using a simplified prompt without code generation
        prompt_str = DEFAULT_CREATE_PROMPT.format(
            type=model.__name__,
            docs=model.__doc__ or "N/A",
            schema=model.model_json_schema(),
        )

        call_messages.append(Message.system(prompt_str))

        return await self._llm.create(model, call_messages)

    # --- Internal helpers for structured decision making ---

    def _create_cot_model(self, name: str, result_cls: Any) -> type[BaseModel]:
        """
        Creates a dynamic Pydantic model for Chain-of-Thought reasoning.

        Args:
            name: The name of the generated model class.
            result_cls: The type of the 'result' field in the model.

        Returns:
            A new Pydantic model class with 'reasoning' and 'result' fields.
        """
        return create_model(name, reasoning=(str, ...), result=(result_cls, ...))

    async def _structured_choice[T](
        self,
        context: Context,
        name: str,
        result_cls: Any,
        prompt_template: str,
        prompt_kwargs: dict[str, Any],
        *instructions: str | Message,
    ) -> Any:
        """
        Internal helper for making a structured choice using Chain-of-Thought reasoning.

        Args:
            context: The conversation context.
            name: Name for the dynamic CoT model.
            result_cls: The type of the result to be selected.
            prompt_template: The template string for the system prompt.
            prompt_kwargs: Keywords to format the prompt template.
            *instructions: Additional instructions for the LLM.

        Returns:
            The 'result' field from the LLM's structured response.
        """
        model_cls = self._create_cot_model(name, result_cls)
        prompt = prompt_template.format(
            format=model_cls.model_json_schema(), **prompt_kwargs
        )
        call_messages = self._expand_content(
            context, *instructions, Message.system(prompt)
        )

        response = await self.create(context, model_cls, *call_messages)
        return response.result

    async def choose[T](
        self, context: Context, options: list[T], *instructions: str | Message
    ) -> T:
        """
        Calls the LLM to choose one item from a list of options.

        Args:
            context: The conversation context.
            options: A list of objects to choose from.
            *instructions: Additional instructions to guide the choice.

        Returns:
            The selected option from the input list.
        """
        # Create a mapping of string representations to original objects
        mapping = {str(option): option for option in options}
        enum_type = Literal[*mapping.keys()]

        result = await self._structured_choice(
            context,
            "Choose",
            enum_type,
            DEFAULT_CHOOSE_PROMPT,
            {"options": "\n".join([f"- {opt}" for opt in mapping.keys()])},
            *instructions,
        )

        return mapping[result]  # type: ignore

    async def decide(self, context: Context, *instructions: str | Message) -> bool:
        """
        Calls the LLM to make a True/False decision.

        Args:
            context: The conversation context.
            *instructions: Instructions describing the decision to be made.

        Returns:
            A boolean representing the LLM's decision.
        """
        return await self._structured_choice(
            context, "Decide", bool, DEFAULT_DECIDE_PROMPT, {}, *instructions
        )

    async def equip(self, context: Context, *tools: Tool) -> Tool:
        """
        Calls the LLM to select the most appropriate Tool
        from the available tool list.

        Args:
            context: The conversation context.
            *tools: Optional specific tools to choose from. Defaults to engine's tools.

        Returns:
            The selected Tool object.
        """
        _tools = list(tools) or self._tools

        if not _tools:
            raise ValueError("No tools available.")

        if len(_tools) == 1:
            return _tools[0]

        tool_map = {tool.name: tool for tool in _tools}
        enum_type = Literal[*tool_map.keys()]

        result = await self._structured_choice(
            context,
            "Equip",
            enum_type,
            DEFAULT_EQUIP_PROMPT,
            {
                "tools": "\n".join([f"- {t.name}: {t.description}" for t in _tools]),
            },
        )

        return tool_map[result]  # type: ignore

    async def invoke(
        self, context: Context, tool: Tool, *instructions: str | Message, **kwargs: Any
    ) -> ToolResult:
        """
        Infers parameters for a tool, executes it, and returns the result.

        1. Calls the LLM to generate parameters for the given Tool.
        2. Merges with provided **kwargs.
        3. Executes the Tool.
        4. Returns a ToolResult (with data or error).

        Args:
            context: The conversation context.
            tool: The tool to invoke.
            *instructions: Additional instructions for parameter inference.
            **kwargs: Manual parameter overrides.

        Returns:
            A ToolResult object containing the output or error.
        """
        try:
            all_params = await self.infer(context, tool, *instructions, **kwargs)

            result = await tool.run(**all_params)
            return ToolResult(tool=tool.name, result=result)

        except Exception as e:
            return ToolResult(tool=tool.name, error=str(e))

    async def infer(
        self, context: Context, tool: Tool, *instructions: str | Message, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Infers parameters for a given tool using the LLM.

        Args:
            context: The conversation context.
            tool: The tool whose parameters need to be inferred.
            *instructions: Additional instructions for inference.
            **kwargs: Manual parameter overrides.

        Returns:
            A dictionary of inferred and overridden parameters.
        """
        parameters: dict[str, Any] = tool.parameters()

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
        Shortcut for equip/invoke. Selects a tool and runs it immediately.

        Args:
            context: The conversation context.
            *tools: Tools available for selection.

        Returns:
            The result of the tool execution.
        """
        tool = await self.equip(context, *tools)
        return await self.invoke(context, tool)

    def stop(self) -> None:
        """
        Stops the current flow by raising `StopFlow`.

        This is caught by Flow.execute(...) to terminate the workflow early.
        DO NOT USE outside a Flow `execute` method or a skill.
        """
        from .flow import StopFlow

        raise StopFlow()

    async def wait(self) -> None:
        """
        Wait on the internal signal queue.
        Used by the main loop to wait for the engine to signal it needs input.
        """
        await self._signal_queue.get()

    async def put(self, msg: str) -> None:
        """
        Put a message on the internal input queue.
        Used by the main loop to provide user input to a waiting engine.

        Args:
            msg: The user input message.
        """
        await self._input_queue.put(msg)

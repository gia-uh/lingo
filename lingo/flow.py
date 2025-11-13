import abc
import asyncio
from typing import Any, Callable, Coroutine, Self, Type
import uuid

from pydantic import BaseModel

from .context import Context
from .engine import Engine  # Import the new Engine
from .llm import LLM, Message
from .tools import Tool


class Node(abc.ABC):
    """
    An abstract base class for a single, declarative step in a Flow.

    This is the abstract "Component" in a Composite design pattern.
    Each node represents one piece of logic that will be executed
    sequentially, operating on and mutating a shared Context object.
    """

    @abc.abstractmethod
    async def execute(self, context: Context, engine: Engine) -> None:
        """
        Executes the node's logic on the given mutable context,
        using the engine to perform LLM operations.
        """
        pass


# --- "Leaf" Nodes (Primitive Operations) ---


class Append(Node):
    """A Leaf node that appends a message to the context."""

    def __init__(self, msg: Message):
        self.msg = msg

    async def execute(self, context: Context, engine: Engine) -> None:
        context.append(self.msg)


class Prepend(Node):
    """A Leaf node that prepends a message to the context."""

    def __init__(self, msg: Message):
        self.msg = msg

    async def execute(self, context: Context, engine: Engine) -> None:
        context.prepend(self.msg)


class Reply(Node):
    """
    A Leaf node that calls the LLM for a response and adds
    that response to the context.
    """

    def __init__(self, *instructions: str | Message):
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> None:
        response = await engine.reply(context, *self.instructions)
        context.append(response)


class Invoke(Node):
    """
    A Leaf node that performs the equip -> invoke logic for tools.
    It selects the best tool, runs it, and adds the ToolResult
    to the context as a tool message.
    """

    def __init__(self, *tools: Tool):
        if not tools:
            raise ValueError("Invoke node must be initialized with at least one Tool.")
        self.tools = tools

    async def execute(self, context: Context, engine: Engine) -> None:
        # 1. Ask the LLM to select the best tool
        selected_tool = await engine.equip(context, *self.tools)

        # 2. Ask the LLM to generate args for and run the tool
        tool_result = await engine.invoke(context, selected_tool)

        # 3. Add the result to the context
        context.append(Message.tool(tool_result.model_dump()))


class NoOp(Node):
    """A Leaf node that does nothing. Used for empty branches."""

    async def execute(self, context: Context, engine: Engine) -> None:
        pass


class Create(Node):
    """A leaf node to create a custom object."""

    def __init__(self, model: Type[BaseModel], *instructions: Message | str) -> None:
        self.model = model
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> None:
        response = await engine.create(context, model=self.model, *self.instructions)
        context.append(Message.system(response))


class FunctionalNode(Node):
    """
    A wrapper Node that executes a user-provided function.
    """

    def __init__(self, func: Callable[[Context, Engine], Coroutine]):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("Flow function must be a coroutine function")

        self.func = func

    async def execute(self, context: Context, engine: Engine) -> None:
        await self.func(context, engine)


# --- "Composite" Nodes (Containers) ---


class Sequence(Node):
    """
    A Composite node that holds an ordered list of child nodes
    and executes them sequentially. This is the core of the
    Composite pattern.
    """

    def __init__(self, *nodes: Node):
        self.nodes: list[Node] = list(nodes)

    async def execute(self, context: Context, engine: Engine) -> None:
        """Executes each child node in order."""
        for node in self.nodes:
            await node.execute(context, engine)

    def then(self, node: Node) -> Self:
        self.nodes.append(node)
        return self


class Decide(Node):
    """
    A Composite node that handles boolean (True/False) branching.
    It calls engine.decide() and then executes one of two
    child nodes (which are typically Sequence or NoOp nodes).
    """

    def __init__(self, yes: Node, no: Node, *instructions: str | Message):
        self.on_true = yes
        self.on_false = no
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> None:
        result = await engine.decide(context, *self.instructions)
        node_to_run = self.on_true if result else self.on_false
        await node_to_run.execute(context, engine)


class Choose(Node):
    """
    A Composite node that handles multi-way branching.
    It calls engine.choose() and executes the matching
    child node from a dictionary.
    """

    def __init__(self, choices: dict[str, Node], *instructions: str | Message):
        self.choices = choices
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> None:
        option_keys = list(self.choices.keys())
        selected_key = await engine.choose(context, option_keys, *self.instructions)

        node_to_run = self.choices.get(selected_key)
        if node_to_run:
            await node_to_run.execute(context, engine)


class Route(Node):
    """
    A container node that automatically routes between
    two or more flows.
    """

    def __init__(self, *flows: "Flow") -> None:
        if len(flows) < 2:
            raise ValueError("Route needs at least two flows.")

        self.flows = list(flows)

    async def execute(self, context: Context, engine: Engine) -> None:
        # Build a description list for the LLM
        # We use the flow's name and description to guide the choice.
        descriptions = []

        for f in self.flows:
            desc = f.description or "No description provided."
            descriptions.append(f"{f.name}: {desc}")

        instruction = (
            "Read the following option descriptions:\n"
            + "\n".join(descriptions)
            + "\n\nSelect the most appropriate option to handle the conversation."
        )

        # context.choose uses str(option) for the list of keys.
        # Since Flow.__str__ returns the name, the keys will be clean names.
        selected_flow = await engine.choose(context, list(self.flows), instruction)

        # Execute the chosen Flow
        await selected_flow.execute(context, engine)


# --- User-Facing Fluent API ---


class Flow(Sequence):
    """
    A fluent, chainable API for building a declarative
    workflow.

    A Flow is itself a 'Sequence' Node, allowing it to be
    composed of other nodes and even nested inside other Flows.
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
    ):
        super().__init__()  # Initialize the Sequence parent
        self.name = name or f"Flow-{str(uuid.uuid4())}"
        self.description = description or ""

    def __str__(self) -> str:
        return self.name

    def append(self, msg: str | Message) -> "Flow":
        """
        Adds a step to append a message to the context.
        Defaults to system message.
        """
        if isinstance(msg, str):
            msg = Message.system(msg)

        return self.then(Append(msg))

    def prepend(self, msg: str | Message) -> "Flow":
        """
        Adds a step to prepend a message to the context.
        Defaults to system message.
        """
        if isinstance(msg, str):
            msg = Message.system(msg)

        return self.then(Prepend(msg))

    def reply(self, *instructions: str | Message) -> "Flow":
        """
        Adds a step to call the LLM for a response.
        The response will be added to the context as an assistant message.

        Args:
            *instructions: Optional, temporary instructions for this
                           specific reply, e.g., Message.system("Be concise").
        """
        return self.then(Reply(*instructions))

    def invoke(self, *tools: Tool) -> "Flow":
        """
        Adds a step to equip and invoke a tool.
        The LLM will select the best tool from the ones provided
        and execute it. The ToolResult is added to the context.

        Args:
            *tools: One or more Tool objects available for this step.
        """
        return self.then(Invoke(*tools))

    def decide(self, prompt: str, yes: Node, no: Node = NoOp()) -> "Flow":
        """
        Adds a conditional branching step (True/False).
        The LLM will make a boolean decision based on the prompt.

        Args:
            prompt: The question for the LLM (e.g., "Is sentiment positive?").
            on_true: The Node (e.g., another Flow) to execute if True.
            on_false: The Node to execute if False. Defaults to NoOp.
        """
        return self.then(Decide(yes, no, prompt))

    def choose(self, prompt: str, choices: dict[str, Node]) -> "Flow":
        """
        Adds a multi-way branching step.
        The LLM will choose one of the string keys from the 'choices' dict.

        Args:
            prompt: The question for the LLM (e.g., "Which topic?").
            choices: A dictionary mapping string choices to the
                     Node (e.g., another Flow) to execute.
        """
        return self.then(Choose(choices, prompt))

    def create(self, model: Type[BaseModel], prompt: str) -> "Flow":
        """
        Adds a step to create a Pydantic model from the LLM's response.

        Args:
            model: A pydantic class to create.
            instructions: Optional sequence of temporal instructions.
        """
        return self.then(Create(model, prompt))

    def custom(self, func: Callable[[Context, Engine], Coroutine]) -> "Flow":
        return self.then(FunctionalNode(func))

    def route(self, *flows: "Flow") -> "Flow":
        return self.then(Route(*flows))

    async def __call__(self, engine: Engine, messages: list[Message]) -> Context:
        """
        Executes the entire defined flow.

        This is the main entry point to run the pipeline. It creates
        a new Context and Engine, and passes them through every
        node in the flow.

        Args:
            llm: The Language Model instance to use.
            messages: The initial list of messages (e.g., the user's
                      first message).

        Returns:
            The final, mutated Context object after all steps
            have been run.
        """
        # Create the pure-state Context
        context = Context(list(messages))

        # Execute the flow in the Engine
        await self.execute(context, engine)
        return context


# Utilities


def flow(func: Callable[[Context, Engine], Coroutine]) -> Flow:
    """
    A decorator that converts a function into a Flow instance.
    The function must be a coroutine taking (Context, Engine).
    Use the function's docstring as the Flow's description.
    """
    return Flow(name=func.__name__, description=func.__doc__).custom(func)

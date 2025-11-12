import abc
from typing import Any

from .context import Context
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
    async def execute(self, context: Context) -> None:
        """
        Executes the node's logic on the given mutable context.

        This method should perform its action (e.g., add a message,
        call the LLM, or run a tool) by using the methods
        on the 'context' object.
        """
        pass


# --- "Leaf" Nodes (Primitive Operations) ---


class AddSystemMessage(Node):
    """A Leaf node that adds a static system message to the context."""

    def __init__(self, content: str):
        self.content = content

    async def execute(self, context: Context) -> None:
        context.add(Message.system(self.content))


class AddUserMessage(Node):
    """A Leaf node that adds a static user message to the context."""

    def __init__(self, content: str):
        self.content = content

    async def execute(self, context: Context) -> None:
        context.add(Message.user(self.content))


class AddAssistantMessage(Node):
    """A Leaf node that adds a static assistant message to the context."""

    def __init__(self, content: str):
        self.content = content

    async def execute(self, context: Context) -> None:
        context.add(Message.assistant(self.content))


class Reply(Node):
    """
    A Leaf node that calls the LLM for a response and adds
    that response to the context.
    """

    def __init__(self, *instructions: str | Message):
        self.instructions = instructions

    async def execute(self, context: Context) -> None:
        response = await context.reply(*self.instructions)
        context.add(response)


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

    async def execute(self, context: Context) -> None:
        # 1. Ask the LLM to select the best tool
        selected_tool = await context.equip(*self.tools)

        # 2. Ask the LLM to generate args for and run the tool
        tool_result = await context.invoke(selected_tool)

        # 3. Add the result to the context
        context.add(Message.tool(tool_result.model_dump()))


class NoOp(Node):
    """A Leaf node that does nothing. Used for empty branches."""

    async def execute(self, context: Context) -> None:
        pass


# --- "Composite" Nodes (Containers) ---


class Sequence(Node):
    """
    A Composite node that holds an ordered list of child nodes
    and executes them sequentially. This is the core of the
    Composite pattern.
    """

    def __init__(self, *nodes: Node):
        self.nodes: list[Node] = list(nodes)

    async def execute(self, context: Context) -> None:
        """Executes each child node in order."""
        for node in self.nodes:
            await node.execute(context)


class Decide(Node):
    """
    A Composite node that handles boolean (True/False) branching.
    It calls context.decide() and then executes one of two
    child nodes (which are typically Sequence or NoOp nodes).
    """

    def __init__(self, on_true: Node, on_false: Node, *instructions: str | Message):
        self.on_true = on_true
        self.on_false = on_false
        self.instructions = instructions

    async def execute(self, context: Context) -> None:
        result = await context.decide(*self.instructions)
        node_to_run = self.on_true if result else self.on_false
        await node_to_run.execute(context)


class Choose(Node):
    """
    A Composite node that handles multi-way branching.
    It calls context.choose() and executes the matching
    child node from a dictionary.
    """

    def __init__(self, choices: dict[str, Node], *instructions: str | Message):
        self.choices = choices
        self.instructions = instructions

    async def execute(self, context: Context) -> None:
        option_keys = list(self.choices.keys())
        selected_key = await context.choose(option_keys, *self.instructions)

        node_to_run = self.choices.get(selected_key)
        if node_to_run:
            await node_to_run.execute(context)


# --- User-Facing Fluent API ---


class Flow(Sequence):
    """
    A fluent, chainable API for building a declarative
    workflow.

    A Flow is itself a 'Sequence' Node, allowing it to be
    composed of other nodes and even nested inside other Flows.
    """

    def system(self, content: str) -> "Flow":
        """
        Adds a step to append a system message to the context.

        Args:
            content: The text content of the system message.
        """
        self.nodes.append(AddSystemMessage(content))
        return self

    def user(self, content: str) -> "Flow":
        """
        Adds a step to append a user message to the context.

        Args:
            content: The text content of the user message.
        """
        self.nodes.append(AddUserMessage(content))
        return self

    def assistant(self, content: str) -> "Flow":
        """
        Adds a step to append an assistant message to the context.

        Args:
            content: The text content of the assistant message.
        """
        self.nodes.append(AddAssistantMessage(content))
        return self

    def reply(self, *instructions: str | Message) -> "Flow":
        """
        Adds a step to call the LLM for a response.
        The response will be added to the context as an assistant message.

        Args:
            *instructions: Optional, temporary instructions for this
                           specific reply, e.g., Message.system("Be concise").
        """
        self.nodes.append(Reply(*instructions))
        return self

    def invoke(self, *tools: Tool) -> "Flow":
        """
        Adds a step to equip and invoke a tool.
        The LLM will select the best tool from the ones provided
        and execute it. The ToolResult is added to the context.

        Args:
            *tools: One or more Tool objects available for this step.
        """
        self.nodes.append(Invoke(*tools))
        return self

    def decide(self, prompt: str, on_true: Node, on_false: Node = NoOp()) -> "Flow":
        """
        Adds a conditional branching step (True/False).
        The LLM will make a boolean decision based on the prompt.

        Args:
            prompt: The question for the LLM (e.g., "Is sentiment positive?").
            on_true: The Node (e.g., another Flow) to execute if True.
            on_false: The Node to execute if False. Defaults to NoOp.
        """
        instruction = Message.system(prompt)
        self.nodes.append(Decide(on_true, on_false, instruction))
        return self

    def choose(self, prompt: str, choices: dict[str, Node]) -> "Flow":
        """
        Adds a multi-way branching step.
        The LLM will choose one of the string keys from the 'choices' dict.

        Args:
            prompt: The question for the LLM (e.g., "Which topic?").
            choices: A dictionary mapping string choices to the
                     Node (e.g., another Flow) to execute.
        """
        instruction = Message.system(prompt)
        self.nodes.append(Choose(choices, instruction))
        return self

    async def run(self, llm: LLM, messages: list[Message]) -> Context:
        """
        Executes the entire defined flow.

        This is the main entry point to run the pipeline. It creates
        a new Context and passes it through every node in the flow.

        Args:
            messages: The initial list of messages (e.g., the user's
                      first message).

        Returns:
            The final, mutated Context object after all steps
            have been run.
        """
        context = Context(llm, list(messages))
        await self.execute(context)
        return context

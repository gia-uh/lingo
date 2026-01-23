from __future__ import annotations

import abc
import asyncio
from typing import Any, Callable, Coroutine, Type, cast
import uuid

from pydantic import BaseModel

from .context import Context
from .engine import Engine
from .llm import Message
from .tools import Tool


class StopFlow[T](BaseException):
    def __init__(self, reason: str | None = None, default: T | None = None):
        self.reason = reason
        self.default = default


class Node[T](abc.ABC):
    """
    An abstract base class for a single, declarative step in a Flow.

    This is the abstract "Component" in a Composite design pattern.
    Each node represents one piece of logic that will be executed
    sequentially, operating on and mutating a shared Context object.
    """

    @abc.abstractmethod
    async def execute(self, context: Context, engine: Engine) -> T:
        """
        Executes the node's logic on the given mutable context,
        using the engine to perform LLM operations.
        """
        pass


# --- "Leaf" Nodes (Primitive Operations) ---


class Append(Node[None]):
    """A Leaf node that appends a message to the context."""

    def __init__(self, msg: Message):
        self.msg = msg

    async def execute(self, context: Context, engine: Engine) -> None:
        context.append(self.msg)


class Prepend(Node[None]):
    """A Leaf node that prepends a message to the context."""

    def __init__(self, msg: Message):
        self.msg = msg

    async def execute(self, context: Context, engine: Engine) -> None:
        context.prepend(self.msg)


class Reply(Node[str]):
    """
    A Leaf node that calls the LLM for a response and adds
    that response to the context.
    """

    def __init__(self, *instructions: str | Message):
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> str:
        response = await engine.reply(context, *self.instructions)
        context.append(response)
        return str(response.content)


class Decide(Node[bool]):
    """
    A leaf node that returns True or False to a given
    prompt.
    """

    def __init__(self, prompt: str):
        self.prompt = prompt

    async def execute(self, context: Context, engine: Engine) -> bool:
        return await engine.decide(context, self.prompt)


class Choose[T](Node[T]):
    """
    A leaf node that returns one of several items.
    """

    def __init__(self, prompt: str, *options: T):
        self.prompt = prompt
        self.options = list(options)

    async def execute(self, context: Context, engine: Engine) -> T:
        return await engine.choose(context, self.options, self.prompt)


class Act(Node[Any]):
    """
    A Leaf node that performs the equip -> invoke logic for tools.
    It selects the best tool, runs it, and adds the ToolResult
    to the context as a tool message.
    """

    def __init__(self, *tools: Tool):
        if not tools:
            raise ValueError("Invoke node must be initialized with at least one Tool.")
        self.tools = tools

    async def execute(self, context: Context, engine: Engine):
        selected_tool = await engine.equip(context, *self.tools)
        tool_result = await engine.invoke(context, selected_tool)
        context.append(Message.tool(tool_result.model_dump()))
        return tool_result


class NoOp(Node[None]):
    """A Leaf node that does nothing. Used for empty branches."""

    async def execute(self, context: Context, engine: Engine) -> None:
        pass


class Create[T: BaseModel](Node[T]):
    """A leaf node to create a custom object."""

    def __init__(self, model: Type[T], *instructions: Message | str) -> None:
        self.model = model
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> T:
        response = await engine.create(context, model=self.model, *self.instructions)
        context.append(Message.system(response))
        return response


class FunctionalNode(Node):
    """
    A wrapper Node that executes a user-provided function.
    """

    def __init__(self, func: Callable[[Context, Engine], Coroutine]):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("Flow function must be a coroutine function")

        self.func = func

    async def execute(self, context: Context, engine: Engine):
        return await self.func(context, engine)


# --- "Composite" Nodes (Containers) ---


class Sequence[T](Node[T]):
    """
    A Composite node that holds an ordered list of child nodes
    and executes them sequentially. This is the core of the
    Composite pattern.
    """

    def __init__(self, *nodes: Node):
        self.nodes: list[Node] = list(nodes)

    async def execute(self, context: Context, engine: Engine) -> T:
        """Executes each child node in order."""
        result = None

        for node in self.nodes:
            result = await node.execute(context, engine)

        return cast(T, result)

    def then[U](self, node: Node[U]) -> Sequence[U]:
        self.nodes.append(node)
        return cast(Sequence[U], self)


class When[T](Node[T]):
    """
    A Composite node that handles boolean (True/False) branching.
    It calls engine.decide() and then executes one of two
    child nodes (which are typically Sequence or NoOp nodes).
    """

    def __init__(self, then: Node[T], otherwise: Node[T], *instructions: str | Message):
        self.then = then
        self.otherwise = otherwise
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> T:
        result = await engine.decide(context, *self.instructions)
        node_to_run = self.then if result else self.otherwise
        return await node_to_run.execute(context, engine)


class Branch[T](Node[T]):
    """
    A Composite node that handles multi-way branching.
    It calls engine.choose() and executes the matching
    child node from a dictionary.
    """

    def __init__(self, *instructions: str | Message, **choices: Node[T]):
        self.choices = choices
        self.instructions = instructions

    async def execute(self, context: Context, engine: Engine) -> T:
        option_keys = list(self.choices.keys())
        selected_key = await engine.choose(context, option_keys, *self.instructions)

        node_to_run = self.choices.get(selected_key)

        if node_to_run:
            return await node_to_run.execute(context, engine)

        raise KeyError(selected_key)


class Route[T](Node[T]):
    """
    A container node that automatically routes between
    two or more flows.
    """

    def __init__(self, *flows: Flow[T], prompt: str | None = None) -> None:
        if len(flows) < 2:
            raise ValueError("Route needs at least two flows.")

        self.flows = list(flows)
        self.prompt = prompt

    async def execute(self, context: Context, engine: Engine) -> T:
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

        if self.prompt:
            instruction += "\n\n" + self.prompt

        # context.choose uses str(option) for the list of keys.
        # Since Flow.__str__ returns the name, the keys will be clean names.
        selected_flow = await engine.choose(context, list(self.flows), instruction)

        # Execute the chosen Flow
        return await selected_flow.execute(context, engine)


class Repeat[T](Node[T]):
    """
    A Composite node that implements an iterative loop.
    It executes the 'body' until the 'until' condition returns True
    or the 'max_repeats' limit is reached.
    """

    def __init__(self, body: Node[T], until: Node[bool], max_repeats: int = 5):
        self.body = body
        self.until = until
        self.max_repeats = max_repeats

    async def execute(self, context: Context, engine: Engine) -> T:
        """
        Executes the loop. Returns the result of the final iteration's body.
        """
        last_result: Any = None

        for _ in range(self.max_repeats):
            # 1. Execute the main body of the loop
            last_result = await self.body.execute(context, engine)

            # 2. Check the exit condition
            if await self.until.execute(context, engine):
                break

        return last_result  # type: ignore


class Fork[T](Node[T]):
    """
    Executes multiple sub-flows in parallel using context clones.
    Synthesizes results using an aggregator Node or string prompt.
    """

    def __init__(
        self,
        branches: list[Node[Any]],
        aggregator: str | Node[T] = "Summarize these inputs",
    ):
        self.branches = branches
        self.aggregator = aggregator

    async def execute(self, context: Context, engine: Engine) -> T:
        # 1. Run all branches in parallel with isolated context clones
        # context.clone() ensures that 'scratchpad' messages stay in the branch
        results = await asyncio.gather(
            *(node.execute(context.clone(), engine) for node in self.branches)
        )

        # 2. Prepare aggregator context (clone of original + branch results)
        agg_context = context.clone()
        for res in results:
            # Format results for the aggregator's context
            content = res.model_dump_json() if isinstance(res, BaseModel) else str(res)
            agg_context.append(Message.system(content))

        # 3. Perform aggregation
        if isinstance(self.aggregator, str):
            # Default string aggregator is wrapped in a Reply node
            agg_node = Reply(self.aggregator)
            result = await agg_node.execute(agg_context, engine)
        else:
            # Custom Node[T] allows for structured data aggregation
            result = await self.aggregator.execute(agg_context, engine)

        # 4. Integrate the final synthesized result into the main context
        # This keeps the main history clean while informing it of the fork's conclusion
        summary_text = (
            result.model_dump_json() if isinstance(result, BaseModel) else str(result)
        )
        context.append(Message.assistant(summary_text))

        return result  # type: ignore


class Retry[T](Node[T]):
    """
    Attempts to execute a node multiple times.
    On failure, rolls back context and runs a 'fixer' node to update
    the context for the next attempt.
    """

    def __init__(self, body: Node[T], fixer: Node[Any], max_retries: int = 3):
        self.body = body
        self.fixer = fixer
        self.max_retries = max_retries

    async def execute(self, context: Context, engine: Engine) -> T:
        for i in range(self.max_retries):
            try:
                # Try the body atomically
                with context.atomic():
                    return await self.body.execute(context, engine)
            except Exception as e:
                # If this was the last attempt, re-raise the error
                if i == self.max_retries - 1:
                    raise e

                # Rollback happened automatically via atomic().
                # Now, run the fixer to add "hints" or error info to the context.
                with context.fork():
                    context.append(
                        Message.system(f"The attempt failed with exception: {e}")
                    )
                    fix = await self.fixer.execute(context, engine)

                context.append(Message.system(fix))

        raise AssertionError("We can't be here")


class Attempt[T](Node[T]):
    """
    Tries to execute the 'body' node. If it fails, the context is rolled back
    and the 'fallback' node is executed instead.
    """

    def __init__(self, body: Node[T], fallback: Node[T]):
        self.body = body
        self.fallback = fallback

    async def execute(self, context: Context, engine: Engine) -> T:
        try:
            # Use the atomic manager to ensure failure doesn't pollute history
            with context.atomic():
                return await self.body.execute(context, engine)
        except Exception:
            # Fallback is executed in the cleaned context
            return await self.fallback.execute(context, engine)


class Compress(Node[None]):
    """
    Prunes the context window.
    - If aggregator is provided: Context becomes [Prefix, Summary].
    - If aggregator is None: Context becomes [Prefix, Last N] (Limit mode).
    """

    def __init__(
        self,
        n: int | None = None,
        prefix_k: int = 1,
        aggregator: str | Node[str] | None = "Summarize the conversation history.",
    ):
        self.n = n
        self.prefix_k = prefix_k
        self.aggregator = aggregator

    async def execute(self, context: Context, engine: Engine) -> None:
        # 1. Identify the 'Anchor' Prefix (e.g., system prompts)
        prefix = context.messages[: self.prefix_k]

        # 2. Identify the 'Working' messages to summarize or keep
        if self.n is not None:
            working = context.messages[-self.n :]
        else:
            working = list(context.messages)

        # Ensure we don't duplicate the prefix in the final message set
        clean_working = [m for m in working if m not in prefix]

        # 3. Handle Limit Mode (Aggregator is None)
        if self.aggregator is None:
            context.messages[:] = prefix + clean_working
            return None

        # 4. Handle Compression Mode
        # Create a temp context of the prefix + working messages to summarize
        summary_ctx = Context(prefix + clean_working)

        if isinstance(self.aggregator, str):
            # Use the engine directly to get a summary string
            summary_msg = await engine.reply(summary_ctx, self.aggregator)
            summary_text = str(summary_msg.content)
        else:
            # Use a custom node/flow for aggregation
            summary_text = await self.aggregator.execute(summary_ctx, engine)

        # 5. Prune and Update the Context
        # Final state: Anchor Prefix + a System message containing the summary
        context.messages[:] = prefix + [Message.system(f"SUMMARY: {summary_text}")]


class Scope[T](Node[T]):
    """
    A Composite node that creates a temporary tool scope.
    It clones the Engine, adds the new tools, and executes
    its body with the new Engine instance.
    """

    def __init__(self, tools: list[Tool], body: Node[T]):
        self.tools = tools
        self.body = body

    async def execute(self, context: Context, engine: Engine) -> T:
        # 1. Create a NEW engine instance with the scoped tools
        # This is safe for concurrency because we don't mutate 'engine'
        scoped_engine = engine.scope(self.tools)

        # 2. Execute the body with the new engine
        return await self.body.execute(context, scoped_engine)


class Flow[T](Sequence[T]):
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

    def append(self, msg: str | Message) -> Flow[None]:
        """
        Adds a step to append a message to the context.
        Defaults to system message.
        """
        if isinstance(msg, str):
            msg = Message.system(msg)

        return self.then(Append(msg))  # type: ignore

    def prepend(self, msg: str | Message) -> Flow[None]:
        """
        Adds a step to prepend a message to the context.
        Defaults to system message.
        """
        if isinstance(msg, str):
            msg = Message.system(msg)

        return self.then(Prepend(msg))  # type: ignore

    def reply(self, *instructions: str | Message) -> Flow[str]:
        """
        Adds a step to call the LLM for a response.
        The response will be added to the context as an assistant message.

        Args:
            *instructions: Optional, temporary instructions for this
                           specific reply, e.g., Message.system("Be concise").
        """
        return self.then(Reply(*instructions))  # type: ignore

    def act(self, *tools: Tool) -> Flow[Any]:
        """
        Adds a step to equip and invoke a tool.
        The LLM will select the best tool from the ones provided
        and execute it. The ToolResult is added to the context.

        Args:
            *tools: One or more Tool objects available for this step.
        """
        return self.then(Act(*tools))  # type: ignore

    def when(self, prompt: str, then: Node[T], otherwise: Node[T] = NoOp()) -> Flow[T]:
        """
        Adds a conditional branching step (True/False).
        The LLM will make a boolean decision based on the prompt.

        Args:
            prompt: The question for the LLM (e.g., "Is sentiment positive?").
            yes: The Node (e.g., another Flow) to execute if True.
            no: The Node to execute if False. Defaults to NoOp.
        """
        return self.then(When(prompt, then=then, otherwise=otherwise))  # type: ignore

    def decide(self, prompt: str) -> Flow[bool]:
        """
        Adds a step that computes a bool responde for
        a specific question.
        """
        return self.then(Decide(prompt))  # type: ignore

    def choose[R](self, prompt: str, *options: R) -> Flow[R]:
        """
        Adds a step that computes a bool responde for
        a specific question.
        """
        return self.then(Choose(prompt, *options))  # type: ignore

    def branch(self, prompt: str, **choices: Node[T]) -> Flow[T]:
        """
        Adds a multi-way branching step.
        The LLM will choose one of the string keys from the 'choices' dict.

        Args:
            prompt: The question for the LLM (e.g., "Which topic?").
            choices: A dictionary mapping string choices to the
                     Node (e.g., another Flow) to execute.
        """
        return self.then(Branch(choices, prompt))  # type: ignore

    def create[R: BaseModel](self, model: Type[R], prompt: str) -> Flow[R]:
        """
        Adds a step to create a Pydantic model from the LLM's response.

        Args:
            model: A pydantic class to create.
            instructions: Optional sequence of temporal instructions.
        """
        return self.then(Create(model, prompt))  # type: ignore

    def custom(self, func: Callable[[Context, Engine], Coroutine]) -> Flow:
        return self.then(FunctionalNode(func))  # type: ignore

    def route[R](self, *flows: Flow[R], prompt: str | None = None) -> Flow[R]:
        return self.then(Route(*flows, prompt=prompt))  # type: ignore

    def repeat[U](
        self, body: Node[U], until: str | Node[bool], max_repeats: int = 5
    ) -> Flow[U]:
        """
        Adds an iterative loop to the flow.

        Args:
            body: The Node or Flow to repeat.
            until: A prompt string (LLM decision) or a Node returning bool.
            max_repeats: Safety limit to prevent infinite loops.
        """
        # Wrap string prompts into a Decide node automatically
        condition = Decide(until) if isinstance(until, str) else until
        return self.then(Repeat(body, condition, max_repeats))  # type: ignore

    def fork[U](
        self, *flows: Node[Any], aggregator: str | Node[U] = "Summarize these inputs"
    ) -> Flow[U]:
        """
        Adds a parallel fork step to the flow.
        The Flow's return type transitions to the type U returned by the aggregator.
        """
        return self.then(Fork(list(flows), aggregator))  # type: ignore

    def retry(self, fixer: Node[Any], max_retries: int = 3) -> Flow[T]:
        """
        Wraps all previously defined steps into a Retry block.
        If an exception occurs, the context is rolled back via atomic(),
        the 'fixer' node is executed to provide feedback, and the flow
        re-starts from the first step.
        """
        # Package existing nodes as the 'body' to be retried
        body = Sequence(*self.nodes)

        # Replace current flow logic with the wrapped Retry node
        self.nodes = [Retry(body, fixer, max_retries)]
        return self

    def fallback(self, fallback: Node[T]) -> Flow[T]:
        """
        Wraps all previously defined steps into an Attempt block.
        If the primary flow fails, the context is rolled back via atomic()
        and the fallback node is executed instead.
        """
        # Package existing nodes as the primary attempt 'body'
        body = Sequence(*self.nodes)

        # Replace current flow logic with the wrapped Attempt node
        # The Flow's return type transitions to match the fallback/body type U
        self.nodes = [Attempt(body, fallback)]
        return self  # type: ignore

    def then[U](self, node: Node[U]) -> Flow[U]:
        return cast(Flow[U], super().then(node))

    def compress(
        self,
        n: int | None = None,
        prefix_k: int = 1,
        aggregator: str | Node[str] | None = "Summarize the history.",
    ) -> Flow[None]:
        """
        Prunes the context history.
        If aggregator is None, it acts as a simple sliding window (Limit).
        Returns the summary string (or None if in Limit mode).
        """
        return self.then(Compress(n, prefix_k, aggregator))  # type: ignore

    async def execute(self, context: Context, engine: Engine) -> T:
        try:
            return await super().execute(context, engine)
        except StopFlow as e:
            return cast(T, e.default)


def flow(func: Callable[[Context, Engine], Coroutine]) -> Flow:
    """
    A decorator that converts a function into a Flow instance.
    The function must be a coroutine taking (Context, Engine).
    Use the function's docstring as the Flow's description.
    """
    return Flow(name=func.__name__, description=func.__doc__).custom(func)

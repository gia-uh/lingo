import asyncio
from typing import Callable, Coroutine, Iterator, Protocol, Optional
from purely import Registry, ensure
from pydantic import BaseModel, Field, create_model

from .skills import Skill
from .flow import Flow, flow
from .llm import LLM, Message
from .tools import Tool, tool
from .context import Context
from .prompts import DEFAULT_SYSTEM_PROMPT
from .engine import Engine
from .state import State


class Conversation(Protocol):
    """
    Protocol defining the expected interface for a conversation history container.
    """

    def append(self, message: Message, /) -> None: ...
    def __iter__(self) -> Iterator[Message]: ...
    def __getitem__(self, index: int, /) -> Message: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...


class Lingo:
    """
    The main orchestrator for a Lingo application.

    Manages the chat session, global message history, and the execution of flows.
    """

    def __init__(
        self,
        name: str = "Lingo",
        description: str = "A friendly chatbot.",
        llm: LLM | None = None,
        skills: list[Skill] | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = False,
        conversation: Conversation | None = None,
        router_prompt: str | None = None,
        state: State | None = None,
    ) -> None:
        """
        Initializes a Lingo instance.

        Args:
            name: The name of the bot.
            description: A brief description of the bot's purpose.
            llm: The LLM client to use. Defaults to a new LLM instance.
            skills: Initial list of skills.
            tools: Initial list of tools.
            system_prompt: The base system prompt template.
            verbose: Whether to enable verbose logging.
            conversation: Custom conversation history container.
            router_prompt: Custom prompt for skill routing.
            state: Global application state.
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt.format(
            name=self.name, description=self.description
        )
        self.llm = llm or LLM()
        self.skills: list[Skill] = skills or []
        self.tools: list[Tool] = tools or []
        self.messages: Conversation = (
            conversation if conversation is not None else list[Message]()
        )
        self._verbose = verbose
        self._router_prompt = router_prompt
        self.state = state

        self.registry = Registry()
        self.registry.register(self)
        self.registry.register(self.llm)
        self.registry.register(self.state)

        self._before_hooks: list[Callable] = []
        self._after_hooks: list[Callable] = []
        self._filters: dict[str, Flow] = {}

        # Session State
        self._runner_task: Optional[asyncio.Task] = None
        self._active_engine: Optional[Engine] = None
        self._active_context: Optional[Context] = None

    def before(self, func: Callable[[Context, Engine], Coroutine]) -> Callable:
        """
        Decorator to register a function to run BEFORE the main flow/skills.

        Useful to, e.g., add few-shot examples to dynamically improve
        skill routing.
        """
        self._before_hooks.append(self.registry.inject(func))
        return func

    def after(self, func: Callable[[Context, Engine], Coroutine]) -> Callable:
        """
        Decorator to register a function to run AFTER the main flow/skills.

        Usable to, e.g., compress or clean up the context.
        """
        self._after_hooks.append(self.registry.inject(func))
        return func

    def skill(self, func: Callable[[Context, Engine], Coroutine]) -> Skill:
        """
        Decorator to register a method as a skill for the chatbot.
        """
        self.skills.append(s := Skill(self.registry, func))
        return s

    def tool(self, func: Callable) -> Tool:
        """
        Decorator to register a function as a tool.
        """
        self.tools.append(t := tool(self.registry.inject(func)))
        return t

    def _build_flow(self) -> Flow:
        """
        Internal helper to construct the main execution flow.

        The flow includes system prompts, before-hooks, filters,
        routed skills, and after-hooks.
        """
        flow_obj = Flow("Main flow")

        for hook in self._before_hooks:
            flow_obj.custom(hook)

        if self._filters:
            flow_obj.then(self._build_filters())

        if len(self.skills) == 1:
            flow_obj.then(self.skills[0].build())
        elif len(self.skills) > 1:
            flow_obj.route(
                *[s.build() for s in self.skills], prompt=self._router_prompt
            )

        for hook in self._after_hooks:
            flow_obj.custom(hook)

        if not self.skills:
            flow_obj.reply()

        return flow_obj

    async def chat(self, msg: str) -> Message:
        """
        Interacts with the bot.

        Handles both starting new sessions and resuming paused ones.
        Synchronizes context back to global history and handles early exits.

        Args:
            msg: The user input message.

        Returns:
            The last message from the assistant.
        """
        # Ensure system prompt is in history if empty
        if not self.messages:
            self.messages.append(Message.system(self.system_prompt))

        user_message = Message.user(msg)

        # 1. Start or Resume turn
        if self._runner_task and not self._runner_task.done():
            # RESUME turn
            self.messages.append(user_message)
            ensure(self._active_context).append(user_message)
            await self._handle_resumption(msg)
        else:
            # START turn
            self.messages.append(user_message)
            await self._handle_new_session()

        # 2. Wait for bot execution to finish or wait for input
        await self._wait_for_execution()

        # 3. Synchronize history
        if self._active_context:
            local_msgs = list(self._active_context.messages)

            # Find divergence point
            i = 0
            while (
                i < len(local_msgs)
                and i < len(self.messages)
                and local_msgs[i] == self.messages[i]
            ):
                i += 1

            if i < len(local_msgs):
                for m in local_msgs[i:]:
                    self.messages.append(m)

        # 4. Cleanup session state if task is done
        if self._runner_task and self._runner_task.done():
            self._clear_session()

        # 5. Return latest assistant message
        # Search backwards for the first assistant message
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message

        return self.messages[-1]

    async def _handle_resumption(self, msg: str) -> None:
        """RESUME: Feed input to the waiting engine."""
        await ensure(self._active_engine).put(msg)

    async def _handle_new_session(self) -> None:
        """START: Create new session and initialize the runner task."""
        context = Context(list(self.messages))
        engine = Engine(self.llm, self.tools)
        flow_obj = self._build_flow()

        # Store active session components
        self._active_engine = engine
        self._active_context = context
        self._runner_task = asyncio.create_task(flow_obj.execute(context, engine))

    async def _wait_for_execution(self) -> None:
        """
        Waits for EITHER the signal that the bot is waiting for input (engine.input called)
        OR the runner task to finish completely.
        """
        signal_task = asyncio.create_task(ensure(self._active_engine).wait())

        # Ensure runner task exists
        runner = ensure(self._runner_task)

        done, pending = await asyncio.wait(
            [signal_task, runner], return_when=asyncio.FIRST_COMPLETED
        )

        # Cleanup the signal waiter if it didn't fire (i.e., task finished)
        if signal_task in pending:
            signal_task.cancel()

        # If the flow crashed, re-raise the exception
        if runner.done() and (exc := runner.exception()):
            self._clear_session()
            raise exc

    def _clear_session(self) -> None:
        """Resets the active session state."""
        self._runner_task = None
        self._active_engine = None
        self._active_context = None

    def _build_filters(self) -> Flow:
        """
        Constructs a dynamic flow to route based on '@when' filters.
        """
        # We need a stable mapping between option_i and the filter keys (conditions)
        filter_keys = list(self._filters.keys())

        fields = {
            f"option_{i}": (bool, Field(description=f"True if {key}"))
            for i, key in enumerate(filter_keys)
        }

        model = create_model("Filter", **fields)

        @flow
        async def router(context: Context, engine: Engine) -> None:
            response: BaseModel = await engine.create(
                context, model, "Determine which of these options is true."
            )
            filters = response.model_dump()

            for i, key in enumerate(filter_keys):
                if filters.get(f"option_{i}"):
                    await self._filters[key].execute(context, engine)

        return router

    def when(self, condition: str) -> Callable:
        """
        Decorator to register a conditional 'reflex' flow.

        Reflexes run before skills and can be used for guardrails or global commands.
        """

        def decorator(func: Callable[[Context, Engine], Coroutine]) -> Flow:
            f = flow(func)
            self._filters[condition] = f
            return f

        return decorator

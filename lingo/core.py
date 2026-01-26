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
    def append(self, message: Message, /): ...
    def __iter__(self) -> Iterator[Message]: ...
    def __getitem__(self, index: int, /) -> Message: ...
    def clear(self): ...
    def __len__(self) -> int: ...


class Lingo:
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

    def before(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to register a function to run BEFORE the main flow/skills.

        Useful to, e.g., add few-shot examples to dynamically improve
        skill routing.
        """
        self._before_hooks.append(self.registry.inject(func))
        return func

    def after(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to register a function to run AFTER the main flow/skills.

        Usable to, e.g., compress or clean up the context.
        """
        self._after_hooks.append(self.registry.inject(func))
        return func

    def skill(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to register a method as a skill for the chatbot.
        """
        self.skills.append(s := Skill(self.registry, func))
        return s

    def tool(self, func: Callable):
        """
        Decorator to register a function as a tool.
        """
        self.tools.append(t := tool(self.registry.inject(func)))
        return t

    def _build_flow(self) -> Flow:
        flow = Flow("Main flow").prepend(self.system_prompt)

        for hook in self._before_hooks:
            flow.custom(hook)

        if self._filters:
            flow.then(self._build_filters())

        if len(self.skills) == 1:
            flow.then(self.skills[0].build())
        elif len(self.skills) > 1:
            flow.route(*[s.build() for s in self.skills], prompt=self._router_prompt)

        for hook in self._after_hooks:
            flow.custom(hook)

        if not self.skills:
            flow.reply()

        return flow

    async def chat(self, msg: str) -> Message:
        """
        Interacts with the bot.
        Handles both starting new sessions and resuming paused ones.
        """
        # 1. Update Global History
        self.messages.append(Message.user(msg))

        # 2. Determine Logic: Resume or Start
        if self._runner_task and not self._runner_task.done():
            # RESUME: Feed input to the waiting engine
            # Note: Engine.input() will handle appending this msg to the local context
            await ensure(self._active_engine).put(msg)
        else:
            # START: Create new session
            context = Context(list(self.messages))
            engine = Engine(self.llm, self.tools)
            flow = self._build_flow()

            # Store active session components
            self._active_engine = engine
            self._active_context = context
            self._runner_task = asyncio.create_task(flow.execute(context, engine))

        # 3. Wait for Bot Response (Signal or Completion)
        # We wait for EITHER the signal that the bot is waiting for input (engine.input called)
        # OR the task to finish completely.

        signal_task = asyncio.create_task(ensure(self._active_engine).wait())

        done, pending = await asyncio.wait(
            [signal_task, self._runner_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cleanup the signal waiter if it didn't fire (i.e., task finished)
        if signal_task in pending:
            signal_task.cancel()

        # 4. Sync State and Return

        # If the flow crashed, re-raise the exception
        if self._runner_task.done() and (exc := self._runner_task.exception()):
            # Clean up before raising
            self._runner_task = None
            self._active_engine = None
            self._active_context = None
            raise exc

        # Check for new messages generated in the context
        if self._active_context:
            # We append only the NEW messages from the context to the global history
            # The logic assumes strict append-only behavior in both lists
            new_messages = self._active_context.messages[len(self.messages) :]
            for m in new_messages:
                self.messages.append(m)

        # If the task finished cleanly, clear session state
        if self._runner_task.done():
            self._runner_task = None
            self._active_engine = None
            self._active_context = None

        return self.messages[-1]

    def _build_filters(self):
        fields = {
            f"option_{i}": (bool, Field(description=f"True if {key}"))
            for i, key in enumerate(self._filters)
        }

        model = create_model("Filter", **fields)

        @flow
        async def router(context: Context, engine: Engine):
            response: BaseModel = await engine.create(
                context, model, "Determine which of these options is true."
            )
            filters = response.model_dump()

            for key, value in filters.items():
                if value:
                    await self._filters[key].execute(context, engine)

        return router

    def when(self, condition: str):
        def decorator(func: Callable[[Context, Engine], Coroutine]) -> Flow:
            f = flow(func)
            self._filters[condition] = f
            return f

        return decorator

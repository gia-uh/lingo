from typing import Callable, Coroutine, Iterator, Protocol
from purely import Registry

from lingo.skills import Skill
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
        self.verbose = verbose
        self.router_prompt = router_prompt
        self.state = state

        self.registry = Registry()
        self.registry.register(self)
        self.registry.register(self.llm)
        self.registry.register(self.state)

        self.before_hooks: list[Callable] = []
        self.after_hooks: list[Callable] = []

    def before(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to register a function to run BEFORE the main flow/skills.

        Useful to, e.g., add few-shot examples to dynamically improve
        skill routing.
        """
        self.before_hooks.append(self.registry.inject(func))
        return func

    def after(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to register a function to run AFTER the main flow/skills.

        Usable to, e.g., compress or clean up the context.
        """
        self.after_hooks.append(self.registry.inject(func))
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
        Automatically injects the LLM if necessary.
        """
        self.tools.append(t := tool(self.registry.inject(func)))
        return t

    def _build_flow(self) -> Flow:
        flow = Flow("Main flow").prepend(self.system_prompt)

        for hook in self.before_hooks:
            flow.custom(hook)

        if not self.skills:
            flow.reply()
        elif len(self.skills) == 1:
            flow.then(self.skills[0].build())
        else:
            flow.route(*[s.build() for s in self.skills])

        for hook in self.after_hooks:
            flow.custom(hook)

        return flow

    async def chat(self, msg: str) -> Message:
        self.messages.append(Message.user(msg))

        context = Context(list(self.messages))
        engine = Engine(self.llm, self.tools)
        flow = self._build_flow()

        await flow.execute(context, engine)

        for m in context.messages[len(self.messages) :]:
            self.messages.append(m)

        return self.messages[-1]

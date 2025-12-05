from typing import Callable, Coroutine, Iterator, Protocol
from .flow import Flow, flow
from .llm import LLM, Message
from .tools import Tool, tool
from .context import Context
from .prompts import DEFAULT_SYSTEM_PROMPT
from .engine import Engine


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
        skills: list[Flow] | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = False,
        conversation: Conversation | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt.format(
            name=self.name, description=self.description
        )
        self.llm = llm or LLM()
        self.skills: list[Flow] = skills or []
        self.tools: list[Tool] = tools or []
        self.messages: Conversation = (
            conversation if conversation is not None else list[Message]()
        )
        self.verbose = verbose

    def skill(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to register a method as a skill for the chatbot.
        """
        self.skills.append(flow(func))

    def tool(self, func: Callable):
        """
        Decorator to register a function as a tool.
        Automatically injects the LLM if necessary.
        """
        self.tools.append(tool(self.llm.wrap(func)))

    def _build_flow(self) -> Flow:
        flow = Flow("Main flow").prepend(self.system_prompt)

        if not self.skills:
            return flow.reply()

        if len(self.skills) == 1:
            return flow.then(self.skills[0])

        return flow.route(*self.skills)

    async def chat(self, msg: str) -> Message:
        self.messages.append(Message.user(msg))

        context = Context(list(self.messages))
        engine = Engine(self.llm, self.tools)
        flow = self._build_flow()

        await flow.execute(context, engine)

        for m in context.messages[len(self.messages) :]:
            self.messages.append(m)

        return self.messages[-1]

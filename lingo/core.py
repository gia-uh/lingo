from typing import Callable, Coroutine

from lingo.utils import tee

from .flow import Flow, flow
from .llm import LLM, Message
from .tools import Tool, tool
from .context import Context
from .prompts import DEFAULT_SYSTEM_PROMPT
from .engine import Engine

import asyncio


class Lingo:
    def __init__(
        self,
        name: str = "Lingo",
        description: str = "A friendly chatbot.",
        llm: LLM | None = None,
        skills: list[Flow] | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt.format(
            name=self.name, description=self.description
        )
        self.llm = llm or LLM()
        self.skills: list[Flow] = skills or []
        self.tools: list[Tool] = tools or []
        self.messages: list[Message] = []

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

        context = Context(self.messages)
        engine = Engine(self.llm, self.tools)
        flow = self._build_flow()

        await flow.execute(context, engine)

        self.messages = context.messages
        return self.messages[-1]

    async def run(self, input_fn=None, output_fn=None):
        """
        Runs this model in the terminal, using optional
        input and output callbacks.

        Args:
            input_fn: If provided, should be a function that
                returns the user message.
                Defaults to Python's builtin input().
            output_fn: If provided, should be a callback to stream
                the LLM chat response.
                Defaults to print().
        """
        if input_fn is None:
            input_fn = lambda: input(">>> ")

        if output_fn is None:
            output_fn = lambda token: print(token, end="", flush=True)

        original_callback = self.llm.callback

        if original_callback:
            self.llm.callback = tee(output_fn, self.llm.callback)
        else:
            self.llm.callback = output_fn

        try:
            while True:
                msg = input_fn()
                await self.chat(msg)
                output_fn("\n\n")
        except EOFError:
            pass
        finally:
            self.llm.callback = original_callback

    def loop(self, input_fn=None, output_fn=None):
        """
        Automatically creates an asyncio event loop and runs
        this chatbot in a simple CLI loop.

        Receives the same arguments as `run`.
        """
        print("Name:", self.name)
        print("Description:", self.description)
        print("\n[Press Ctrl+D to exit]\n")
        asyncio.run(self.run(input_fn, output_fn))

from typing import Callable, Coroutine
from pydantic import BaseModel

from lingo.utils import tee

from .flow import Flow, flow
from .llm import LLM, Message  # StreamType is removed
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
        verbose: bool = False,
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
                the LLM chat response tokens (as strings).
                Defaults to print().
        """
        if input_fn is None:
            input_fn = lambda: input(">>> ")

        if output_fn is None:
            # Default output_fn just takes a string
            output_fn = lambda token: print(token, end="", flush=True)

        # The handler is simplified, as it only receives strings
        cli_token_handler = output_fn

        original_on_token = self.llm.on_token
        original_on_create = self.llm.on_create

        self.llm.on_token = (
            tee(cli_token_handler, original_on_token)
            if original_on_token
            else cli_token_handler
        )

        def _verbose_on_create(model: BaseModel):
            """Callback to pretty-print parsed Pydantic models."""
            output_fn("\n------- [Thinking] -------")
            output_fn(model.model_dump_json(indent=2))
            output_fn("--------------------------\n")

        if self.verbose:
            self.llm.on_create = (
                tee(original_on_create, _verbose_on_create)
                if original_on_create
                else _verbose_on_create
            )

        try:
            while True:
                msg = input_fn()
                await self.chat(msg)
                output_fn("\n\n")
        except EOFError:
            pass
        finally:
            # Restore the original on_token callback
            self.llm.on_token = original_on_token

    def loop(self, input_fn=None, output_fn=None):
        """
        Automatically creates an asyncio event loop and runs
        this chatbot in a simple CLI loop.

        Receives the same arguments as `run`.
        """
        print("Name:", self.name)
        print("Description:", self.description)

        if self.verbose:
            print("Mode: Verbose")

        print("\n[Press Ctrl+D to exit]\n")
        asyncio.run(self.run(input_fn, output_fn))

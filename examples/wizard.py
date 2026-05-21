"""wizard.py — Lingo's conversational-modeling primitives.

This is the quickstart from the README turned into a runnable file. It
demonstrates the four "structured" engine methods:

  - ``engine.ask(ctx, question)``   pause + wait for user input
  - ``engine.decide(ctx, ...)``     LLM returns a boolean
  - ``engine.choose(ctx, options)`` LLM picks one item from a list
  - ``engine.create(ctx, model)``   LLM returns a parsed Pydantic model

The Python stack IS your state machine — ``await engine.ask(...)`` suspends
the coroutine until the next user message arrives, and local variables
persist naturally across the pause.

For agent-style applications where the LLM decides which tools to call,
see ``native_tool_call.py``.

Run:
    OPENROUTER_API_KEY=... python examples/wizard.py
"""

from pydantic import BaseModel, Field

from lingo import Lingo, Context, Engine
from lingo.llm import Message
from lingo.cli import loop

import dotenv

dotenv.load_dotenv()


class Account(BaseModel):
    """The shape we want the LLM to extract from a free-text description."""

    name: str = Field(description="Full legal name")
    email: str = Field(description="Email address")
    use_case: str = Field(description="Why they want an account, in one sentence")


bot = Lingo(
    "OnboardingWizard",
    description="Walks the user through creating an account.",
)


@bot.skill
async def onboarding(ctx: Context, eng: Engine):
    """Onboard a new user. The pause/resume primitives let us write the
    flow as a linear Python script — the LLM and user input are just
    await-points."""
    # 1. engine.choose: LLM picks one item from a list.
    await eng.reply(ctx, "Welcome! Let's set up your account.")

    role = await eng.choose(
        ctx,
        ["developer", "researcher", "founder", "student"],
        "What best describes the user's role?",
    )
    await eng.reply(ctx, f"Got it — setting up the {role} experience.")

    # 2. engine.ask: pause execution and wait for the next user message.
    #    The coroutine suspends here; local variables persist across the pause.
    raw = await eng.ask(
        ctx,
        "Tell me your name, email, and (briefly) what you want to use this for.",
    )
    # Append the user's reply so the LLM sees it when we call engine.create.
    ctx.append(Message.user(raw))

    # 3. engine.create: ask the LLM to extract a typed Pydantic object.
    account = await eng.create(ctx, Account)
    await eng.reply(
        ctx, f"Thanks {account.name}! I have your email as {account.email}."
    )

    # 4. engine.decide: LLM returns True or False over the conversation.
    if await eng.decide(ctx, "Did the user mention an enterprise or team context?"):
        await eng.reply(ctx, "I'll flag this as a team account for our sales team.")
    else:
        await eng.reply(ctx, "Setting up a personal account.")

    await eng.reply(ctx, "All done!")


def main():
    loop(bot)


if __name__ == "__main__":
    main()

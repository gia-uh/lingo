"""index_wizard.py — procedural skill with pause/resume.

Demonstrates the four structured-output engine methods:
  - eng.ask(ctx, question)     pause execution, wait for user input
  - eng.decide(ctx, ...)       LLM returns bool (with chain-of-thought)
  - eng.choose(ctx, options)   LLM picks one item from a typed list
  - eng.create(ctx, Model)     LLM fills a Pydantic model

Run:
    API_KEY=... python examples/index_wizard.py
"""
from pydantic import BaseModel, Field
from lingo import Lingo, Context, Engine
from lingo.llm import Message
from lingo.cli import loop

app = Lingo("OnboardingWizard", description="Walks a new user through account setup.")


class UserProfile(BaseModel):
    """Information to collect about the new user."""
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    use_case: str = Field(description="What they want to use this for, one sentence")


@app.skill
async def onboarding(ctx: Context, eng: Engine):
    """Guide a new user through account setup.

    The skill is a linear script — await-points are the only state.
    No session IDs, no database lookups mid-flow, no manual resume logic.
    """
    role = await eng.choose(
        ctx,
        ["developer", "researcher", "founder", "student"],
        "What best describes the user's role based on the conversation?",
    )
    await eng.reply(ctx, f"Setting up the {role} experience.")

    raw = await eng.ask(ctx, "Tell me your name, email, and what you want to use this for.")
    ctx.append(Message.user(raw))

    profile = await eng.create(ctx, UserProfile)
    await eng.reply(ctx, f"Got it, {profile.name}. I'll reach you at {profile.email}.")

    if await eng.decide(ctx, "Did the user mention a team or enterprise context?"):
        await eng.reply(ctx, "I'll flag this as a team account.")
    else:
        await eng.reply(ctx, "Setting up a personal account.")

    await eng.reply(ctx, "All done!")


def main():
    loop(app)

if __name__ == "__main__":
    main()

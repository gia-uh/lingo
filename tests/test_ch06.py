import pytest
from lingo import Lingo, Message, flow
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM

def make_multi_skill_bot() -> Lingo:
    bot = Lingo(name="Assistant", llm=MockLLM(["Done."] * 10))

    @bot.skill
    async def answer_questions(context: Context, engine: Engine):
        """Answer factual questions."""
        await engine.reply(context)

    @bot.skill
    async def write_code(context: Context, engine: Engine):
        """Write and explain code."""
        context.append("Respond with runnable code.")
        await engine.reply(context)

    return bot

def test_bot_registers_skills():
    bot = make_multi_skill_bot()
    assert len(bot.skills) == 2

def make_bot_with_hooks() -> Lingo:
    bot = Lingo(name="HookBot", llm=MockLLM(["Hi!"] * 5))

    @bot.before
    async def inject_date(context: Context, engine: Engine):
        context.append("Today is 2026-07-12.")

    @bot.skill
    async def chat(context: Context, engine: Engine):
        """Chat with the user."""
        await engine.reply(context)

    @bot.after
    async def log_turn(context: Context, engine: Engine):
        context.append("[turn logged]")

    return bot

@pytest.mark.asyncio
async def test_hooks_run_around_skill():
    bot = make_bot_with_hooks()
    await bot.chat("Hello")
    contents = [str(m.content) for m in bot.messages]
    assert any("[turn logged]" in c for c in contents)

def make_filtered_bot() -> Lingo:
    bot = Lingo(name="FilterBot", llm=MockLLM(["Answer."] * 5))

    @bot.when("The user is asking in Spanish")
    async def translate_first(context: Context, engine: Engine):
        context.append("Translate the user message to English first.")

    @bot.skill
    async def answer(context: Context, engine: Engine):
        """Answer any question."""
        await engine.reply(context)

    return bot

def test_filtered_bot_has_filter():
    bot = make_filtered_bot()
    assert "The user is asking in Spanish" in bot._filters

def make_wizard_bot() -> Lingo:
    bot = Lingo(name="Wizard", llm=MockLLM(["What is your name?", "Nice to meet you!"]))

    @bot.skill
    async def wizard(context: Context, engine: Engine):
        """A multi-step wizard that collects information."""
        # engine.reply does NOT auto-append — do it manually inside skills
        question = await engine.reply(context, "What is your name?")
        context.append(question)
        name = await engine.input()
        context.append(f"User's name is: {name}")
        answer = await engine.reply(context, f"Nice to meet you, {name}!")
        context.append(answer)

    return bot

@pytest.mark.asyncio
async def test_wizard_collects_name():
    bot = make_wizard_bot()
    reply1 = await bot.chat("Start wizard")
    assert reply1.role == "assistant"
    reply2 = await bot.chat("Alice")
    assert reply2.role == "assistant"


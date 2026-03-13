import pytest
from lingo import Lingo
from lingo.mock import MockLLM


@pytest.mark.asyncio
async def test_lingo_chat_new_session():
    """Tests starting a new chat session."""
    llm = MockLLM(["Hello!"])
    bot = Lingo(llm=llm)

    resp = await bot.chat("Hi")

    assert resp.role == "assistant"
    assert resp.content == "Hello!"
    # Verify messages are present in history
    assert any(msg.role == "user" and msg.content == "Hi" for msg in bot.messages)
    assert any(
        msg.role == "assistant" and msg.content == "Hello!" for msg in bot.messages
    )


@pytest.mark.asyncio
async def test_lingo_before_after_hooks():
    """Tests that before and after hooks are executed."""
    llm = MockLLM(["Response"])
    bot = Lingo(llm=llm)

    before_called = False
    after_called = False

    @bot.before
    async def before_hook(ctx, eng):
        nonlocal before_called
        before_called = True

    @bot.after
    async def after_hook(ctx, eng):
        nonlocal after_called
        after_called = True

    await bot.chat("Hi")

    assert before_called is True
    assert after_called is True


@pytest.mark.asyncio
async def test_lingo_when_filter():
    """Tests the @when filter functionality."""
    from pydantic import BaseModel, Field

    class Filter(BaseModel):
        option_0: bool = Field(description="True if user says stop")

    # Responses:
    # 1. engine.create(Filter)
    # 2. engine.reply() inside the filter
    # 3. automatic engine.reply() at the end (because skills list is empty)
    llm = MockLLM([Filter(option_0=True), "Stopped.", "End."])
    bot = Lingo(llm=llm)

    filter_executed = False

    @bot.when("user says stop")
    async def stop_filter(ctx, eng):
        nonlocal filter_executed
        filter_executed = True
        msg = await eng.reply(ctx, "Stopped.")
        ctx.append(msg)

    await bot.chat("stop")

    assert filter_executed is True
    assert any("Stopped." in str(msg.content) for msg in bot.messages)


@pytest.mark.asyncio
async def test_lingo_session_resumption():
    """Tests that a session is resumed when input is requested."""
    # Use UNIQUE strings to trace exactly what is happening
    llm = MockLLM(["QUESTION_1", "RESPONSE_2"])
    bot = Lingo(llm=llm)

    @bot.skill
    async def age_skill(ctx, eng):
        # Turn 1 starts here
        age = await eng.ask(ctx, "How old are you?")
        # Turn 2 starts here (resumption)
        msg = await eng.reply(ctx, f"You are {age}")
        ctx.append(msg)

    # Turn 1
    resp1 = await bot.chat("Start")
    assert resp1.content == "QUESTION_1"

    # Turn 2
    await bot.chat("25")

    # Trace history
    # 0: System Prompt
    # 1: User "Start"
    # 2: Assistant "QUESTION_1"
    # 3: User "25"
    # 4: Assistant "RESPONSE_2"

    contents = [str(m.content) for m in bot.messages]
    assert "Start" in contents
    assert "QUESTION_1" in contents
    assert "25" in contents
    assert "RESPONSE_2" in contents

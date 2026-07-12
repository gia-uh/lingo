import pytest
from lingo import LLM, Message, Lingo
from lingo.mock import MockLLM

# LLM accepts callbacks for token-level events.
# Useful for streaming UIs or logging.
# llm = LLM(
#     on_token=lambda t: print(t, end="", flush=True),
#     on_message=lambda msg: print("\n--- done ---"),
# )

def test_message_factories():
    system_msg  = Message.system("You are a helpful assistant.")
    user_msg    = Message.user("What is 2 + 2?")
    assist_msg  = Message.assistant("4")
    assert system_msg.role == "system"
    assert user_msg.role == "user"
    assert assist_msg.role == "assistant"

def make_bot(*responses) -> Lingo:
    llm = MockLLM(list(responses) if responses else ["Hello!"])
    return Lingo(name="TestBot", description="A test bot.", llm=llm)

@pytest.mark.asyncio
async def test_basic_chat():
    bot = make_bot("4")
    reply = await bot.chat("What is 2 + 2?")
    assert "4" in str(reply.content)

@pytest.mark.asyncio
async def test_history_grows():
    bot = make_bot("Hi!", "Doing well!")
    await bot.chat("Hello")
    await bot.chat("How are you?")
    # user + assistant for each turn
    assert len(bot.messages) == 4

@pytest.mark.asyncio
async def test_custom_system_prompt():
    bot = Lingo(
        name="Poet",
        description="A poetry bot.",
        system_prompt="You are {name}. {description} Speak only in verse.",
        llm=MockLLM(["Roses are red."]),
    )
    reply = await bot.chat("Say hello.")
    assert reply.content  # not empty


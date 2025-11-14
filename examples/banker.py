from lingo import Lingo, Context, Engine, Message
import dotenv

from lingo.cli import loop


dotenv.load_dotenv()


# Simulate the bank
class Account:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("Insufficient funds")

        self.balance -= amount
        return self.balance


account = Account(1000)


bot = Lingo(
    name="Banker",
    description="A helpful assistant that can execute bank transactions and reply with the account information.",
    verbose=True,
)


# Add a basic chat skill
@bot.skill
async def casual_chat(context: Context, engine: Engine):
    """Casual chat with the user.

    Use this skill when the user asks a general question or engages
    in casual chat.
    """
    await engine.reply(context)


@bot.skill
async def banker(context: Context, engine: Engine):
    """Interact with the bank account.

    Use this skill when the user asks for information about the bank account,
    such as balance, or asks to deposit or withdraw.
    """
    tool = await engine.equip(context)
    result = await engine.invoke(context, tool)
    await engine.reply(
        context,
        Message.system(result),
        Message.system("Inform the user the result of the operation."),
    )


@bot.tool
async def check_balance() -> dict:
    """Returns the balance in the user account."""
    return dict(balance=account.balance)


@bot.tool
async def deposit(ammount: int) -> dict:
    """Deposit money into the user account.
    Returns the new balance.
    """
    return dict(balance=account.deposit(ammount), deposited=ammount)


@bot.tool
async def withdraw(ammount: int) -> dict:
    """Withdraw money from the user account.
    Returns the new balance.
    """
    try:
        return dict(balance=account.withdraw(ammount), withdrawn=ammount)
    except:
        return dict(error="Insufficient funds.", balance=account.balance)


loop(bot)

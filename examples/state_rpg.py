import dotenv
from lingo import Lingo, State, Message, depends, Context, Engine
from lingo.cli import loop

dotenv.load_dotenv()


class GameData(State):
    """
    The persistent state of the RPG session.
    Subclassing 'State' gives us type hints and dot-notation access.
    """

    hp: int = 100
    gold: int = 50
    inventory: list[str] = ["wooden sword"]
    location: str = "Town Square"


bot = Lingo("DungeonMaster", state=GameData(), verbose=True)


@bot.tool
def buy_item(item_name: str, cost: int, state=depends(GameData)):
    """
    Buys an item from the shop.
    """
    if state.gold < cost:
        return f"Not enough gold! You have {state.gold}, but need {cost}."

    state.gold -= cost
    state.inventory.append(item_name)
    return f"Bought {item_name} for {cost} gold. Remaining gold: {state.gold}"


@bot.tool
def drink_potion(state=depends(GameData)):
    """
    Drinks a potion from inventory to restore HP.
    """
    if "potion" not in state.inventory:
        return "You don't have a potion!"

    state.inventory.remove("potion")
    state.hp = min(100, state.hp + 20)
    return f"Glug glug! HP restored to {state.hp}."


@bot.tool
def check_status(state=depends(GameData)):
    """
    Checks your current HP, Gold, and Inventory.
    """
    return state.render()


@bot.skill
async def game_loop(context: Context, engine: Engine, state=depends(GameData)):
    """
    The main game master skill.
    """
    # We cast to GameData for IDE autocomplete (optional but helpful)
    with context.fork():
        context.append(
            Message.system(
                f"CURRENT STATUS:\n{state.render('hp', 'gold', 'location')}\n"
                f"INVENTORY: {', '.join(state.inventory)}"
            )
        )

        # Run the standard reasoning loop
        # The LLM will decide to call tools (buy_item, drink_potion) based on user input
        tool = await engine.equip(context)
        result = await engine.invoke(context, tool)

        context.append(Message.system(result.model_dump_json()))
        msg = await engine.reply(context)

    context.append(msg)


if __name__ == "__main__":
    print("Welcome to LingoRPG! You are in the Town Square.")
    loop(bot)

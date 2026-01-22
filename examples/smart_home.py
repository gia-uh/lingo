import asyncio
import dotenv
from lingo import Lingo, Context, Engine, Message
from lingo.cli import loop

# Load environment variables (e.g. OPENAI_API_KEY)
dotenv.load_dotenv()

# 1. Initialize the Bot
bot = Lingo(
    name="HomeBot",
    description="A smart home assistant that manages devices in different rooms.",
    verbose=True,
)

# -------------------------------------------------------------------------
# PARENT SKILL
# -------------------------------------------------------------------------


@bot.skill
async def smart_home(context: Context, engine: Engine):
    """
    The main controller for the smart home.
    It routes commands to specific rooms (Kitchen, Living Room).
    """
    # The parent skill function primarily acts as a container/router in this pattern.
    # Lingo automatically handles routing to subskills after this function executes.
    pass


@smart_home.tool
def emergency_shutdown():
    """
    GLOBAL: Turns off power to the entire house.
    Available in all rooms.
    """
    return "WARNING: All power cut to the house."


# -------------------------------------------------------------------------
# SUB-SKILL: KITCHEN
# -------------------------------------------------------------------------


@smart_home.subskill
async def kitchen(context: Context, engine: Engine):
    """
    Manages devices in the Kitchen.
    Use this for coffee, cooking, or kitchen appliances.
    """
    # Attempt to find a relevant tool (including scoped tools from parent)
    tool = await engine.equip(context)

    if tool:
        # Execute the tool
        result = await engine.invoke(context, tool)
        await engine.reply(context, str(result))
    else:
        # Fallback if no tool matches
        await engine.reply(
            context, "I'm in the kitchen, but I don't know how to do that."
        )


@kitchen.tool
def coffee_maker(state: str):
    """
    KITCHEN ONLY: Controls the coffee maker.
    State can be 'on' or 'off'.
    """
    return f"Coffee maker turned {state}."


# -------------------------------------------------------------------------
# SUB-SKILL: LIVING ROOM
# -------------------------------------------------------------------------


@smart_home.subskill
async def living_room(context: Context, engine: Engine):
    """
    Manages devices in the Living Room.
    Use this for TV, entertainment, or relaxing.
    """
    tool = await engine.equip(context)

    if tool:
        result = await engine.invoke(context, tool)
        await engine.reply(context, str(result))
    else:
        await engine.reply(context, "I'm in the living room, but I can't do that.")


@living_room.tool
def tv_remote(command: str):
    """
    LIVING ROOM ONLY: Controls the Television.
    Commands: 'power', 'volume_up', 'channel_next'.
    """
    return f"TV Remote sent: {command}"


# -------------------------------------------------------------------------
# RUN LOOP
# -------------------------------------------------------------------------

if __name__ == "__main__":
    loop(bot)

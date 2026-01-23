from lingo import Lingo, Context, Engine
from lingo.fsm import StateMachine
from lingo.cli import loop

import dotenv

dotenv.load_dotenv()

# 1. Setup
bot = Lingo("SupportBot")

# Create the FSM and register it so tools can ask for it
fsm = StateMachine(bot.registry)
bot.registry.register(fsm)


# --- STATE 1: TRIAGE (The Default) ---
@fsm.state
async def triage(ctx: Context, eng: Engine):
    """
    The initial state. Classifies the user's issue.
    """
    await eng.reply(
        ctx, "👋 I am the Triage Bot. Is your issue related to 'Billing' or 'Tech'?"
    )
    tool = await eng.equip(ctx)
    await eng.invoke(ctx, tool)


@triage.tool
def route_request(topic: str, fsm: StateMachine):
    """
    Routes the user based on their topic (billing/tech).
    """
    topic = topic.lower()

    if "bill" in topic or "money" in topic:
        # Hot Handoff (restart=True):
        # Switch to billing AND run the billing welcome message immediately.
        fsm.goto(billing, restart=True)

    elif "tech" in topic or "bug" in topic:
        fsm.goto(tech, restart=True)

    return "Please specify 'Billing' or 'Tech'."


# --- STATE 2: BILLING ---
@fsm.state
async def billing(ctx: Context, eng: Engine):
    """
    Secure environment for financial transactions.
    """
    # The LLM only sees billing tools here.
    await eng.reply(ctx, "💰 BILLING DEPT: How can I help with your account?")
    tool = await eng.equip(ctx)
    await eng.invoke(ctx, tool)


@billing.tool
def refund_transaction(transaction_id: str):
    """Refunds a transaction."""
    return f"Refund processed for ID: {transaction_id}."


@billing.tool
def exit_billing(fsm: StateMachine):
    """Returns to the main menu."""
    fsm.goto(triage, restart=True)
    return "Exiting..."


# --- STATE 3: TECH SUPPORT ---
@fsm.state
async def tech(ctx: Context, eng: Engine):
    """
    Technical support mode.
    """
    await eng.reply(ctx, "🛠️ TECH SUPPORT: Have you tried turning it off and on?")
    tool = await eng.equip(ctx)
    await eng.invoke(ctx, tool)


@tech.tool
def restart_server(server_name: str):
    """Restarts a server."""
    return f"Server '{server_name}' has been rebooted."


@tech.tool
def escalate_to_human(fsm: StateMachine):
    """Gives up and goes back to triage to ask for help."""
    fsm.goto(triage, restart=False)  # Cold handoff (waits for user input)
    return "Escalating ticket..."


# --- MAIN ENTRY POINT ---
@bot.skill
async def main(ctx, eng):
    # We simply execute the FSM.
    # It handles state persistence and routing automatically.
    await fsm.execute(ctx, eng)


if __name__ == "__main__":
    loop(bot)

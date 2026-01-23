import asyncio
from lingo import Lingo, Context, Engine
from lingo.cli import loop
from lingo.llm import Message
from dotenv import load_dotenv

load_dotenv()

# Initialize the bot
bot = Lingo(
    name="HelpDesk", description="A helpful customer support agent for a SaaS platform."
)

# --- PATTERN 1: PREEMPTIVE GUARDRAIL ---
# If this triggers, we reply and STOP. The main skill never runs.


@bot.when("The user is asking for illegal actions or using abusive language")
async def security_guardrail(ctx: Context, engine: Engine):
    print(">>> [GUARDRAIL TRIGGERED] Blocking request.")

    # 1. Reply to the user directly
    await engine.reply(
        ctx, "I cannot assist with that request as it violates our policy."
    )

    # 2. STOP execution immediately.
    # This raises StopFlow, preventing the main 'support_skill' from running.
    engine.stop()


# --- PATTERN 2: PASSIVE CONTEXT INJECTION ---
# If this triggers, we just add info to the context and let the flow continue.


@bot.when("The user seems frustrated or angry")
async def sentiment_analyzer(ctx: Context, engine: Engine):
    print(">>> [SENTIMENT DETECTED] Injecting guidance.")

    # We do NOT reply to the user. We add a system instruction for the Agent.
    # Because we do NOT call engine.stop(), the flow continues to the main skill.
    ctx.append(
        Message.system(
            "SYSTEM NOTE: The user is frustrated. "
            "Prioritize empathy and keep responses concise."
        )
    )


# --- MAIN SKILL ---
# This only runs if the Guardrail didn't stop the flow.


@bot.skill
async def support_skill(ctx: Context, engine: Engine):
    # Standard ReAct behavior or simple reply
    await engine.reply(ctx, "How can I help you with your account today?")


# --- SIMULATION ---

loop(bot)

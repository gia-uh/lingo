# User Guide: Building Applications with Lingo

This guide provides "recipes" or "blueprints" for common LLM application patterns. Each recipe is based on core Lingo primitives like `Lingo`, `Engine`, `Context`, and `Flow`.

## 👩‍🍳 Recipe 1: The Stateful Wizard (Linear Flows)

**Best for: Multi-step data collection, onboarding, forms.**

Use `await engine.ask()` to pause and wait for user input. Variables persist in memory across turns, so you don't need to manage session state manually.

```python
@app.skill
async def register(ctx, eng):
    name = await eng.ask(ctx, "Name?")
    email = await eng.ask(ctx, f"Hi {name}, what's your email?")
    
    # engine.decide() forces a boolean response from the LLM
    if await eng.decide(ctx, f"Is {email} a valid corporate email?"):
        await eng.reply(ctx, "Success!")
    else:
        await eng.reply(ctx, "Personal emails not allowed.")
```

## 🛠️ Recipe 2: The Tool-User (Function Calling)

**Best for: Interacting with external APIs, databases, or local scripts.**

Register functions as tools and use `engine.equip()` and `engine.invoke()` to execute them.

```python
@app.tool
def get_weather(city: str):
    """Returns the weather for a city."""
    return f"It's sunny in {city}."

@app.skill
async def weather_assistant(ctx, eng):
    tool = await eng.equip(ctx) # LLM selects the best tool
    result = await eng.invoke(ctx, tool) # Executes it and returns the result
    await eng.reply(ctx, f"The weather report is: {result}")
```

## 🏗️ Recipe 3: The Structured Architect (StateMachine)

**Best for: Complex business logic, secure environments, and deterministic workflows.**

Use `StateMachine` to define strict states and transitions.

```python
from lingo.fsm import StateMachine

fsm = StateMachine(app.registry)

@fsm.state
async def home(ctx, eng):
    await eng.reply(ctx, "Main menu. Say 'Help' or 'Settings'.")
    fsm.goto(settings) # Transition to another state

@app.skill
async def main_loop(ctx, eng):
    await fsm.execute(ctx, eng)
```

## ⚡ Recipe 4: The Reactive Agent (Reflexive Patterns)

**Best for: Guardrails, interruptions, global commands.**

Use `@app.when()` to intercept messages before they reach regular skills.

```python
@app.when("User wants to stop or cancel")
async def stop_handler(ctx, eng):
    await eng.reply(ctx, "Stopping everything.")
    eng.stop() # Immediately terminates the current flow
```

## 🏗️ Recipe 5: Scaling with Sub-skills

**Best for: Large applications with many components (e.g., a smart home).**

Organize logic into parent and child skills with scoped tools.

```python
@app.skill
async def kitchen(ctx, eng):
    """Manages the kitchen."""
    pass

@kitchen.tool
def start_oven(temp: int):
    return f"Oven heating to {temp}."
```

## 💾 Recipe 6: Managing Persistent State

**Best for: Games, RPGs, or applications with complex, structured data.**

Subclass `State` (a Pydantic model) to define your application's data schema.

```python
from lingo import State

class GameData(State):
    hp: int = 100
    gold: int = 50

bot = Lingo("GameBot", state=GameData())

@bot.tool
def take_damage(damage: int, state=depends(GameData)):
    state.hp -= damage
    return f"Ouch! HP: {state.hp}"
```

## 📝 Recipe 7: Prompt Engineering with Lingo

Lingo uses the `Context` object to manage the conversation history. You can steer the LLM by manually appending messages to the context.

- **`Message.system(text)`**: Internal instructions or status updates.
- **`Message.user(text)`**: Simulates user input.
- **`Message.assistant(text)`**: Simulates the bot's response.

```python
@app.skill
async def steering_skill(ctx, eng):
    # Inject a hidden instruction before the LLM generates a reply
    ctx.append(Message.system("Respond in the style of a pirate."))
    await eng.reply(ctx)
```

### Pro Tip: `context.fork()`
Use `with context.fork():` to create a temporary branch of the conversation history. Any messages appended inside the block are discarded after the block exits, which is perfect for "scratchpad" reasoning or speculative execution.

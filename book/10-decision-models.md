# Chapter 10: Decision Models

Chapters 3 and 6 showed the Engine choosing a skill, picking a tool and answering
yes/no questions with the LLM. Each of those calls asks the model to reason in
text and then name its answer, and the app pays for every generated token.

A *decision model* answers the same kind of question without generating text. It
reads a state and a set of typed questions, and returns a probability for every
option in one forward pass. The questions come in three types:

- **choice**: one option from a set, each with a description;
- **noul**: yes or no, answered with the probability of yes;
- **score**: a level on an ordered rubric.

This is the *System One* contract. lingo speaks it over HTTP through `SystemOne`,
and knows nothing about who serves it: a hosted gateway such as OpenRouter, or a
local decision server, fits behind the same URL.

## Using it for routing

```python
import os
from lingo import Lingo, LLM, SystemOne

bot = Lingo(
    llm=LLM(model="anthropic/claude-haiku-5.5"),
    decider=SystemOne(
        model="typesafe/jev-1.14",
        api_key=os.environ["OPENROUTER_API_KEY"],
        threshold=0.8,
    ),
)

@bot.skill
async def stores(context, engine):
    """Where the shops are: addresses and opening hours by city."""
    ...

@bot.skill
async def other(context, engine):
    """Anything else."""
    ...
```

Nothing else changes. Skill routing, and every `Route`, `Choose`, `Decide` and
`Act` node, call `Engine.choose`, `Engine.decide` or `Engine.equip`, and those ask
the decider first:

- the options become the question's criteria, each described by the skill's or
  tool's docstring;
- the state is the last `history` user and assistant messages (default 6);
- an answer at or above `threshold` is used as is;
- an answer below it, or a decider that fails, falls back to the LLM path of the
  earlier chapters.

Confidence is the chosen option's probability for a choice, and `max(p, 1 - p)`
for a yes/no question, so one threshold reads the same for both.

## Asking several questions at once

When a flow needs more than one decision about the same message, ask them
together. `Engine.ask_many` returns every answer with its probabilities and
applies no threshold:

```python
from lingo.decider import choice, noul

decision = await engine.ask_many(context, {
    "route": choice("What is the message about?", {
        "stores": "where shops are",
        "products": "a product in the catalog",
        "other": "anything else",
    }),
    "urgent": noul("Does it need an answer today?"),
})

decision["route"].choice       # 'stores'
decision["route"].confidence   # 0.97
decision["urgent"].noul        # probability of yes
```

`SystemOne.decide(state, questions)` does the same with a state you build
yourself, a string or a JSON object. Pass `on_decision=` to see every decision,
for example to count how often the LLM was still needed.

## What it does not do

A decision model answers closed questions. `Engine.create` and `Engine.infer`
produce open values, such as a tool's free-text arguments, and stay on the LLM.

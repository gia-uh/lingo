# Chapter 8: Patterns

End-to-end examples showing how the pieces compose into real applications.

## Pattern 1: A classification pipeline

Extract structured data from unstructured text using `engine.create`.

```python {name=ch08_ticket_model}
class Ticket(BaseModel):
    """A parsed support ticket."""
    priority: str
    category: str
    summary: str
```

```python {name=ch08_test_classify}
@flow
async def classify_ticket(context: Context, engine: Engine):
    """Classify a support ticket into structured fields."""
    ticket = await engine.create(
        context,
        Ticket,
        "Extract the ticket details from the conversation.",
    )
    context.append(ticket.model_dump_json())
    return ticket

@pytest.mark.asyncio
async def test_classify_pipeline():
    expected = Ticket(priority="high", category="billing", summary="Wrong charge")
    engine = Engine(MockLLM([expected]))
    c = Context([Message.user("I was charged twice for my subscription!")])
    result = await classify_ticket.execute(c, engine)
    assert isinstance(result, Ticket)
    assert result.priority == "high"
    assert result.category == "billing"
```

## Pattern 2: Retry with self-correction

Use `Flow.retry(fixer)` to automatically fix failures. The `fixer` flow is called
with the error appended to the context, giving the LLM a chance to correct its approach.

```python {name=ch08_test_retry}
attempt_count = {"n": 0}

@flow
async def risky_parse(context: Context, engine: Engine):
    attempt_count["n"] += 1
    if attempt_count["n"] < 2:
        raise ValueError("Malformed output — try again")
    context.append("Parsed successfully.")

@flow
async def suggest_fix(context: Context, engine: Engine):
    return "Simplify your output and try again."

@pytest.mark.asyncio
async def test_retry_recovers():
    attempt_count["n"] = 0
    f = Flow("parse").then(risky_parse).retry(fixer=suggest_fix, max_retries=3)
    c = Context([Message.user("Parse this input")])
    await f.execute(c, Engine(MockLLM(["hint"] * 3)))
    assert attempt_count["n"] == 2  # failed once, succeeded on retry
```

## Pattern 3: Parallel research + synthesis

`fork` runs independent research flows in parallel and synthesizes the results.
Each branch gets a clone of the context, keeping the main history clean.

```python {name=ch08_test_fork_synthesis}
@flow
async def research_pros(context: Context, engine: Engine):
    """Research the advantages."""
    context.append("Focus: list the pros.")
    return await engine.reply(context)

@flow
async def research_cons(context: Context, engine: Engine):
    """Research the disadvantages."""
    context.append("Focus: list the cons.")
    return await engine.reply(context)

@pytest.mark.asyncio
async def test_parallel_research():
    f = Flow("research").fork(
        research_pros,
        research_cons,
        aggregator="Write a balanced summary of pros and cons.",
    )
    c = Context([Message.user("Should I use microservices?")])
    await f.execute(c, Engine(MockLLM(["Pro: scalable", "Con: complex", "Balanced view."])))
    assert c.messages[-1].role == "assistant"
    assert "Balanced" in c.messages[-1].content
```

## Pattern 4: Stateful quiz bot

A multi-turn bot that tracks score in `State`, asks questions one at a time,
and evaluates answers.

```python {name=ch08_quiz_state}
class QuizState(State):
    score: int = 0
    question_index: int = 0

QUESTIONS = [
    ("What is 2+2?", "4"),
    ("Capital of France?", "Paris"),
]
```

```python {name=ch08_test_quiz}
def make_quiz_bot() -> Lingo:
    state = QuizState()
    bot = Lingo(
        name="QuizBot",
        llm=MockLLM(["What is 2+2?", "Correct!", "Capital of France?", "Correct!"]),
        state=state,
    )

    @bot.skill
    async def quiz(context: Context, engine: Engine):
        """Run a quiz question."""
        if state.question_index >= len(QUESTIONS):
            await engine.reply(context, f"Quiz done! Score: {state.score}/{len(QUESTIONS)}")
            return

        question, _ = QUESTIONS[state.question_index]
        await engine.reply(context, question)
        answer = await engine.input()

        _, expected = QUESTIONS[state.question_index]
        if expected.lower() in answer.lower():
            state.score += 1

        state.question_index += 1
        await engine.reply(context, "Correct!" if expected.lower() in answer.lower() else f"The answer was {expected}.")

    return bot

@pytest.mark.asyncio
async def test_quiz_bot_scores():
    bot = make_quiz_bot()
    await bot.chat("Start quiz")   # bot asks Q1
    await bot.chat("4")            # correct answer
    assert bot.state.score == 1
    assert bot.state.question_index == 1
```

## Test file

```python {export=tests/test_ch08.py}
import pytest
from pydantic import BaseModel
from lingo import Lingo, Message, Flow, flow
from lingo.context import Context
from lingo.engine import Engine
from lingo.mock import MockLLM
from lingo.state import State

<<ch08_ticket_model>>
<<ch08_test_classify>>
<<ch08_test_retry>>
<<ch08_test_fork_synthesis>>
<<ch08_quiz_state>>
<<ch08_test_quiz>>
```

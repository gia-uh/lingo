---
number: 12
title: "Add High-Level Cognitive Patterns Module (`lingo.patterns`)"
state: open
labels:
---

## Context & Motivation

Currently, `lingo` provides excellent low-level primitives (`Flow`, `Node`, `Repeat`, `Fork`, `Create`) for building agentic behaviors. However, users often need to implement standard cognitive architectures (like ReAct, Tree of Thoughts, or Refinement loops) from scratch.

This issue proposes adding a new module, `lingo.patterns`, containing **factory functions** that implement these standard patterns. This will standardize best practices and drastically reduce boilerplate for common use cases.

## Proposed Solution

Introduce `lingo/patterns.py` (or a `lingo.patterns` package) that exports high-level flow generators. These functions will return pre-configured `Flow` objects ready for execution.

## Detailed Design & Patterns

### 1. `react` (The Agent)

The fundamental loop for general-purpose tool use. It alternates between reasoning ("Thought"), acting ("Action"), and observing results until a goal is met.

```python
def react(goal: str, tools: list[Tool], max_steps: int = 10) -> Flow:
    """
    Creates a ReAct (Reason + Act) loop.

    1. The LLM reasons about the current state.
    2. The LLM selects a tool (or speaks).
    3. The loop continues until the stop_condition is met or max_steps reached.
    """
    step = Flow("Step")
    step.reply("Assess the situation and decide the next step.")
    step.act(*tools)

    wrapper = Flow("ReAct Loop", description=f"Solves: {goal}")
    wrapper.prepend(f"GOAL: {goal}")
    wrapper.repeat(step, until=f"Is the goal '{goal}' achieved?", max_repeats=max_steps)
    return wrapper

```

### 2. `optimizer` (Refinement Loop)

A general loop for **Self-Correction**, **Reflexion**, and **Chain of Verification**. It takes a generator node and a verifier node, looping until the verifier approves the output.

```python
def optimizer(generator: Node, verifier: Node, max_retries: int = 3) -> Flow:
    """
    Generic 'Generate -> Verify -> Improve' loop.

    Args:
        generator: The node that produces the initial result (and subsequent improvements).
        verifier: A node that inspects context and returns/adds a critique.
    """
    body = Flow("Optimization Step")
    body.then(generator)
    body.then(verifier)

    loop = Flow("Optimizer")
    loop.repeat(
        body=body,
        until="Did the verifier explicitly state the result is satisfactory?",
        max_repeats=max_retries
    )
    return loop

```

### 3. `planner` (Plan & Execute)

Implements **Dynamic Planning** and **Least-to-Most Prompting**. It first generates a structured list of steps using `Create`, then sequentially executes a factory function for each step.

```python
def planner(goal: str, executor: Callable[[str], Node]) -> Flow:
    """
    1. Generates a structured plan (list of steps).
    2. Executes the 'executor' node factory for each step sequentially.
    """
    class Plan(BaseModel):
        steps: list[str]

    f = Flow("Planner")
    f.create(Plan, f"Create a step-by-step plan to achieve: {goal}")

    async def run_plan(context: Context, engine: Engine):
        last_msg = context.messages[-1]
        plan = Plan.model_validate_json(last_msg.content)

        for i, step_desc in enumerate(plan.steps):
            context.append(Message.system(f"Executing Step {i+1}: {step_desc}"))
            step_node = executor(step_desc)
            await step_node.execute(context, engine)

    f.custom(run_plan)
    return f

```

### 4. `brainstorm` (Parallel Consensus)

Consolidates **Tree of Thoughts**, **Self-Consistency**, and **Debate**. It uses `Fork` to spawn parallel reasoning paths and aggregates them using a consensus prompt.

```python
def brainstorm(
    prompts: list[str] | str,
    aggregator: str = "Synthesize the best aspects of these options."
) -> Flow:
    """
    Runs parallel branches to explore a problem space, then aggregates.

    Args:
        prompts: If a list, spawns one branch per prompt (Debate).
                 If a string, spawns N identical branches (Self-Consistency/ToT).
    """
    branches = []

    # Handle both single-prompt (Self-Consistency) and multi-prompt (Debate)
    if isinstance(prompts, str):
        prompt_list = [prompts] * 3 # Default to 3 paths
    else:
        prompt_list = prompts

    for p in prompt_list:
        b = Flow()
        b.reply(p)
        branches.append(b)

    return Flow("Brainstorm").fork(*branches, aggregator=aggregator)

```

### 5. `interviewer` (Slot Filling)

Designed for **State Machine** behaviors where specific information must be gathered from the user before proceeding.

```python
def interviewer(schema: type[BaseModel], goal: str) -> Flow:
    """
    Loops until the Pydantic schema is fully populated by the user.
    """
    f = Flow("Interviewer")

    # Attempt to extract current state
    f.create(schema, f"Extract info for {goal}. Use null for missing fields.")

    # Custom functional node to check completeness
    async def check_missing(context: Context, engine: Engine) -> bool:
        last_msg = context.messages[-1]
        data = schema.model_validate_json(last_msg.content)
        missing = [k for k, v in data.model_dump().items() if not v]

        if not missing:
            return True

        # Inject advice for the next loop iteration
        context.append(Message.system(f"Missing: {missing}. Ask the user."))
        return False

    f.repeat(
        body=Flow().reply("Ask the user for missing fields."),
        until=Flow().custom(check_missing),
        max_repeats=10
    )
    return f

```

### 6. `batch_processor` (Map-Reduce)

A dynamic Map-Reduce implementation for processing lists of items (e.g., summarizing documents) where the number of items is not known at compile time.

*Implementation requires a new `MapReduce` Node class or a functional node wrapper.*

### 7. `resilient` (Error Handling Wrapper)

A wrapper that applies the **Resilient Actor** pattern. It wraps a flow in a `Retry` block with a standard "Fixer" node that analyzes exceptions.

```python
def resilient(node: Node, max_retries: int = 3) -> Flow:
    fixer = Flow("Error Handler")
    fixer.reply("Analyze the error and state what parameters to change for the next attempt.")

    wrapper = Flow("Resilient Wrapper")
    wrapper.retry(body=node, fixer=fixer, max_retries=max_retries)
    return wrapper

```

## Usage Examples

```python
from lingo import Lingo
from lingo.patterns import react, brainstorm, resilient

bot = Lingo("SmartBot")

@bot.skill
def solve_complex_problem(context, engine):
    # 1. Brainstorm ideas
    ideas_flow = brainstorm("How can we reduce latency in this system?")

    # 2. Execute the best idea using ReAct, with auto-healing
    execution_flow = resilient(
        react(goal="Implement the chosen strategy", tools=my_tools)
    )

    return Flow("Main").then(ideas_flow).then(execution_flow)

```

## Checklist

* [ ] Create `lingo/patterns.py`.
* [ ] Implement `react`, `optimizer`, `planner`, `brainstorm`, `interviewer`, `resilient`.
* [ ] Implement `MapReduce` node/helper.
* [ ] Add unit tests for each pattern in `tests/test_patterns.py`.
* [ ] Update `README.md` with a "Patterns" section.
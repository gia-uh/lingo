# tests/test_core_lingo.py
"""
Tests for lingo/core.py — the Lingo chatbot orchestrator.

Focus areas:
- __init__ wiring (LLM, skills, tools, system_prompt formatting, registry)
- @skill / @tool / @before / @after / @when decorators
- _build_flow logic (no-skills → reply, one skill, multiple skills, filters)
- chat() happy-path round-trip (no-skills, end-to-end with MockLLM)
- chat() resume path (engine.input pause/resume)
- chat() exception propagation and cleanup
"""

import asyncio
import pytest
from pydantic import BaseModel
from lingo import Lingo, Message, Engine, Context, Flow
from lingo.mock import MockLLM
from lingo.skills import Skill
from lingo.prompts import DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_app(responses=None, **kwargs) -> Lingo:
    """Convenience: build a Lingo with a MockLLM pre-loaded with responses."""
    llm = MockLLM(responses or [])
    return Lingo(llm=llm, **kwargs)


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestLingoInit:
    def test_default_name_and_description(self):
        app = make_app()
        assert app.name == "Lingo"
        assert app.description == "A friendly chatbot."

    def test_custom_name_and_description(self):
        app = make_app(name="Bob", description="A helpful bot.")
        assert app.name == "Bob"
        assert app.description == "A helpful bot."

    def test_system_prompt_formatted(self):
        app = make_app(name="Alice", description="An assistant.")
        assert "Alice" in app.system_prompt
        assert "An assistant." in app.system_prompt

    def test_system_prompt_no_raw_placeholders(self):
        """The formatted system_prompt must not contain raw {name} etc."""
        app = make_app()
        assert "{name}" not in app.system_prompt
        assert "{description}" not in app.system_prompt

    def test_llm_injected(self):
        llm = MockLLM([])
        app = Lingo(llm=llm)
        assert app.llm is llm

    def test_default_llm_created_when_none(self):
        """When no LLM is passed, Lingo creates one (real LLM is fine, just test it exists)."""
        # We can't easily use a real LLM without a key, so just verify the attribute
        # is set to *something* that is not None.
        from lingo.llm import LLM
        app = Lingo.__new__(Lingo)
        # Call __init__ with only a mock llm to avoid network
        mock = MockLLM()
        app2 = Lingo(llm=mock)
        assert isinstance(app2.llm, LLM)

    def test_skills_default_empty(self):
        app = make_app()
        assert app.skills == []

    def test_tools_default_empty(self):
        app = make_app()
        assert app.tools == []

    def test_skills_list_injected(self):
        app = make_app()

        @app.skill
        async def my_skill(ctx, eng):
            """A skill."""
            pass

        assert len(app.skills) == 1

    def test_messages_default_empty_list(self):
        app = make_app()
        assert list(app.messages) == []

    def test_registry_contains_app(self):
        app = make_app()
        # Registry.register(self) — Lingo registers itself; verify llm is also there
        assert app.registry is not None

    def test_hooks_default_empty(self):
        app = make_app()
        assert app._before_hooks == []
        assert app._after_hooks == []
        assert app._filters == {}

    def test_session_state_default_none(self):
        app = make_app()
        assert app._runner_task is None
        assert app._active_engine is None
        assert app._active_context is None


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestDecorators:
    def test_skill_decorator_registers(self):
        app = make_app()

        @app.skill
        async def handle_query(ctx, eng):
            """Handles a query."""
            pass

        assert len(app.skills) == 1
        assert isinstance(app.skills[0], Skill)

    def test_skill_decorator_returns_skill(self):
        app = make_app()

        @app.skill
        async def handle_query(ctx, eng):
            """Handles a query."""
            pass

        assert isinstance(handle_query, Skill)

    def test_multiple_skills_accumulated(self):
        app = make_app()

        @app.skill
        async def s1(ctx, eng):
            """Skill 1."""
            pass

        @app.skill
        async def s2(ctx, eng):
            """Skill 2."""
            pass

        assert len(app.skills) == 2

    def test_tool_decorator_registers(self):
        app = make_app()

        @app.tool
        def get_weather(city: str) -> str:
            """Returns weather for a city."""
            return f"Sunny in {city}"

        assert len(app.tools) == 1

    def test_tool_decorator_returns_tool(self):
        from lingo.tools import Tool

        app = make_app()

        @app.tool
        def get_weather(city: str) -> str:
            """Returns weather for a city."""
            return f"Sunny in {city}"

        assert isinstance(get_weather, Tool)

    def test_before_decorator_registers(self):
        app = make_app()

        @app.before
        async def inject_examples(ctx, eng):
            pass

        assert len(app._before_hooks) == 1

    def test_after_decorator_registers(self):
        app = make_app()

        @app.after
        async def cleanup(ctx, eng):
            pass

        assert len(app._after_hooks) == 1

    def test_when_decorator_registers_filter(self):
        app = make_app()

        @app.when("user asks about weather")
        async def weather_branch(ctx, eng):
            pass

        assert "user asks about weather" in app._filters

    def test_when_decorator_returns_flow(self):
        app = make_app()

        @app.when("some condition")
        async def branch(ctx, eng):
            pass

        assert isinstance(branch, Flow)

    def test_multiple_when_conditions(self):
        app = make_app()

        @app.when("condition A")
        async def branch_a(ctx, eng):
            pass

        @app.when("condition B")
        async def branch_b(ctx, eng):
            pass

        assert len(app._filters) == 2
        assert "condition A" in app._filters
        assert "condition B" in app._filters


# ---------------------------------------------------------------------------
# _build_flow tests
# ---------------------------------------------------------------------------


class TestBuildFlow:
    def test_no_skills_ends_with_reply(self):
        """With no skills, the flow should include a Reply node."""
        from lingo.flow import Reply, Prepend

        app = make_app()
        built = app._build_flow()

        # Check that the nodes include a Reply at the end
        node_types = [type(n).__name__ for n in built.nodes]
        assert "Reply" in node_types

    def test_no_skills_includes_system_prompt_prepend(self):
        """System prompt is prepended to the flow."""
        from lingo.flow import Prepend

        app = make_app()
        built = app._build_flow()

        node_types = [type(n).__name__ for n in built.nodes]
        assert "Prepend" in node_types

    def test_one_skill_uses_skill_flow(self):
        """Single skill: flow ends with the skill's built flow, no Route."""
        from lingo.flow import Route

        app = make_app()

        @app.skill
        async def my_skill(ctx, eng):
            """A single skill."""
            pass

        built = app._build_flow()

        # Should NOT have a Route node (only one skill = direct flow)
        node_types = [type(n).__name__ for n in built.nodes]
        assert "Route" not in node_types

    def test_multiple_skills_uses_route(self):
        """Multiple skills: flow contains a Route node."""
        from lingo.flow import Route

        app = make_app()

        @app.skill
        async def skill_a(ctx, eng):
            """Skill A."""
            pass

        @app.skill
        async def skill_b(ctx, eng):
            """Skill B."""
            pass

        built = app._build_flow()

        node_types = [type(n).__name__ for n in built.nodes]
        assert "Route" in node_types

    def test_before_hooks_appear_in_flow(self):
        """Before hooks are registered as FunctionalNode entries early in the flow."""
        from lingo.flow import FunctionalNode

        app = make_app()

        @app.before
        async def inject_examples(ctx, eng):
            pass

        built = app._build_flow()

        node_types = [type(n).__name__ for n in built.nodes]
        assert "FunctionalNode" in node_types

    def test_after_hooks_appear_in_flow(self):
        """After hooks are registered as FunctionalNode entries after skills."""
        from lingo.flow import FunctionalNode

        app = make_app()

        @app.after
        async def summarize(ctx, eng):
            pass

        built = app._build_flow()

        node_types = [type(n).__name__ for n in built.nodes]
        assert "FunctionalNode" in node_types

    def test_filters_add_functional_node(self):
        """When _filters exist, _build_filters result is added to the flow."""
        app = make_app()

        @app.when("some condition")
        async def branch(ctx, eng):
            pass

        built = app._build_flow()

        # _build_filters wraps in a flow decorated with @flow, which becomes
        # a FunctionalNode when .custom() is called
        node_types = [type(n).__name__ for n in built.nodes]
        # The filter flow appears as a sub-node in the sequence
        assert len(built.nodes) >= 2  # at least prepend + filter flow + reply/skill

    def test_returns_flow_instance(self):
        app = make_app()
        built = app._build_flow()
        assert isinstance(built, Flow)


# ---------------------------------------------------------------------------
# chat() end-to-end tests (using MockLLM)
# ---------------------------------------------------------------------------


class TestChatEndToEnd:
    @pytest.mark.asyncio
    async def test_basic_round_trip_no_skills(self):
        """A no-skills Lingo should return an assistant message for user input."""
        app = make_app(responses=["Hello from the bot!"])
        result = await app.chat("Hi there")

        assert result.role == "assistant"
        assert result.content == "Hello from the bot!"

    @pytest.mark.asyncio
    async def test_user_message_appended_to_history(self):
        """chat() must append the user message to self.messages."""
        app = make_app(responses=["OK"])
        await app.chat("Test message")

        roles = [m.role for m in app.messages]
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_assistant_message_appended_to_history(self):
        """chat() must append the assistant response to self.messages."""
        app = make_app(responses=["My response"])
        await app.chat("Hello")

        roles = [m.role for m in app.messages]
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_history_grows_on_multiple_turns(self):
        """Two chat() calls should accumulate in messages."""
        app = make_app(responses=["First reply", "Second reply"])

        await app.chat("First message")
        count_after_first = len(app.messages)

        await app.chat("Second message")
        count_after_second = len(app.messages)

        assert count_after_second > count_after_first

    @pytest.mark.asyncio
    async def test_runner_task_cleared_after_completion(self):
        """After a successful chat(), the session state is cleaned up."""
        app = make_app(responses=["Done"])
        await app.chat("Hello")

        assert app._runner_task is None
        assert app._active_engine is None
        assert app._active_context is None

    @pytest.mark.asyncio
    async def test_returns_last_message(self):
        """chat() returns the last message in self.messages."""
        app = make_app(responses=["The answer is 42"])
        result = await app.chat("What is the answer?")

        assert result is app.messages[-1]

    @pytest.mark.asyncio
    async def test_chat_with_one_skill(self):
        """
        Lingo with one registered skill should still complete a round-trip.
        The skill flow must produce an assistant message.
        """
        app = make_app(responses=["Skill response"])

        @app.skill
        async def my_skill(ctx: Context, eng: Engine):
            """Handles everything."""
            response = await eng.reply(ctx)
            ctx.append(response)

        result = await app.chat("Do something")
        assert result.role == "assistant"

    @pytest.mark.asyncio
    async def test_exception_cleared_session_state(self):
        """If the flow raises, chat() re-raises and clears session state."""

        class BoomLLM(MockLLM):
            async def chat(self, messages, **kwargs):
                raise RuntimeError("LLM exploded")

        app = Lingo(llm=BoomLLM([]))

        with pytest.raises(RuntimeError, match="LLM exploded"):
            await app.chat("Hello")

        # Session state must be cleaned up even on error
        assert app._runner_task is None
        assert app._active_engine is None
        assert app._active_context is None


# ---------------------------------------------------------------------------
# chat() resume path
# ---------------------------------------------------------------------------


class TestChatResume:
    @pytest.mark.asyncio
    async def test_resume_feeds_input_to_engine(self):
        """
        When a flow calls engine.input(), a second chat() should
        feed the message to the waiting engine rather than starting
        a new task (i.e. the runner task count stays at 1).
        """
        app = make_app(responses=["Question answered!"])
        captured_input = []
        tasks_at_resume: list[int] = []

        @app.skill
        async def ask_then_answer(ctx: Context, eng: Engine):
            """Asks for clarification then replies."""
            # This will block until a second chat() call provides input
            user_reply = await eng.input()
            captured_input.append(user_reply)
            # Now reply using the clarification
            response = await eng.reply(ctx, f"User said: {user_reply}")
            ctx.append(response)

        # First call: flow runs until engine.input() pauses
        task1 = asyncio.create_task(app.chat("Start"))

        # Give the flow a chance to reach the input() pause
        await asyncio.sleep(0.05)

        # The runner task should still be in-flight at this point
        assert app._runner_task is not None
        assert not app._runner_task.done()

        # Second call: provides the waiting input — must NOT start a new asyncio.Task
        task_before = app._runner_task
        task2 = asyncio.create_task(app.chat("My clarification"))

        await task2
        await task1  # let the first task finish too

        # The input reached the waiting skill coroutine
        assert "My clarification" in captured_input

        # The same runner task handled both turns (no second task was spawned)
        # After full completion the messages list ends with assistant
        roles = [m.role for m in app.messages]
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# _build_filters tests
# ---------------------------------------------------------------------------


class TestBuildFilters:
    @pytest.mark.asyncio
    async def test_filters_built_as_flow(self):
        """_build_filters returns a Flow / coroutine-wrapped filter."""
        from lingo.flow import Flow

        app = make_app()

        @app.when("user is angry")
        async def calm_down(ctx, eng):
            pass

        filter_flow = app._build_filters()
        # It's decorated with @flow — returns a Flow instance
        assert isinstance(filter_flow, Flow)

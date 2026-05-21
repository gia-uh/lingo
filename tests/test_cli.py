# tests/test_cli.py
"""
Tests for lingo/cli.py — run() and loop() CLI helpers.

Focus areas:
- run() basic round-trip with mocked input/output
- run() with verbose=True fires _on_create path
- run() with default (None) input_fn / output_fn via monkeypatching
- loop() shim works the same as run()
"""

import asyncio
import pytest
from lingo import Lingo
from lingo.mock import MockLLM
from lingo.cli import run, loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cli_app(responses=None, verbose=False) -> Lingo:
    llm = MockLLM(responses or [])
    return Lingo(llm=llm, name="TestBot", description="Test chatbot.", verbose=verbose)


class OneShotInputFn:
    """Callable that returns one message then raises EOFError."""

    def __init__(self, message: str):
        self.message = message
        self._called = False

    def __call__(self) -> str:
        if not self._called:
            self._called = True
            return self.message
        raise EOFError


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------


class TestRunBasic:
    @pytest.mark.asyncio
    async def test_run_produces_output(self, capsys):
        """run() should call chat() once and write output tokens."""
        app = make_cli_app(responses=["Hello there!"])
        tokens: list[str] = []

        await run(app, input_fn=OneShotInputFn("Hi"), output_fn=tokens.append)

        # The MockLLM emits each word as a token plus a trailing newline pair
        full_output = "".join(tokens)
        assert "Hello" in full_output or "there" in full_output

    @pytest.mark.asyncio
    async def test_run_captures_all_tokens(self):
        """All streaming tokens should arrive in the output_fn."""
        app = make_cli_app(responses=["One Two Three"])
        tokens: list[str] = []

        await run(app, input_fn=OneShotInputFn("Go"), output_fn=tokens.append)

        full_output = "".join(tokens)
        # MockLLM emits "One " "Two " "Three " then the "\n\n" appended by run()
        assert "One" in full_output
        assert "Two" in full_output
        assert "Three" in full_output

    @pytest.mark.asyncio
    async def test_run_prints_header(self, capsys):
        """run() always prints name/description header to stdout."""
        app = make_cli_app(responses=["Hi"])
        await run(app, input_fn=OneShotInputFn("Hello"), output_fn=lambda t: None)

        captured = capsys.readouterr()
        assert "TestBot" in captured.out
        assert "Test chatbot." in captured.out

    @pytest.mark.asyncio
    async def test_run_restores_on_token_after_completion(self):
        """run() must restore the original _on_token callback after finishing."""
        # Must be callable — tee() will invoke it during token streaming
        original_tokens: list[str] = []
        sentinel = original_tokens.append  # a callable original handler

        llm = MockLLM(["Response"])
        llm._on_token = sentinel

        app = Lingo(llm=llm)
        await run(app, input_fn=OneShotInputFn("test"), output_fn=lambda t: None)

        assert app.llm._on_token is sentinel

    @pytest.mark.asyncio
    async def test_run_restores_on_token_on_exception(self):
        """run() must restore _on_token even when an exception occurs."""
        sentinel = None  # no-op original
        llm = MockLLM([])
        llm._on_token = sentinel

        app = Lingo(llm=llm)

        class BustedInput:
            def __call__(self):
                raise RuntimeError("bust")

        try:
            await run(app, input_fn=BustedInput(), output_fn=lambda t: None)
        except Exception:
            pass

        assert app.llm._on_token is sentinel

    @pytest.mark.asyncio
    async def test_run_eofError_exits_cleanly(self):
        """EOFError from input_fn should cause run() to return without raising."""
        app = make_cli_app(responses=[])

        def immediate_eof():
            raise EOFError

        # Should not raise
        await run(app, input_fn=immediate_eof, output_fn=lambda t: None)


# ---------------------------------------------------------------------------
# run() verbose mode
# ---------------------------------------------------------------------------


class TestRunVerbose:
    @pytest.mark.asyncio
    async def test_verbose_mode_prints_mode_header(self, capsys):
        """With verbose=True the header prints 'Mode: Verbose'."""
        app = make_cli_app(responses=["OK"], verbose=True)
        await run(app, input_fn=OneShotInputFn("Hi"), output_fn=lambda t: None)

        captured = capsys.readouterr()
        assert "Verbose" in captured.out

    @pytest.mark.asyncio
    async def test_verbose_on_create_fires(self):
        """With verbose=True, _on_create is wired to emit structured output."""
        from pydantic import BaseModel

        class Thought(BaseModel):
            text: str

        tokens: list[str] = []

        # Build a mock LLM that will fire on_create with a Thought
        class CreateFiringLLM(MockLLM):
            async def chat(self, messages, **kwargs):
                # Fire on_create manually
                await self.on_create(Thought(text="reasoning"))
                return await super().chat(messages, **kwargs)

        llm = CreateFiringLLM(responses=["Answer"])
        app = Lingo(llm=llm, verbose=True)

        await run(app, input_fn=OneShotInputFn("Hi"), output_fn=tokens.append)

        full_output = "".join(tokens)
        # The verbose handler writes "------- [Thinking] -------"
        assert "Thinking" in full_output


# ---------------------------------------------------------------------------
# run() with default input_fn/output_fn
# ---------------------------------------------------------------------------


class TestRunDefaultCallbacks:
    @pytest.mark.asyncio
    async def test_run_with_default_output_fn_prints(self, monkeypatch, capsys):
        """When output_fn is None, run() defaults to print() — tokens appear on stdout."""
        app = make_cli_app(responses=["PrintedToken"])

        call_count = 0

        def fake_input(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "hello"
            raise EOFError

        monkeypatch.setattr("builtins.input", fake_input)
        await run(app, input_fn=None, output_fn=None)

        captured = capsys.readouterr()
        # The token "PrintedToken" should have been printed (plus the "\n\n" separator)
        assert "PrintedToken" in captured.out

    @pytest.mark.asyncio
    async def test_run_with_default_input_fn(self, monkeypatch, capsys):
        """When input_fn is None, run() defaults to input() — one prompt fires."""
        app = make_cli_app(responses=["Response"])
        calls = []

        def fake_input(prompt=""):
            calls.append(prompt)
            raise EOFError  # immediately stop

        monkeypatch.setattr("builtins.input", fake_input)
        await run(app, input_fn=None, output_fn=lambda t: None)

        # Default input_fn calls input(">>> ")
        assert len(calls) >= 1


# ---------------------------------------------------------------------------
# loop() shim
# ---------------------------------------------------------------------------


class TestLoop:
    def test_loop_runs_synchronously(self):
        """loop() is a sync wrapper around run(); it should complete without error."""
        app = make_cli_app(responses=["Hi from loop"])
        tokens: list[str] = []

        loop(app, input_fn=OneShotInputFn("Hello"), output_fn=tokens.append)

        full_output = "".join(tokens)
        assert "Hi" in full_output or "loop" in full_output

    def test_loop_produces_same_result_as_run(self):
        """loop() and run() should produce the same token stream."""
        tokens_loop: list[str] = []
        app1 = make_cli_app(responses=["Token stream"])
        loop(app1, input_fn=OneShotInputFn("Go"), output_fn=tokens_loop.append)

        tokens_run: list[str] = []
        app2 = make_cli_app(responses=["Token stream"])
        asyncio.run(run(app2, input_fn=OneShotInputFn("Go"), output_fn=tokens_run.append))

        assert "".join(tokens_loop) == "".join(tokens_run)

    def test_loop_eofError_exits_cleanly(self):
        """loop() must not propagate EOFError to the caller."""
        app = make_cli_app(responses=[])

        def immediate_eof():
            raise EOFError

        # Should not raise
        loop(app, input_fn=immediate_eof, output_fn=lambda t: None)

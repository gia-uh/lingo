"""Tests for the lightweight Playground command-line interface."""

import sys
from types import ModuleType

import pytest

from lingo import Lingo
from lingo.mock import MockLLM
from lingo.playground.cli import build_parser, main


def test_parser_uses_safe_local_defaults():
    args = build_parser().parse_args(["examples.playground_app:create_app"])

    assert args.target == "examples.playground_app:create_app"
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.open_browser is False
    assert args.log_level == "info"


def test_parser_accepts_server_options():
    args = build_parser().parse_args(
        [
            "package.application:create_app",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--open",
            "--log-level",
            "debug",
        ]
    )

    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.open_browser is True
    assert args.log_level == "debug"


@pytest.mark.parametrize("port", ["0", "65536", "not-a-number"])
def test_parser_rejects_invalid_port(port: str):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["package:app", "--port", port])

    assert exc_info.value.code == 2


def test_main_validates_target_before_loading_optional_server(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["malformed-target"])

    assert exc_info.value.code == 2
    assert "module:attribute" in capsys.readouterr().err


def test_main_builds_app_and_runs_uvicorn(monkeypatch):
    module = ModuleType("playground_cli_target")
    module.create_app = lambda: Lingo(llm=MockLLM(["Hello"]))
    monkeypatch.setitem(sys.modules, module.__name__, module)

    invocation = {}

    def run(app, **kwargs):
        invocation["app"] = app
        invocation["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", run)

    result = main(
        [
            f"{module.__name__}:create_app",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--log-level",
            "debug",
        ]
    )

    assert result == 0
    assert invocation["app"].state.session_manager.target.mode == "factory"
    assert invocation["kwargs"] == {
        "host": "0.0.0.0",
        "port": 9000,
        "log_level": "debug",
    }

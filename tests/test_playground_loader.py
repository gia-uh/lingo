"""Tests for loading factory-backed and singleton Playground targets."""

import sys
from types import ModuleType

import pytest

from lingo import Lingo
from lingo.mock import MockLLM
from lingo.playground.loader import TargetLoadError, load_target


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes) -> None:
    """Install a temporary importable module with the requested attributes."""
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)


def _make_lingo(name: str = "Playground test") -> Lingo:
    """Create a Lingo instance without external API credentials or calls."""
    return Lingo(name, llm=MockLLM())


def test_load_target_accepts_zero_argument_factory(monkeypatch: pytest.MonkeyPatch):
    def create_app() -> Lingo:
        return _make_lingo()

    _install_module(monkeypatch, "playground_factory_target", create_app=create_app)

    target = load_target("playground_factory_target:create_app")

    first = target.create()
    second = target.create()
    assert target.mode == "factory"
    assert first is not second
    assert first.name == "Playground test"


def test_load_target_accepts_lingo_singleton(monkeypatch: pytest.MonkeyPatch):
    app = _make_lingo("Singleton")
    _install_module(monkeypatch, "playground_singleton_target", app=app)

    target = load_target("playground_singleton_target:app")

    assert target.mode == "singleton"
    assert target.create() is app


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("missing_separator", "module:attribute"),
        (":attribute", "module:attribute"),
        ("module:", "module:attribute"),
    ],
)
def test_load_target_rejects_malformed_reference(reference: str, message: str):
    with pytest.raises(TargetLoadError, match=message):
        load_target(reference)


def test_load_target_reports_import_failure():
    with pytest.raises(TargetLoadError, match="Could not import module"):
        load_target("module_that_does_not_exist_for_lingo_playground:app")


def test_load_target_reports_missing_attribute(monkeypatch: pytest.MonkeyPatch):
    _install_module(monkeypatch, "playground_missing_attribute")

    with pytest.raises(TargetLoadError, match="has no attribute 'app'"):
        load_target("playground_missing_attribute:app")


def test_load_target_rejects_factory_with_required_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    def create_app(required: str) -> Lingo:
        return _make_lingo(required)

    _install_module(monkeypatch, "playground_argument_target", create_app=create_app)

    with pytest.raises(TargetLoadError, match="must be callable without arguments"):
        load_target("playground_argument_target:create_app")


def test_factory_result_must_be_lingo(monkeypatch: pytest.MonkeyPatch):
    def create_app() -> object:
        return object()

    _install_module(
        monkeypatch, "playground_wrong_factory_target", create_app=create_app
    )
    target = load_target("playground_wrong_factory_target:create_app")

    with pytest.raises(TargetLoadError, match="must return a Lingo instance"):
        target.create()


def test_load_target_rejects_async_factory(monkeypatch: pytest.MonkeyPatch):
    async def create_app() -> Lingo:
        return _make_lingo()

    _install_module(monkeypatch, "playground_async_target", create_app=create_app)

    with pytest.raises(TargetLoadError, match="must be synchronous"):
        load_target("playground_async_target:create_app")


def test_load_target_rejects_non_callable_non_lingo_attribute(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_module(monkeypatch, "playground_wrong_target", app="not an app")

    with pytest.raises(TargetLoadError, match="must be a Lingo instance or a factory"):
        load_target("playground_wrong_target:app")


def test_bundled_playground_example_uses_factory_contract():
    target = load_target("examples.playground_app:create_app")

    assert target.mode == "factory"
    assert target.create() is not target.create()

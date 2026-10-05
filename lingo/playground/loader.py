"""Resolve Python application targets for the local Lingo Playground."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Callable, Literal, cast

from lingo.core import Lingo


class TargetLoadError(ValueError):
    """Raised when a Playground target cannot produce a valid Lingo app."""


@dataclass(frozen=True, slots=True)
class LoadedTarget:
    """A validated singleton or factory used to create Playground sessions."""

    reference: str
    mode: Literal["factory", "singleton"]
    _target: Lingo | Callable[[], Lingo]

    def create(self) -> Lingo:
        """Return the Lingo instance assigned to a new Playground session."""
        if self.mode == "singleton":
            return cast(Lingo, self._target)

        factory = cast(Callable[[], Lingo], self._target)
        try:
            app = factory()
        except Exception as exc:
            raise TargetLoadError(
                f"Factory '{self.reference}' failed while creating the Lingo app: {exc}"
            ) from exc

        if inspect.isawaitable(app):
            if inspect.iscoroutine(app):
                app.close()
            raise TargetLoadError(
                f"Factory '{self.reference}' must be synchronous and return a "
                "Lingo instance directly."
            )
        if not isinstance(app, Lingo):
            raise TargetLoadError(
                f"Factory '{self.reference}' must return a Lingo instance, "
                f"not {type(app).__name__}."
            )
        return app


def _validate_factory(reference: str, factory: Callable) -> None:
    """Ensure a target callable can be invoked without arguments."""
    if inspect.iscoroutinefunction(factory):
        raise TargetLoadError(
            f"Factory '{reference}' must be synchronous and return a Lingo "
            "instance directly."
        )
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise TargetLoadError(
            f"Factory '{reference}' has no inspectable call signature."
        ) from exc

    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    if required:
        names = ", ".join(parameter.name for parameter in required)
        raise TargetLoadError(
            f"Factory '{reference}' must be callable without arguments; "
            f"required parameters: {names}."
        )


def load_target(reference: str) -> LoadedTarget:
    """Load and validate a `module:attribute` Playground target.

    A target may be an existing :class:`Lingo` instance or a zero-argument
    callable that creates one. Factories are preferred because they give each
    browser session an isolated runtime.
    """
    if reference.count(":") != 1:
        raise TargetLoadError(
            "Playground target must use the 'module:attribute' format."
        )

    module_name, attribute_name = reference.split(":", maxsplit=1)
    if not module_name or not attribute_name:
        raise TargetLoadError(
            "Playground target must use the 'module:attribute' format."
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise TargetLoadError(
            f"Could not import module '{module_name}' for target '{reference}': {exc}"
        ) from exc

    try:
        target = getattr(module, attribute_name)
    except AttributeError as exc:
        raise TargetLoadError(
            f"Module '{module_name}' has no attribute '{attribute_name}'."
        ) from exc

    if isinstance(target, Lingo):
        return LoadedTarget(reference, "singleton", target)

    if callable(target):
        _validate_factory(reference, target)
        return LoadedTarget(reference, "factory", target)

    raise TargetLoadError(
        f"Target '{reference}' must be a Lingo instance or a factory, "
        f"not {type(target).__name__}."
    )

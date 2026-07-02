import abc
import asyncio
import inspect
import re
from purely.di import _Depends
from pydantic import BaseModel
from typing import Callable, Any, get_type_hints


class ToolResult(BaseModel):
    """Data model for the result of a tool execution."""

    tool: str
    error: str | None = None
    result: Any | None = None


def _parse_args_docstring(doc: str) -> dict[str, str]:
    """Parse a Google-style ``Args:`` block into ``{name: one-line description}``.

    Tolerant: returns ``{}`` when there is no ``Args:`` section. Continuation
    lines (indented further than the param line) are folded into the param's
    description. Stops at the next section header (``Returns:``, ``Raises:``…).
    """
    if not doc or "Args:" not in doc:
        return {}
    lines = doc.splitlines()
    out: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for raw in lines:
        stripped = raw.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        # A new section header ends the Args block.
        if stripped.endswith(":") and " " not in stripped and stripped != "Args:":
            break
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)$", raw)
        if m:
            current = m.group(1)
            out[current] = m.group(2).strip()
        elif current and stripped:
            out[current] = (out[current] + " " + stripped).strip()
    return out


class Tool(abc.ABC):
    """Abstract Base Class for a Tool."""

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description
        # Optional pre-built JSON Schema for parameters. When set, the OpenAI
        # schema builder uses it verbatim instead of deriving from parameters().
        self.json_schema: dict | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @abc.abstractmethod
    def parameters(self) -> dict[str, type]:
        """Returns a dict of parameter names to types."""
        pass

    def defaults(self) -> dict[str, Any]:
        """Parameter names → default values (only params that have a default)."""
        return {}

    def param_docs(self) -> dict[str, str]:
        """Parameter names → one-line descriptions parsed from the docstring."""
        return {}

    @abc.abstractmethod
    async def run(self, **kwargs) -> Any:
        """Executes the tool's logic."""
        pass


class DelegateTool(Tool):
    """A Tool implemented from a decorated method."""

    def __init__(self, name, description, target: Callable):
        super().__init__(name, description)
        self._target = target

    def parameters(self) -> dict[str, type]:
        """
        Extracts parameters from the function's type annotations,
        filtering out internal (_) and dependency-injected parameters.
        """
        sig = inspect.signature(self._target)
        try:
            hints = get_type_hints(self._target)
        except (AttributeError, TypeError, NameError):
            # Fallback if type hints are complex or unavailable
            hints = getattr(self._target, "__annotations__", {})

        params = {}
        for name, param in sig.parameters.items():
            # Exclude internal/manual parameters (starting with _)
            if name.startswith("_"):
                continue

            # Exclude parameters marked with dependency injection
            if isinstance(param.default, _Depends):
                continue

            # Exclude the return annotation
            if name == "return":
                continue

            # Add valid tool parameters to the schema
            params[name] = hints.get(name, Any)

        return params

    def defaults(self) -> dict[str, Any]:
        """Params that carry a signature default (excluding _/DI params)."""
        sig = inspect.signature(self._target)
        out: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name.startswith("_"):
                continue
            if isinstance(param.default, _Depends):
                continue
            if param.default is not inspect.Parameter.empty:
                out[name] = param.default
        return out

    def param_docs(self) -> dict[str, str]:
        """Parse the wrapped function's Google-style ``Args:`` docstring block."""
        return _parse_args_docstring(self._description or "")

    async def run(self, **kwargs) -> Any:
        """Runs the wrapped function."""
        return await self._target(**kwargs)


def tool(func: Callable) -> Tool:
    """
    A decorator to convert an async or sync function into a DelegateTool.
    The function's name and docstring are used as the Tool's
    name and description, respectively.
    """
    name = func.__name__
    description = func.__doc__ or "No description provided."

    # if method is async, just wrap it directly
    if asyncio.iscoroutinefunction(func):
        return DelegateTool(name, description, func)
    else:
        # if method is sync, wrap it to make it async
        async def async_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return DelegateTool(name, description, async_wrapper)

from __future__ import annotations

from typing import Callable, Coroutine
from purely import Registry
from .flow import Flow, Scope, flow
from .context import Context
from .engine import Engine
from .tools import Tool, tool


class Skill:
    """
    A specialized Flow that supports attaching subskills, cleanup hooks,
    and scoped tools.

    It bridges the Imperative (function-based) and Declarative (flow-based)
    APIs by allowing users to decorate functions that become Flow nodes,
    while also attaching additional declarative steps.
    """

    def __init__(
        self,
        registry: Registry,
        method: Callable[[Context, Engine], Coroutine],
    ):
        self.registry = registry
        self.method = self.registry.inject(method)
        self.tools: list[Tool] = []
        self.subskills: list[Skill] = []
        self.callbacks: list[Callable] = []

    def tool(self, func: Callable):
        """
        Decorator to register a tool specifically for this skill.
        The tool is only available when this skill is executing.
        """
        self.tools.append(t := tool(self.registry.inject(func)))
        return t

    def subskill(self, func: Callable[[Context, Engine], Coroutine]) -> Skill:
        """
        Decorator to attach a sub-skill to this skill.
        If multiple subskills are added, they are routed automatically.
        """
        # Inject dependencies using the parent skill's registry
        self.subskills.append(s := Skill(self.registry, func))
        return s

    def after(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to attach a cleanup/post-processing step to this skill.
        This runs after the main skill body and any sub-routing is complete.
        """
        self.callbacks.append(func)

    def build(self) -> Flow:
        """
        Compile a Skill into a Flow object linking all subskills
        and tools.
        """
        f = flow(self.registry.inject(self.method))

        if self.subskills:
            f.route(*[s.build() for s in self.subskills])

        for c in self.callbacks:
            f.custom(self.registry.inject(c))

        if self.tools:
            # We create a new container Flow to hold the Scope
            # This ensures the structure remains a Flow object
            wrapper = Flow(name=f.name, description=f.description)
            wrapper.then(Scope(self.tools, f))
            return wrapper

        return f

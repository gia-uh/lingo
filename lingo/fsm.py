from __future__ import annotations

from typing import Callable, Coroutine
from purely import Registry, ensure
from lingo.context import Context
from lingo.engine import Engine
from lingo.skills import Skill
from lingo.flow import Node


class ChangeState(BaseException):
    """
    Control-flow exception raised when a state transition occurs.
    Inherits from BaseException so it isn't caught by standard 'except Exception' blocks
    in the engine or tools, ensuring the transition bubbles up to the FSM.
    """

    def __init__(self, state: State, restart: bool):
        self.state = state
        self.restart = restart


class State(Skill):
    """
    Represents a specific state in the Finite State Machine.
    It is a full-featured Skill, meaning it can have its own tools,
    subskills, and lifecycle hooks.
    """

    def __init__(self, name: str, registry: Registry, func: Callable):
        super().__init__(registry, func)
        self.name = name

    def __repr__(self):
        return f"<State: {self.name}>"


class StateMachine(Node):
    """
    A Finite State Machine implemented as a Flow Node.

    It manages a collection of States (Skills) and executes the active one.
    It supports deterministic transitions via 'goto()' which can optionally
    restart execution immediately (Hot Handoff).
    """

    def __init__(
        self,
        registry: Registry,
    ):
        """
        Args:
            registry: The tool registry to bind states to.
            state_key: The key in context.state where the current status string is stored.
        """
        self.registry = registry
        self._states: dict[str, State] = {}
        self._initial_state: State | None = None
        self._current_state: State | None = None

        # Holds the active context during execution to allow parameter-less goto()
        self.context: Context | None = None

    def state(self, func: Callable[[Context, Engine], Coroutine]):
        """
        Decorator to register a function as a State.

        Usage:
            @fsm.state
            async def my_state(ctx, eng): ...
        """
        name = func.__name__
        return self._register(name, func)

    def _register(self, name: str, func: Callable) -> State:
        if name in self._states:
            raise ValueError(f"State '{name}' is already defined.")

        state = State(name, self.registry, func)
        self._states[name] = state

        # The first registered state becomes the default start state
        if self._initial_state is None:
            self._initial_state = state

        return state

    def goto(self, state: State, restart: bool = False):
        """
        Transitions the FSM to the target State and interrupts current execution.

        Args:
            state: The target State object (must be registered).
            restart: If True, the FSM immediately executes the new state's logic
                     in the same turn (Hot Handoff). If False, execution stops
                     and waits for the next turn.
        """
        if self.context is None:
            raise RuntimeError("fsm.goto() called outside of FSM execution loop.")

        if not isinstance(state, State):
            raise TypeError(f"Expected State object, got {type(state)}")

        if state.name not in self._states:
            raise ValueError(f"State '{state.name}' is not registered with this FSM.")

        # Raise the signal to stop execution and bubble up to execute()
        raise ChangeState(state=state, restart=restart)

    async def execute(self, context: Context, engine: Engine):
        """
        Executes the FSM logic:
        1. Binds context to self.
        2. Resolves current state.
        3. Executes the State's flow.
        4. Catches StateChangeException to handle transitions.
        """
        # Bind context for the duration of this execution
        self.context = context

        if not self._states:
            raise RuntimeError("FSM has no states defined.")

        if self._current_state is None:
            self._current_state = self._initial_state

        try:
            # We loop to handle 'restart=True' transitions (Hot Handoffs)
            while True:
                state = ensure(self._current_state)

                try:
                    # 2. Build & Execute the State's Flow
                    # Since State is a Skill, build() returns a Flow.
                    await state.build().execute(context, engine)

                    # If we finish without exception, we are done for this turn.
                    return

                except ChangeState as e:
                    # 3. Handle Transition Signal
                    self._current_state = e.state

                    if not e.restart:
                        return

        finally:
            # Cleanup: Unbind context
            self.context = None

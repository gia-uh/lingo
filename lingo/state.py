from collections import UserDict
import contextlib
import copy
from typing import Any, Type, Self
from pydantic import BaseModel, ValidationError


class State[T: BaseModel](UserDict):
    """
    A smart dictionary for conversation state.

    Features:
    - **Attribute Access**: Use `state.count` instead of `state['count']`.
    - **IDE Support**: Subclass this to get autocompletion.
    - **Transaction Safety**: Use `atomic()` and `fork()` for rollbacks.
    - **Serialization**: It's just a dict at runtime, so it serializes easily.
    """

    def __init__(
        self,
        data: dict | None = None,
        schema: Type[T] | None = None,
        shared_keys: set[str] | None = None,
        **kwargs
    ):
        """
        Args:
            data: Initial dictionary data.
            schema: Optional Pydantic model for validation.
            shared_keys: Keys that should not be deep-copied (e.g., database connections).
            **kwargs: Additional initial keys.
        """
        initial_data = data or {}
        initial_data.update(kwargs)
        super().__init__(initial_data)

        self._schema = schema
        self._shared_keys = shared_keys or set()

        if self._schema:
            self.validate()

    def validate(self):
        """Validates the current state against the Pydantic schema."""
        if self._schema:
            try:
                # Validate by instantiating the model
                validated = self._schema(**self).model_dump()
                self.update(validated)
            except ValidationError as e:
                raise ValueError(f"State validation failed: {e}")

    def __getattr__(self, key: str) -> Any:
        """Enables `state.key` access."""
        try:
            return self[key]
        except KeyError:
            # Fallback to allow methods/properties of the class to work
            raise AttributeError(f"'State' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        """Enables `state.key = value` assignment."""
        # Private attributes (internal state) go to the object's __dict__
        if key.startswith("_"):
            super().__setattr__(key, value)
        # Public attributes go to the dictionary
        else:
            self[key] = value

    def __delattr__(self, key: str):
        """Enables `del state.key`."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

    def _smart_copy(self) -> dict:
        """Deep copies data, preserving shared keys by reference."""
        new_data = {}
        for k, v in self.items():
            if k in self._shared_keys:
                new_data[k] = v
            else:
                new_data[k] = copy.deepcopy(v)
        return new_data

    def clone(self) -> Self:
        """Returns an independent copy of the State (for parallel branches)."""
        new_state = self.__class__(
            self._smart_copy(),
            schema=self._schema,
            shared_keys=self._shared_keys
        )
        return new_state

    @contextlib.contextmanager
    def atomic(self):
        """
        Savepoint: Rollback changes only if an exception occurs.
        """
        snapshot = self._smart_copy()
        try:
            yield self
            # Optional: Validate on successful exit
            if self._schema:
                self.validate()
        except Exception:
            self.clear()
            self.update(snapshot)
            raise

    @contextlib.contextmanager
    def fork(self):
        """
        Temporary Scope: Always rollback changes on exit.
        """
        snapshot = self._smart_copy()
        try:
            yield self
        finally:
            self.clear()
            self.update(snapshot)

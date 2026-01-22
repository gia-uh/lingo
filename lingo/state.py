import contextlib
import copy
from typing import Any, Type, TypeVar, Self
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class State(dict):
    """
    A smart dictionary for conversation state.

    Features:
    - **Attribute Access**: Use `state.count` instead of `state['count']`.
    - **Recursion Safe**: Strict guards on magic methods.
    - **Transaction Safety**: Use `atomic()` and `fork()` for rollbacks.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Intervention to make Pydantic-style defaults work with Dict storage.
        We strip assignments like `score = 0` from the class so they don't block
        __getattr__, and store them in `_class_defaults` to be applied in __init__.
        """
        super().__init_subclass__(**kwargs)

        defaults = {}
        # Iterate over class dict to find assigned values
        for k in list(cls.__dict__.keys()):
            if k.startswith("_"):
                continue

            v = cls.__dict__[k]
            # Skip methods, properties, and descriptors
            if callable(v) or hasattr(v, "__get__"):
                continue

            # It's a default value (e.g. score: int = 0)
            defaults[k] = v
            # Remove from class so instance access triggers __getattr__
            delattr(cls, k)

        # Merge with parent defaults for inheritance support
        parent_defaults = getattr(cls, "_class_defaults", {})
        final_defaults = parent_defaults.copy()
        final_defaults.update(defaults)

        # Use object.__setattr__ to avoid any interference
        type.__setattr__(cls, "_class_defaults", final_defaults)

    def __init__(
        self,
        data: dict | None = None,
        schema: Type[T] | None = None,
        shared_keys: set[str] | None = None,
        **kwargs,
    ):
        # 1. Initialize the dict content first
        initial_data = data or {}
        initial_data.update(kwargs)
        super().__init__(initial_data)

        # 2. Use object.__setattr__ to avoid triggering our own __setattr__ logic
        #    This ensures these exist in __dict__ before any property access happens.
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_shared_keys", shared_keys or set())

        # 3. Validate if schema is present
        if self._schema:
            self.validate()

    def validate(self):
        """Validates the current state against the Pydantic schema."""
        if self._schema:
            try:
                # We validate by constructing the model from the dict content
                validated = self._schema(**self).model_dump()
                self.update(validated)
            except ValidationError as e:
                raise ValueError(f"State validation failed: {e}")

    def __getattr__(self, key: str) -> Any:
        """Enables `state.key` access."""
        # CRITICAL FIX: Immediately fail for private/magic methods.
        # This prevents infinite recursion when pickling/copying checks for __getstate__, etc.
        if key.startswith("_"):
            raise AttributeError(key)

        try:
            return self[key]
        except KeyError:
            # Must raise AttributeError, not KeyError, for getattr protocol
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key: str, value: Any):
        """Enables `state.key = value` assignment."""
        # Private attributes go directly to the instance's __dict__
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        # Public attributes go to the dictionary content
        else:
            self[key] = value

    def __delattr__(self, key: str):
        """Enables `del state.key`."""
        if key.startswith("_"):
            object.__delattr__(self, key)
        else:
            try:
                del self[key]
            except KeyError:
                raise AttributeError(key)

    # --- Copying & Serialization Logic ---

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
            self._smart_copy(), schema=self._schema, shared_keys=self._shared_keys
        )
        return new_state

    def __copy__(self):
        """Support for copy.copy()"""
        return self.clone()

    def __deepcopy__(self, memo):
        """Support for copy.deepcopy()"""
        # We manually implement this to use our smart copy logic
        # and avoid recursive pickling checks.
        return self.clone()

    # --- Context Managers ---

    @contextlib.contextmanager
    def atomic(self):
        """
        Savepoint: Rollback changes only if an exception occurs.
        """
        snapshot = self._smart_copy()
        try:
            yield self
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

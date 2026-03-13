import yaml
import contextlib
import copy
from typing import Any, Type, Self, Iterator
from pydantic import BaseModel, ValidationError


class State[T: BaseModel](dict):
    """
    A smart dictionary for managing conversation and application state.

    Features:
    - **Attribute Access**: Access keys as attributes (e.g., `state.count` instead of `state['count']`).
    - **Pydantic Integration**: Optionally validate state against a Pydantic schema.
    - **Transaction Safety**: Use `atomic()` for rollback on error, or `fork()` for temporary modifications.
    - **Prompt Ready**: Easily render state to YAML for injection into LLM prompts.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Customizes subclass creation to support Pydantic-style field defaults.

        Assignments like `score: int = 0` are moved from the class dictionary to
        an internal `_class_defaults` map, ensuring that instance-level attribute
        access triggers `__getattr__` correctly.
        """
        super().__init_subclass__(**kwargs)

        defaults = {}
        for k in list(cls.__dict__.keys()):
            if k.startswith("_"):
                continue

            v = cls.__dict__[k]
            if callable(v) or hasattr(v, "__get__"):
                continue

            defaults[k] = v
            delattr(cls, k)

        parent_defaults = getattr(cls, "_class_defaults", {})
        final_defaults = parent_defaults.copy()
        final_defaults.update(defaults)

        type.__setattr__(cls, "_class_defaults", final_defaults)

    def __init__(
        self,
        data: dict | None = None,
        schema: Type[T] | None = None,
        shared_keys: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the State.

        Args:
            data: Initial dictionary of data.
            schema: Optional Pydantic model for validation.
            shared_keys: Keys that should be shared by reference during cloning.
            **kwargs: Initial data passed as keyword arguments.
        """
        final_data = getattr(self.__class__, "_class_defaults", {}).copy()
        if data:
            final_data.update(data)
        final_data.update(kwargs)

        super().__init__(final_data)

        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_shared_keys", shared_keys or set())

        if self._schema:
            self.validate()

    def validate(self) -> None:
        """
        Validates the current dictionary content against the assigned Pydantic schema.

        Raises:
            ValueError: If validation fails.
        """
        if self._schema:
            try:
                validated = self._schema(**self).model_dump()
                self.update(validated)
            except ValidationError as e:
                raise ValueError(f"State validation failed: {e}")

    def __getattr__(self, key: str) -> Any:
        """
        Enables attribute-style access to dictionary keys.

        Args:
            key: The attribute name.

        Returns:
            The value associated with the key.

        Raises:
            AttributeError: If the key does not exist or is private.
        """
        if key.startswith("_"):
            raise AttributeError(key)

        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Enables attribute-style assignment.

        Args:
            key: The attribute name.
            value: The value to assign.
        """
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __delattr__(self, key: str) -> None:
        """
        Enables attribute-style deletion.

        Args:
            key: The attribute name.

        Raises:
            AttributeError: If the key does not exist.
        """
        if key.startswith("_"):
            object.__delattr__(self, key)
        else:
            try:
                del self[key]
            except KeyError:
                raise AttributeError(key)

    def _smart_copy(self) -> dict:
        """
        Internal helper to deep-copy data while preserving shared keys by reference.
        """
        new_data = {}
        for k, v in self.items():
            if k in self._shared_keys:
                new_data[k] = v
            else:
                new_data[k] = copy.deepcopy(v)
        return new_data

    def clone(self) -> Self:
        """
        Returns an independent copy of the State.
        Useful for branching conversation history in parallel flows.
        """
        return self.__class__(
            self._smart_copy(), schema=self._schema, shared_keys=self._shared_keys
        )

    def __copy__(self) -> Self:
        """Standard copy support."""
        return self.clone()

    def __deepcopy__(self, memo: Any) -> Self:
        """Standard deepcopy support, using smart copy logic."""
        return self.clone()

    @contextlib.contextmanager
    def atomic(self) -> Iterator[Self]:
        """
        Transactional context manager.

        Rolls back all changes made within the block if an exception occurs.
        If a schema is present, validates the final state before completing.
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
    def fork(self) -> Iterator[Self]:
        """
        Scoped context manager.

        Always rolls back all changes made within the block upon exit, regardless
        of whether an exception occurred. Useful for speculative execution.
        """
        snapshot = self._smart_copy()
        try:
            yield self
        finally:
            self.clear()
            self.update(snapshot)

    def render(self, *keys: str) -> str:
        """
        Returns a clean YAML string representation of the state.

        Args:
            *keys: Optional specific keys to include. If empty, all keys are rendered.

        Returns:
            A YAML-formatted string.
        """
        if keys:
            data = {k: self[k] for k in keys if k in self}
        else:
            data = dict(self)

        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False).strip()

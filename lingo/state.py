import yaml
import contextlib
import copy
from typing import Any, Type, Self
from pydantic import BaseModel, ValidationError


class State[T: BaseModel](dict):
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
        # 1. Gather all data sources
        # Priority: kwargs > data > subclass_defaults
        final_data = getattr(self.__class__, "_class_defaults", {}).copy()
        if data:
            final_data.update(data)
        final_data.update(kwargs)

        # 2. Initialize dict
        super().__init__(final_data)

        # 3. Setup internals (bypass __setattr__)
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_shared_keys", shared_keys or set())

        # 4. Validate
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

    def render(self, *keys: str) -> str:
        """
        Returns a clean YAML string representation of the state.
        Useful for injecting state data into prompt contexts.

        Usage:
            context.append(Message.system(f"CURRENT STATE:\n{state.render('score', 'history')}"))
        """

        # 1. Filter data
        if keys:
            # Only include keys that exist to avoid KeyErrors, or let it crash if strictness is preferred.
            # Here we skip missing keys to be safe for prompting.
            data = {k: self[k] for k in keys if k in self}
        else:
            # Convert to plain dict to avoid YAML object tags
            data = dict(self)

        # 2. Render
        # default_flow_style=False -> Uses block style (lists/dicts are expanded)
        # sort_keys=False -> Preserves insertion order (better for context logic)
        return yaml.safe_dump(data, default_flow_style=False, sort_keys=False).strip()

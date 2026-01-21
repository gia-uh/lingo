import contextlib
from .llm import Message


class Context:
    """
    A *mutable* object representing a single interaction.
    It holds the message history and provides methods
    for manipulating that history.
    """

    def __init__(self, messages: list[Message]):
        self._messages = messages

    @property
    def messages(self) -> list[Message]:
        """Gets the mutable list of messages for this turn."""
        return self._messages

    def append(self, message: Message | str) -> None:
        """
        Mutates the context by appending a message to its
        internal list.
        """
        if isinstance(message, str):
            message = Message.system(message)

        self._messages.append(message)

    def prepend(self, message: Message | str) -> None:
        """
        Mutates the context by prepending a message to its
        internal list.
        """
        if isinstance(message, str):
            message = Message.system(message)

        self._messages.insert(0, message)

    def clone(self) -> "Context":
        """
        Returns a new, independent Context instance with a *shallow copy*
        of the current messages, allowing for durable branching.
        """
        return Context(list(self._messages))

    @contextlib.contextmanager
    def fork(self):
        """
        A context manager for temporary, "what-if" state.
        All mutations (like .append()) inside the 'with'
        block will be discarded upon exit.
        """
        # Save the current list of messages
        snapshot = list(self._messages)

        try:
            yield self
        finally:
            # Restore the original list of messages
            self._messages = snapshot

    @contextlib.contextmanager
    def atomic(self):
        """
        Rolls back mutations ONLY if an exception is raised.
        If successful, the mutations are committed to the history.
        """
        # Snapshot the current state
        snapshot = list(self._messages)

        try:
            yield self
        except Exception:
            # Rollback: restore the snapshot on failure
            self._messages = snapshot
            raise

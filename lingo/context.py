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
        self._state_stack: list[list[Message]] = []

    @property
    def messages(self) -> list[Message]:
        """Gets the mutable list of messages for this turn."""
        return self._messages

    def append(self, message: Message) -> None:
        """
        Mutates the context by appending a message to its
        internal list.
        """
        self._messages.append(message)

    def prepend(self, message: Message) -> None:
        """
        Mutates the context by prepending a message to its
        internal list.
        """
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
        self._state_stack.append(list(self._messages))

        try:
            yield self
        finally:
            # Restore the original list of messages
            self._messages = self._state_stack.pop()

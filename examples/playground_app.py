"""A factory-backed application for the Lingo Playground.

Run it with:

    lingo-playground examples.playground_app:create_app --open
"""

from lingo import Lingo


def create_app() -> Lingo:
    """Create an isolated assistant for one Playground browser session."""
    return Lingo(
        "Local Assistant",
        description=(
            "A concise assistant running in an isolated Lingo Playground session."
        ),
    )

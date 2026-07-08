"""index_hello.py — the minimal lingo application.

A Lingo with no skills configured simply replies to every message.
This is the floor — the simplest thing that works.

Run:
    API_KEY=... python examples/index_hello.py
"""

from lingo import Lingo
from lingo.cli import loop

app = Lingo("Assistant", description="A helpful assistant.")


def main():
    loop(app)


if __name__ == "__main__":
    main()

"""Lightweight local browser interface for Lingo applications.

Install the optional dependencies with ``pip install lingo-ai[playground]``.
The target loader remains importable without those dependencies so base Lingo
installations do not eagerly import a web framework.
"""

from .loader import LoadedTarget, TargetLoadError, load_target


__all__ = ["LoadedTarget", "TargetLoadError", "load_target"]

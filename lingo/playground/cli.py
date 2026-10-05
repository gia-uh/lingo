"""Command-line entry point for the local Lingo Playground."""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Sequence
import webbrowser

from .loader import TargetLoadError, load_target


_LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _port(value: str) -> int:
    """Parse and validate a TCP port for argparse."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    """Build the Playground argument parser."""
    parser = argparse.ArgumentParser(
        prog="lingo-playground",
        description="Run a Lingo application in a lightweight local web UI.",
    )
    parser.add_argument(
        "target",
        metavar="TARGET",
        help="Lingo application or factory in module:attribute form",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="server bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        default=8765,
        type=_port,
        help="server bind port (default: 8765)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="open the Playground in the default browser after startup",
    )
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="info",
        help="Uvicorn log level (default: info)",
    )
    return parser


def _open_browser(port: int) -> None:
    """Open the loopback Playground URL after Uvicorn has started."""
    timer = threading.Timer(
        0.5,
        webbrowser.open,
        args=(f"http://127.0.0.1:{port}",),
    )
    timer.daemon = True
    timer.start()


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a target and run its local Playground server."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        target = load_target(args.target)
    except TargetLoadError as exc:
        parser.error(str(exc))

    try:
        import uvicorn
    except ModuleNotFoundError:
        parser.error(
            "Playground dependencies are not installed. "
            "Install them with: pip install 'lingo-ai[playground]'"
        )

    try:
        from .app import create_playground_app
    except ModuleNotFoundError as exc:
        if exc.name == "starlette":
            parser.error(
                "Playground dependencies are not installed. "
                "Install them with: pip install 'lingo-ai[playground]'"
            )
        raise

    if args.host not in _LOOPBACK_HOSTS:
        print(
            "Warning: Lingo Playground is a local development tool and is not "
            "hardened for public hosting.",
            file=sys.stderr,
        )

    if args.open_browser:
        _open_browser(args.port)

    app = create_playground_app(target)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0

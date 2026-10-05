"""Starlette application for the local Lingo Playground."""

from __future__ import annotations

from importlib import resources
import json
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .events import ErrorEvent, event_payload
from .loader import LoadedTarget
from .session import SessionManager, SessionUnavailableError


MAX_MESSAGE_LENGTH = 20_000
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; connect-src 'self' ws: wss:; "
        "img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
_ASSETS = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


def _asset_response(filename: str) -> Response:
    """Read one known packaged asset without exposing arbitrary paths."""
    content = (
        resources.files("lingo.playground").joinpath("static", filename).read_bytes()
    )
    return Response(
        content,
        media_type=_ASSETS[filename],
        headers=_SECURITY_HEADERS,
    )


async def _index(request: Request) -> Response:
    """Serve the Playground application shell."""
    return _asset_response("index.html")


async def _static(request: Request) -> Response:
    """Serve a fixed CSS or JavaScript asset."""
    filename = request.path_params["filename"]
    if filename not in {"styles.css", "app.js"}:
        return Response("Not found", status_code=404, headers=_SECURITY_HEADERS)
    return _asset_response(filename)


async def _health(request: Request) -> JSONResponse:
    """Return a dependency-free liveness response."""
    return JSONResponse({"status": "ok"}, headers=_SECURITY_HEADERS)


def _origin_is_allowed(websocket: WebSocket) -> bool:
    """Accept absent origins or the exact origin that served this socket."""
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    host = websocket.headers.get("host")
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.netloc == host


async def _send_error(
    websocket: WebSocket,
    code: str,
    message: str,
    *,
    recoverable: bool = True,
) -> None:
    """Send a validated safe error event."""
    await websocket.send_json(
        event_payload(ErrorEvent(code=code, message=message, recoverable=recoverable))
    )


async def _websocket(websocket: WebSocket) -> None:
    """Bridge one browser WebSocket to an isolated Playground session."""
    if not _origin_is_allowed(websocket):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    manager: SessionManager = websocket.app.state.session_manager

    async def send(event) -> None:
        await websocket.send_json(event_payload(event))

    try:
        session = await manager.open(send)
    except SessionUnavailableError as exc:
        await _send_error(
            websocket,
            "session_unavailable",
            str(exc),
            recoverable=False,
        )
        await websocket.close(code=1013)
        return

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except json.JSONDecodeError:
                await _send_error(websocket, "invalid_event", "Expected a JSON event.")
                continue

            if not isinstance(payload, dict):
                await _send_error(
                    websocket, "invalid_event", "Unsupported Playground event."
                )
                continue

            event_type = payload.get("type")
            if event_type == "reset":
                await session.reset()
                continue

            if event_type != "user_message":
                await _send_error(
                    websocket, "invalid_event", "Unsupported Playground event."
                )
                continue

            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                await _send_error(
                    websocket,
                    "invalid_message",
                    "Message content must be a non-empty string.",
                )
                continue
            if len(content) > MAX_MESSAGE_LENGTH:
                await _send_error(
                    websocket,
                    "message_too_large",
                    f"Message content cannot exceed {MAX_MESSAGE_LENGTH} characters.",
                )
                continue

            await session.submit(content)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.close(session.id)


def create_playground_app(target: LoadedTarget) -> Starlette:
    """Create the local ASGI application for a loaded Lingo target."""
    app = Starlette(
        debug=False,
        routes=[
            Route("/", _index),
            Route("/health", _health),
            Route("/static/{filename}", _static),
            WebSocketRoute("/ws", _websocket),
        ],
    )
    app.state.session_manager = SessionManager(target)
    return app

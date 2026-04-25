"""WebSocket broadcast emitter.

The Layer 2 pipeline runs synchronously (numpy / scipy), so this emitter
hosts an asyncio loop on a dedicated daemon thread.  Connected WebSocket
clients receive the JSON payload as text frames.

Usage::

    emitter = WebSocketEmitter(host="localhost", port=9001)
    emitter.start()
    try:
        emitter.emit({"command": "SELECT", "frequency": 12.0, ...})
    finally:
        emitter.stop()
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol, serve

from layer2_processing.logging_config import get_logger

logger = get_logger(__name__)


class WebSocketEmitter:
    """Broadcasts the same JSON message to every connected client."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9001,
    ) -> None:
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[WebSocketServerProtocol] = set()
        self._server = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def start(self, timeout_s: float = 5.0) -> None:
        """Spin up the asyncio loop on a daemon thread and bind the server."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run_loop, name="WS-emitter", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError(
                f"WebSocket server did not become ready within {timeout_s:.1f}s"
            )
        if self._startup_error is not None:
            raise RuntimeError(
                f"WebSocket server failed to start: {self._startup_error}"
            ) from self._startup_error
        logger.info(
            "WebSocket emitter listening on ws://%s:%d", self._host, self._port
        )

    def stop(self, timeout_s: float = 5.0) -> None:
        if self._loop is None or not self._loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result(timeout_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebSocket shutdown error: %s", exc)
        if self._thread is not None:
            self._thread.join(timeout_s)
        self._thread = None
        self._loop = None
        logger.info("WebSocket emitter stopped.")

    def emit(self, payload: dict[str, Any]) -> None:
        """Schedule a broadcast of ``payload`` to all connected clients."""
        if self._loop is None or not self._clients:
            return
        msg = json.dumps(payload, separators=(",", ":"))
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    # ----- internals ------------------------------------------------------

    async def _broadcast(self, message: str) -> None:
        if not self._clients:
            return
        results = await asyncio.gather(
            *(c.send(message) for c in list(self._clients)),
            return_exceptions=True,
        )
        for client, res in zip(list(self._clients), results):
            if isinstance(res, Exception):
                self._clients.discard(client)

    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        logger.info("WS client connected (%d total)", len(self._clients))
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)
            logger.info("WS client disconnected (%d total)", len(self._clients))

    async def _shutdown(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for c in list(self._clients):
            try:
                await c.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()
        if self._stop_event is not None:
            self._stop_event.set()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve_forever())
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()  # unblock the waiter so start() raises
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _serve_forever(self) -> None:
        self._stop_event = asyncio.Event()
        try:
            self._server = await serve(self._handler, self._host, self._port)
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        await self._stop_event.wait()

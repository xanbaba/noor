"""FastAPI application for Layer 3 MVP.

Endpoints:

- ``GET  /``        — serves ``index.html`` (the SSVEP flicker page)
- ``GET  /health``  — liveness probe ``{"ok": true}``
- ``WS   /ws``      — browser clients connect here; receives re-broadcast
                       SELECT payloads from the :class:`Layer2Bridge`

The bridge is started as a FastAPI *lifespan* background task so it connects
to Layer 2 as soon as the server boots and tears down cleanly on shutdown.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from layer3_backend.config import BackendConfig
from layer3_backend.layer2_bridge import Broadcaster, Layer2Bridge

logger = logging.getLogger(__name__)

# Module-level singletons populated by create_app()
_broadcaster: Broadcaster | None = None
_bridge: Layer2Bridge | None = None


def create_app(cfg: BackendConfig) -> FastAPI:
    """Build and return the FastAPI application."""
    global _broadcaster, _bridge  # noqa: PLW0603

    _broadcaster = Broadcaster()
    _bridge = Layer2Bridge(
        upstream_url=cfg.layer2_ws_url,
        broadcaster=_broadcaster,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await _bridge.start()
        yield
        await _bridge.stop()

    static_path = Path(__file__).parent / cfg.static_dir
    app = FastAPI(title="SSVEP-BCI Layer 3 MVP", lifespan=lifespan)

    # ── Routes ────────────────────────────────────────────────────────────

    @app.get("/", response_class=FileResponse)
    async def index():
        return FileResponse(static_path / "index.html", media_type="text/html")

    @app.get("/health")
    async def health():
        return JSONResponse({"ok": True})

    @app.get("/config")
    async def config():
        """Expose the stimulus frequency list so the frontend stays in sync."""
        return JSONResponse({
            "stimulus_frequencies_hz": cfg.stimulus_frequencies_hz,
        })

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        _broadcaster.add(websocket)
        logger.info("Browser WS connected (%d total)", _broadcaster.client_count)

        # Send the last known payload immediately so the page doesn't
        # wait for the next epoch to populate the status box.
        last = _broadcaster.last_payload
        if last is not None:
            try:
                await websocket.send_text(
                    json.dumps(last, separators=(",", ":"))
                )
            except Exception:  # noqa: BLE001
                _broadcaster.remove(websocket)
                return

        try:
            while True:
                # Keep the connection alive; we only send, never receive
                # meaningful data, but we must drain pings/pongs.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            _broadcaster.remove(websocket)
            logger.info(
                "Browser WS disconnected (%d total)", _broadcaster.client_count
            )

    # Static assets (JS, CSS) — mounted AFTER explicit routes so /ws
    # doesn't get shadowed.
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    return app

"""FastAPI application for Layer 3 MVP.

Endpoints:

- ``GET  /``           — serves ``index.html`` (the SSVEP flicker page)
- ``GET  /health``     — liveness probe ``{"ok": true}``
- ``GET  /config``     — stimulus frequencies + phrase cards for the frontend
- ``POST /api/speak``  — ElevenLabs TTS by ``phrase_id`` (API key from env)
- ``WS   /ws``         — browser clients; receives Layer 2 SELECT + ``confirmed`` events

The bridge is started as a FastAPI *lifespan* background task so it connects
to Layer 2 as soon as the server boots and tears down cleanly on shutdown.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from layer3_backend.config import BackendConfig
from layer3_backend.confirmation import normalise_frequency_hz
from layer3_backend.elevenlabs_tts import synthesize_speech_mpeg
from layer3_backend.layer2_bridge import Broadcaster, Layer2Bridge

logger = logging.getLogger(__name__)

# Module-level singletons populated by create_app()
_broadcaster: Broadcaster | None = None
_bridge: Layer2Bridge | None = None

# Simple per-client-IP cooldown for TTS (seconds)
_SPEAK_COOLDOWN_S = 2.5
_speak_last_mono: dict[str, float] = {}


class SpeakRequest(BaseModel):
    phrase_id: str


class SpeakTextRequest(BaseModel):
    text: str


def create_app(cfg: BackendConfig) -> FastAPI:
    """Build and return the FastAPI application."""
    global _broadcaster, _bridge  # noqa: PLW0603

    phrase_by_norm_frequency = {
        normalise_frequency_hz(p.frequency_hz): p for p in cfg.phrases
    }

    _broadcaster = Broadcaster()
    _bridge = Layer2Bridge(
        upstream_url=cfg.layer2_ws_url,
        broadcaster=_broadcaster,
        phrase_by_norm_frequency=phrase_by_norm_frequency,
        streak_required=5,
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
        """Stimulus frequencies and phrase cards for dynamic tile layout."""
        phrases_out = [
            {
                "id": p.id,
                "label": p.label,
                "frequency_hz": p.frequency_hz,
                "color": p.color,
                "utterance": p.utterance,
            }
            for p in cfg.phrases
        ]
        return JSONResponse(
            {
                "stimulus_frequencies_hz": cfg.stimulus_frequencies_hz,
                "phrases": phrases_out,
            }
        )

    @app.post("/api/speak")
    async def api_speak(request: Request, body: SpeakRequest) -> Response:
        """Synthesize ``phrase_id`` via ElevenLabs; returns ``audio/mpeg``."""
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="TTS unavailable: set ELEVENLABS_API_KEY environment variable.",
            )

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        last = _speak_last_mono.get(client_ip, 0.0)
        if now - last < _SPEAK_COOLDOWN_S:
            raise HTTPException(status_code=429, detail="Too many TTS requests; try again shortly.")
        _speak_last_mono[client_ip] = now

        phrase = next((p for p in cfg.phrases if p.id == body.phrase_id), None)
        if phrase is None:
            raise HTTPException(status_code=404, detail=f"Unknown phrase_id: {body.phrase_id!r}")

        voice = os.environ.get("ELEVENLABS_VOICE_ID") or cfg.elevenlabs_voice_id
        try:
            audio = await synthesize_speech_mpeg(
                phrase.utterance,
                voice_id=voice,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ElevenLabs TTS error: %s", exc)
            raise HTTPException(status_code=502, detail="Upstream TTS failed.") from exc

        return Response(content=audio, media_type="audio/mpeg")

    @app.post("/api/speak-text")
    async def api_speak_text(request: Request, body: SpeakTextRequest) -> Response:
        """Synthesize arbitrary text via ElevenLabs TTS; returns ``audio/mpeg``.

        Used by the multi-page frontend for dynamic utterances (wheelchair
        commands, food/water/caregiver requests, individual letters, etc.)
        rather than the static phrase-card utterances used by ``/api/speak``.
        """
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="TTS unavailable: set ELEVENLABS_API_KEY environment variable.",
            )

        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty.")

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        last = _speak_last_mono.get(client_ip, 0.0)
        if now - last < _SPEAK_COOLDOWN_S:
            raise HTTPException(status_code=429, detail="Too many TTS requests; try again shortly.")
        _speak_last_mono[client_ip] = now

        voice = os.environ.get("ELEVENLABS_VOICE_ID") or cfg.elevenlabs_voice_id
        try:
            audio = await synthesize_speech_mpeg(
                text,
                voice_id=voice,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ElevenLabs TTS error: %s", exc)
            raise HTTPException(status_code=502, detail="Upstream TTS failed.") from exc

        return Response(content=audio, media_type="audio/mpeg")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        _broadcaster.add(websocket)
        logger.info("Browser WS connected (%d total)", _broadcaster.client_count)

        last = _broadcaster.last_payload
        if last is not None:
            try:
                await websocket.send_text(json.dumps(last, separators=(",", ":")))
            except Exception:  # noqa: BLE001
                _broadcaster.remove(websocket)
                return

        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            _broadcaster.remove(websocket)
            logger.info(
                "Browser WS disconnected (%d total)", _broadcaster.client_count
            )

    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    return app

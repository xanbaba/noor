"""Bridge between Layer 2's WebSocket output and the Layer 3 broadcaster.

Maintains a persistent connection to ``ws://localhost:9001`` (or whatever the
config says) with exponential-backoff reconnect.  Every SELECT payload is
parsed, cached as ``last_payload``, and forwarded to the :class:`Broadcaster`
which fans it out to every connected browser WebSocket client.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets

from layer3_backend.config import PhraseCard
from layer3_backend.confirmation import StreakTracker

logger = logging.getLogger(__name__)


class Broadcaster:
    """Thread-safe set of connected browser WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set = set()
        self._last_payload: dict[str, Any] | None = None

    @property
    def last_payload(self) -> dict[str, Any] | None:
        return self._last_payload

    @last_payload.setter
    def last_payload(self, value: dict[str, Any]) -> None:
        self._last_payload = value

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def add(self, ws) -> None:
        self._clients.add(ws)

    def remove(self, ws) -> None:
        self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        self._last_payload = payload
        if not self._clients:
            return
        msg = json.dumps(payload, separators=(",", ":"))
        results = await asyncio.gather(
            *(c.send_text(msg) for c in list(self._clients)),
            return_exceptions=True,
        )
        for client, res in zip(list(self._clients), results):
            if isinstance(res, Exception):
                self._clients.discard(client)


class Layer2Bridge:
    """Asyncio task that subscribes to Layer 2 and feeds the broadcaster."""

    def __init__(
        self,
        upstream_url: str,
        broadcaster: Broadcaster,
        phrase_by_norm_frequency: dict[float, PhraseCard] | None = None,
        streak_required: int = 5,
    ) -> None:
        self._url = upstream_url
        self._broadcaster = broadcaster
        self._phrase_by_freq = phrase_by_norm_frequency or {}
        self._streak = StreakTracker(required=streak_required)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="layer2-bridge")
        logger.info("Layer2Bridge started | upstream=%s", self._url)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Layer2Bridge stopped.")

    async def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self._url, open_timeout=2.0
                ) as ws:
                    logger.info("Connected to Layer 2 at %s", self._url)
                    backoff = 0.5
                    async for message in ws:
                        if self._stop.is_set():
                            return
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            logger.warning("Non-JSON from Layer 2: %s", message[:80])
                            continue
                        await self._broadcaster.broadcast(payload)
                        await self._maybe_broadcast_confirmed(payload)
            except asyncio.CancelledError:
                return
            except (OSError, websockets.WebSocketException) as exc:
                if self._stop.is_set():
                    return
                logger.warning(
                    "Layer 2 connection lost (%s); reconnecting in %.1fs",
                    exc.__class__.__name__,
                    backoff,
                )
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=backoff
                    )
                    return  # stop was set during the wait
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 5.0)

    async def _maybe_broadcast_confirmed(self, payload: dict[str, Any]) -> None:
        if payload.get("command") != "SELECT":
            return
        freq = payload.get("frequency")
        if not isinstance(freq, (int, float)):
            return
        if not self._phrase_by_freq:
            return
        confirmed_f = self._streak.feed(float(freq))
        if confirmed_f is None:
            return
        card = self._phrase_by_freq.get(confirmed_f)
        if card is None:
            logger.debug(
                "Confirmed frequency %.1f Hz has no phrase card; skipping extra WS",
                confirmed_f,
            )
            return
        confirmed_msg = {
            "type": "confirmed",
            "phrase_id": card.id,
            "frequency_hz": confirmed_f,
            "utterance": card.utterance,
            "streak": self._streak.required,
        }
        await self._broadcaster.broadcast(confirmed_msg)

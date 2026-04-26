"""Tests for layer3_backend.layer2_bridge — Broadcaster + Layer2Bridge."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
import websockets
from websockets.server import serve

from layer3_backend.config import PhraseCard
from layer3_backend.layer2_bridge import Broadcaster, Layer2Bridge


_PAYLOAD = {
    "command": "SELECT",
    "frequency": 6.0,
    "snr_db": 5.0,
    "confidence": 0.6,
    "epoch_ms": 2000,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Broadcaster ──────────────────────────────────────────────────────


class TestBroadcaster:
    @pytest.mark.asyncio
    async def test_broadcast_updates_last_payload(self):
        b = Broadcaster()
        assert b.last_payload is None
        await b.broadcast(_PAYLOAD)
        assert b.last_payload == _PAYLOAD

    @pytest.mark.asyncio
    async def test_add_remove_client(self):
        b = Broadcaster()
        sentinel = object()
        b.add(sentinel)
        assert b.client_count == 1
        b.remove(sentinel)
        assert b.client_count == 0


class _CollectBroadcaster(Broadcaster):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[dict] = []

    async def broadcast(self, payload):  # type: ignore[override]
        self.sent.append(payload)
        await super().broadcast(payload)


# ── Layer2Bridge ─────────────────────────────────────────────────────


class TestLayer2Bridge:
    @pytest.mark.asyncio
    async def test_bridge_receives_and_forwards(self):
        """Spin up a fake Layer 2 WS server, connect the bridge, assert the
        broadcaster receives the payload."""
        port = _free_port()
        received: list[dict] = []

        async def fake_layer2(websocket):
            await websocket.send(json.dumps(_PAYLOAD))
            # Keep connection alive until the test finishes
            await asyncio.sleep(5)

        async with serve(fake_layer2, "127.0.0.1", port):
            broadcaster = Broadcaster()
            bridge = Layer2Bridge(
                upstream_url=f"ws://127.0.0.1:{port}",
                broadcaster=broadcaster,
            )
            await bridge.start()

            # Wait until the broadcaster has a payload (with timeout)
            for _ in range(50):
                if broadcaster.last_payload is not None:
                    break
                await asyncio.sleep(0.1)

            await bridge.stop()

        assert broadcaster.last_payload is not None
        assert broadcaster.last_payload["frequency"] == 6.0
        assert broadcaster.last_payload["command"] == "SELECT"

    @pytest.mark.asyncio
    async def test_five_identical_selects_emit_confirmed(self):
        port = _free_port()
        card = PhraseCard(
            id="water",
            label="Water",
            frequency_hz=6.0,
            color="#00e5ff",
            utterance="Water, please.",
        )
        phrase_map = {6.0: card}

        async def fake_layer2(websocket):
            for _ in range(5):
                await websocket.send(json.dumps(_PAYLOAD))
            await asyncio.sleep(3)

        async with serve(fake_layer2, "127.0.0.1", port):
            broadcaster = _CollectBroadcaster()
            bridge = Layer2Bridge(
                upstream_url=f"ws://127.0.0.1:{port}",
                broadcaster=broadcaster,
                phrase_by_norm_frequency=phrase_map,
                streak_required=5,
            )
            await bridge.start()
            for _ in range(80):
                if len(broadcaster.sent) >= 6:
                    break
                await asyncio.sleep(0.05)
            await bridge.stop()

        assert len(broadcaster.sent) >= 6
        assert broadcaster.sent[0]["command"] == "SELECT"
        last = broadcaster.sent[-1]
        assert last.get("type") == "confirmed"
        assert last.get("phrase_id") == "water"
        assert last.get("frequency_hz") == 6.0
        assert last.get("streak") == 5

    @pytest.mark.asyncio
    async def test_bridge_reconnects_on_disconnect(self):
        """If the upstream closes, the bridge should reconnect."""
        port = _free_port()
        connect_count = 0

        async def fake_layer2(websocket):
            nonlocal connect_count
            connect_count += 1
            await websocket.send(json.dumps(_PAYLOAD))
            if connect_count == 1:
                await websocket.close()
                return
            await asyncio.sleep(5)

        async with serve(fake_layer2, "127.0.0.1", port):
            broadcaster = Broadcaster()
            bridge = Layer2Bridge(
                upstream_url=f"ws://127.0.0.1:{port}",
                broadcaster=broadcaster,
            )
            await bridge.start()

            # Wait for at least 2 connections
            for _ in range(60):
                if connect_count >= 2:
                    break
                await asyncio.sleep(0.1)

            await bridge.stop()

        assert connect_count >= 2, f"Expected >=2 connections, got {connect_count}"

    @pytest.mark.asyncio
    async def test_bridge_stop_is_clean(self):
        """Stopping without a running upstream should not hang or raise."""
        broadcaster = Broadcaster()
        bridge = Layer2Bridge(
            upstream_url="ws://127.0.0.1:1",  # nothing there
            broadcaster=broadcaster,
        )
        await bridge.start()
        await asyncio.sleep(0.3)
        await bridge.stop()

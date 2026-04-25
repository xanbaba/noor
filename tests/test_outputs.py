"""Tests for layer2_processing.outputs — WebSocketEmitter & OscEmitter.

All tests are headless (no Cyton, no Quest).
"""

from __future__ import annotations

import asyncio
import json
import socket
import time

import pytest

from layer2_processing.outputs.osc_emitter import OscEmitter
from layer2_processing.outputs.websocket_emitter import WebSocketEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAYLOAD = {
    "command": "SELECT",
    "frequency": 12.0,
    "snr_db": 5.3,
    "confidence": 0.82,
    "epoch_ms": 2000,
}


def _free_port() -> int:
    """Bind on port 0 and return the assigned ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# WebSocketEmitter
# ---------------------------------------------------------------------------

class TestWebSocketEmitter:
    def test_start_stop(self):
        port = _free_port()
        emitter = WebSocketEmitter(port=port)
        emitter.start(timeout_s=5.0)
        assert emitter.client_count == 0
        emitter.stop()

    def test_client_receives_message(self):
        port = _free_port()
        emitter = WebSocketEmitter(port=port)
        emitter.start(timeout_s=5.0)
        received: list[str] = []

        async def _collect():
            import websockets
            uri = f"ws://localhost:{port}"
            async with websockets.connect(uri) as ws:
                # Give the server loop time to register this client
                await asyncio.sleep(0.1)
                emitter.emit(_PAYLOAD)
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                received.append(msg)

        try:
            asyncio.run(_collect())
        finally:
            emitter.stop()

        assert len(received) == 1
        parsed = json.loads(received[0])
        assert parsed["command"] == "SELECT"
        assert parsed["frequency"] == 12.0

    def test_multiple_clients_all_receive(self):
        port = _free_port()
        emitter = WebSocketEmitter(port=port)
        emitter.start(timeout_s=5.0)
        received: list[list[str]] = [[], []]

        async def _collect_two():
            import websockets
            uri = f"ws://localhost:{port}"
            async with (
                websockets.connect(uri) as ws1,
                websockets.connect(uri) as ws2,
            ):
                await asyncio.sleep(0.1)  # let server register both
                emitter.emit(_PAYLOAD)
                msg1 = await asyncio.wait_for(ws1.recv(), timeout=3.0)
                msg2 = await asyncio.wait_for(ws2.recv(), timeout=3.0)
                received[0].append(msg1)
                received[1].append(msg2)

        try:
            asyncio.run(_collect_two())
        finally:
            emitter.stop()

        assert len(received[0]) == 1 and len(received[1]) == 1
        assert json.loads(received[0][0]) == json.loads(received[1][0])

    def test_emit_before_start_does_not_crash(self):
        emitter = WebSocketEmitter(port=_free_port())
        emitter.emit(_PAYLOAD)  # should be a no-op, not raise


# ---------------------------------------------------------------------------
# OscEmitter
# ---------------------------------------------------------------------------

class TestOscEmitter:
    def test_emit_reaches_udp_listener(self):
        """A raw UDP socket captures the datagram sent by OscEmitter."""
        port = _free_udp_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        sock.settimeout(2.0)

        emitter = OscEmitter(host="127.0.0.1", port=port, address="/bci/command")
        emitter.emit(_PAYLOAD)

        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            pytest.fail("UDP datagram not received within timeout")
        finally:
            sock.close()

        assert len(data) > 0, "Expected non-empty UDP datagram"
        # OSC packets start with the address string
        assert b"/bci/command" in data

    def test_emit_payload_contains_json(self):
        """The JSON payload must be embedded inside the OSC message bytes."""
        port = _free_udp_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        sock.settimeout(2.0)

        emitter = OscEmitter(host="127.0.0.1", port=port, address="/bci/command")
        emitter.emit(_PAYLOAD)

        try:
            data, _ = sock.recvfrom(4096)
        finally:
            sock.close()

        # Find the JSON string inside the binary OSC packet
        start = data.find(b"{")
        end = data.rfind(b"}") + 1
        assert start >= 0 and end > start, "No JSON object found in OSC packet"
        parsed = json.loads(data[start:end])
        assert parsed["command"] == "SELECT"
        assert parsed["frequency"] == 12.0

    def test_address_property(self):
        emitter = OscEmitter(address="/test/addr")
        assert emitter.address == "/test/addr"

    def test_target_property(self):
        emitter = OscEmitter(host="192.168.1.1", port=7000)
        assert emitter.target == ("192.168.1.1", 7000)

    def test_context_manager(self):
        with OscEmitter(port=_free_udp_port()) as emitter:
            emitter.emit(_PAYLOAD)  # should not raise

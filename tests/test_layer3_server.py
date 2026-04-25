"""Tests for layer3_backend.server — FastAPI app endpoints."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
from httpx import ASGITransport, AsyncClient

from layer3_backend.config import BackendConfig
from layer3_backend.server import create_app
from layer3_backend.layer2_bridge import Broadcaster


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(**overrides) -> BackendConfig:
    defaults = dict(
        layer2_ws_url="ws://127.0.0.1:1",  # intentionally unreachable
        host="127.0.0.1",
        port=_free_port(),
        stimulus_frequencies_hz=[12.0, 15.0],
        static_dir="static",
    )
    defaults.update(overrides)
    return BackendConfig(**defaults)


# ── HTTP endpoints ────────────────────────────────────────────────────

class TestHTTPEndpoints:
    @pytest.mark.asyncio
    async def test_health(self):
        app = create_app(_cfg())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_config_endpoint(self):
        app = create_app(_cfg())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/config")
        assert resp.status_code == 200
        body = resp.json()
        assert body["stimulus_frequencies_hz"] == [12.0, 15.0]

    @pytest.mark.asyncio
    async def test_index_returns_html(self):
        app = create_app(_cfg())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "SSVEP" in resp.text

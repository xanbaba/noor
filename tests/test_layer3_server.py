"""Tests for layer3_backend.server — FastAPI app endpoints."""

from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from layer3_backend.config import BackendConfig, PhraseCard
from layer3_backend.server import create_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _phrases_two() -> list[PhraseCard]:
    return [
        PhraseCard(
            id="water",
            label="Water, please",
            frequency_hz=6.0,
            color="#00e5ff",
            utterance="Water.",
        ),
        PhraseCard(
            id="pain",
            label="Pain",
            frequency_hz=20.0,
            color="#ff3d71",
            utterance="I am in pain.",
        ),
    ]


def _cfg(**overrides) -> BackendConfig:
    defaults = dict(
        layer2_ws_url="ws://127.0.0.1:1",  # intentionally unreachable
        host="127.0.0.1",
        port=_free_port(),
        stimulus_frequencies_hz=[6.0, 20.0],
        static_dir="static",
        phrases=_phrases_two(),
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
        assert body["stimulus_frequencies_hz"] == [6.0, 20.0]
        assert len(body["phrases"]) == 2
        assert body["phrases"][0]["id"] == "water"

    @pytest.mark.asyncio
    async def test_index_returns_html(self):
        app = create_app(_cfg())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "SSVEP" in resp.text

    @pytest.mark.asyncio
    async def test_speak_unknown_phrase_returns_404(self):
        app = create_app(_cfg())
        transport = ASGITransport(app=app)
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "fake-key-for-route"}):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/api/speak",
                    json={"phrase_id": "nope"},
                )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_speak_without_api_key_returns_503(self, monkeypatch):
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        app = create_app(_cfg())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/speak",
                json={"phrase_id": "water"},
            )
        assert resp.status_code == 503

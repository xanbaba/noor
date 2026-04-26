"""ElevenLabs text-to-speech (server-side, API key from environment)."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


async def synthesize_speech_mpeg(
    text: str,
    *,
    voice_id: str,
    api_key: str,
    model_id: str = "eleven_multilingual_v2",
) -> bytes:
    """Return MP3 bytes for ``text``."""
    url = TTS_URL.format(voice_id=voice_id)
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            headers={
                "xi-api-key": api_key,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            json={"text": text, "model_id": model_id},
        )
    if resp.status_code >= 400:
        logger.warning(
            "ElevenLabs TTS failed | status=%s | body=%s",
            resp.status_code,
            resp.text[:200],
        )
    resp.raise_for_status()
    return resp.content

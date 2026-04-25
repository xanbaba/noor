"""OSC (UDP) emitter for SELECT commands.

Wraps :class:`pythonosc.udp_client.SimpleUDPClient` and serialises the payload
as a single JSON-encoded string argument under the configured address
(default ``/bci/command``).
"""

from __future__ import annotations

import json
from typing import Any

from pythonosc.udp_client import SimpleUDPClient

from layer2_processing.logging_config import get_logger

logger = get_logger(__name__)


class OscEmitter:
    """Fire-and-forget OSC publisher (UDP, no acknowledgements)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        address: str = "/bci/command",
    ) -> None:
        self._host = host
        self._port = port
        self._address = address
        self._client = SimpleUDPClient(host, port)
        logger.info(
            "OSC emitter ready | target=%s:%d | address=%s",
            host,
            port,
            address,
        )

    @property
    def address(self) -> str:
        return self._address

    @property
    def target(self) -> tuple[str, int]:
        return self._host, self._port

    def emit(self, payload: dict[str, Any]) -> None:
        msg = json.dumps(payload, separators=(",", ":"))
        try:
            self._client.send_message(self._address, msg)
        except OSError as exc:
            logger.warning("OSC send failed: %s", exc)

    def close(self) -> None:  # parity with WebSocketEmitter
        return None

    def __enter__(self) -> "OscEmitter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

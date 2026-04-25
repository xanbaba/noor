"""LSL inlet for the Layer 2 pipeline.

Resolves a named EEG outlet (default ``BCI_RawEEG``) and exposes a
``pull_chunk`` wrapper that returns a ``(channels, n_samples)`` ``float32``
array plus the corresponding LSL timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from layer2_processing.logging_config import get_logger

logger = get_logger(__name__)


class StreamNotFoundError(RuntimeError):
    """Raised when the desired LSL stream cannot be resolved within the timeout."""


@dataclass
class InletInfo:
    name: str
    type: str
    channel_count: int
    nominal_srate: float
    source_id: str


class RawEegInlet:
    """Thin wrapper around ``pylsl.StreamInlet`` for the Layer 2 pipeline."""

    def __init__(
        self,
        stream_name: str = "BCI_RawEEG",
        resolve_timeout_s: float = 5.0,
        max_buflen_s: int = 30,
    ) -> None:
        self._stream_name = stream_name
        self._resolve_timeout_s = resolve_timeout_s
        self._max_buflen_s = max_buflen_s
        self._inlet: StreamInlet | None = None
        self._info: InletInfo | None = None

    @property
    def info(self) -> InletInfo:
        if self._info is None:
            raise RuntimeError("Inlet not opened — call open() first.")
        return self._info

    @property
    def channel_count(self) -> int:
        return self.info.channel_count

    @property
    def sample_rate_hz(self) -> float:
        return self.info.nominal_srate

    def open(self) -> InletInfo:
        """Resolve the LSL stream by name and open an inlet."""
        logger.info(
            "Resolving LSL stream '%s' (timeout %.1fs)…",
            self._stream_name,
            self._resolve_timeout_s,
        )
        streams = resolve_byprop(
            "name", self._stream_name, timeout=self._resolve_timeout_s
        )
        if not streams:
            raise StreamNotFoundError(
                f"No LSL stream named '{self._stream_name}' found within "
                f"{self._resolve_timeout_s:.1f}s."
            )

        info = streams[0]
        self._inlet = StreamInlet(info, max_buflen=self._max_buflen_s)
        self._info = InletInfo(
            name=info.name(),
            type=info.type(),
            channel_count=info.channel_count(),
            nominal_srate=info.nominal_srate(),
            source_id=info.source_id(),
        )
        logger.info(
            "Inlet open | name=%s | channels=%d | rate=%.1f Hz | source=%s",
            self._info.name,
            self._info.channel_count,
            self._info.nominal_srate,
            self._info.source_id,
        )
        return self._info

    def pull_chunk(
        self, timeout_s: float = 0.05, max_samples: int = 1024
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pull a chunk from the inlet.

        Returns:
            data: ``(channels, n_samples)`` float32 array (may be empty).
            timestamps: ``(n_samples,)`` float64 LSL clock timestamps.
        """
        if self._inlet is None:
            raise RuntimeError("Inlet not opened — call open() first.")

        chunk, timestamps = self._inlet.pull_chunk(
            timeout=timeout_s, max_samples=max_samples
        )
        if not chunk:
            return (
                np.empty((self.channel_count, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float64),
            )

        # pylsl returns a list[list[float]] in (n_samples, channels) order
        data = np.asarray(chunk, dtype=np.float32).T  # (channels, n)
        ts = np.asarray(timestamps, dtype=np.float64)
        return data, ts

    def close(self) -> None:
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:  # noqa: BLE001
                pass
            self._inlet = None
            logger.info("Inlet closed.")

    def __enter__(self) -> "RawEegInlet":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

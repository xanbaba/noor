"""LSL outlet wrapper for the BCI_RawEEG stream.

Creates and owns the pylsl StreamOutlet that constitutes the sole contractual
output of Layer 1.  All downstream layers (Layer 2, verify scripts, etc.)
discover the stream by name ``BCI_RawEEG`` — never by source_id or type alone.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from pylsl import StreamInfo, StreamOutlet, cf_float32, local_clock

from .config import AcquisitionConfig

logger = logging.getLogger(__name__)

# Channel metadata units for the XML description injected into the StreamInfo.
_UNIT = "microvolts"
_CHANNEL_TYPE = "EEG"


class RawEegOutlet:
    """Owns a single pylsl StreamOutlet conforming to the Layer 1 contract.

    Stream contract (frozen — changing any field breaks Layer 2):
      name         = cfg.lsl_stream_name   (default: "BCI_RawEEG")
      type         = cfg.lsl_stream_type   (default: "EEG")
      channel_count= cfg.channel_count     (8)
      nominal_srate= cfg.sample_rate_hz    (500)
      channel_format = float32
      source_id    = "layer1_<board>_<uuid4>"
    """

    def __init__(self, cfg: AcquisitionConfig, board_name: str) -> None:
        self._cfg = cfg
        source_id = f"layer1_{board_name}_{uuid.uuid4().hex[:8]}"

        info = StreamInfo(
            name=cfg.lsl_stream_name,
            type=cfg.lsl_stream_type,
            channel_count=cfg.channel_count,
            nominal_srate=float(cfg.sample_rate_hz),
            channel_format=cf_float32,
            source_id=source_id,
        )

        self._annotate_channels(info, cfg.channel_labels)

        # max_buffered: buffer 360 seconds of data in the LSL kernel — ample
        # headroom if a consumer lags or reconnects.
        self._outlet = StreamOutlet(info, max_buffered=360)

        logger.info(
            "LSL outlet created: name='%s', type='%s', channels=%d, "
            "rate=%d Hz, source_id='%s'.",
            cfg.lsl_stream_name,
            cfg.lsl_stream_type,
            cfg.channel_count,
            cfg.sample_rate_hz,
            source_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_chunk(self, samples: np.ndarray) -> None:
        """Push a chunk of EEG samples to the LSL outlet.

        Args:
            samples: Array of shape (channel_count, n_samples), dtype float32.
                     Empty chunks (n_samples == 0) are silently ignored.
        """
        if samples.shape[1] == 0:
            return

        # pylsl push_chunk expects (n_samples, channel_count) — transpose.
        chunk_t = samples.T.tolist()
        timestamp = local_clock()
        self._outlet.push_chunk(chunk_t, timestamp)

    def push_sample(self, sample: np.ndarray) -> None:
        """Push a single EEG sample (shape: (channel_count,)) to the outlet."""
        self._outlet.push_sample(sample.tolist(), local_clock())

    @property
    def info(self) -> StreamInfo:
        return self._outlet.get_info()

    @property
    def stream_name(self) -> str:
        return self._outlet.get_info().name()

    @property
    def stream_type(self) -> str:
        return self._outlet.get_info().type()

    @property
    def stream_channel_count(self) -> int:
        return self._outlet.get_info().channel_count()

    @property
    def stream_nominal_srate(self) -> float:
        return self._outlet.get_info().nominal_srate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_channels(info: StreamInfo, labels: list[str]) -> None:
        """Inject per-channel metadata (label, unit, type) into the StreamInfo XML."""
        channels_node = info.desc().append_child("channels")
        for label in labels:
            ch = channels_node.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", _UNIT)
            ch.append_child_value("type", _CHANNEL_TYPE)

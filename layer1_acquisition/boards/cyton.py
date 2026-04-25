"""OpenBCI Cyton board driver (BrainFlow BoardIds.CYTON_BOARD = 0)."""

from __future__ import annotations

import time
import logging

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.exit_codes import BrainFlowError

from .base import AbstractBoard
from ..config import AcquisitionConfig

logger = logging.getLogger(__name__)


class CytonBoard(AbstractBoard):
    """Wraps BrainFlow's Cyton driver with the AbstractBoard contract."""

    sample_rate_hz: int = 500
    channel_count: int = 8

    def __init__(self, cfg: AcquisitionConfig) -> None:
        self._cfg = cfg
        self.channel_labels = cfg.channel_labels
        self.sample_rate_hz = cfg.sample_rate_hz
        self.channel_count = cfg.channel_count

        params = BrainFlowInputParams()
        if cfg.serial_port != "auto":
            params.serial_port = cfg.serial_port

        BoardShim.enable_dev_board_logger()
        BoardShim.set_log_level(3)  # WARN — suppress debug spam

        self._shim = BoardShim(BoardIds.CYTON_BOARD, params)
        self._eeg_channels: list[int] = BoardShim.get_eeg_channels(
            BoardIds.CYTON_BOARD
        )

    # ------------------------------------------------------------------
    # AbstractBoard implementation
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        logger.info("Preparing Cyton session…")
        self._shim.prepare_session()
        logger.info("Cyton session ready.")

    def start_stream(self) -> None:
        # Ring buffer: 45 seconds of samples at 500 Hz = 22 500 samples.
        self._shim.start_stream(45 * self._cfg.sample_rate_hz)
        logger.info(
            "Cyton stream started at %d Hz on %d channels.",
            self._cfg.sample_rate_hz,
            self._cfg.channel_count,
        )

    def get_chunk(self) -> np.ndarray:
        """Pull all available samples from BrainFlow's ring buffer.

        Returns shape (channel_count, n_samples), dtype float32.
        """
        try:
            n = self._shim.get_board_data_count()
        except BrainFlowError:
            return np.empty((self._cfg.channel_count, 0), dtype=np.float32)
        if n == 0:
            return np.empty((self._cfg.channel_count, 0), dtype=np.float32)

        data = self._shim.get_board_data(n)
        # BrainFlow returns (total_channels, n); slice to the 8 EEG channels.
        eeg = data[self._eeg_channels, :].astype(np.float32)
        return eeg

    def stop(self) -> None:
        try:
            self._shim.stop_stream()
        except Exception:
            pass
        try:
            self._shim.release_session()
        except Exception:
            pass
        logger.info("Cyton session released.")

    def _safe_config_board(self, cmd: str) -> None:
        """Send a config command to the Cyton, tolerating non-UTF-8 response bytes.

        BrainFlow's config_board writes the command to the serial port, then
        reads the board's response and decodes it as UTF-8.  After a stream
        session, residual binary EEG packets (start byte 0xA0) remain in the
        serial RX buffer and cause UnicodeDecodeError on the response decode.
        The command itself was already sent successfully by this point.
        """
        try:
            self._shim.config_board(cmd)
        except UnicodeDecodeError:
            logger.debug(
                "config_board('%s') response contained non-UTF-8 bytes "
                "(residual EEG packets); command was still sent.", cmd
            )

    def impedance_kohm(self) -> dict[str, float]:
        """Measure per-channel impedance using the OpenBCI Cyton lead-off injection method.

        For each active channel:
          1. Enable 6 nA AC lead-off drive: ``z<ch>01Z``
          2. Start stream, collect ~1 s of 60 Hz-injected EEG data, stop stream
          3. Disable lead-off drive: ``z<ch>00Z``
          4. Compute RMS of the EEG channel amplitude and convert to kΩ:
             ``impedance_kohm = (sqrt(2) * rms_µV * 1e-6) / 6e-9 / 1000``

        Reference: OpenBCI Cyton SDK lead-off commands + OpenBCI Forum impedance formula.
        BrainFlow 5.x does not expose a native impedance API for CYTON_BOARD
        (get_resistance_channels raises UNSUPPORTED_BOARD_ERROR), so this
        signal-based approach is the correct method.
        """
        import math as _math

        results: dict[str, float] = {}
        active_indices = self._cfg.active_channel_indices
        eeg_rows = self._eeg_channels  # e.g. [1, 2, 3, 4, 5, 6, 7, 8]

        n_collect = int(self._cfg.sample_rate_hz * 1.2)

        for ch_idx in active_indices:
            label = self._cfg.channel_labels[ch_idx]
            ch_num = ch_idx + 1  # Cyton protocol is 1-based

            cmd_on = f"z{ch_num}01Z"
            cmd_off = f"z{ch_num}00Z"

            try:
                self._safe_config_board(cmd_on)
                time.sleep(0.1)

                self._shim.start_stream(n_collect * 4)
                time.sleep(1.2)
                raw = self._shim.get_board_data()
                self._shim.stop_stream()
                time.sleep(0.5)

                self._safe_config_board(cmd_off)
                time.sleep(0.1)

                eeg_row = eeg_rows[ch_idx]
                channel_data = raw[eeg_row, :]

                if channel_data.size == 0:
                    logger.warning("No data collected for %s; assuming high impedance.", label)
                    results[label] = 50.0
                    continue

                rms_uv = float(np.sqrt(np.mean(channel_data ** 2)))
                impedance_ohm = (_math.sqrt(2) * rms_uv * 1e-6) / 6e-9
                impedance_k = impedance_ohm / 1000.0

                logger.debug("  %s  rms=%.1f µV  → %.1f kΩ", label, rms_uv, impedance_k)
                results[label] = impedance_k

            except Exception as exc:
                logger.error("Impedance check failed for ch %d (%s): %s", ch_num, label, exc)
                try:
                    self._shim.stop_stream()
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    self._safe_config_board(cmd_off)
                except Exception:
                    pass
                results[label] = 50.0

        return results

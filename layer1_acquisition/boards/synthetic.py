"""Synthetic board driver using BrainFlow's built-in synthetic board.

Used for CI pipelines and bench development when no Cyton dongle is attached.
The synthetic board generates plausible EEG-like data at the configured rate,
allowing all downstream code paths (LSL outlet, acquisition loop, tests) to run
without any hardware.
"""

from __future__ import annotations

import logging

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.exit_codes import BrainFlowError

from .base import AbstractBoard
from ..config import AcquisitionConfig

logger = logging.getLogger(__name__)


class SyntheticBoard(AbstractBoard):
    """Wraps BrainFlow's SYNTHETIC_BOARD with the AbstractBoard contract."""

    def __init__(self, cfg: AcquisitionConfig) -> None:
        self._cfg = cfg
        self.sample_rate_hz = cfg.sample_rate_hz
        self.channel_count = cfg.channel_count
        self.channel_labels = cfg.channel_labels

        BoardShim.set_log_level(3)
        params = BrainFlowInputParams()
        self._shim = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
        self._eeg_channels: list[int] = BoardShim.get_eeg_channels(
            BoardIds.SYNTHETIC_BOARD
        )

    def prepare(self) -> None:
        logger.info("Preparing synthetic board session…")
        self._shim.prepare_session()
        logger.info("Synthetic board ready.")

    def start_stream(self) -> None:
        self._shim.start_stream(45 * self._cfg.sample_rate_hz)
        logger.info(
            "Synthetic stream started at %d Hz.", self._cfg.sample_rate_hz
        )

    def get_chunk(self) -> np.ndarray:
        try:
            n = self._shim.get_board_data_count()
        except BrainFlowError:
            return np.empty((self._cfg.channel_count, 0), dtype=np.float32)
        if n == 0:
            return np.empty((self._cfg.channel_count, 0), dtype=np.float32)

        data = self._shim.get_board_data(n)

        # Synthetic board may expose more or fewer EEG channels than Cyton.
        # Clip or pad to match the configured channel_count so the LSL contract
        # (fixed 8-channel shape) is always honoured.
        eeg_raw = data[self._eeg_channels, :].astype(np.float32)
        n_avail = eeg_raw.shape[0]
        n_need = self._cfg.channel_count

        if n_avail >= n_need:
            return eeg_raw[:n_need, :]
        else:
            pad = np.zeros((n_need - n_avail, eeg_raw.shape[1]), dtype=np.float32)
            return np.vstack([eeg_raw, pad])

    def stop(self) -> None:
        try:
            self._shim.stop_stream()
        except Exception:
            pass
        try:
            self._shim.release_session()
        except Exception:
            pass
        logger.info("Synthetic board session released.")

    def impedance_kohm(self) -> dict[str, float]:
        """Synthetic board has no real impedance measurement.

        Returns a constant 5.0 kΩ (below the 10 kΩ gate) for all active channels
        so that the impedance gate always passes in test/CI environments.
        """
        return {
            self._cfg.channel_labels[i]: 5.0
            for i in self._cfg.active_channel_indices
        }

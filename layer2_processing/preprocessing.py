"""Real-time preprocessing for the Layer 2 pipeline.

Two collaborators:

- :class:`Preprocessor` — notch (60 Hz) → bandpass (5–45 Hz) → CAR → artefact
  gate.  All filters use zero-phase :func:`scipy.signal.sosfiltfilt`.
- :class:`EpochBuffer` — fixed-length sliding window over the incoming chunk
  stream.  Pre-allocated buffer, emits ready epochs as numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy import signal

from layer2_processing.config import ProcessingConfig


@dataclass
class EpochResult:
    """Output of :meth:`Preprocessor.process`."""

    data: np.ndarray  # (channels, n_samples) float32, after notch+bandpass+CAR
    artefactual: bool
    peak_to_peak_uv: np.ndarray  # (channels,) float32


class Preprocessor:
    """Stateless per-epoch preprocessor: notch → bandpass → CAR → artefact.

    Filters are pre-computed in second-order-section form so each call is a
    pair of cheap :func:`sosfiltfilt` invocations.
    """

    def __init__(self, cfg: ProcessingConfig) -> None:
        self._fs = float(cfg.sample_rate_hz)

        # IIR notch (returns b, a) → convert to SOS for sosfiltfilt
        b_notch, a_notch = signal.iirnotch(
            w0=cfg.notch_freq_hz, Q=cfg.notch_q, fs=self._fs
        )
        self._sos_notch = signal.tf2sos(b_notch, a_notch)

        self._sos_band = signal.butter(
            N=cfg.bandpass_order,
            Wn=[cfg.bandpass_low_hz, cfg.bandpass_high_hz],
            btype="band",
            fs=self._fs,
            output="sos",
        )

        self._artefact_uv = float(cfg.artefact_threshold_uv)

    def filter(self, epoch: np.ndarray) -> np.ndarray:
        """Notch → bandpass → common-average reference. Returns float32 copy."""
        x = np.asarray(epoch, dtype=np.float64)
        x = signal.sosfiltfilt(self._sos_notch, x, axis=1)
        x = signal.sosfiltfilt(self._sos_band, x, axis=1)
        # Common Average Reference: subtract the per-sample mean across channels
        x = x - x.mean(axis=0, keepdims=True)
        return x.astype(np.float32, copy=False)

    def peak_to_peak(self, epoch: np.ndarray) -> np.ndarray:
        return (epoch.max(axis=1) - epoch.min(axis=1)).astype(np.float32, copy=False)

    def is_artefactual(self, epoch: np.ndarray) -> bool:
        ptp = self.peak_to_peak(epoch)
        return bool((ptp > self._artefact_uv).any())

    def process(self, epoch: np.ndarray) -> EpochResult:
        clean = self.filter(epoch)
        ptp = self.peak_to_peak(clean)
        return EpochResult(
            data=clean,
            artefactual=bool((ptp > self._artefact_uv).any()),
            peak_to_peak_uv=ptp,
        )


class EpochBuffer:
    """Sliding-window buffer that emits fixed-length epochs at a fixed step.

    The buffer is pre-allocated and only stores the most-recent
    ``epoch_samples`` samples.  After every ``step_samples`` of new data we
    emit a fresh ``(channels, epoch_samples)`` snapshot.

    Designed to be called once per chunk pulled from LSL: ``append(chunk)``
    then iterate ``epochs()`` until exhausted.
    """

    def __init__(
        self,
        channels: int,
        epoch_samples: int,
        step_samples: int,
    ) -> None:
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if epoch_samples <= 0 or step_samples <= 0:
            raise ValueError("epoch_samples and step_samples must be > 0")
        if step_samples > epoch_samples:
            raise ValueError("step_samples must be <= epoch_samples")

        self._channels = channels
        self._epoch_samples = int(epoch_samples)
        self._step_samples = int(step_samples)
        self._buf = np.zeros((channels, self._epoch_samples), dtype=np.float32)
        self._n_written = 0
        self._n_at_last_emit: int | None = None  # None until first epoch fires

    @property
    def n_written(self) -> int:
        return self._n_written

    @property
    def is_full(self) -> bool:
        return self._n_written >= self._epoch_samples

    def append(self, chunk: np.ndarray) -> None:
        """Append a (channels, n) chunk to the buffer (no-op if empty)."""
        if chunk.size == 0:
            return
        if chunk.ndim != 2 or chunk.shape[0] != self._channels:
            raise ValueError(
                f"Expected chunk of shape ({self._channels}, n); "
                f"got {chunk.shape}"
            )

        n = chunk.shape[1]
        if n >= self._epoch_samples:
            # Chunk is at least as long as the window — keep only the tail.
            self._buf[:] = chunk[:, -self._epoch_samples :].astype(
                np.float32, copy=False
            )
        else:
            # Shift left by n then write the new chunk at the tail.
            # np.roll allocates a new array so this is overlap-safe.
            self._buf = np.roll(self._buf, -n, axis=1)
            self._buf[:, -n:] = chunk.astype(np.float32, copy=False)
        self._n_written += n

    def epochs(self) -> Iterator[np.ndarray]:
        """Yield ready epochs (most-recent ``epoch_samples`` snapshot).

        Fires once when the buffer first fills, then once per
        ``step_samples`` of additional data.
        """
        if self._n_written < self._epoch_samples:
            return

        if self._n_at_last_emit is None:
            yield self._buf.copy()
            self._n_at_last_emit = self._n_written
            return

        while (self._n_written - self._n_at_last_emit) >= self._step_samples:
            yield self._buf.copy()
            self._n_at_last_emit += self._step_samples

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._n_written = 0
        self._n_at_last_emit = None

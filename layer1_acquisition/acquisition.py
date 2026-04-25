"""Main acquisition loop — pulls EEG from the board and pushes to the LSL outlet."""

from __future__ import annotations

import logging
import signal
import threading
import time

import numpy as np

from .boards.base import AbstractBoard
from .config import AcquisitionConfig
from .lsl_outlet import RawEegOutlet

logger = logging.getLogger(__name__)


class AcquisitionLoop:
    """Real-time acquisition loop with pre-allocated buffers and health logging.

    Design notes:
    - Buffers are pre-allocated at __init__ time to avoid GC pauses in the
      hot path (per architecture §2.4 guidance, applied here too).
    - Pull cadence is cfg.pull_interval_ms (default 10 ms = ~5 samples at
      500 Hz), keeping ring-buffer latency low.
    - Dropped-sample detection is based on a monotonic expected-sample counter
      compared against BrainFlow's returned chunk sizes.
    - A health summary is logged every cfg.log_interval_s seconds.
    - SIGINT / SIGTERM trigger a clean shutdown via _running flag.
    """

    def __init__(
        self,
        board: AbstractBoard,
        outlet: RawEegOutlet,
        cfg: AcquisitionConfig,
    ) -> None:
        self._board = board
        self._outlet = outlet
        self._cfg = cfg
        self._running = False

        # Pre-allocate a working buffer large enough for 1 second of data
        # to avoid per-chunk allocation in the hot path.
        self._buf: np.ndarray = np.empty(
            (cfg.channel_count, cfg.sample_rate_hz), dtype=np.float32
        )

        self._total_samples = 0
        self._dropped_samples = 0
        self._last_log_time = 0.0
        self._loop_start_time = 0.0
        self._first_chunk_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block until stopped (SIGINT/SIGTERM) or stop() is called."""
        self._install_signal_handlers()
        self._running = True
        self._loop_start_time = time.monotonic()
        self._last_log_time = self._loop_start_time

        pull_interval_s = self._cfg.pull_interval_ms / 1000.0
        logger.info(
            "Acquisition loop started (pull every %d ms, log every %d s).",
            self._cfg.pull_interval_ms,
            self._cfg.log_interval_s,
        )

        try:
            while self._running:
                t0 = time.monotonic()

                chunk = self._board.get_chunk()
                n_new = chunk.shape[1]

                if n_new > 0:
                    if self._first_chunk_time is None:
                        self._first_chunk_time = time.monotonic()
                    self._outlet.push_chunk(chunk)
                    self._total_samples += n_new
                    self._detect_drops()

                self._maybe_log_health()

                # Sleep for the remainder of the pull interval.
                elapsed = time.monotonic() - t0
                sleep_s = pull_interval_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Request a graceful shutdown from another thread."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_drops(self) -> None:
        """Estimate dropped samples by comparing received vs. expected count.

        Counting begins from the first chunk arrival to exclude board warm-up
        latency (BrainFlow's SYNTHETIC_BOARD takes ~0.4 s to start delivering
        samples after start_stream is called).
        """
        if self._first_chunk_time is None:
            return
        elapsed_since_first = time.monotonic() - self._first_chunk_time
        expected = int(elapsed_since_first * self._cfg.sample_rate_hz)
        gap = expected - self._total_samples
        if gap > self._cfg.sample_rate_hz * 0.30:  # >30% of a second to avoid noise
            self._dropped_samples += gap
            logger.warning(
                "Possible sample drop detected: expected ~%d total, received %d "
                "(gap=%d samples).",
                expected,
                self._total_samples,
                gap,
            )

    def _maybe_log_health(self) -> None:
        now = time.monotonic()
        if now - self._last_log_time >= self._cfg.log_interval_s:
            ref = self._first_chunk_time or self._loop_start_time
            elapsed = now - ref
            rate = self._total_samples / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Health | elapsed=%.1fs | samples=%d | rate=%.1f Hz | "
                "estimated_drops=%d",
                elapsed,
                self._total_samples,
                rate,
                self._dropped_samples,
            )
            self._last_log_time = now

    def _shutdown(self) -> None:
        logger.info(
            "Acquisition loop stopping. Total samples pushed: %d, "
            "estimated drops: %d.",
            self._total_samples,
            self._dropped_samples,
        )
        self._board.stop()

    def _install_signal_handlers(self) -> None:
        # Signal handlers can only be registered from the main thread.
        # When run() is called from a worker thread (e.g. in tests), skip
        # signal registration silently — callers must use stop() instead.
        if threading.current_thread() is not threading.main_thread():
            return

        def _handler(signum, frame):  # noqa: ANN001
            logger.info("Signal %d received — initiating graceful shutdown.", signum)
            self._running = False

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

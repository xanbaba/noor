"""Layer 2 main pipeline.

Wires together: LSL inlet → epoch buffer → preprocessor → classifier → SNR
gate → WebSocket / OSC emitters.

Output payload (frozen, per ARCHITECTURE §2.4)::

    {"command":"SELECT","frequency":12.0,"snr_db":4.1,"confidence":0.87,"epoch_ms":2000}
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from layer2_processing.classifiers.base import AbstractClassifier
from layer2_processing.classifiers.factory import create_classifier
from layer2_processing.config import ProcessingConfig
from layer2_processing.logging_config import get_logger
from layer2_processing.lsl_inlet import RawEegInlet
from layer2_processing.outputs import OscEmitter, WebSocketEmitter
from layer2_processing.preprocessing import EpochBuffer, Preprocessor
from layer2_processing.snr import compute_ssvep_snr_db

logger = get_logger(__name__)


@dataclass
class PipelineStats:
    epochs_seen: int = 0
    epochs_artefactual: int = 0
    epochs_below_snr: int = 0
    commands_emitted: int = 0
    # Peak-to-peak (µV, max over channels) after filters on the last processed epoch
    last_max_ptp_uv: float = 0.0


def build_payload(
    frequency_hz: float,
    snr_db: float,
    confidence: float,
    epoch_ms: int,
    command: str = "SELECT",
) -> dict[str, Any]:
    """Construct the SELECT payload (frozen contract)."""
    return {
        "command": command,
        "frequency": round(float(frequency_hz), 4),
        "snr_db": round(float(snr_db), 4),
        "confidence": round(float(confidence), 4),
        "epoch_ms": int(epoch_ms),
    }


class Pipeline:
    """Synchronous pipeline driven by the LSL inlet poll loop."""

    def __init__(
        self,
        cfg: ProcessingConfig,
        inlet: RawEegInlet,
        classifier: AbstractClassifier,
        ws_emitter: WebSocketEmitter | None,
        osc_emitter: OscEmitter | None,
        on_emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._inlet = inlet
        self._classifier = classifier
        self._ws = ws_emitter
        self._osc = osc_emitter
        self._on_emit = on_emit

        self._preproc = Preprocessor(cfg)
        self._buffer = EpochBuffer(
            channels=inlet.channel_count,
            epoch_samples=cfg.epoch_length_samples,
            step_samples=cfg.epoch_step_samples,
        )
        self._epoch_ms = int(round(cfg.epoch_length_s * 1000.0))

        self.stats = PipelineStats()
        self._stop = threading.Event()
        self._last_log = time.monotonic()
        self._last_ptp_eval_max: float = 0.0

    def stop(self) -> None:
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def _handler(signum, _frame):
            logger.info("Received signal %d; shutting down…", signum)
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    def run(self) -> None:
        """Block until :meth:`stop` is called or the inlet dies."""
        self._install_signal_handlers()
        logger.info(
            "Pipeline running | epoch=%.2fs/%.2fs | freqs=%s | SNR≥%.1f dB",
            self._cfg.epoch_length_s,
            self._cfg.epoch_step_s,
            self._cfg.stimulus_frequencies_hz,
            self._cfg.snr_min_db,
        )

        try:
            while not self._stop.is_set():
                chunk, _ts = self._inlet.pull_chunk(
                    timeout_s=self._cfg.inlet_pull_timeout_s
                )
                if chunk.size:
                    self._buffer.append(chunk)
                    for epoch in self._buffer.epochs():
                        self._handle_epoch(epoch)

                self._maybe_log_health()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info(
                "Pipeline exit | epochs=%d | artefactual=%d | snr_rejected=%d | emitted=%d",
                self.stats.epochs_seen,
                self.stats.epochs_artefactual,
                self.stats.epochs_below_snr,
                self.stats.commands_emitted,
            )

    def _handle_epoch(self, epoch) -> None:
        self.stats.epochs_seen += 1

        result = self._preproc.process(epoch)
        ptp = result.peak_to_peak_uv
        idx = self._cfg.artefact_channel_indices
        if idx is None:
            self._last_ptp_eval_max = float(ptp.max()) if ptp.size else 0.0
        else:
            valid = [i for i in idx if 0 <= i < int(ptp.shape[0])]
            self._last_ptp_eval_max = float(ptp[valid].max()) if valid else float(ptp.max())
        self.stats.last_max_ptp_uv = float(ptp.max()) if ptp.size else 0.0

        if result.artefactual:
            self.stats.epochs_artefactual += 1
            logger.debug(
                "Epoch dropped (artefact) | max_ptp_eval=%.1f µV (threshold=%.1f on %s)",
                self._last_ptp_eval_max,
                self._cfg.artefact_threshold_uv,
                "all ch" if idx is None else f"indices {list(idx)}",
            )
            return

        prediction = self._classifier.predict(result.data)
        snr_db = compute_ssvep_snr_db(
            result.data,
            target_freq_hz=prediction.frequency_hz,
            fs=self._cfg.sample_rate_hz,
            channel_idx=self._cfg.snr_channel_index,
            n_harmonics=self._cfg.n_harmonics,
            noise_band_hz=self._cfg.snr_noise_band_hz,
        )

        if snr_db < self._cfg.snr_min_db:
            self.stats.epochs_below_snr += 1
            logger.debug(
                "Epoch dropped (SNR %.2f dB < %.2f dB) | freq=%.1f Hz",
                snr_db,
                self._cfg.snr_min_db,
                prediction.frequency_hz,
            )
            return

        payload = build_payload(
            frequency_hz=prediction.frequency_hz,
            snr_db=snr_db,
            confidence=prediction.confidence,
            epoch_ms=self._epoch_ms,
        )
        self._emit(payload)
        self.stats.commands_emitted += 1
        logger.info(
            "SELECT | freq=%.2f Hz | SNR=%.2f dB | conf=%.2f",
            prediction.frequency_hz,
            snr_db,
            prediction.confidence,
        )

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._ws is not None:
            self._ws.emit(payload)
        if self._osc is not None:
            self._osc.emit(payload)
        if self._on_emit is not None:
            try:
                self._on_emit(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_emit hook raised: %s", exc)

    def _maybe_log_health(self) -> None:
        now = time.monotonic()
        if now - self._last_log < self._cfg.log_interval_s:
            return
        self._last_log = now
        logger.info(
            "Health | epochs=%d | artefactual=%d | snr_rejected=%d | emitted=%d | "
            "last_max_ptp=%.1f µV (eval_max=%.1f)",
            self.stats.epochs_seen,
            self.stats.epochs_artefactual,
            self.stats.epochs_below_snr,
            self.stats.commands_emitted,
            self.stats.last_max_ptp_uv,
            self._last_ptp_eval_max,
        )


def build_pipeline(
    cfg: ProcessingConfig,
    classifier_name: str | None = None,
    start_outputs: bool = True,
) -> tuple[Pipeline, RawEegInlet, WebSocketEmitter, OscEmitter]:
    """Construct a full pipeline + inlet + emitters from a validated config."""
    inlet = RawEegInlet(
        stream_name=cfg.lsl_stream_name,
        resolve_timeout_s=cfg.inlet_resolve_timeout_s,
    )
    inlet.open()

    classifier = create_classifier(cfg, classifier_name)

    ws = WebSocketEmitter(host=cfg.websocket_host, port=cfg.websocket_port)
    osc = OscEmitter(host=cfg.osc_host, port=cfg.osc_port, address=cfg.osc_address)
    if start_outputs:
        ws.start()

    pipeline = Pipeline(
        cfg=cfg,
        inlet=inlet,
        classifier=classifier,
        ws_emitter=ws,
        osc_emitter=osc,
    )
    return pipeline, inlet, ws, osc

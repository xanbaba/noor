"""Layer 2 main pipeline.

Wires together: LSL inlet → epoch buffer → preprocessor → classifier → SNR
gate → WebSocket / OSC emitters.

Output payload (frozen, per ARCHITECTURE §2.4)::

    {"command":"SELECT","frequency":6.0,"snr_db":4.1,"confidence":0.87,"epoch_ms":2000}
"""

from __future__ import annotations

import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from layer2_processing.classifiers.base import AbstractClassifier, ClassifierResult
from layer2_processing.classifiers.factory import create_classifier
from layer2_processing.config import ProcessingConfig
from layer2_processing.logging_config import get_logger
from layer2_processing.lsl_inlet import RawEegInlet
from layer2_processing.outputs import OscEmitter, WebSocketEmitter
from layer2_processing.preprocessing import EpochBuffer, Preprocessor
from layer2_processing.snr import compute_ssvep_snr_db, compute_ssvep_snr_db_aggregate
from layer2_processing.smoothing import smoothed_frequency_hz

logger = get_logger(__name__)

_SNRELAX_STEP_DB = 0.25
_SNRELAX_CAP = 8


@dataclass
class PipelineStats:
    epochs_seen: int = 0
    epochs_artefactual: int = 0
    epochs_below_snr: int = 0
    commands_emitted: int = 0
    no_decisions_emitted: int = 0
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
        self._consecutive_snr_rejects = 0
        sw = int(cfg.prediction_smoothing_window)
        self._smooth_maxlen = max(1, sw) if sw > 1 else 0
        self._freq_window: deque[float] = deque(maxlen=self._smooth_maxlen or 1)
        self._freqs = [float(f) for f in cfg.stimulus_frequencies_hz]

        # Adaptive decision extension state (confidence gating across epochs).
        self._decision_log_scores: np.ndarray | None = None
        self._decision_epochs = 0

        # Log one-time warnings for channel-weight vector shape mismatch.
        self._warned_w6_shape = False
        self._warned_w15_shape = False

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
        snr_line = (
            f"SNR≥{self._cfg.snr_min_db:.1f} dB (adaptive to "
            f"{self._cfg.snr_adaptive_floor_db:.1f} dB floor)"
            if self._cfg.snr_adaptive_relax
            else f"SNR≥{self._cfg.snr_min_db:.1f} dB"
        )
        logger.info(
            "Pipeline running | epoch=%.2fs/%.2fs | freqs=%s | %s | artefact=%s | CAR=%s | conf_min=%.2f | max_extra=%d | w6_15=%s",
            self._cfg.epoch_length_s,
            self._cfg.epoch_step_s,
            self._cfg.stimulus_frequencies_hz,
            snr_line,
            self._cfg.artefact_policy,
            self._cfg.use_car,
            self._cfg.decision_confidence_min,
            self._cfg.decision_max_extra_epochs,
            self._cfg.enable_6_15_weighting,
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
                "Pipeline exit | epochs=%d | artefactual=%d | snr_rejected=%d | emitted=%d | no_decision=%d",
                self.stats.epochs_seen,
                self.stats.epochs_artefactual,
                self.stats.epochs_below_snr,
                self.stats.commands_emitted,
                self.stats.no_decisions_emitted,
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
            if self._cfg.artefact_policy == "drop":
                logger.debug(
                    "Epoch dropped (artefact) | max_ptp_eval=%.1f µV (threshold=%.1f on %s)",
                    self._last_ptp_eval_max,
                    self._cfg.artefact_threshold_uv,
                    "all ch" if idx is None else f"indices {list(idx)}",
                )
                return
            if self._cfg.artefact_policy == "penalize":
                logger.debug(
                    "Epoch artefactual (penalize) | max_ptp_eval=%.1f µV (threshold=%.1f on %s)",
                    self._last_ptp_eval_max,
                    self._cfg.artefact_threshold_uv,
                    "all ch" if idx is None else f"indices {list(idx)}",
                )

        prediction = self._classifier.predict(result.data)
        prediction = self._predict_with_optional_6_15_weighting(result.data, prediction)
        snr_db = self._compute_snr_db(result.data, prediction.frequency_hz)

        if self._cfg.snr_gate_enabled:
            effective_min = self._effective_snr_min_db()
            if snr_db < effective_min:
                self.stats.epochs_below_snr += 1
                if self._cfg.snr_adaptive_relax:
                    self._consecutive_snr_rejects += 1
                logger.debug(
                    "Epoch dropped (SNR %.2f dB < %.2f dB effective) | freq=%.1f Hz",
                    snr_db,
                    effective_min,
                    prediction.frequency_hz,
                )
                return

            if self._cfg.snr_adaptive_relax:
                self._consecutive_snr_rejects = 0

        confidence = float(prediction.confidence)
        raw_scores = np.asarray(prediction.raw_scores, dtype=np.float64)
        if result.artefactual and self._cfg.artefact_policy == "penalize":
            confidence *= float(self._cfg.artefact_penalty)
            raw_scores = raw_scores * float(self._cfg.artefact_penalty)

        # Optionally defer low-confidence decisions for up to N extra epochs.
        emit_now, no_decision, decided_freq, decided_conf = self._decision_gate(
            raw_scores=raw_scores,
            fallback_freq=float(prediction.frequency_hz),
            fallback_conf=confidence,
        )
        if not emit_now:
            return

        if no_decision:
            payload = build_payload(
                frequency_hz=decided_freq,
                snr_db=snr_db,
                confidence=decided_conf,
                epoch_ms=self._epoch_ms,
                command=self._cfg.decision_no_decision_command,
            )
            self._emit(payload)
            self.stats.commands_emitted += 1
            self.stats.no_decisions_emitted += 1
            logger.info(
                "%s | freq=%.2f Hz | SNR=%.2f dB | conf=%.2f",
                self._cfg.decision_no_decision_command,
                decided_freq,
                snr_db,
                decided_conf,
            )
            return

        raw_freq = float(decided_freq)
        confidence = float(decided_conf)
        if self._smooth_maxlen <= 1:
            out_freq = raw_freq
        else:
            self._freq_window.append(raw_freq)
            out_freq = smoothed_frequency_hz(self._freq_window)

        # ``snr_db`` is for the classifier frequency this epoch; ``out_freq`` may differ
        # briefly when temporal smoothing is enabled.
        payload = build_payload(
            frequency_hz=out_freq,
            snr_db=snr_db,
            confidence=confidence,
            epoch_ms=self._epoch_ms,
        )
        self._emit(payload)
        self.stats.commands_emitted += 1
        logger.info(
            "SELECT | freq=%.2f Hz | SNR=%.2f dB | conf=%.2f",
            out_freq,
            snr_db,
            confidence,
        )

    def _find_freq_index(self, target_hz: float) -> int | None:
        for i, f in enumerate(self._freqs):
            if abs(float(f) - float(target_hz)) < 1e-6:
                return i
        return None

    def _fit_channel_weights(
        self,
        raw_weights: list[float] | None,
        n_channels: int,
        which: str,
    ) -> np.ndarray:
        # Default: equal channel contribution.
        if raw_weights is None:
            return np.ones(n_channels, dtype=np.float64)

        arr = np.asarray(raw_weights, dtype=np.float64)
        if arr.size == n_channels:
            pass
        elif arr.size < n_channels:
            pad = np.ones(n_channels - arr.size, dtype=np.float64)
            arr = np.concatenate([arr, pad], axis=0)
            if which == "6" and not self._warned_w6_shape:
                logger.warning(
                    "weights_6hz_by_channel length=%d does not match channels=%d; padded with 1.0",
                    arr.size - pad.size,
                    n_channels,
                )
                self._warned_w6_shape = True
            if which == "15" and not self._warned_w15_shape:
                logger.warning(
                    "weights_15hz_by_channel length=%d does not match channels=%d; padded with 1.0",
                    arr.size - pad.size,
                    n_channels,
                )
                self._warned_w15_shape = True
        else:
            arr = arr[:n_channels]
            if which == "6" and not self._warned_w6_shape:
                logger.warning(
                    "weights_6hz_by_channel longer than channels=%d; truncated",
                    n_channels,
                )
                self._warned_w6_shape = True
            if which == "15" and not self._warned_w15_shape:
                logger.warning(
                    "weights_15hz_by_channel longer than channels=%d; truncated",
                    n_channels,
                )
                self._warned_w15_shape = True

        arr = np.maximum(arr, 1e-6)
        # Normalize to mean=1 so scaling does not explode/vanish scores.
        return arr / float(np.mean(arr))

    def _predict_with_optional_6_15_weighting(
        self,
        epoch: np.ndarray,
        prediction: ClassifierResult,
    ) -> ClassifierResult:
        if not self._cfg.enable_6_15_weighting:
            return prediction

        i6 = self._find_freq_index(6.0)
        i15 = self._find_freq_index(15.0)
        if i6 is None or i15 is None:
            return prediction

        n_channels = int(epoch.shape[0])
        w6 = self._fit_channel_weights(
            self._cfg.weights_6hz_by_channel,
            n_channels,
            which="6",
        )
        w15 = self._fit_channel_weights(
            self._cfg.weights_15hz_by_channel,
            n_channels,
            which="15",
        )

        s6 = np.zeros(n_channels, dtype=np.float64)
        s15 = np.zeros(n_channels, dtype=np.float64)
        for ch in range(n_channels):
            s6[ch] = compute_ssvep_snr_db(
                epoch,
                target_freq_hz=6.0,
                fs=self._cfg.sample_rate_hz,
                channel_idx=ch,
                n_harmonics=self._cfg.n_harmonics,
                noise_band_hz=self._cfg.snr_noise_band_hz,
            )
            s15[ch] = compute_ssvep_snr_db(
                epoch,
                target_freq_hz=15.0,
                fs=self._cfg.sample_rate_hz,
                channel_idx=ch,
                n_harmonics=self._cfg.n_harmonics,
                noise_band_hz=self._cfg.snr_noise_band_hz,
            )

        s6 = np.nan_to_num(s6, nan=0.0, posinf=50.0, neginf=-50.0)
        s15 = np.nan_to_num(s15, nan=0.0, posinf=50.0, neginf=-50.0)

        score6 = float(np.dot(w6, s6))
        score15 = float(np.dot(w15, s15))

        raw = np.asarray(prediction.raw_scores, dtype=np.float64).copy()
        if raw.size != len(self._freqs):
            return prediction
        raw[i6] = score6
        raw[i15] = score15

        # Binary confidence between 6 and 15 Hz only.
        pair = np.array([score6, score15], dtype=np.float64)
        pair = pair - float(np.max(pair))
        probs = np.exp(pair)
        probs = probs / float(np.sum(probs))
        winner = i6 if score6 >= score15 else i15
        conf = float(np.max(probs))

        return ClassifierResult(
            frequency_hz=float(self._freqs[winner]),
            confidence=conf,
            raw_scores=raw.astype(np.float32),
        )

    def _decision_from_log_scores(self, log_scores: np.ndarray) -> tuple[float, float]:
        if log_scores.size != len(self._freqs):
            return float(self._freqs[0]), 0.0
        shifted = log_scores - float(np.max(log_scores))
        probs = np.exp(shifted)
        total = float(np.sum(probs))
        if total <= 0.0:
            return float(self._freqs[0]), 0.0
        probs /= total
        idx = int(np.argmax(probs))
        return float(self._freqs[idx]), float(probs[idx])

    def _decision_gate(
        self,
        raw_scores: np.ndarray,
        fallback_freq: float,
        fallback_conf: float,
    ) -> tuple[bool, bool, float, float]:
        # No adaptive gate: always emit current epoch decision immediately.
        if (
            self._cfg.decision_confidence_min <= 0.0
            and self._cfg.decision_max_extra_epochs <= 0
            and not self._cfg.decision_emit_no_decision
        ):
            return True, False, fallback_freq, fallback_conf

        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.size != len(self._freqs):
            return True, False, fallback_freq, fallback_conf

        # Keep native score ratios when already positive; only offset when the
        # vector contains non-positive values.
        min_score = float(np.min(scores))
        if min_score <= 0.0:
            scores = scores - min_score + 1e-6

        if self._decision_log_scores is None:
            self._decision_log_scores = np.zeros_like(scores, dtype=np.float64)
            self._decision_epochs = 0

        self._decision_log_scores += np.log(np.clip(scores, 1e-12, None))
        self._decision_epochs += 1

        freq, conf = self._decision_from_log_scores(self._decision_log_scores)
        if conf >= float(self._cfg.decision_confidence_min):
            self._decision_log_scores = None
            self._decision_epochs = 0
            return True, False, freq, conf

        allowed = 1 + int(self._cfg.decision_max_extra_epochs)
        if self._decision_epochs < allowed:
            return False, False, fallback_freq, fallback_conf

        # Out of extra epochs and still low confidence.
        self._decision_log_scores = None
        self._decision_epochs = 0
        if self._cfg.decision_emit_no_decision:
            return True, True, fallback_freq, conf
        return False, False, fallback_freq, conf

    def _effective_snr_min_db(self) -> float:
        base = float(self._cfg.snr_min_db)
        if not self._cfg.snr_adaptive_relax:
            return base
        relax = _SNRELAX_STEP_DB * min(self._consecutive_snr_rejects, _SNRELAX_CAP)
        return max(float(self._cfg.snr_adaptive_floor_db), base - relax)

    def _snr_indices_for_epoch(self, n_channels: int) -> list[int]:
        raw = self._cfg.snr_channel_indices
        if raw is None:
            return list(range(n_channels))
        out = [i for i in raw if 0 <= i < n_channels]
        return out if out else list(range(n_channels))

    def _compute_snr_db(self, data, target_freq_hz: float) -> float:
        if self._cfg.snr_aggregate == "single":
            return compute_ssvep_snr_db(
                data,
                target_freq_hz=target_freq_hz,
                fs=self._cfg.sample_rate_hz,
                channel_idx=self._cfg.snr_channel_index,
                n_harmonics=self._cfg.n_harmonics,
                noise_band_hz=self._cfg.snr_noise_band_hz,
            )
        ch_ix = self._snr_indices_for_epoch(int(data.shape[0]))
        mode = self._cfg.snr_aggregate
        return compute_ssvep_snr_db_aggregate(
            data,
            target_freq_hz=target_freq_hz,
            fs=float(self._cfg.sample_rate_hz),
            channel_indices=ch_ix,
            mode=mode,  # type: ignore[arg-type]
            n_harmonics=self._cfg.n_harmonics,
            noise_band_hz=self._cfg.snr_noise_band_hz,
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

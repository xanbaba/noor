"""End-to-end Layer 2 test.

Topology:

    synthetic SSVEP LSL outlet  →  Layer 2 Pipeline  →  on_emit callback

No Cyton, no WebSocket client, no OSC listener required.  The on_emit hook
captures every SELECT command so we can assert on frequency and SNR.

The test stream name is ``BCI_RawEEG_E2E`` to avoid colliding with a live
Layer 1 session that might be running on the same machine.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Try importing dependencies; skip whole module if pylsl is absent.
# ---------------------------------------------------------------------------
try:
    from pylsl import StreamInfo, StreamOutlet, cf_float32, local_clock
except ImportError:
    pytest.skip("pylsl not installed", allow_module_level=True)

from layer2_processing.config import load_config
from layer2_processing.classifiers.factory import create_classifier
from layer2_processing.lsl_inlet import RawEegInlet, StreamNotFoundError
from layer2_processing.outputs.osc_emitter import OscEmitter
from layer2_processing.outputs.websocket_emitter import WebSocketEmitter
from layer2_processing.pipeline import Pipeline
from layer2_processing.preprocessing import EpochBuffer, Preprocessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STREAM_NAME = "BCI_RawEEG_E2E"
_FS = 500
_N_CHANNELS = 8
_TARGET_FREQ = 6.0
_AMPLITUDE_UV = 30.0
_NOISE_UV = 3.0       # low noise so FBCCA can detect it quickly
_CHUNK_SIZE = 20      # 40 ms / push
_LABELS = ("Oz", "O1", "O2", "Pz", "--", "--", "--", "--")
_TIMEOUT_S = 20.0     # generous — CCA is slow on the first few epochs


# ---------------------------------------------------------------------------
# Synthetic source fixture (module scope so one outlet services all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ssvep_outlet():
    """Open a BCI_RawEEG_E2E LSL outlet streaming 6 Hz SSVEP on Oz."""
    info = StreamInfo(
        name=_STREAM_NAME,
        type="EEG",
        channel_count=_N_CHANNELS,
        nominal_srate=float(_FS),
        channel_format=cf_float32,
        source_id=f"e2e_{uuid.uuid4().hex[:6]}",
    )
    ch_node = info.desc().append_child("channels")
    for lbl in _LABELS:
        ch = ch_node.append_child("channel")
        ch.append_child_value("label", lbl)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")

    outlet = StreamOutlet(info, max_buffered=30)

    rng = np.random.default_rng(0)
    stop = threading.Event()

    def _stream():
        n_sent = 0
        chunk_dt = _CHUNK_SIZE / _FS
        next_tick = time.monotonic()
        while not stop.is_set():
            t_idx = np.arange(n_sent, n_sent + _CHUNK_SIZE) / _FS
            # Fundamental + 1st harmonic on Oz (row 0)
            ssvep = (
                _AMPLITUDE_UV * np.sin(2 * np.pi * _TARGET_FREQ * t_idx)
                + (_AMPLITUDE_UV / 2) * np.sin(2 * np.pi * 2 * _TARGET_FREQ * t_idx)
            ).astype(np.float32)
            noise = (rng.standard_normal((_N_CHANNELS, _CHUNK_SIZE)) * _NOISE_UV).astype(np.float32)
            noise[0] += ssvep
            outlet.push_chunk(noise.T.tolist(), local_clock())
            n_sent += _CHUNK_SIZE
            next_tick += chunk_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    t = threading.Thread(target=_stream, daemon=True)
    t.start()

    # Give pylsl a moment to announce the stream
    time.sleep(0.5)
    yield outlet

    stop.set()
    t.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e_cfg():
    default = Path("configs/layer2_default.yaml")
    if not default.exists():
        pytest.skip("configs/layer2_default.yaml not found — run from project root.")
    return load_config(
        default,
        overrides={
            "lsl_stream_name": _STREAM_NAME,
            "inlet_resolve_timeout_s": 5.0,
            "snr_min_db": 3.5,
            # Override sub_bands so the highest cutoff (90 Hz) stays below nyquist=250 Hz
            # (the defaults are fine at 500 Hz SR — this is just to be explicit)
        },
    )


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

def test_pipeline_emits_correct_frequency(ssvep_outlet, e2e_cfg):
    """
    Full stack:
        synthetic 6 Hz outlet → RawEegInlet → EpochBuffer → Preprocessor
        → FBCCAClassifier → SNR gate → on_emit callback
    Assert that at least one SELECT at 6.0 Hz arrives within TIMEOUT_S seconds.
    """
    received: list[dict] = []
    done = threading.Event()

    def _on_emit(payload: dict) -> None:
        received.append(payload)
        if payload.get("frequency") == _TARGET_FREQ:
            done.set()

    inlet = RawEegInlet(
        stream_name=_STREAM_NAME,
        resolve_timeout_s=5.0,
    )
    try:
        inlet.open()
    except StreamNotFoundError as exc:
        pytest.skip(f"LSL stream not found: {exc}")

    classifier = create_classifier(e2e_cfg)
    pipeline = Pipeline(
        cfg=e2e_cfg,
        inlet=inlet,
        classifier=classifier,
        ws_emitter=None,
        osc_emitter=None,
        on_emit=_on_emit,
    )

    t = threading.Thread(target=pipeline.run, daemon=True)
    t.start()

    succeeded = done.wait(timeout=_TIMEOUT_S)
    pipeline.stop()
    t.join(timeout=5.0)
    inlet.close()

    assert succeeded, (
        f"No correct SELECT received within {_TIMEOUT_S}s. "
        f"Got {len(received)} payloads: {received[:5]}"
    )

    matching = [p for p in received if p.get("frequency") == _TARGET_FREQ]
    assert matching, "No payload with the correct 6.0 Hz frequency"
    assert matching[0]["snr_db"] >= e2e_cfg.snr_min_db, (
        f"SNR {matching[0]['snr_db']:.2f} dB is below gate {e2e_cfg.snr_min_db} dB"
    )
    assert 0.0 <= matching[0]["confidence"] <= 1.0


def test_pipeline_stats_after_run(ssvep_outlet, e2e_cfg):
    """Pipeline stats must be non-zero after processing some data."""
    received: list[dict] = []
    done = threading.Event()

    def _on_emit(payload: dict) -> None:
        received.append(payload)
        done.set()

    inlet = RawEegInlet(stream_name=_STREAM_NAME, resolve_timeout_s=5.0)
    try:
        inlet.open()
    except StreamNotFoundError:
        pytest.skip("LSL stream not found")

    classifier = create_classifier(e2e_cfg)
    pipeline = Pipeline(
        cfg=e2e_cfg,
        inlet=inlet,
        classifier=classifier,
        ws_emitter=None,
        osc_emitter=None,
        on_emit=_on_emit,
    )
    t = threading.Thread(target=pipeline.run, daemon=True)
    t.start()
    done.wait(timeout=_TIMEOUT_S)
    pipeline.stop()
    t.join(timeout=5.0)
    inlet.close()

    stats = pipeline.stats
    assert stats.epochs_seen >= 1, "Pipeline should have seen at least one epoch"
    assert stats.commands_emitted >= 1, "At least one command should have been emitted"

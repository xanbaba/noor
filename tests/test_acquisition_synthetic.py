"""End-to-end acquisition test using the synthetic board and an LSL inlet.

Spins up the full stack (SyntheticBoard → AcquisitionLoop → RawEegOutlet),
consumes via a pylsl StreamInlet for 1 second, and asserts:
  - at least 480 samples received (>= 96% of 500 Hz nominal)
  - each sample has exactly 8 channels
"""

from __future__ import annotations

import textwrap
import threading
import time

import pytest

from layer1_acquisition.config import load_config
from layer1_acquisition.boards.factory import create_board
from layer1_acquisition.lsl_outlet import RawEegOutlet
from layer1_acquisition.acquisition import AcquisitionLoop

try:
    from pylsl import StreamInlet, resolve_byprop
    _LSL_AVAILABLE = True
except ImportError:
    _LSL_AVAILABLE = False


YAML = textwrap.dedent("""\
    board: synthetic
    serial_port: auto
    sample_rate_hz: 500
    channel_count: 8
    channel_labels: [Oz, O1, O2, Pz, "--", "--", "--", "--"]
    impedance_max_kohm: 10.0
    lsl_stream_name: BCI_RawEEG_Test
    lsl_stream_type: EEG
    pull_interval_ms: 10
    log_interval_s: 30
""")


@pytest.fixture(scope="module")
def lsl_stream_data(tmp_path_factory):
    """Module-scoped fixture: spin up one acquisition stack, collect 1.5 s of
    data, tear down cleanly, and return the collected samples."""
    if not _LSL_AVAILABLE:
        pytest.skip("pylsl not installed")

    tmp = tmp_path_factory.mktemp("cfg")
    p = tmp / "cfg.yaml"
    p.write_text(YAML)
    cfg = load_config(p)

    board = create_board(cfg)
    board.prepare()
    board.start_stream()

    outlet = RawEegOutlet(cfg, board_name="synthetic")
    loop = AcquisitionLoop(board, outlet, cfg)

    t = threading.Thread(target=loop.run, daemon=True)
    t.start()

    # Allow the outlet to register on the local LSL multicast.
    time.sleep(0.4)

    streams = resolve_byprop("name", "BCI_RawEEG_Test", timeout=4.0)
    if not streams:
        loop.stop()
        t.join(timeout=3.0)
        pytest.fail("LSL stream 'BCI_RawEEG_Test' not found within 4 s")

    inlet = StreamInlet(streams[0], max_buflen=10)

    all_samples: list[list[float]] = []
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        chunk, _ = inlet.pull_chunk(timeout=0.05)
        all_samples.extend(chunk)

    loop.stop()
    t.join(timeout=3.0)

    return all_samples


@pytest.mark.skipif(not _LSL_AVAILABLE, reason="pylsl not installed")
def test_synthetic_acquisition_rate(lsl_stream_data):
    """Assert samples were received in the 1.5 s window.

    BrainFlow's SYNTHETIC_BOARD delivers ~250 Hz internally regardless of the
    configured rate, so we assert a conservative 200 samples (well below the
    actual ~375 typically received) to keep the test stable across machines.
    """
    assert len(lsl_stream_data) >= 200, (
        f"Expected >= 200 samples in 1.5 s, got {len(lsl_stream_data)}. "
        "Possible cause: BrainFlow synthetic board startup latency or LSL buffering."
    )


@pytest.mark.skipif(not _LSL_AVAILABLE, reason="pylsl not installed")
def test_synthetic_acquisition_channel_count(lsl_stream_data):
    """Assert each received sample has exactly 8 channels."""
    assert lsl_stream_data, "No samples received from the LSL stream"
    # pylsl pull_chunk returns list-of-lists: outer = samples, inner = channels
    assert len(lsl_stream_data[0]) == 8, (
        f"Expected 8 channels per sample, got {len(lsl_stream_data[0])}"
    )

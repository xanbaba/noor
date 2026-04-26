"""AcquisitionLoop.total_samples is readable from another thread."""

from __future__ import annotations

import textwrap
import threading
import time
from pathlib import Path

from layer1_acquisition.acquisition import AcquisitionLoop
from layer1_acquisition.boards.factory import create_board
from layer1_acquisition.config import load_config
from layer1_acquisition.lsl_outlet import RawEegOutlet


def test_total_samples_while_running(tmp_path: Path) -> None:
    yaml = textwrap.dedent(
        """\
        board: synthetic
        serial_port: auto
        sample_rate_hz: 500
        channel_count: 8
        channel_labels: [Oz, O1, O2, Pz, "--", "--", "--", "--"]
        impedance_max_kohm: 10.0
        lsl_stream_name: BCI_RawEEG_TotalSamples
        lsl_stream_type: EEG
        pull_interval_ms: 10
        log_interval_s: 60
        """
    )
    p = tmp_path / "c.yaml"
    p.write_text(yaml)
    cfg = load_config(p)
    board = create_board(cfg)
    board.prepare()
    board.start_stream()
    outlet = RawEegOutlet(cfg, board_name="synthetic")
    loop = AcquisitionLoop(board, outlet, cfg)

    def run_loop() -> None:
        loop.run()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    time.sleep(0.35)
    a = loop.total_samples
    time.sleep(0.25)
    b = loop.total_samples
    loop.stop()
    t.join(timeout=5.0)
    assert b >= a > 0

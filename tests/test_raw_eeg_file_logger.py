"""Tests for layer1_acquisition.raw_eeg_file_logger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from layer1_acquisition.config import load_config
from layer1_acquisition.raw_eeg_file_logger import RawEegFileLogger


def test_append_two_chunks_csv(tmp_path: Path) -> None:
    labels = ["Oz", "O1", "O2", "Pz", "--", "--", "--", "--"]
    out = tmp_path / "eeg.csv"
    log = RawEegFileLogger(out, labels, 500, "csv")
    a = np.arange(8 * 10, dtype=np.float32).reshape(8, 10)
    b = np.ones((8, 5), dtype=np.float32) * 3.5
    log.append_chunk(a)
    log.append_chunk(b)
    log.close()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("sample_index,monotonic_s,")
    assert "Oz" in lines[0] and "Pz" in lines[0]
    assert len(lines) == 1 + 10 + 5


def test_append_chunk_wrong_channel_count_raises(tmp_path: Path) -> None:
    log = RawEegFileLogger(tmp_path / "x.csv", ["A", "B"], 500)
    with pytest.raises(ValueError, match="channels"):
        log.append_chunk(np.zeros((3, 4), dtype=np.float32))
    log.close()


def test_load_config_raw_log_keys_roundtrip(tmp_path: Path) -> None:
    import textwrap

    y = textwrap.dedent(
        """\
        board: synthetic
        serial_port: auto
        sample_rate_hz: 500
        channel_count: 2
        channel_labels: [Oz, O1]
        impedance_max_kohm: 10.0
        lsl_stream_name: BCI_RawEEG
        lsl_stream_type: EEG
        pull_interval_ms: 10
        log_interval_s: 5
        raw_eeg_log_path: /tmp/ignored.csv
        raw_eeg_log_format: csv
        """
    )
    p = tmp_path / "c.yaml"
    p.write_text(y)
    cfg = load_config(p)
    assert cfg.raw_eeg_log_path == "/tmp/ignored.csv"
    assert cfg.raw_eeg_log_format == "csv"


def test_load_config_rejects_unknown_log_format(tmp_path: Path) -> None:
    import textwrap

    y = textwrap.dedent(
        """\
        board: synthetic
        serial_port: auto
        sample_rate_hz: 500
        channel_count: 2
        channel_labels: [Oz, O1]
        impedance_max_kohm: 10.0
        lsl_stream_name: BCI_RawEEG
        lsl_stream_type: EEG
        pull_interval_ms: 10
        log_interval_s: 5
        raw_eeg_log_format: f32bin
        """
    )
    p = tmp_path / "c.yaml"
    p.write_text(y)
    with pytest.raises(ValueError, match="raw_eeg_log_format"):
        load_config(p)

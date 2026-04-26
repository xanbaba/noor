"""Tests for configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from layer1_acquisition.config import AcquisitionConfig, load_config


VALID_YAML = textwrap.dedent("""\
    board: synthetic
    serial_port: auto
    sample_rate_hz: 500
    channel_count: 8
    channel_labels: [Oz, O1, O2, Pz, "--", "--", "--", "--"]
    impedance_max_kohm: 10.0
    lsl_stream_name: BCI_RawEEG
    lsl_stream_type: EEG
    pull_interval_ms: 10
    log_interval_s: 5
""")


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.yaml"
    p.write_text(content)
    return p


def test_load_valid_config(tmp_path):
    path = _write_yaml(tmp_path, VALID_YAML)
    cfg = load_config(path)
    assert cfg.board == "synthetic"
    assert cfg.sample_rate_hz == 500
    assert cfg.channel_count == 8
    assert cfg.lsl_stream_name == "BCI_RawEEG"
    assert cfg.raw_eeg_log_path is None
    assert cfg.raw_eeg_log_format == "csv"


def test_active_channel_indices(tmp_path):
    path = _write_yaml(tmp_path, VALID_YAML)
    cfg = load_config(path)
    # Labels: Oz, O1, O2, Pz, --, --, --, --  →  indices 0–3
    assert cfg.active_channel_indices == [0, 1, 2, 3]


def test_override_board(tmp_path):
    path = _write_yaml(tmp_path, VALID_YAML)
    cfg = load_config(path, overrides={"board": "synthetic"})
    assert cfg.board == "synthetic"


def test_sample_rate_too_low_raises(tmp_path):
    bad = VALID_YAML.replace("sample_rate_hz: 500", "sample_rate_hz: 200")
    path = _write_yaml(tmp_path, bad)
    with pytest.raises(ValueError, match="250 Hz minimum"):
        load_config(path)


def test_wrong_label_count_raises(tmp_path):
    bad = VALID_YAML.replace(
        "channel_labels: [Oz, O1, O2, Pz, \"--\", \"--\", \"--\", \"--\"]",
        "channel_labels: [Oz, O1]",
    )
    path = _write_yaml(tmp_path, bad)
    with pytest.raises(ValueError, match="channel_labels length"):
        load_config(path)


def test_unknown_board_raises(tmp_path):
    bad = VALID_YAML.replace("board: synthetic", "board: unicorn")
    path = _write_yaml(tmp_path, bad)
    with pytest.raises(ValueError, match="Unknown board"):
        load_config(path)


def test_negative_impedance_raises(tmp_path):
    bad = VALID_YAML.replace("impedance_max_kohm: 10.0", "impedance_max_kohm: -1")
    path = _write_yaml(tmp_path, bad)
    with pytest.raises(ValueError, match="impedance_max_kohm"):
        load_config(path)

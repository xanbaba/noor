"""Tests for the board factory dispatch."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from layer1_acquisition.config import load_config
from layer1_acquisition.boards.factory import create_board
from layer1_acquisition.boards.synthetic import SyntheticBoard


YAML = textwrap.dedent("""\
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


def _cfg(tmp_path, board="synthetic"):
    p = tmp_path / "cfg.yaml"
    p.write_text(YAML.replace("board: synthetic", f"board: {board}"))
    return load_config(p)


def test_factory_returns_synthetic(tmp_path):
    cfg = _cfg(tmp_path)
    board = create_board(cfg)
    assert isinstance(board, SyntheticBoard)


def test_factory_unknown_board_raises(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.board = "nonexistent"
    with pytest.raises(ValueError, match="Unknown board"):
        create_board(cfg)


def test_synthetic_board_attributes(tmp_path):
    cfg = _cfg(tmp_path)
    board = create_board(cfg)
    assert board.sample_rate_hz == 500
    assert board.channel_count == 8
    assert len(board.channel_labels) == 8

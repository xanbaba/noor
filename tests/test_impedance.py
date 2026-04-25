"""Tests for impedance gating logic."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from layer1_acquisition.config import load_config
from layer1_acquisition.impedance import check_impedance, ImpedanceGateError


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


@pytest.fixture()
def cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(YAML)
    return load_config(p)


def _mock_board(readings: dict[str, float]):
    board = MagicMock()
    board.impedance_kohm.return_value = readings
    return board


def test_gate_passes_all_below_threshold(cfg):
    board = _mock_board({"Oz": 5.0, "O1": 4.2, "O2": 3.8, "Pz": 7.1})
    result = check_impedance(board, cfg, skip=False)
    assert result == {"Oz": 5.0, "O1": 4.2, "O2": 3.8, "Pz": 7.1}


def test_gate_fails_when_channel_exceeds_threshold(cfg):
    board = _mock_board({"Oz": 5.0, "O1": 15.0, "O2": 3.8, "Pz": 7.1})
    with pytest.raises(ImpedanceGateError, match="O1"):
        check_impedance(board, cfg, skip=False)


def test_gate_skip_returns_empty_and_no_error(cfg):
    board = _mock_board({"Oz": 50.0})
    result = check_impedance(board, cfg, skip=True)
    assert result == {}
    board.impedance_kohm.assert_not_called()


def test_gate_raises_not_implemented_when_board_unsupported(cfg):
    board = MagicMock()
    board.impedance_kohm.side_effect = NotImplementedError
    with pytest.raises(NotImplementedError):
        check_impedance(board, cfg, skip=False)


def test_gate_exactly_at_threshold_passes(cfg):
    board = _mock_board({"Oz": 10.0, "O1": 10.0, "O2": 10.0, "Pz": 10.0})
    result = check_impedance(board, cfg, skip=False)
    assert all(v <= 10.0 for v in result.values())

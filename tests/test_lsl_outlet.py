"""Tests for LSL outlet metadata and push behaviour."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from layer1_acquisition.config import load_config
from layer1_acquisition.lsl_outlet import RawEegOutlet


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


def test_outlet_stream_name(cfg):
    outlet = RawEegOutlet(cfg, board_name="synthetic")
    assert outlet.stream_name == "BCI_RawEEG"


def test_outlet_stream_type(cfg):
    outlet = RawEegOutlet(cfg, board_name="synthetic")
    assert outlet.stream_type == "EEG"


def test_outlet_channel_count(cfg):
    outlet = RawEegOutlet(cfg, board_name="synthetic")
    assert outlet.stream_channel_count == 8


def test_outlet_nominal_rate(cfg):
    outlet = RawEegOutlet(cfg, board_name="synthetic")
    assert outlet.stream_nominal_srate == 500.0


def test_outlet_push_chunk_empty_is_no_op(cfg):
    outlet = RawEegOutlet(cfg, board_name="synthetic")
    empty = np.empty((8, 0), dtype=np.float32)
    # Should not raise.
    outlet.push_chunk(empty)


def test_outlet_push_chunk_valid_shape(cfg):
    outlet = RawEegOutlet(cfg, board_name="synthetic")
    chunk = np.random.randn(8, 50).astype(np.float32)
    outlet.push_chunk(chunk)  # Should not raise.

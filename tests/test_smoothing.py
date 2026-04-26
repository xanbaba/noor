"""Tests for layer2_processing.smoothing."""

from __future__ import annotations

import pytest

from layer2_processing.smoothing import smoothed_frequency_hz


def test_mode_single_value():
    assert smoothed_frequency_hz([12.0]) == 12.0


def test_mode_clear_winner():
    assert smoothed_frequency_hz([12.0, 12.0, 15.0, 12.0]) == 12.0


def test_tie_prefers_most_recent():
    assert smoothed_frequency_hz([12.0, 15.0, 12.0, 15.0]) == 15.0


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        smoothed_frequency_hz([])

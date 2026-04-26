"""Tests for layer3_backend.confirmation — StreakTracker."""

from __future__ import annotations

import pytest

from layer3_backend.confirmation import StreakTracker, normalise_frequency_hz


def test_normalise_frequency_hz():
    assert normalise_frequency_hz(5.999) == 6.0


def test_streak_fires_on_fifth_identical():
    t = StreakTracker(required=5)
    assert t.feed(6.0) is None
    assert t.feed(6.0) is None
    assert t.feed(6.0) is None
    assert t.feed(6.0) is None
    assert t.feed(6.0) == 6.0


def test_streak_resets_on_frequency_change():
    t = StreakTracker(required=5)
    for _ in range(4):
        assert t.feed(6.0) is None
    assert t.feed(20.0) is None
    assert t.feed(6.0) is None
    for _ in range(3):
        assert t.feed(6.0) is None
    # Fifth consecutive 6 Hz after the 20 Hz interruption confirms once.
    assert t.feed(6.0) == 6.0


def test_streak_requires_positive_n():
    with pytest.raises(ValueError):
        StreakTracker(required=0)

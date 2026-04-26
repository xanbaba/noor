"""Trial schedule helper for SSVEP ML experiments."""

from __future__ import annotations

import random

import pytest

from experiments.ssvep_ml.schedule import (
    label_to_frequency_hz,
    make_trial_schedule,
)


def test_make_trial_schedule_counts_and_bounds():
    rng = random.Random(0)
    s = make_trial_schedule(12, rng=rng)
    assert len(s) == 24
    assert s.count("6hz") == 12
    assert s.count("20hz") == 12
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            assert run <= 4
        else:
            run = 1


def test_make_trial_schedule_rejects_zero():
    with pytest.raises(ValueError):
        make_trial_schedule(0)


def test_label_to_frequency_hz():
    assert label_to_frequency_hz("6hz") == 6.0
    assert label_to_frequency_hz("20hz") == 20.0
    with pytest.raises(ValueError):
        label_to_frequency_hz("10hz")

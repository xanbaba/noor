"""Unit tests for ``scripts/synthetic_signal_model`` (no pylsl)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SSM_PATH = _ROOT / "scripts" / "synthetic_signal_model.py"


@pytest.fixture(scope="module")
def ssm():
    spec = importlib.util.spec_from_file_location("synthetic_signal_model", _SSM_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Required so @dataclass can resolve postponed annotations during import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_apply_adc_clip(ssm):
    x = np.array([[-500.0, 0.0, 80.0], [10.0, -200.0, 3.0]], dtype=np.float64)
    y = ssm.apply_adc_clip(x, 50.0)
    assert np.all(y >= -50.0) and np.all(y <= 50.0)
    assert y[0, 2] == 50.0
    z = ssm.apply_adc_clip(x, 0.0)
    assert z.shape == x.shape


def test_blink_waveform_length_and_peak(ssm):
    fs = 500.0
    dur_s = 0.2
    w = ssm.blink_waveform(fs, dur_s)
    assert w.size == int(round(dur_s * fs))
    assert abs(float(w.max()) - 1.0) < 1e-9
    assert float(w.min()) >= -1e-9


def test_add_blink_occipital_spatial_mix(ssm):
    n_ch, n = 4, 20
    data = np.zeros((n_ch, n), dtype=np.float64)
    prof = np.zeros(n, dtype=np.float64)
    prof[10] = 1.0
    spatial = ssm.occipital_spatial_blink_weights(n_ch)
    ssm.add_blink_occipital(data, prof, 100.0, spatial)
    assert data[0, 10] == pytest.approx(100.0 * spatial[0])
    assert data[3, 10] == pytest.approx(100.0 * spatial[3])


def test_inject_pop_adds_step(ssm):
    data = np.zeros((2, 50), dtype=np.float64)
    before = data[1, 20]
    ssm.inject_pop_step_decay(data, 1, 20, -1.0, 80.0, 500.0)
    assert data[1, 20] < before
    assert data[0, 20] == 0.0


def test_eog_ramp_advances_phase(ssm):
    state = ssm.EogRampState(phase=0.1)
    n_ch, n = 4, 30
    spatial = ssm.occipital_spatial_eog_weights(n_ch)
    data = np.zeros((n_ch, n), dtype=np.float64)
    ssm.eog_ramp_add(data, 0, 500.0, 1.0, 10.0, spatial, state)
    ph_after = state.phase
    assert ph_after != pytest.approx(0.1)
    expected = (0.1 + 2.0 * math.pi * 1.0 * n / 500.0) % (2.0 * math.pi)
    assert ph_after == pytest.approx(expected)


def test_poisson_next_interval_nonpositive_rate(ssm):
    rng = np.random.default_rng(1)
    assert math.isinf(ssm.poisson_next_interval(0.0, rng))

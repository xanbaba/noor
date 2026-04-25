"""Tests for layer2_processing.classifiers.fbcca — FBCCAClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from layer2_processing.classifiers.fbcca import FBCCAClassifier
from layer2_processing.classifiers.factory import create_classifier, registered_classifiers


FS = 500
N = 1000   # 2 s


def _cfg(**overrides):
    from types import SimpleNamespace
    defaults = dict(
        sample_rate_hz=FS,
        classifier="fbcca",
        stimulus_frequencies_hz=[9.0, 12.0, 15.0],
        sub_bands_hz=[[6, 90], [14, 90], [22, 90], [30, 90]],
        sub_band_filter_order=5,
        weight_a=1.25,
        weight_b=0.25,
        n_harmonics=3,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _ssvep_epoch(
    target_hz: float,
    n_channels: int = 8,
    amplitude: float = 30.0,
    noise_level: float = 1.0,
    n_harmonics: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Create a (channels, N) epoch with SSVEP on channel 0 plus broadband noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    ssvep = np.zeros(N, dtype=np.float64)
    for h in range(1, n_harmonics + 2):
        ssvep += (amplitude / h) * np.sin(2 * np.pi * h * target_hz * t)
    noise = rng.standard_normal((n_channels, N)) * noise_level
    epoch = noise.astype(np.float32)
    epoch[0] += ssvep.astype(np.float32)
    return epoch


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_registers_fbcca():
    assert "fbcca" in registered_classifiers()


def test_factory_creates_fbcca():
    clf = create_classifier(_cfg())
    assert isinstance(clf, FBCCAClassifier)


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown classifier"):
        create_classifier(_cfg(), name="trca")


# ---------------------------------------------------------------------------
# Correct frequency detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_hz", [9.0, 12.0, 15.0])
def test_detects_correct_frequency(target_hz):
    clf = FBCCAClassifier(_cfg())
    epoch = _ssvep_epoch(target_hz)
    result = clf.predict(epoch)
    assert result.frequency_hz == target_hz, (
        f"Expected {target_hz} Hz, got {result.frequency_hz} Hz "
        f"(confidence={result.confidence:.3f})"
    )


@pytest.mark.parametrize("target_hz", [9.0, 12.0, 15.0])
def test_confidence_above_threshold_for_ssvep(target_hz):
    clf = FBCCAClassifier(_cfg())
    epoch = _ssvep_epoch(target_hz)
    result = clf.predict(epoch)
    assert result.confidence > 0.5, (
        f"Expected confidence > 0.5, got {result.confidence:.3f} for {target_hz} Hz"
    )


# ---------------------------------------------------------------------------
# Pure noise gives low confidence
# ---------------------------------------------------------------------------

def test_noise_epoch_low_max_score():
    """Pure white noise must not yield high confidence for any frequency."""
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((8, N)).astype(np.float32) * 10.0
    clf = FBCCAClassifier(_cfg())
    result = clf.predict(noise)
    # With noise, raw_scores should all be small
    assert float(result.raw_scores.max()) < 10.0, (
        f"Noise yielded unexpectedly high score: {result.raw_scores.max():.3f}"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_predict_wrong_ndim_raises():
    clf = FBCCAClassifier(_cfg())
    with pytest.raises(ValueError, match="2-D"):
        clf.predict(np.zeros((8, N, 2)))


# ---------------------------------------------------------------------------
# Classifier result shape
# ---------------------------------------------------------------------------

def test_result_raw_scores_shape():
    clf = FBCCAClassifier(_cfg())
    epoch = _ssvep_epoch(12.0)
    result = clf.predict(epoch)
    assert result.raw_scores.shape == (3,)  # 3 stimulus frequencies
    assert result.frequency_hz in [9.0, 12.0, 15.0]
    assert 0.0 <= result.confidence <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Stimulus frequency list is config-driven
# ---------------------------------------------------------------------------

def test_custom_frequency_list():
    """Classifier respects the config frequency list — no hard-coded defaults."""
    custom_freqs = [7.5, 20.0, 30.0]
    clf = FBCCAClassifier(_cfg(stimulus_frequencies_hz=custom_freqs))
    assert clf.stimulus_frequencies_hz == custom_freqs

    epoch = _ssvep_epoch(20.0)
    result = clf.predict(epoch)
    assert result.frequency_hz in custom_freqs

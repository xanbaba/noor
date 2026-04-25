"""Tests for layer2_processing.snr — compute_ssvep_snr_db."""

from __future__ import annotations

import numpy as np
import pytest

from layer2_processing.snr import compute_ssvep_snr_db

FS = 500.0
N = 1000   # 2 s


def _pure_sine(freq_hz: float, amplitude: float = 50.0) -> np.ndarray:
    """Single-channel (1, N) epoch containing only a pure sine."""
    t = np.arange(N) / FS
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).reshape(1, -1).astype(np.float32)


def _white_noise_epoch(amplitude: float = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, N)) * amplitude).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure sine → high SNR
# ---------------------------------------------------------------------------

def test_pure_sine_high_snr():
    """A pure 12 Hz sine with no noise should produce a very high SNR."""
    epoch = _pure_sine(freq_hz=12.0)
    snr = compute_ssvep_snr_db(epoch, target_freq_hz=12.0, fs=FS)
    assert snr > 20.0, f"Expected SNR > 20 dB for pure sine, got {snr:.2f} dB"


def test_pure_sine_above_gate():
    """A 3.5 dB gate should pass for a clean sine at any SSVEP frequency."""
    for f in [9.0, 12.0, 15.0, 18.0, 30.0]:
        epoch = _pure_sine(freq_hz=f)
        snr = compute_ssvep_snr_db(epoch, target_freq_hz=f, fs=FS)
        assert snr > 3.5, f"SNR gate should pass for pure sine at {f} Hz, got {snr:.2f} dB"


# ---------------------------------------------------------------------------
# Pure white noise → low / negative SNR
# ---------------------------------------------------------------------------

def test_white_noise_low_snr():
    """White noise alone should have low SNR on average.

    White noise is flat in expectation, so the signal bin has the same
    expected power as noise bins.  A single realisation can be a few dB
    either way; we allow up to 10 dB as a conservative single-sample limit.
    The companion test ``test_white_noise_below_gate_reliably`` checks the
    average more stringently.
    """
    epoch = _white_noise_epoch(seed=7)
    snr = compute_ssvep_snr_db(epoch, target_freq_hz=12.0, fs=FS)
    assert snr <= 10.0, (
        f"White noise SNR unexpectedly high: {snr:.2f} dB (should be well below 10 dB)"
    )


def test_white_noise_below_gate_reliably():
    """Averaged over multiple random seeds, noise SNR should be < gate (3.5 dB)."""
    snrs = [
        compute_ssvep_snr_db(
            _white_noise_epoch(seed=i), target_freq_hz=12.0, fs=FS
        )
        for i in range(10)
    ]
    mean_snr = float(np.mean(snrs))
    assert mean_snr < 3.5, f"Mean noise SNR {mean_snr:.2f} dB should be below 3.5 dB"


# ---------------------------------------------------------------------------
# Multi-channel: channel_idx selects the right row
# ---------------------------------------------------------------------------

def test_channel_idx_selection():
    """SNR is measured on the designated channel, not across all channels."""
    rng = np.random.default_rng(42)
    epoch = (rng.standard_normal((4, N)) * 3.0).astype(np.float32)
    # Place a strong sine only on channel 2
    t = np.arange(N) / FS
    epoch[2] += (80.0 * np.sin(2 * np.pi * 12.0 * t)).astype(np.float32)

    snr_ch2 = compute_ssvep_snr_db(epoch, target_freq_hz=12.0, fs=FS, channel_idx=2)
    snr_ch0 = compute_ssvep_snr_db(epoch, target_freq_hz=12.0, fs=FS, channel_idx=0)

    assert snr_ch2 > 10.0, f"Signal channel SNR should be high, got {snr_ch2:.2f} dB"
    assert snr_ch2 > snr_ch0, "Signal channel SNR must exceed noise-only channel SNR"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_wrong_ndim_raises():
    with pytest.raises(ValueError, match="2-D"):
        compute_ssvep_snr_db(np.zeros((N,)), target_freq_hz=12.0, fs=FS)


def test_channel_idx_out_of_range_raises():
    epoch = _pure_sine(12.0)
    with pytest.raises(ValueError, match="channel_idx"):
        compute_ssvep_snr_db(epoch, target_freq_hz=12.0, fs=FS, channel_idx=5)

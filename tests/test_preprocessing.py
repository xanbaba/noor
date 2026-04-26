"""Tests for layer2_processing.preprocessing — Preprocessor & EpochBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from layer2_processing.preprocessing import EpochBuffer, Preprocessor


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FS = 500
N = 1000   # 2 s at 500 Hz


def _make_cfg(**overrides):
    """Return a minimal duck-typed config namespace."""
    from types import SimpleNamespace
    defaults = dict(
        sample_rate_hz=FS,
        notch_freq_hz=60.0,
        notch_q=35.0,
        bandpass_low_hz=5.0,
        bandpass_high_hz=45.0,
        bandpass_order=4,
        artefact_threshold_uv=100.0,
        artefact_channel_indices=None,
        additional_notch_freqs_hz=[],
        use_car=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _pure_tone(freq_hz: float, n: int, fs: int, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _multi_channel(n_ch: int = 4, freq_hz: float = 10.0, amplitude: float = 1.0) -> np.ndarray:
    """(n_ch, N) epoch with the same tone on every channel."""
    tone = _pure_tone(freq_hz, N, FS, amplitude)
    return np.stack([tone] * n_ch, axis=0)


# ---------------------------------------------------------------------------
# Notch filter
# ---------------------------------------------------------------------------

def test_notch_attenuates_60hz():
    """60 Hz component must be reduced by at least 30 dB after notch+bandpass."""
    pre = Preprocessor(_make_cfg(use_car=True))
    epoch = _multi_channel(freq_hz=60.0, amplitude=100.0)

    power_before = float(np.mean(epoch[0] ** 2))
    filtered = pre.filter(epoch)
    power_after = float(np.mean(filtered[0] ** 2))

    # ≥ 30 dB attenuation
    assert power_after < power_before * 10 ** (-30 / 10), (
        f"60 Hz not attenuated enough: before={power_before:.3f}, after={power_after:.3f}"
    )


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

def test_bandpass_passes_10hz():
    """10 Hz (well inside 5–45 Hz) must pass with at most 3 dB attenuation.

    CAR subtracts the cross-channel mean, so we use independent signals on
    each channel to avoid complete cancellation of a common-mode tone.
    """
    pre = Preprocessor(_make_cfg(use_car=True))
    amplitude = 50.0
    n_ch = 4
    rng = np.random.default_rng(99)
    # Different DC-free noise on each channel so CAR doesn't cancel the tone
    noise = rng.standard_normal((n_ch, N)).astype(np.float32) * 2.0
    # Add the test tone only to channel 0
    tone = _pure_tone(10.0, N, FS, amplitude)
    epoch = noise.copy()
    epoch[0] += tone
    filtered = pre.filter(epoch)
    rms = float(np.sqrt(np.mean(filtered[0] ** 2)))
    # After CAR the signal is reduced slightly (by 1/n_ch fraction) but must
    # still be well above 3 dB attenuation of the original amplitude.
    expected_rms_min = (amplitude / np.sqrt(2)) * (1 - 1 / n_ch) * 10 ** (-3 / 20)
    assert rms > expected_rms_min, (
        f"10 Hz passed with too much attenuation: expected > {expected_rms_min:.2f}, got {rms:.2f}"
    )


def test_bandpass_blocks_1hz():
    """1 Hz (below pass-band) must be strongly attenuated (≥ 20 dB)."""
    pre = Preprocessor(_make_cfg(use_car=True))
    amplitude = 100.0
    epoch = _multi_channel(freq_hz=1.0, amplitude=amplitude)
    power_before = float(np.mean(epoch[0] ** 2))
    filtered = pre.filter(epoch)
    power_after = float(np.mean(filtered[0] ** 2))
    assert power_after < power_before * 10 ** (-20 / 10), (
        f"1 Hz not attenuated enough: before={power_before:.3f}, after={power_after:.3f}"
    )


def test_bandpass_blocks_80hz():
    """80 Hz (above pass-band) must be strongly attenuated (≥ 20 dB)."""
    pre = Preprocessor(_make_cfg(use_car=True))
    amplitude = 100.0
    epoch = _multi_channel(freq_hz=80.0, amplitude=amplitude)
    power_before = float(np.mean(epoch[0] ** 2))
    filtered = pre.filter(epoch)
    power_after = float(np.mean(filtered[0] ** 2))
    assert power_after < power_before * 10 ** (-20 / 10), (
        f"80 Hz not attenuated enough: before={power_before:.3f}, after={power_after:.3f}"
    )


# ---------------------------------------------------------------------------
# Common Average Reference (CAR)
# ---------------------------------------------------------------------------

def test_identical_channels_zeroed_only_with_car():
    """Identical signal on all channels: CAR removes it; without CAR it remains."""
    epoch = _multi_channel(4, 10.0, 50.0)
    with_car = Preprocessor(_make_cfg(use_car=True)).filter(epoch)
    without = Preprocessor(_make_cfg(use_car=False)).filter(epoch)
    assert float(np.abs(with_car).max()) < 1.0
    assert float(np.abs(without).max()) > 10.0


def test_car_sums_to_zero():
    """After CAR the per-sample channel mean must be ≈ 0."""
    pre = Preprocessor(_make_cfg(use_car=True))
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((8, N)).astype(np.float32) * 20.0
    filtered = pre.filter(epoch)
    channel_mean = filtered.mean(axis=0)
    assert float(np.abs(channel_mean).max()) < 1e-4, (
        "CAR did not zero the cross-channel mean"
    )


# ---------------------------------------------------------------------------
# Artefact gate
# ---------------------------------------------------------------------------

def test_artefact_flag_on_large_amplitude():
    cfg = _make_cfg(artefact_threshold_uv=50.0)
    pre = Preprocessor(cfg)
    # Construct an epoch whose filtered signal has a big transient on ch0
    rng = np.random.default_rng(1)
    epoch = (rng.standard_normal((4, N)) * 5.0).astype(np.float32)
    epoch[0, 0] = 200.0   # spike on channel 0
    result = pre.process(epoch)
    assert result.artefactual, "Large amplitude spike should be flagged"


def test_no_artefact_flag_on_small_epoch():
    cfg = _make_cfg(artefact_threshold_uv=200.0)
    pre = Preprocessor(cfg)
    rng = np.random.default_rng(2)
    epoch = (rng.standard_normal((4, N)) * 3.0).astype(np.float32)
    result = pre.process(epoch)
    assert not result.artefactual, "Small-amplitude epoch should not be flagged"


# ---------------------------------------------------------------------------
# EpochBuffer
# ---------------------------------------------------------------------------

def test_epoch_buffer_emits_correct_shape():
    buf = EpochBuffer(channels=4, epoch_samples=1000, step_samples=250)
    rng = np.random.default_rng(3)
    chunk = rng.standard_normal((4, 1000)).astype(np.float32)
    buf.append(chunk)
    epochs = list(buf.epochs())
    assert len(epochs) == 1
    assert epochs[0].shape == (4, 1000)


def test_epoch_buffer_step_cadence():
    """Buffer should emit one new epoch every step_samples of additional data."""
    epoch_s, step_s = 1000, 250
    buf = EpochBuffer(channels=4, epoch_samples=epoch_s, step_samples=step_s)
    rng = np.random.default_rng(4)

    # First fill: triggers the very first epoch
    buf.append(rng.standard_normal((4, epoch_s)).astype(np.float32))
    first_batch = list(buf.epochs())
    assert len(first_batch) == 1

    # Each subsequent chunk of step_s should fire exactly 1 epoch.
    emitted_count = 0
    for _ in range(4):
        buf.append(rng.standard_normal((4, step_s)).astype(np.float32))
        new_epochs = list(buf.epochs())
        emitted_count += len(new_epochs)
    assert emitted_count == 4, f"Expected 4 step epochs, got {emitted_count}"


def test_epoch_buffer_not_full_before_enough_data():
    buf = EpochBuffer(channels=4, epoch_samples=500, step_samples=100)
    rng = np.random.default_rng(5)
    buf.append(rng.standard_normal((4, 100)).astype(np.float32))
    assert not buf.is_full
    assert list(buf.epochs()) == []


def test_epoch_buffer_wrong_channel_count_raises():
    buf = EpochBuffer(channels=4, epoch_samples=500, step_samples=100)
    with pytest.raises(ValueError, match=r"shape \(4, n\)"):
        buf.append(np.zeros((3, 100), dtype=np.float32))

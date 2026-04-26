"""Signal-to-noise ratio gate for the Layer 2 pipeline.

Implements the *sum-of-harmonics* SNR estimator commonly used in SSVEP
literature: the signal power is the sum of bin powers at f, 2f, …, n_harm·f
on the gaze channel; the noise power is the average bin power inside a small
window around each harmonic, **excluding** the signal bin itself.

Returned in decibels:

    SNR_dB = 10 · log10(P_sig / P_noise_per_bin)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SnrAggregateMode = Literal["max", "mean", "median"]


def compute_ssvep_snr_db(
    epoch: np.ndarray,
    target_freq_hz: float,
    fs: float,
    channel_idx: int = 0,
    n_harmonics: int = 3,
    noise_band_hz: float = 1.0,
) -> float:
    """Compute SSVEP sum-of-harmonics SNR in dB on a single channel.

    Args:
        epoch: ``(channels, n_samples)`` epoch.
        target_freq_hz: Stimulus frequency to test.
        fs: Sampling rate in Hz.
        channel_idx: Row index in ``epoch`` to analyse (typically Oz).
        n_harmonics: Number of harmonic bins summed for the signal.
        noise_band_hz: Half-width of the noise window around each harmonic.

    Returns:
        SNR in dB.  Returns ``-inf`` if the noise floor is exactly zero.
    """
    if epoch.ndim != 2:
        raise ValueError("epoch must be 2-D (channels, n_samples)")
    if channel_idx < 0 or channel_idx >= epoch.shape[0]:
        raise ValueError(
            f"channel_idx={channel_idx} out of range for "
            f"{epoch.shape[0]} channels"
        )

    x = np.asarray(epoch[channel_idx], dtype=np.float64)
    n = x.size

    # Power spectral density via single-sided FFT
    psd = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    p_sig = 0.0
    p_noise = 0.0
    n_noise_bins = 0

    nyq = fs / 2.0
    for h in range(1, n_harmonics + 1):
        f_h = h * target_freq_hz
        if f_h >= nyq:
            break
        bin_sig = int(np.argmin(np.abs(freqs - f_h)))
        # Noise window: bins within ±noise_band_hz of the harmonic, excluding
        # the signal bin itself.
        window_mask = (np.abs(freqs - f_h) <= noise_band_hz) & (
            np.arange(freqs.size) != bin_sig
        )
        p_sig += float(psd[bin_sig])
        p_noise += float(psd[window_mask].sum())
        n_noise_bins += int(window_mask.sum())

    if n_noise_bins == 0 or p_noise <= 0.0:
        return float("inf") if p_sig > 0 else float("-inf")

    noise_per_bin = p_noise / max(n_noise_bins, 1)
    if noise_per_bin <= 0:
        return float("inf")
    return float(10.0 * np.log10(p_sig / noise_per_bin))


def compute_ssvep_snr_db_aggregate(
    epoch: np.ndarray,
    target_freq_hz: float,
    fs: float,
    channel_indices: list[int],
    mode: SnrAggregateMode,
    n_harmonics: int = 3,
    noise_band_hz: float = 1.0,
) -> float:
    """Per-channel sum-of-harmonics SNR, then aggregate across rows.

    Args:
        epoch: ``(channels, n_samples)``.
        target_freq_hz: Stimulus frequency to test.
        fs: Sampling rate in Hz.
        channel_indices: Row indices (each passed to :func:`compute_ssvep_snr_db`).
        mode: ``max``, ``mean``, or ``median`` across channel SNRs.

    Returns:
        Aggregated SNR in dB.
    """
    if not channel_indices:
        raise ValueError("channel_indices must be non-empty")
    values = [
        compute_ssvep_snr_db(
            epoch,
            target_freq_hz=target_freq_hz,
            fs=fs,
            channel_idx=idx,
            n_harmonics=n_harmonics,
            noise_band_hz=noise_band_hz,
        )
        for idx in channel_indices
    ]
    arr = np.asarray(values, dtype=np.float64)
    if mode == "max":
        return float(np.nanmax(arr))
    if mode == "mean":
        return float(np.nanmean(arr))
    if mode == "median":
        return float(np.median(arr))
    raise ValueError(f"unknown aggregate mode: {mode!r}")

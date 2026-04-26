"""Pure NumPy helpers for synthetic EEG + artifact injection.

Used by ``synthetic_ssvep_source.py`` and unit-tested without pylsl.
Not anatomically faithful — phenomenological stress for Layer 2 only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def apply_adc_clip(data: np.ndarray, clip_uv: float) -> np.ndarray:
    """Hard clip all channels to ``[-clip_uv, +clip_uv]``. No-op if ``clip_uv`` <= 0."""
    if clip_uv <= 0:
        return np.asarray(data, dtype=np.float64, order="C")
    return np.clip(np.asarray(data, dtype=np.float64), -clip_uv, clip_uv)


def blink_waveform(fs: float, duration_s: float) -> np.ndarray:
    """One blink template, length ``round(duration_s * fs)``, peak = 1.0."""
    n = max(3, int(round(duration_s * fs)))
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = 0.5 * (1.0 - np.cos(2.0 * math.pi * t))
    m = float(w.max())
    if m > 1e-12:
        w /= m
    return w


def raised_cosine_blink_profile(n_samples: int, fs: float, duration_s: float) -> np.ndarray:
    """Return a length-``n_samples`` 1-D pulse (centered), peak = 1.0.

    For whole-chunk overlays; prefer :func:`blink_waveform` for streaming blink.
    """
    n_pulse = max(3, int(round(duration_s * fs)))
    n_pulse = min(n_pulse, n_samples)
    if n_pulse < 3:
        return np.zeros(n_samples, dtype=np.float64)
    w = blink_waveform(fs, n_pulse / fs)
    out = np.zeros(n_samples, dtype=np.float64)
    start = (n_samples - n_pulse) // 2
    out[start : start + n_pulse] = w
    m = float(out.max())
    if m > 1e-12:
        out /= m
    return out


def add_blink_occipital(
    data: np.ndarray,
    profile_1d: np.ndarray,
    amplitude_uv: float,
    spatial: np.ndarray,
) -> None:
    """Add blink-shaped waveform: ``data[ch] += spatial[ch] * amplitude * profile``."""
    n_ch, n = data.shape
    prof = np.asarray(profile_1d, dtype=np.float64)
    if prof.size != n:
        raise ValueError(f"profile length {prof.size} != data columns {n}")
    if spatial.shape != (n_ch,):
        raise ValueError("spatial must be (n_channels,)")
    for ch in range(n_ch):
        data[ch] += spatial[ch] * amplitude_uv * prof


def inject_pop_step_decay(
    data: np.ndarray,
    channel: int,
    start: int,
    sign: float,
    step_uv: float,
    fs: float,
    decay_tau_s: float = 0.08,
) -> None:
    """Electrode pop: one-sample step + exponential decay on tail."""
    n = data.shape[1]
    if not (0 <= channel < data.shape[0] and 0 <= start < n):
        return
    s = int(sign)
    if s == 0:
        s = 1
    data[channel, start] += s * step_uv
    tail = n - start - 1
    if tail <= 0:
        return
    t = np.arange(1, tail + 1, dtype=np.float64) / fs
    decay = np.exp(-t / max(decay_tau_s, 1e-4))
    data[channel, start + 1 :] += s * step_uv * 0.35 * decay[: data[channel, start + 1 :].size]


@dataclass
class EogRampState:
    """Phase accumulator for a slow shared sinusoid."""

    phase: float = 0.0

    def advance(self, n_samples: int, fs: float, hz: float) -> None:
        self.phase = (self.phase + 2.0 * math.pi * hz * n_samples / fs) % (2.0 * math.pi)


def eog_ramp_add(
    data: np.ndarray,
    t_sample_start: int,
    fs: float,
    hz: float,
    peak_uv: float,
    spatial: np.ndarray,
    state: EogRampState,
) -> None:
    """Add ``peak_uv * sin(2π f t + φ)`` spatially weighted; updates ``state.phase``."""
    n_ch, n = data.shape
    if spatial.shape != (n_ch,):
        raise ValueError("spatial must be (n_channels,)")
    t = (t_sample_start + np.arange(n, dtype=np.float64)) / fs
    wave = peak_uv * np.sin(2.0 * math.pi * hz * t + state.phase)
    state.advance(n, fs, hz)
    data += spatial[:, np.newaxis] * wave


def occipital_spatial_blink_weights(n_ch: int) -> np.ndarray:
    """Oz, O1, O2, Pz relative weights; rest zero."""
    w = np.zeros(n_ch, dtype=np.float64)
    if n_ch > 0:
        w[0] = 1.0
    if n_ch > 1:
        w[1] = 0.62
    if n_ch > 2:
        w[2] = 0.62
    if n_ch > 3:
        w[3] = 0.38
    return w


def occipital_spatial_eog_weights(n_ch: int) -> np.ndarray:
    """Lateral gradient + small common mode on occipital ring."""
    w = np.zeros(n_ch, dtype=np.float64)
    if n_ch > 0:
        w[0] = 0.35
    if n_ch > 1:
        w[1] = 1.0
    if n_ch > 2:
        w[2] = -0.85
    if n_ch > 3:
        w[3] = 0.25
    # normalize max abs to 1 for predictable peak scaling
    m = float(np.max(np.abs(w))) or 1.0
    w /= m
    return w


def poisson_next_interval(rate_per_s: float, rng: np.random.Generator) -> float:
    if rate_per_s <= 0:
        return float("inf")
    return float(-math.log(max(rng.random(), 1e-12)) / rate_per_s)

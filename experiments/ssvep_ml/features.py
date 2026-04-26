"""Band-power features for SSVEP baseline classifiers."""

from __future__ import annotations

import numpy as np
from scipy import signal


def log_bandpower(
    epoch: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
) -> float:
    """Mean log power in ``band_hz`` over channels (epoch shape ``(C, T)``)."""
    lo, hi = band_hz
    powers: list[float] = []
    nperseg = min(epoch.shape[1], int(fs))
    if nperseg < 8:
        nperseg = epoch.shape[1]
    for c in range(epoch.shape[0]):
        f, pxx = signal.welch(epoch[c], fs=fs, nperseg=nperseg, axis=-1)
        mask = (f >= lo) & (f <= hi)
        if not np.any(mask):
            powers.append(0.0)
        else:
            powers.append(float(np.mean(pxx[mask])))
    return float(np.mean(np.log(np.maximum(powers, 1e-20))))


def bandpower_feature_matrix(
    X: np.ndarray,
    fs: float,
    bands_hz: list[tuple[float, float]],
) -> np.ndarray:
    """``X`` shape ``(n_epochs, n_channels, n_samples)`` → ``(n_epochs, n_bands)``."""
    n = X.shape[0]
    feats = np.empty((n, len(bands_hz)), dtype=np.float64)
    for i in range(n):
        for j, band in enumerate(bands_hz):
            feats[i, j] = log_bandpower(X[i], fs, band)
    return feats
